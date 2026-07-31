"""Import conversation and memory exports as review-gated proposals (#431).

Someone arriving with existing agent history — a claude.ai or gemini or
perplexity chat export, a memory dump from a prior memory tool — starts with an
empty KB and re-teaches everything by hand. That is the biggest friction on
adoption, and it is the gap ``chatgpt_import`` (one vendor) and the note-vault
importers (files, not conversations) leave open.

Two readers, both tolerant:

* ``chat-json`` normalises the common JSON conversation shapes — openai's
  branching ``mapping`` tree (delegated to ``chatgpt_import``), claude.ai's
  ``chat_messages``, and the generic ``messages: [{role, content}]`` an export
  from almost anything else emits — into the same ``Conversation`` /
  ``Exchange`` pair the chatgpt importer already uses. One PENDING page per
  conversation, cited to a per-conversation source.
* ``memory-export`` reads a prior tool's memory dump — a JSON array, a JSON
  object of records, JSONL, or one memory per line — and files each memory as a
  receipt-backed claim that quotes its own source verbatim.

``markdown-vault`` is the third format the issue names; it is the note-vault
importer, registered here as an alias rather than reimplemented.

Three things keep an import from becoming a reviewer's problem:

* **``--max-proposals``** caps a single run and says so in the report, so a
  ten-year history cannot flood the queue in one shot.
* **Dedup** drops a candidate that an approved claim or an already-pending
  proposal covers. Lexical first, because a base install has no ``[embeddings]``
  extra and dedup that silently stops working is how an unattended import
  floods a queue anyway; the embedding hits (#147) fold in on top when
  available.
* **``--dry-run``** reports what a real run would file and touches nothing.

Never calls ``approve()``. Everything lands PENDING, and a human drains the
queue exactly as for any other write.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .chatgpt_import import Conversation, Exchange, _parse_conversation
from .extract import extract_receipt_claims
from .models import Proposal, ProposalKind, ProposalStatus
from .proposals import default_scope, propose_page, propose_quoted_claim
from .storage import KBStore

# Deliberately not one of admission's AUTO_CAPTURE_ACTORS: an import is a human
# choosing to file their own history, so admission verdicts stay advisory and
# the pages reach review instead of being auto-rejected as capture noise.
CONVERSATION_ACTOR = "conversation-import"

PAGE_TYPE = "session"
FORMATS = ("chat-json", "memory-export", "markdown-vault")

# Mirrors the ceiling in `chatgpt_import` and the note-vault importers.
_MAX_EXPORT_BYTES = 200 * 1024 * 1024
_MAX_TURN_CHARS = 2_000
_MAX_EXCHANGES_PER_PAGE = 50
_MAX_TITLE_CHARS = 120
_MAX_MEMORY_CHARS = 2_000
_MIN_MEMORY_CHARS = 20
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_WORD_RE = re.compile(r"[a-z0-9']+")

# Same shape as the stoplist in `contradictions` — words too common to say
# anything about whether two candidates are the same knowledge.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "and", "or", "but", "with", "that",
    "this", "it", "as", "at", "by", "from", "into", "than", "then", "we",
    "you", "i", "they", "he", "she", "our", "your", "my",
})

# Above this token overlap with an approved claim or a pending proposal, a
# candidate is the same knowledge and is dropped rather than filed again.
DEFAULT_DEDUP_THRESHOLD = 0.75

_ROLE_ALIASES = {
    "user": "user", "human": "user", "prompter": "user", "me": "user",
    "assistant": "assistant", "model": "assistant", "ai": "assistant",
    "bot": "assistant", "gpt": "assistant", "claude": "assistant",
}

_MEMORY_KEYS = ("memory", "text", "content", "fact", "value", "note", "body")


class ConversationImportError(RuntimeError):
    """Raised when an export can't be read or doesn't parse as one.

    The CLI turns this into a clean ``Error: ...`` line via ``_cli_errors``;
    nothing is written to the KB when it's raised.
    """


@dataclass
class Memory:
    """One remembered fact from a prior tool's memory dump."""

    text: str
    key: str
    created_at: str | None = None
    tags: list[str] = field(default_factory=list)


# --- shared -----------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {
        w for w in _WORD_RE.findall(text.lower())
        if w not in _STOPWORDS and len(w) > 2
    }


def overlap(a: str, b: str) -> float:
    """Jaccard overlap of two texts' significant tokens, 0.0 to 1.0."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _read_export_bytes(path: Path) -> bytes:
    """The export's bytes, from a bare file or the first JSON entry in a ZIP.

    Size is checked before reading — for a ZIP against the entry's *declared*
    uncompressed size, so a small archive cannot smuggle in a zip-bombed
    payload.
    """
    try:
        is_zip = zipfile.is_zipfile(path)
    except OSError as e:
        raise ConversationImportError(f"cannot read export file {path}: {e}") from e
    if is_zip:
        with zipfile.ZipFile(path) as zf:
            entries = [
                i for i in sorted(zf.infolist(), key=lambda i: i.filename)
                if not i.is_dir() and i.filename.lower().endswith((".json", ".jsonl"))
            ]
            if not entries:
                raise ConversationImportError(
                    f"{path.name} holds no .json export — pass the export file "
                    f"itself if the archive is laid out differently"
                )
            info = entries[0]
            if info.file_size > _MAX_EXPORT_BYTES:
                raise ConversationImportError(
                    f"{info.filename} is too large to import "
                    f"({info.file_size} bytes > {_MAX_EXPORT_BYTES} byte limit)"
                )
            return zf.read(info)
    try:
        if path.stat().st_size > _MAX_EXPORT_BYTES:
            raise ConversationImportError(
                f"{path.name} is too large to import "
                f"(> {_MAX_EXPORT_BYTES} byte limit)"
            )
        return path.read_bytes()
    except OSError as e:
        raise ConversationImportError(f"cannot read export file {path}: {e}") from e


def _load_json(path: Path) -> Any:
    raw = _read_export_bytes(path).decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # JSONL is the other thing every tool emits. One bad line is skipped,
        # not fatal — an export is someone else's file.
        rows = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if rows:
            return rows
        raise ConversationImportError(
            f"{path.name} is neither JSON nor JSONL — this does not look like "
            f"an export"
        ) from None


def _clip(text: str, limit: int = _MAX_TURN_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1].rstrip() + "…"


def _iso(value: Any) -> str | None:
    """An export timestamp (epoch seconds, epoch millis, or ISO) as ISO-8601."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    seconds = float(value)
    if seconds > 1e11:  # milliseconds
        seconds /= 1000.0
    try:
        return datetime.fromtimestamp(seconds, tz=UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


# --- chat-json --------------------------------------------------------------


def _block_text(content: Any) -> str:
    """Message content as text, whether it is a string or typed blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, dict) and str(block.get("text", "")).strip()
        ]
        return "\n\n".join(parts)
    if isinstance(content, dict):
        return _block_text(content.get("text") or content.get("parts") or "")
    return ""


def _turns(raw_messages: Any) -> list[tuple[str, str]]:
    """Normalise a flat message list into ``(role, text)`` pairs."""
    out: list[tuple[str, str]] = []
    if not isinstance(raw_messages, list):
        return out
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        raw_role = message.get("role") or message.get("sender") or message.get("author")
        if isinstance(raw_role, dict):
            raw_role = raw_role.get("role")
        role = _ROLE_ALIASES.get(str(raw_role or "").strip().lower())
        if role is None:
            continue
        text = _block_text(
            message.get("content")
            if message.get("content") is not None
            else message.get("text")
        )
        if text:
            out.append((role, _clip(text)))
    return out


def _pair(turns: list[tuple[str, str]]) -> list[Exchange]:
    """Pair each user turn with the assistant turn that answered it."""
    exchanges: list[Exchange] = []
    pending: str | None = None
    for role, text in turns:
        if role == "user":
            pending = text
        elif pending is not None:
            exchanges.append(Exchange(user=pending, assistant=text))
            pending = None
    return exchanges


def _messages_of(raw: dict[str, Any]) -> Any:
    for key in ("chat_messages", "messages", "turns", "conversation", "history"):
        if isinstance(raw.get(key), list):
            return raw[key]
    return None


def _generic_conversation(raw: Any, index: int) -> Conversation | None:
    if not isinstance(raw, dict):
        return None
    messages = _messages_of(raw)
    if messages is None:
        return None
    conv_id = raw.get("uuid") or raw.get("id") or raw.get("conversation_id")
    conv_id = str(conv_id).strip() if conv_id else f"conversation-{index}"
    title = raw.get("name") or raw.get("title")
    exchanges = _pair(_turns(messages))
    return Conversation(
        conversation_id=conv_id,
        title=str(title).strip()[:_MAX_TITLE_CHARS] if title else None,
        created_at=_iso(raw.get("created_at") or raw.get("create_time")),
        updated_at=_iso(raw.get("updated_at") or raw.get("update_time")),
        exchanges=exchanges,
    )


def parse_chat_json(path: Path) -> list[Conversation]:
    """Parse a JSON conversation export into normalised conversations.

    Handles, in order of specificity: openai's branching ``mapping`` tree (via
    ``chatgpt_import``), claude.ai's ``chat_messages``, the generic
    ``messages: [{role, content}]`` shape, and a bare list of messages with no
    conversation wrapper at all. Entries that match none of them are skipped —
    an export is someone else's file and its schema drifts.
    """
    data = _load_json(path)
    if isinstance(data, dict):
        for key in ("conversations", "chats", "data", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ConversationImportError(
            f"{path.name}: expected a JSON array of conversations (or an object "
            f"wrapping one) — this does not look like a chat export"
        )

    # A bare list of messages, with no conversation wrapper.
    if data and all(
        isinstance(row, dict) and _messages_of(row) is None and "mapping" not in row
        for row in data
    ):
        turns = _turns(data)
        if turns:
            return [Conversation(
                conversation_id=path.stem or "conversation",
                title=path.stem or None,
                exchanges=_pair(turns),
            )]

    out: list[Conversation] = []
    for index, raw in enumerate(data):
        conv: Conversation | None = None
        if isinstance(raw, dict) and isinstance(raw.get("mapping"), dict):
            conv = _parse_conversation(raw)
        if conv is None:
            conv = _generic_conversation(raw, index)
        if conv is not None:
            out.append(conv)
    if not out:
        raise ConversationImportError(
            f"{path.name}: no conversations found — every entry was missing a "
            f"recognisable message list"
        )
    return out


# --- memory-export ----------------------------------------------------------


def _memory_of(raw: Any, index: int) -> Memory | None:
    if isinstance(raw, str):
        text = raw.strip()
        return Memory(text=text, key=f"memory-{index}") if text else None
    if not isinstance(raw, dict):
        return None
    text = ""
    for key in _MEMORY_KEYS:
        candidate = raw.get(key)
        if isinstance(candidate, str) and candidate.strip():
            text = candidate.strip()
            break
    if not text:
        return None
    identifier = raw.get("id") or raw.get("uuid") or raw.get("key")
    raw_tags = raw.get("tags") or raw.get("categories") or raw.get("labels")
    tags = [
        str(t).strip() for t in raw_tags
        if isinstance(raw_tags, list) and str(t).strip()
    ][:20] if isinstance(raw_tags, list) else []
    return Memory(
        text=_clip(text, _MAX_MEMORY_CHARS),
        key=str(identifier).strip() if identifier else f"memory-{index}",
        created_at=_iso(raw.get("created_at") or raw.get("timestamp")),
        tags=tags,
    )


def parse_memory_export(path: Path) -> list[Memory]:
    """Parse a prior tool's memory dump into normalised memories.

    Accepts a JSON array of strings or records, an object wrapping one under a
    ``memories``/``facts``/``items`` key, an object *of* records, JSONL, and —
    when the file is not JSON at all — one memory per non-empty line. Memories
    shorter than ``_MIN_MEMORY_CHARS`` are dropped as acknowledgements rather
    than knowledge.
    """
    try:
        data: Any = _load_json(path)
    except ConversationImportError:
        text = _read_export_bytes(path).decode("utf-8", errors="replace")
        data = [line.strip() for line in text.splitlines() if line.strip()]

    if isinstance(data, dict):
        for key in ("memories", "facts", "items", "data", "records"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [
                {"id": k, **v} if isinstance(v, dict) else {"id": k, "memory": v}
                for k, v in data.items()
            ]
    if not isinstance(data, list):
        raise ConversationImportError(
            f"{path.name}: expected a list of memories — this does not look "
            f"like a memory export"
        )

    out: list[Memory] = []
    seen: set[str] = set()
    for index, raw in enumerate(data):
        memory = _memory_of(raw, index)
        if memory is None or len(memory.text) < _MIN_MEMORY_CHARS:
            continue
        if memory.text in seen:  # the same fact twice in one dump
            continue
        seen.add(memory.text)
        out.append(memory)
    if not out:
        raise ConversationImportError(
            f"{path.name}: no memories found — entries were empty, too short, "
            f"or carried no recognisable text field"
        )
    return out


# --- dedup ------------------------------------------------------------------


def already_known(
    store: KBStore, text: str, *, threshold: float = DEFAULT_DEDUP_THRESHOLD
) -> str | None:
    """The id of an approved claim or pending proposal that already says this.

    Lexical first on purpose: the embedding path (#147) needs the
    ``[embeddings]`` extra, and dedup that silently stops working on a base
    install is precisely how an unattended import floods a review queue. The
    embedding hits fold in on top when the extra is present.
    """
    stripped = text.strip()
    if not stripped:
        return None
    for claim in store.list_claims():
        if overlap(stripped, claim.text) >= threshold:
            return claim.id
    for proposal in store.list_proposals(ProposalStatus.PENDING):
        if proposal.kind is not ProposalKind.CLAIM:
            continue
        existing = str(proposal.payload.get("text", ""))
        if overlap(stripped, existing) >= threshold:
            return proposal.id
    try:
        from .embeddings.similarity import find_similar_on_propose

        for warning in find_similar_on_propose(store, stripped):
            artifact_id = warning.get("artifact_id")
            if isinstance(artifact_id, str):
                return artifact_id
    except ImportError:
        pass
    return None


# --- conversations -> proposals ---------------------------------------------


def session_key(conversation_id: str) -> str:
    """The session id one conversation dedups on, across every re-import."""
    return f"chat-import-{conversation_id}"


def _page_slug(conversation_id: str) -> str:
    slug = _SLUG_RE.sub("-", conversation_id.lower()).strip("-") or "conversation"
    return f"chat-{slug}"[:80]


def build_page_title(conv: Conversation) -> str:
    if conv.title:
        return f"conversation: {conv.title}"[:_MAX_TITLE_CHARS]
    return f"conversation: {conv.conversation_id}"[:_MAX_TITLE_CHARS]


def build_page_body(conv: Conversation, *, generated_at: str | None = None) -> str:
    stamp = generated_at or datetime.now(UTC).isoformat()
    lines = [
        f"# {build_page_title(conv)}",
        "",
        f"- imported: {stamp}",
        f"- conversation: {conv.conversation_id}",
    ]
    if conv.created_at:
        lines.append(f"- started: {conv.created_at}")
    if conv.updated_at:
        lines.append(f"- last-active: {conv.updated_at}")
    lines.append(f"- exchanges: {len(conv.exchanges)}")
    lines.extend(["", "## exchanges", ""])
    shown = conv.exchanges[:_MAX_EXCHANGES_PER_PAGE]
    for exchange in shown:
        lines.extend([f"**you:** {exchange.user}", "", f"**assistant:** {exchange.assistant}", ""])
    hidden = len(conv.exchanges) - len(shown)
    if hidden > 0:
        lines.append(
            f"(… {hidden} more exchange(s) — full conversation in the cited source)"
        )
    return "\n".join(lines).rstrip() + "\n"


def _source_content(conv: Conversation) -> bytes:
    """The conversation's full text as deterministic JSON bytes.

    Deterministic serialization means an unchanged conversation hashes to the
    same source id on every import — ``put_source`` dedups on content, so
    re-imports never pile up copies.
    """
    payload = {
        "conversation_id": conv.conversation_id,
        "title": conv.title,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "exchanges": [
            {"user": e.user, "assistant": e.assistant} for e in conv.exchanges
        ],
    }
    return json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")


def _comparable_body(body: str) -> str:
    return "\n".join(
        line for line in body.splitlines() if not line.startswith("- imported:")
    )


def _find_existing_page(store: KBStore, sid: str) -> Proposal | None:
    for proposal in store.list_proposals(None):
        if proposal.kind == ProposalKind.PAGE and proposal.session_id == sid:
            return proposal
    return None


@dataclass
class _Budget:
    """The `--max-proposals` cap, and whether a run actually hit it."""

    limit: int | None
    spent: int = 0
    hit: bool = False

    def take(self, n: int = 1) -> bool:
        if self.limit is None:
            self.spent += n
            return True
        if self.spent + n > self.limit:
            self.hit = True
            return False
        self.spent += n
        return True


def import_conversations(
    store: KBStore,
    path: Path,
    *,
    actor: str | None = None,
    limit: int | None = None,
    max_proposals: int | None = None,
    max_claims: int = 0,
    dry_run: bool = False,
    dedup: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Import a chat export into PENDING proposals. Never calls ``approve()``.

    One page per conversation, deduped on a stable per-conversation session id:
    a decided proposal blocks re-import, a still-PENDING one refreshes in place
    when the conversation grew, an unchanged one is a no-op. ``max_claims`` > 0
    additionally files that many receipt-backed claims per conversation from
    the assistant's own answers.

    ``max_proposals`` caps the whole run — the report says whether the cap was
    hit, so an operator can see the import was truncated rather than finished.
    """
    conversations = parse_chat_json(path)
    if limit is not None and limit >= 0:
        conversations = conversations[:limit]
    resolved_actor = actor or os.environ.get("VOUCH_AGENT") or CONVERSATION_ACTOR
    budget = _Budget(max_proposals)

    rows: list[dict[str, Any]] = []
    counts = {"imported": 0, "updated": 0, "skipped": 0}
    claims_filed = 0

    for conv in conversations:
        sid = session_key(conv.conversation_id)
        row: dict[str, Any] = {
            "conversation": conv.conversation_id,
            "title": build_page_title(conv),
            "session_id": sid,
        }
        if not conv.exchanges:
            row.update(action="skipped", reason="no exchanges")
            counts["skipped"] += 1
            rows.append(row)
            continue
        existing = _find_existing_page(store, sid)
        if existing is not None and existing.status != ProposalStatus.PENDING:
            row.update(action="skipped", reason="already-imported", proposal_id=existing.id)
            counts["skipped"] += 1
            rows.append(row)
            continue
        body = build_page_body(conv, generated_at=generated_at)
        if existing is not None and _comparable_body(body) == _comparable_body(
            str(existing.payload.get("body", ""))
        ):
            row.update(action="skipped", reason="unchanged", proposal_id=existing.id)
            counts["skipped"] += 1
            rows.append(row)
            continue
        if not budget.take():
            row.update(action="skipped", reason="max-proposals")
            counts["skipped"] += 1
            rows.append(row)
            continue
        if dry_run:
            action = "updated" if existing is not None else "imported"
            row.update(action=action, dry_run=True)
            counts[action] += 1
            rows.append(row)
            continue

        source = store.put_source(
            _source_content(conv),
            title=row["title"],
            locator=f"chat:{conv.conversation_id}",
            media_type="application/json",
            tags=["chat-import", "conversation-import"],
            scope=default_scope(store),
        )
        if existing is not None:
            refreshed = existing.model_copy(deep=True)
            refreshed.payload["title"] = row["title"]
            refreshed.payload["body"] = body
            refreshed.payload["sources"] = [source.id]
            store.update_proposal(refreshed)
            row.update(action="updated", proposal_id=existing.id)
            counts["updated"] += 1
        else:
            proposal = propose_page(
                store,
                title=row["title"],
                body=body,
                page_type=PAGE_TYPE,
                source_ids=[source.id],
                proposed_by=resolved_actor,
                tags=["chat-import"],
                session_id=sid,
                slug_hint=_page_slug(conv.conversation_id),
                rationale="imported conversation export",
            )
            row.update(action="imported", proposal_id=proposal.id)
            counts["imported"] += 1

        if max_claims > 0:
            filed = _claims_from_answers(
                store, conv, actor=resolved_actor, max_claims=max_claims,
                budget=budget, dedup=dedup,
            )
            row["claims"] = filed
            claims_filed += filed
        rows.append(row)

    return {
        "format": "chat-json",
        "conversations": len(conversations),
        "imported": counts["imported"],
        "updated": counts["updated"],
        "skipped": counts["skipped"],
        "claims": claims_filed,
        "proposals": budget.spent,
        "max_proposals": max_proposals,
        "capped": budget.hit,
        "dry_run": dry_run,
        "rows": rows,
    }


def _claims_from_answers(
    store: KBStore,
    conv: Conversation,
    *,
    actor: str,
    max_claims: int,
    budget: _Budget,
    dedup: bool,
) -> int:
    """Receipt-backed claims from the assistant's answers in one conversation.

    The answer is registered as its own source and each quotable span quotes it
    verbatim, so the receipt verifies by construction — the same path session
    answer-memory uses. Claims are what a reader wants out of a chat history;
    the page is the context they came from.
    """
    filed = 0
    for exchange in conv.exchanges:
        if filed >= max_claims:
            break
        answer = exchange.assistant.strip()
        if len(answer) < _MIN_MEMORY_CHARS:
            continue
        if dedup and already_known(store, answer) is not None:
            continue
        if not budget.take():
            break
        source = store.put_source(
            answer.encode("utf-8"),
            title=f"answer from {conv.conversation_id}",
            locator=f"chat:{conv.conversation_id}#answer",
            tags=["chat-import"],
            scope=default_scope(store),
        )
        results = extract_receipt_claims(
            store, source.id, proposed_by=actor,
            max_claims=max_claims - filed, limit=max_claims - filed,
        )
        filed += len(results)
        # `extract_receipt_claims` files its own proposals; charge the budget
        # for what it actually filed rather than the one slot reserved above.
        budget.spent += max(0, len(results) - 1)
    return filed


# --- memories -> proposals --------------------------------------------------


def import_memories(
    store: KBStore,
    path: Path,
    *,
    actor: str | None = None,
    limit: int | None = None,
    max_proposals: int | None = None,
    dry_run: bool = False,
    dedup: bool = True,
) -> dict[str, Any]:
    """Import a memory dump into PENDING claim proposals. No ``approve()``.

    Each memory is registered as its own source and filed as a claim quoting
    that source verbatim, so the receipt verifies by construction and the
    imported fact is citable rather than asserted. A memory an approved claim
    or a pending proposal already covers is dropped.
    """
    memories = parse_memory_export(path)
    if limit is not None and limit >= 0:
        memories = memories[:limit]
    resolved_actor = actor or os.environ.get("VOUCH_AGENT") or CONVERSATION_ACTOR
    budget = _Budget(max_proposals)

    rows: list[dict[str, Any]] = []
    counts = {"imported": 0, "skipped": 0}
    for memory in memories:
        row: dict[str, Any] = {"memory": memory.key, "text": _clip(memory.text, 120)}
        duplicate = already_known(store, memory.text) if dedup else None
        if duplicate is not None:
            row.update(action="skipped", reason="already-known", duplicate_of=duplicate)
            counts["skipped"] += 1
            rows.append(row)
            continue
        if not budget.take():
            row.update(action="skipped", reason="max-proposals")
            counts["skipped"] += 1
            rows.append(row)
            continue
        if dry_run:
            row.update(action="imported", dry_run=True)
            counts["imported"] += 1
            rows.append(row)
            continue
        source = store.put_source(
            memory.text.encode("utf-8"),
            title=f"memory {memory.key}",
            locator=f"memory:{memory.key}",
            tags=["memory-import", *memory.tags],
            metadata={"memory_key": memory.key, "created_at": memory.created_at},
            scope=default_scope(store),
        )
        result = propose_quoted_claim(
            store, text=memory.text, source_id=source.id, quote=memory.text,
            proposed_by=resolved_actor,
            rationale="imported memory export",
        )
        if result is None:
            # The verbatim check failed (e.g. bytes mangled on decode) — the
            # claim is dropped rather than filed without a working receipt.
            row.update(action="skipped", reason="unquotable")
            counts["skipped"] += 1
            budget.spent -= 1
            rows.append(row)
            continue
        row.update(action="imported", proposal_id=result.proposal.id)
        counts["imported"] += 1
        rows.append(row)

    return {
        "format": "memory-export",
        "memories": len(memories),
        "imported": counts["imported"],
        "skipped": counts["skipped"],
        "proposals": budget.spent,
        "max_proposals": max_proposals,
        "capped": budget.hit,
        "dry_run": dry_run,
        "rows": rows,
    }
