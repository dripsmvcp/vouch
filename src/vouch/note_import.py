"""Import a note vault into review-gated proposals (issue #612).

People arrive with years of notes already written — obsidian vaults, joplin
archives, apple notes and google keep exports, plain markdown folders — and a
fresh KB has nothing to say to them. ``vouch import obsidian <vault>`` (and its
four siblings) files **one PENDING page proposal per note**, each cited to a
source registered from the note's own bytes.

That last part is the point, and it is what an embedding-based importer
structurally cannot offer: the source content is the note verbatim, so every
claim extracted from it quotes real bytes at real offsets and its receipt
verifies. Imported knowledge is citable, not paraphrased.

Re-importing the same vault is safe. Each note gets a stable identity derived
from the origin's own identifier — the vault-relative path, or joplin's/keep's
own note id where the format carries one — recorded on the source ``locator``
and used as the proposal's session id. A note that changed refreshes its still
PENDING proposal in place; an unchanged one is a flat no-op; a decided proposal
is history and blocks re-import. Nothing duplicates, so a big vault can be
imported in ``--limit`` slices and resumed.

Wikilinks become ``references`` relation proposals, but only where the target
resolves to a note that was actually imported — a link to a note that does not
exist yet is dropped rather than filed as a dangling edge.

Claims are opt-in and bounded. ``--max-claims`` is the density selection knob
(``extract.select_spans``): a ten-thousand-note vault must not turn into ten
thousand pending claims, so the default files pages only.

Never calls ``approve()``. An import is a proposal firehose, not a write — a
human reviews it with ``vouch review`` like everything else.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml

from . import audit
from .extract import extract_receipt_claims
from .models import Proposal, ProposalKind, ProposalStatus
from .proposals import ProposalError, default_scope, propose_page, propose_relation
from .storage import KBStore

# Deliberately NOT one of admission's AUTO_CAPTURE_ACTORS: an import is a human
# choosing to file their own notes, so admission verdicts stay advisory and the
# pages reach review instead of being auto-rejected as capture noise. Mirrors
# `chatgpt_import.CHATGPT_ACTOR`.
NOTE_IMPORT_ACTOR = "note-import"

# Each page is a review surface over exactly one imported note, cited to it.
PAGE_TYPE = "source-summary"

KINDS = ("obsidian", "joplin", "notes", "keep", "md")

# Mirrors the ceilings elsewhere (fetch, codex_rollout, chatgpt_import): one
# oversized archive must not exhaust memory.
_MAX_VAULT_BYTES = 200 * 1024 * 1024
_MAX_NOTE_BYTES = 2 * 1024 * 1024
_MAX_BODY_CHARS = 4_000
_MAX_LINKS_PER_NOTE = 50
_MAX_TITLE_CHARS = 120
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# `[[Target]]`, `[[Target|alias]]`, `[[Target#heading]]`, `[[Target^block]]`.
_WIKILINK_RE = re.compile(r"\[\[([^\]\[|#^]+)(?:[#^][^\]\[|]*)?(?:\|[^\]\[]*)?\]\]")
# Joplin's internal link form: `[title](:/32-hex-id)`.
_JOPLIN_LINK_RE = re.compile(r"\]\(:/([0-9a-fA-F]{32})\)")

_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".mdown", ".txt"})
_NOTES_SUFFIXES = frozenset({".html", ".htm", ".txt", ".md"})
# Directories a vault carries that are machinery, not notes.
_SKIP_DIRS = frozenset(
    {".obsidian", ".trash", ".git", ".stfolder", "_resources", "node_modules"}
)


class NoteImportError(RuntimeError):
    """Raised when a vault can't be read or doesn't parse as one.

    The CLI translates this into a clean ``Error: ...`` line via
    ``_cli_errors``; nothing is written to the KB when it's raised.
    """


@dataclass
class Note:
    """One imported note, normalised across the five source formats."""

    key: str
    """Stable identity from the origin — relative path, or the vault's own id."""

    title: str
    body: str
    raw: bytes
    """The note's own bytes, registered verbatim so receipts verify."""

    locator: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    links: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    media_type: str = "text/markdown"


# --- shared parsing --------------------------------------------------------


def _clip(text: str, limit: int = _MAX_BODY_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _title_from(stem: str, frontmatter: dict[str, Any], body: str) -> str:
    """Frontmatter title, else the first `# heading`, else the filename."""
    raw = frontmatter.get("title")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()[:_MAX_TITLE_CHARS]
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()[:_MAX_TITLE_CHARS] or stem
        if stripped:
            break
    return stem[:_MAX_TITLE_CHARS]


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a leading ``---`` YAML frontmatter block off a note body.

    Tolerant by design: a malformed or non-mapping block is left in the body
    rather than raising, because a vault is other people's files and one bad
    note must not fail an import of ten thousand.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    for idx in range(1, len(lines)):
        if lines[idx].strip() in {"---", "..."}:
            block = "".join(lines[1:idx])
            try:
                loaded = yaml.safe_load(block)
            except yaml.YAMLError:
                return {}, text
            if not isinstance(loaded, dict):
                return {}, text
            return loaded, "".join(lines[idx + 1 :])
    return {}, text


def _tags_from_frontmatter(frontmatter: dict[str, Any]) -> list[str]:
    raw = frontmatter.get("tags") or frontmatter.get("tag")
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(",", " ").split()]
    elif isinstance(raw, list):
        parts = [str(p).strip() for p in raw]
    else:
        return []
    return [p.lstrip("#") for p in parts if p and p.strip("#")][:20]


def _wikilinks(body: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _WIKILINK_RE.finditer(body):
        target = match.group(1).strip()
        if not target or target in seen:
            continue
        seen.add(target)
        out.append(target)
        if len(out) >= _MAX_LINKS_PER_NOTE:
            break
    return out


def _read_text(path: Path) -> str:
    raw = path.read_bytes()[:_MAX_NOTE_BYTES]
    return raw.decode("utf-8", errors="replace")


def _walk_notes(root: Path, suffixes: frozenset[str]) -> list[Path]:
    """Every note file under ``root``, skipping vault machinery, sorted.

    Sorted so an interrupted `--limit` run resumes deterministically instead of
    re-walking in filesystem order and picking a different slice.
    """
    out: list[Path] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        parents = path.relative_to(root).parts[:-1]
        if any(part in _SKIP_DIRS or part.startswith(".") for part in parents):
            continue
        if path.name.startswith("."):
            continue
        try:
            total += path.stat().st_size
        except OSError:
            continue
        if total > _MAX_VAULT_BYTES:
            raise NoteImportError(
                f"vault exceeds the {_MAX_VAULT_BYTES // (1024 * 1024)} MiB import "
                f"ceiling at {path} — import a subfolder at a time"
            )
        out.append(path)
    return out


# --- obsidian / plain markdown ---------------------------------------------


def load_markdown_vault(root: Path, *, kind: str = "obsidian") -> list[Note]:
    """Obsidian vaults and plain markdown folders — the same shape.

    Obsidian's on-disk format *is* a folder of markdown with YAML frontmatter
    and `[[wikilinks]]`; the only difference from a plain folder is the
    `.obsidian/` config directory, which `_walk_notes` skips either way. One
    loader, two front doors.
    """
    if not root.is_dir():
        raise NoteImportError(f"not a directory: {root}")
    notes: list[Note] = []
    for path in _walk_notes(root, _MARKDOWN_SUFFIXES):
        text = _read_text(path)
        frontmatter, body = split_frontmatter(text)
        rel = path.relative_to(root).as_posix()
        key = rel[: -len(path.suffix)] if path.suffix else rel
        stat = path.stat()
        notes.append(
            Note(
                key=key,
                title=_title_from(path.stem, frontmatter, body),
                body=body,
                raw=path.read_bytes()[:_MAX_NOTE_BYTES],
                locator=f"{kind}:{rel}",
                frontmatter=frontmatter,
                links=_wikilinks(body),
                tags=_tags_from_frontmatter(frontmatter),
                created_at=_iso(getattr(stat, "st_birthtime", None) or stat.st_mtime),
                updated_at=_iso(stat.st_mtime),
            )
        )
    if not notes:
        raise NoteImportError(f"no markdown notes found under {root}")
    return notes


def _iso(epoch: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(epoch), tz=UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _iso_ms(millis: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(millis) / 1000.0, tz=UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return None


# --- joplin ----------------------------------------------------------------


def _parse_joplin_note(text: str, fallback_key: str) -> Note | None:
    """One joplin RAW/JEX note: title line, body, then a `key: value` footer.

    Joplin appends its metadata as trailing `key: value` lines after the last
    blank line. `type_: 1` marks a note; folders, tags and resources use other
    type codes and are skipped — importing a folder record as a note is the
    classic mistake with this format.
    """
    lines = text.splitlines()
    meta: dict[str, str] = {}
    cut = len(lines)
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if not line.strip():
            cut = idx
            break
        if ":" not in line:
            return None
        key, _, value = line.partition(":")
        if not key.strip() or " " in key.strip():
            return None
        meta[key.strip()] = value.strip()
    if meta.get("type_") not in {"1", None} or "type_" not in meta:
        return None
    head = lines[:cut]
    title = head[0].strip() if head else ""
    body = "\n".join(head[1:]).strip()
    note_id = meta.get("id") or fallback_key
    return Note(
        key=note_id,
        title=(title or note_id)[:_MAX_TITLE_CHARS],
        body=body,
        raw=text.encode("utf-8"),
        locator=f"joplin:{note_id}",
        frontmatter={
            k: v
            for k, v in meta.items()
            if k in {"id", "parent_id", "source_url", "author", "is_todo"}
        },
        links=list(dict.fromkeys(_JOPLIN_LINK_RE.findall(body)))[:_MAX_LINKS_PER_NOTE],
        created_at=meta.get("user_created_time") or meta.get("created_time"),
        updated_at=meta.get("user_updated_time") or meta.get("updated_time"),
    )


def load_joplin(path: Path) -> list[Note]:
    """A joplin export: a `.jex` tar archive, or the folder it unpacks to."""
    notes: list[Note] = []
    if path.is_file():
        if not tarfile.is_tarfile(path):
            raise NoteImportError(
                f"{path} is not a joplin .jex archive (expected a tar) — pass the "
                f"exported folder instead"
            )
        with tarfile.open(path) as tar:
            total = 0
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".md"):
                    continue
                total += member.size
                if total > _MAX_VAULT_BYTES:
                    raise NoteImportError(
                        f"{path} exceeds the "
                        f"{_MAX_VAULT_BYTES // (1024 * 1024)} MiB import ceiling"
                    )
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                text = handle.read(_MAX_NOTE_BYTES).decode("utf-8", errors="replace")
                note = _parse_joplin_note(text, Path(member.name).stem)
                if note is not None:
                    notes.append(note)
    elif path.is_dir():
        for note_path in _walk_notes(path, frozenset({".md"})):
            note = _parse_joplin_note(_read_text(note_path), note_path.stem)
            if note is not None:
                notes.append(note)
    else:
        raise NoteImportError(f"no such joplin export: {path}")
    if not notes:
        raise NoteImportError(
            f"no joplin notes found in {path} — a joplin export is a folder of .md "
            f"files with a trailing metadata block, or the .jex tar of one"
        )
    notes.sort(key=lambda n: n.key)
    return notes


# --- apple notes -----------------------------------------------------------


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text: apple's exporter emits styled html, not markdown.

    No dependency is added for this — an import must not require a parser the
    rest of vouch has no use for. Block-level tags become newlines; everything
    else contributes its text.
    """

    _BLOCK = frozenset({"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # a malformed note must not fail the import
        return re.sub(r"<[^>]+>", " ", html).strip()
    return parser.text()


def load_apple_notes(root: Path) -> list[Note]:
    """An apple notes export: a folder of .html (or .txt/.md) per note."""
    if not root.is_dir():
        raise NoteImportError(f"not a directory: {root}")
    notes: list[Note] = []
    for path in _walk_notes(root, _NOTES_SUFFIXES):
        raw = path.read_bytes()[:_MAX_NOTE_BYTES]
        text = raw.decode("utf-8", errors="replace")
        body = html_to_text(text) if path.suffix.lower() in {".html", ".htm"} else text
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        notes.append(
            Note(
                key=rel[: -len(path.suffix)] if path.suffix else rel,
                title=_title_from(path.stem, {}, body),
                body=body,
                raw=body.encode("utf-8"),  # the extracted text is what claims quote
                locator=f"notes:{rel}",
                links=_wikilinks(body),
                created_at=_iso(getattr(stat, "st_birthtime", None) or stat.st_mtime),
                updated_at=_iso(stat.st_mtime),
                media_type="text/plain",
            )
        )
    if not notes:
        raise NoteImportError(f"no apple-notes files found under {root}")
    return notes


# --- google keep -----------------------------------------------------------


def _keep_note(payload: dict[str, Any], fallback_key: str) -> Note | None:
    if payload.get("isTrashed"):
        return None
    title = str(payload.get("title") or "").strip()
    text = str(payload.get("textContent") or "").strip()
    checklist = payload.get("listContent")
    if isinstance(checklist, list):
        rows = [
            f"- [{'x' if item.get('isChecked') else ' '}] {item.get('text', '')}"
            for item in checklist
            if isinstance(item, dict)
        ]
        text = "\n".join(filter(None, [text, *rows])).strip()
    if not title and not text:
        return None
    labels = payload.get("labels")
    tags = [
        str(item["name"])
        for item in labels or []
        if isinstance(item, dict) and item.get("name")
    ][:20]
    body = text
    return Note(
        key=fallback_key,
        title=(title or fallback_key)[:_MAX_TITLE_CHARS],
        body=body,
        raw=body.encode("utf-8"),
        locator=f"keep:{fallback_key}",
        frontmatter={
            k: payload[k]
            for k in ("isPinned", "isArchived", "color")
            if k in payload
        },
        tags=tags,
        created_at=_iso_ms(payload.get("createdTimestampUsec", 0) / 1000)
        if isinstance(payload.get("createdTimestampUsec"), (int, float))
        else None,
        updated_at=_iso_ms(payload.get("userEditedTimestampUsec", 0) / 1000)
        if isinstance(payload.get("userEditedTimestampUsec"), (int, float))
        else None,
        media_type="text/plain",
    )


def load_google_keep(path: Path) -> list[Note]:
    """A google takeout keep export: `<title>.json` per note, or the zip."""
    notes: list[Note] = []
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            total = 0
            for info in sorted(zf.infolist(), key=lambda i: i.filename):
                if info.is_dir() or not info.filename.lower().endswith(".json"):
                    continue
                if "/Keep/" not in f"/{info.filename}" and "keep" not in info.filename.lower():
                    continue
                total += info.file_size
                if total > _MAX_VAULT_BYTES:
                    raise NoteImportError(
                        f"{path} exceeds the "
                        f"{_MAX_VAULT_BYTES // (1024 * 1024)} MiB import ceiling"
                    )
                try:
                    payload = json.loads(zf.read(info).decode("utf-8", errors="replace"))
                except (json.JSONDecodeError, OSError):
                    continue
                if isinstance(payload, dict):
                    note = _keep_note(payload, Path(info.filename).stem)
                    if note is not None:
                        notes.append(note)
    elif path.is_dir():
        for json_path in _walk_notes(path, frozenset({".json"})):
            try:
                payload = json.loads(_read_text(json_path))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                note = _keep_note(payload, json_path.stem)
                if note is not None:
                    notes.append(note)
    else:
        raise NoteImportError(f"no such keep export: {path}")
    if not notes:
        raise NoteImportError(
            f"no google keep notes found in {path} — a takeout keep export is a "
            f"folder of one .json per note (or the takeout .zip holding it)"
        )
    notes.sort(key=lambda n: n.key)
    return notes


# --- vault -> proposals ----------------------------------------------------


_LOADERS = {
    "obsidian": lambda p: load_markdown_vault(p, kind="obsidian"),
    "md": lambda p: load_markdown_vault(p, kind="md"),
    "joplin": load_joplin,
    "notes": load_apple_notes,
    "keep": load_google_keep,
}


def load_vault(kind: str, path: Path) -> list[Note]:
    """Parse a vault of ``kind`` into normalised notes. No KB writes."""
    loader = _LOADERS.get(kind)
    if loader is None:
        raise NoteImportError(
            f"unknown vault kind {kind!r} — one of {', '.join(KINDS)}"
        )
    if not path.exists():
        raise NoteImportError(f"no such path: {path}")
    return loader(path)


def session_key(kind: str, note_key: str) -> str:
    """The session id one note dedups on, across every re-import.

    Derived from the origin's own identifier, hashed only to bound the length —
    a deep vault path or a keep title can be arbitrarily long, and the session
    id is an index key, not a display string. The readable identifier survives
    on the source ``locator`` and in the page body.
    """
    digest = hashlib.sha256(f"{kind}:{note_key}".encode()).hexdigest()[:16]
    return f"note-{kind}-{digest}"


def _page_slug(kind: str, note_key: str) -> str:
    slug = _SLUG_RE.sub("-", note_key.lower()).strip("-") or "note"
    return f"{kind}-{slug}"[:80]


def build_page_body(
    note: Note, *, kind: str, generated_at: str | None = None
) -> str:
    """The review surface for one note: provenance, then the note itself."""
    stamp = generated_at or datetime.now(UTC).isoformat()
    lines = [
        f"# {note.title}",
        "",
        f"- imported: {stamp}",
        f"- from: {kind}",
        f"- note: {note.key}",
    ]
    if note.created_at:
        lines.append(f"- created: {note.created_at}")
    if note.updated_at:
        lines.append(f"- updated: {note.updated_at}")
    if note.tags:
        lines.append(f"- tags: {', '.join(note.tags)}")
    if note.links:
        lines.append(f"- links: {', '.join(note.links[:20])}")
    extra = {
        k: v for k, v in note.frontmatter.items() if k not in {"title", "tags", "tag"}
    }
    if extra:
        lines.extend(["", "## frontmatter", ""])
        lines.extend(f"- {k}: {v}" for k, v in sorted(extra.items())[:30])
    lines.extend(["", "## note", "", _clip(note.body) or "(empty note)"])
    return "\n".join(lines).rstrip() + "\n"


def _comparable_body(body: str) -> str:
    """The page body minus its import timestamp, so re-importing an unchanged
    note compares equal across runs."""
    return "\n".join(
        line for line in body.splitlines() if not line.startswith("- imported:")
    )


def _find_existing_page(store: KBStore, session_id: str) -> Proposal | None:
    """The page proposal (any status) already filed for this note.

    Filtered to PAGE kind for the same reason `chatgpt_import` filters: an
    approved import can spawn follow-on proposals under the same session id
    (the wikilink relations, for one), and those must never be mistaken for the
    page when a re-import looks for its dedup target.
    """
    for proposal in store.list_proposals(None):
        if proposal.kind == ProposalKind.PAGE and proposal.session_id == session_id:
            return proposal
    return None


def _resolve_link(target: str, by_key: dict[str, str], by_title: dict[str, str]) -> str | None:
    """A wikilink target -> the imported note's key, or None if unresolvable.

    Obsidian resolves `[[Note]]` by basename anywhere in the vault, and by full
    path when one is given; joplin links by id. Titles are matched
    case-insensitively and only when unambiguous — a link that could mean two
    notes is dropped rather than guessed at.
    """
    if target in by_key:
        return by_key[target]
    stripped = target.removesuffix(".md")
    if stripped in by_key:
        return by_key[stripped]
    return by_title.get(stripped.lower())


def import_vault(
    store: KBStore,
    kind: str,
    path: Path,
    *,
    actor: str | None = None,
    limit: int | None = None,
    max_claims: int = 0,
    dry_run: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Import one note vault into PENDING proposals. Never calls ``approve()``.

    One page proposal per note, deduped on a stable per-note session id: a
    decided proposal blocks re-import, a still-PENDING one refreshes in place
    when the note changed, and an unchanged note is a no-op. ``limit`` bounds
    how many notes are considered, in sorted order, so a large vault imports in
    resumable slices.

    ``max_claims`` > 0 additionally files that many receipt-backed claims per
    note through the density-selection knob. It defaults to 0 — pages only —
    because a ten-thousand-note vault must not become ten thousand pending
    claims on someone's first command.

    With ``dry_run`` nothing is written; the report shows what a real run would
    file.
    """
    notes = load_vault(kind, path)
    if limit is not None and limit >= 0:
        notes = notes[:limit]
    resolved_actor = actor or os.environ.get("VOUCH_AGENT") or NOTE_IMPORT_ACTOR

    rows: list[dict[str, Any]] = []
    counts = {"imported": 0, "updated": 0, "skipped": 0}
    claims_filed = 0
    # note key -> registered source id, for the wikilink pass. Only notes that
    # actually got a source land here, so a link into a `--limit`-truncated
    # slice resolves to nothing and is dropped rather than filed dangling.
    source_by_key: dict[str, str] = {}
    by_key = {note.key: note.key for note in notes}
    titles: dict[str, list[str]] = {}
    for note in notes:
        titles.setdefault(note.title.lower(), []).append(note.key)
    by_title = {title: keys[0] for title, keys in titles.items() if len(keys) == 1}

    for note in notes:
        sid = session_key(kind, note.key)
        row: dict[str, Any] = {"note": note.key, "title": note.title, "session_id": sid}
        existing = _find_existing_page(store, sid)
        if existing is not None and existing.status != ProposalStatus.PENDING:
            row.update(action="skipped", reason="already-imported", proposal_id=existing.id)
            counts["skipped"] += 1
            rows.append(row)
            continue
        body = build_page_body(note, kind=kind, generated_at=generated_at)
        if existing is not None and _comparable_body(body) == _comparable_body(
            str(existing.payload.get("body", ""))
        ):
            row.update(action="skipped", reason="unchanged", proposal_id=existing.id)
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
            note.raw,
            title=note.title,
            locator=note.locator,
            media_type=note.media_type,
            tags=[f"{kind}-import", "note-import"],
            metadata={
                "note_key": note.key,
                "import_kind": kind,
                **({"frontmatter": note.frontmatter} if note.frontmatter else {}),
            },
            scope=default_scope(store),
        )
        source_by_key[note.key] = source.id

        if existing is not None:
            refreshed = existing.model_copy(deep=True)
            refreshed.payload["title"] = note.title
            refreshed.payload["body"] = body
            refreshed.payload["sources"] = [source.id]
            store.update_proposal(refreshed)
            audit.log_event(
                store.kb_dir,
                event="proposal.page.update",
                actor=resolved_actor,
                object_ids=[existing.id],
                data={"reason": f"{kind} re-import", "note": note.key},
            )
            row.update(action="updated", proposal_id=existing.id)
            counts["updated"] += 1
        else:
            proposal = propose_page(
                store,
                title=note.title,
                body=body,
                page_type=PAGE_TYPE,
                source_ids=[source.id],
                proposed_by=resolved_actor,
                tags=[f"{kind}-import", *note.tags],
                session_id=sid,
                slug_hint=_page_slug(kind, note.key),
                rationale=f"imported {kind} note",
            )
            row.update(action="imported", proposal_id=proposal.id)
            counts["imported"] += 1

        if max_claims > 0:
            filed = extract_receipt_claims(
                store, source.id, proposed_by=resolved_actor, max_claims=max_claims,
            )
            row["claims"] = len(filed)
            claims_filed += len(filed)
        rows.append(row)

    relations = 0
    if not dry_run:
        relations = _propose_links(
            store, notes, source_by_key, by_key, by_title, actor=resolved_actor, kind=kind
        )

    return {
        "kind": kind,
        "notes": len(notes),
        "imported": counts["imported"],
        "updated": counts["updated"],
        "skipped": counts["skipped"],
        "claims": claims_filed,
        "relations": relations,
        "dry_run": dry_run,
        "rows": rows,
    }


def _propose_links(
    store: KBStore,
    notes: list[Note],
    source_by_key: dict[str, str],
    by_key: dict[str, str],
    by_title: dict[str, str],
    *,
    actor: str,
    kind: str,
) -> int:
    """File a `references` relation per resolvable wikilink. Returns the count.

    Runs after every source is registered, so a link is judged against the
    whole imported set rather than against whatever happened to come first in
    the walk. Unresolvable targets are dropped: a vault is full of links to
    notes that were never written, and a dangling edge is not knowledge.
    """
    filed = 0
    seen: set[tuple[str, str]] = set()
    for note in notes:
        src_id = source_by_key.get(note.key)
        if src_id is None:
            continue
        for target in note.links:
            target_key = _resolve_link(target, by_key, by_title)
            if target_key is None or target_key == note.key:
                continue
            target_id = source_by_key.get(target_key)
            if target_id is None or (src_id, target_id) in seen:
                continue
            seen.add((src_id, target_id))
            try:
                propose_relation(
                    store,
                    src=src_id,
                    relation="references",
                    target=target_id,
                    proposed_by=actor,
                    rationale=f"{kind} link: {note.key} -> {target_key}",
                    session_id=session_key(kind, note.key),
                )
            except ProposalError:
                # A duplicate or otherwise-rejected edge is not worth failing
                # a ten-thousand-note import over.
                continue
            filed += 1
    return filed
