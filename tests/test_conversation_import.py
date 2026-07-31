"""Conversation and memory export importers (#431).

The load-bearing invariant is the one the issue names: an importer has no path
to `approve`. Everything else here is the two things that keep an unattended
import from becoming a reviewer's problem — the per-run cap and dedup — plus
the tolerance every reader needs, because an export is someone else's file.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from vouch import conversation_import as ci
from vouch.cli import cli
from vouch.conversation_import import (
    ConversationImportError,
    import_conversations,
    import_memories,
    parse_chat_json,
    parse_memory_export,
)
from vouch.models import ProposalKind, ProposalStatus
from vouch.storage import KBStore

ANSWER_A = (
    "Deploys run every second Tuesday from the release branch, never from main."
)
ANSWER_B = (
    "The staging environment refreshes nightly at 02:00 UTC from a sanitised dump."
)


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    return KBStore.init(tmp_path / "kb")


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def claude_export(tmp_path: Path) -> Path:
    """A claude.ai-shaped export: `chat_messages` with `sender` and blocks."""
    return _write(tmp_path / "claude.json", [
        {
            "uuid": "conv-1",
            "name": "Deploy cadence",
            "created_at": "2026-07-01T09:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "when do we deploy?"},
                {"sender": "assistant", "content": [{"type": "text", "text": ANSWER_A}]},
                {"sender": "human", "text": "and staging?"},
                {"sender": "assistant", "text": ANSWER_B},
            ],
        },
        {
            "uuid": "conv-2",
            "name": "Empty one",
            "chat_messages": [{"sender": "human", "text": "hello?"}],
        },
    ])


# --- the invariant ---------------------------------------------------------


def test_everything_lands_pending_and_nothing_is_approved(
    store: KBStore, claude_export: Path
) -> None:
    report = import_conversations(store, claude_export, max_claims=2)
    assert report["imported"] == 1
    assert all(
        p.status is ProposalStatus.PENDING for p in store.list_proposals(None)
    )
    assert store.list_pages() == []
    assert store.list_claims() == []


def test_the_module_has_no_path_to_approve() -> None:
    source = Path(ci.__file__).read_text(encoding="utf-8")
    assert not hasattr(ci, "approve")
    assert "approve" not in {
        line.split()[-1] for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    }


# --- chat-json -------------------------------------------------------------


def test_claude_shaped_export_pairs_turns(claude_export: Path) -> None:
    conversations = parse_chat_json(claude_export)
    assert [c.conversation_id for c in conversations] == ["conv-1", "conv-2"]
    first = conversations[0]
    assert first.title == "Deploy cadence"
    assert [e.user for e in first.exchanges] == ["when do we deploy?", "and staging?"]
    assert first.exchanges[0].assistant == ANSWER_A
    assert conversations[1].exchanges == []  # a question with no answer


def test_a_conversation_with_no_exchanges_is_skipped(
    store: KBStore, claude_export: Path
) -> None:
    report = import_conversations(store, claude_export)
    skipped = next(r for r in report["rows"] if r["conversation"] == "conv-2")
    assert skipped["reason"] == "no exchanges"


def test_generic_role_content_shape(store: KBStore, tmp_path: Path) -> None:
    export = _write(tmp_path / "gemini.json", {"conversations": [{
        "id": "g-1", "title": "Retries",
        "messages": [
            {"role": "user", "content": "how many retries?"},
            {"role": "model", "content": "Five, with exponential backoff."},
        ],
    }]})
    conversations = parse_chat_json(export)
    assert conversations[0].exchanges[0].assistant == "Five, with exponential backoff."


def test_a_bare_message_list_becomes_one_conversation(tmp_path: Path) -> None:
    export = _write(tmp_path / "session.json", [
        {"role": "user", "content": "what is the retry limit?"},
        {"role": "assistant", "content": "Five."},
    ])
    conversations = parse_chat_json(export)
    assert len(conversations) == 1
    assert conversations[0].conversation_id == "session"


def test_an_openai_mapping_tree_is_delegated(tmp_path: Path) -> None:
    """The branching export shape stays `chatgpt_import`'s job — this reader
    recognises it and hands it over rather than parsing it a second way."""
    export = _write(tmp_path / "conversations.json", [{
        "conversation_id": "o-1", "title": "About deploys",
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["u1"]},
            "u1": {"id": "u1", "parent": "root", "children": ["a1"], "message": {
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["when?"]},
            }},
            "a1": {"id": "a1", "parent": "u1", "children": [], "message": {
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [ANSWER_A]},
            }},
        },
    }])
    conversations = parse_chat_json(export)
    assert conversations[0].exchanges[0].assistant == ANSWER_A


def test_jsonl_and_zip_exports_are_read(tmp_path: Path) -> None:
    jsonl = tmp_path / "chats.jsonl"
    jsonl.write_text("\n".join([
        json.dumps({"id": "j-1", "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": ANSWER_A},
        ]}),
        "{ not json — skipped",
    ]), encoding="utf-8")
    assert parse_chat_json(jsonl)[0].conversation_id == "j-1"

    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("export/chats.json", jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert parse_chat_json(archive)[0].conversation_id == "j-1"


def test_an_unrecognisable_file_is_an_actionable_error(tmp_path: Path) -> None:
    junk = tmp_path / "junk.json"
    junk.write_text("this is not json at all", encoding="utf-8")
    with pytest.raises(ConversationImportError, match="neither JSON nor JSONL"):
        parse_chat_json(junk)

    scalar = _write(tmp_path / "scalar.json", 7)
    with pytest.raises(ConversationImportError, match="does not look like a chat"):
        parse_chat_json(scalar)

    shapeless = _write(tmp_path / "shapeless.json", [{"nothing": "useful"}])
    with pytest.raises(ConversationImportError, match="no conversations found"):
        parse_chat_json(shapeless)


def test_the_page_cites_the_whole_conversation(
    store: KBStore, claude_export: Path
) -> None:
    import_conversations(store, claude_export)
    page = next(
        p for p in store.list_proposals(None) if p.kind == ProposalKind.PAGE
    )
    (source_id,) = page.payload["sources"]
    content = store.read_source_content(source_id).decode("utf-8")
    assert ANSWER_A in content
    assert ANSWER_B in content
    assert store.get_source(source_id).locator == "chat:conv-1"
    assert ANSWER_A in page.payload["body"]


def test_reimport_is_idempotent_and_refreshes_a_grown_conversation(
    store: KBStore, tmp_path: Path
) -> None:
    path = tmp_path / "claude.json"
    _write(path, [{"uuid": "c-1", "name": "T", "chat_messages": [
        {"sender": "human", "text": "q1"}, {"sender": "assistant", "text": ANSWER_A},
    ]}])
    first = import_conversations(store, path, generated_at="2026-07-31T00:00:00Z")
    assert first["imported"] == 1

    unchanged = import_conversations(store, path, generated_at="2026-08-01T00:00:00Z")
    assert unchanged["skipped"] == 1
    assert unchanged["rows"][0]["reason"] == "unchanged"

    _write(path, [{"uuid": "c-1", "name": "T", "chat_messages": [
        {"sender": "human", "text": "q1"}, {"sender": "assistant", "text": ANSWER_A},
        {"sender": "human", "text": "q2"}, {"sender": "assistant", "text": ANSWER_B},
    ]}])
    grown = import_conversations(store, path)
    assert grown["updated"] == 1
    pages = [p for p in store.list_proposals(None) if p.kind == ProposalKind.PAGE]
    assert len(pages) == 1  # refreshed in place, not re-filed
    assert ANSWER_B in pages[0].payload["body"]


def test_a_decided_proposal_blocks_reimport(
    store: KBStore, claude_export: Path
) -> None:
    from vouch.proposals import reject

    import_conversations(store, claude_export)
    page = next(p for p in store.list_proposals(None) if p.kind == ProposalKind.PAGE)
    reject(store, page.id, rejected_by="reviewer-example", reason="not wanted")
    report = import_conversations(store, claude_export)
    row = next(r for r in report["rows"] if r["conversation"] == "conv-1")
    assert row["reason"] == "already-imported"


def test_claims_are_receipt_backed_and_off_by_default(
    store: KBStore, claude_export: Path
) -> None:
    from vouch import receipts

    off = import_conversations(store, claude_export)
    assert off["claims"] == 0
    assert not [p for p in store.list_proposals(None) if p.kind == ProposalKind.CLAIM]

    fresh = KBStore.init(claude_export.parent / "kb2")
    on = import_conversations(fresh, claude_export, max_claims=2)
    assert on["claims"] > 0
    claims = [p for p in fresh.list_proposals(None) if p.kind == ProposalKind.CLAIM]
    assert claims
    for proposal in claims:
        evidence = fresh.get_evidence(proposal.payload["evidence"][0])
        result = receipts.verify_receipt(
            evidence, fresh.read_source_content(evidence.source_id)
        )
        assert result.status is receipts.ReceiptStatus.VERIFIED


# --- memory-export ---------------------------------------------------------


MEMORIES = [
    "The user prefers tabs over spaces in every language.",
    "Deploys are cut from the release branch on alternate Tuesdays.",
]


def test_memory_array_of_records(store: KBStore, tmp_path: Path) -> None:
    export = _write(tmp_path / "memories.json", [
        {"id": "m1", "memory": MEMORIES[0], "tags": ["prefs"]},
        {"id": "m2", "text": MEMORIES[1], "created_at": 1767225600},
    ])
    report = import_memories(store, export)
    assert report["imported"] == 2
    claims = [p for p in store.list_proposals(None) if p.kind == ProposalKind.CLAIM]
    assert {p.payload["text"] for p in claims} == set(MEMORIES)
    assert all(p.status is ProposalStatus.PENDING for p in claims)


def test_memory_claims_quote_their_own_source(store: KBStore, tmp_path: Path) -> None:
    from vouch import receipts

    export = _write(tmp_path / "memories.json", [MEMORIES[0]])
    import_memories(store, export)
    proposal = next(
        p for p in store.list_proposals(None) if p.kind == ProposalKind.CLAIM
    )
    evidence = store.get_evidence(proposal.payload["evidence"][0])
    result = receipts.verify_receipt(
        evidence, store.read_source_content(evidence.source_id)
    )
    assert result.status is receipts.ReceiptStatus.VERIFIED


def test_memory_object_of_records_and_wrapper_key(
    store: KBStore, tmp_path: Path
) -> None:
    wrapped = _write(tmp_path / "wrapped.json", {"memories": [MEMORIES[0]]})
    assert [m.text for m in parse_memory_export(wrapped)] == [MEMORIES[0]]

    keyed = _write(tmp_path / "keyed.json", {
        "m1": MEMORIES[0],
        "m2": {"fact": MEMORIES[1]},
    })
    assert {m.text for m in parse_memory_export(keyed)} == set(MEMORIES)


def test_memory_plain_text_lines(store: KBStore, tmp_path: Path) -> None:
    export = tmp_path / "memories.txt"
    export.write_text("\n".join([*MEMORIES, "", "short"]), encoding="utf-8")
    parsed = parse_memory_export(export)
    assert [m.text for m in parsed] == MEMORIES  # "short" is below the floor


def test_memory_export_drops_repeats_within_one_dump(tmp_path: Path) -> None:
    export = _write(tmp_path / "dupes.json", [MEMORIES[0], MEMORIES[0]])
    assert len(parse_memory_export(export)) == 1


def test_an_empty_memory_export_is_an_actionable_error(tmp_path: Path) -> None:
    empty = _write(tmp_path / "empty.json", [{"unrelated": 1}, "tiny"])
    with pytest.raises(ConversationImportError, match="no memories found"):
        parse_memory_export(empty)
    scalar = _write(tmp_path / "scalar.json", 7)
    with pytest.raises(ConversationImportError, match="expected a list of memories"):
        parse_memory_export(scalar)


# --- the two guards --------------------------------------------------------


def test_dedup_drops_a_memory_an_approved_claim_already_covers(
    store: KBStore, tmp_path: Path
) -> None:
    from vouch.proposals import approve, propose_claim

    src = store.put_source(MEMORIES[0].encode("utf-8"))
    pr = propose_claim(
        store, text=MEMORIES[0], evidence=[src.id], proposed_by="agent-a"
    )
    approve(store, pr.proposal.id, approved_by="human-b")

    export = _write(tmp_path / "memories.json", MEMORIES)
    report = import_memories(store, export)
    assert report["imported"] == 1
    dropped = next(r for r in report["rows"] if r["action"] == "skipped")
    assert dropped["reason"] == "already-known"


def test_dedup_drops_a_memory_a_pending_proposal_already_covers(
    store: KBStore, tmp_path: Path
) -> None:
    export = _write(tmp_path / "memories.json", [MEMORIES[0]])
    assert import_memories(store, export)["imported"] == 1
    assert import_memories(store, export)["skipped"] == 1


def test_dedup_can_be_turned_off(store: KBStore, tmp_path: Path) -> None:
    export = _write(tmp_path / "memories.json", [MEMORIES[0]])
    import_memories(store, export)
    assert import_memories(store, export, dedup=False)["imported"] == 1


def test_dedup_folds_in_embedding_hits_when_the_extra_is_present(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The embedding half of #147 is additive on top of the lexical guard, and
    is stubbed here so the branch runs with or without the extra installed."""
    import sys
    import types

    module = types.ModuleType("vouch.embeddings.similarity")
    module.find_similar_on_propose = lambda store, text: [  # type: ignore[attr-defined]
        {"artifact_id": None},          # ignored: not a string
        {"artifact_id": "semantic-twin"},
    ]
    monkeypatch.setitem(sys.modules, "vouch.embeddings.similarity", module)
    export = _write(tmp_path / "memories.json", [MEMORIES[0]])
    report = import_memories(store, export)
    assert report["imported"] == 0
    assert report["rows"][0]["duplicate_of"] == "semantic-twin"


def test_dedup_still_works_on_a_base_install(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the `[embeddings]` extra the import fails and the lexical guard
    is all there is — which is why it runs first."""
    import sys

    monkeypatch.setitem(sys.modules, "vouch.embeddings.similarity", None)
    export = _write(tmp_path / "memories.json", [MEMORIES[0]])
    assert import_memories(store, export)["imported"] == 1
    assert import_memories(store, export)["skipped"] == 1


def test_max_proposals_caps_a_run_and_reports_it(
    store: KBStore, tmp_path: Path
) -> None:
    export = _write(tmp_path / "memories.json", MEMORIES)
    report = import_memories(store, export, max_proposals=1)
    assert report["imported"] == 1
    assert report["capped"] is True
    assert report["skipped"] == 1
    assert report["rows"][1]["reason"] == "max-proposals"
    assert len(store.list_proposals(ProposalStatus.PENDING)) == 1

    # rerunning continues where it left off — the first is deduped, the
    # second lands. that is what makes a large history importable at all.
    second = import_memories(store, export)
    assert second["imported"] == 1


def test_max_proposals_caps_conversations_too(
    store: KBStore, claude_export: Path
) -> None:
    report = import_conversations(store, claude_export, max_proposals=0)
    assert report["imported"] == 0
    assert report["capped"] is True


def test_an_uncapped_run_reports_no_cap(store: KBStore, tmp_path: Path) -> None:
    export = _write(tmp_path / "memories.json", MEMORIES)
    report = import_memories(store, export)
    assert report["capped"] is False
    assert report["max_proposals"] is None
    assert report["proposals"] == 2


def test_dry_run_reports_without_enqueuing(
    store: KBStore, tmp_path: Path, claude_export: Path
) -> None:
    export = _write(tmp_path / "memories.json", MEMORIES)
    memories = import_memories(store, export, dry_run=True)
    conversations = import_conversations(store, claude_export, dry_run=True)
    assert memories["imported"] == 2
    assert conversations["imported"] == 1
    assert memories["dry_run"] is True
    assert store.list_proposals(None) == []
    assert not list(store.list_sources())


def test_limit_slices_the_export(store: KBStore, tmp_path: Path) -> None:
    export = _write(tmp_path / "memories.json", MEMORIES)
    assert import_memories(store, export, limit=1)["imported"] == 1


def test_an_unquotable_memory_is_dropped_rather_than_filed(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claim whose receipt cannot be located is not knowledge — the importer
    drops it instead of filing a citation that would fail verification."""
    monkeypatch.setattr(ci, "propose_quoted_claim", lambda *a, **k: None)
    export = _write(tmp_path / "memories.json", [MEMORIES[0]])
    report = import_memories(store, export)
    assert report["imported"] == 0
    assert report["rows"][0]["reason"] == "unquotable"
    assert report["proposals"] == 0


def test_overlap_is_zero_without_shared_signal(store: KBStore) -> None:
    # all stopwords / too-short tokens on one side -> nothing to score against
    assert ci.overlap("we do it", MEMORIES[0]) == 0.0
    assert ci.overlap(MEMORIES[0], "") == 0.0
    assert ci.already_known(store, "   ") is None


# --- cli -------------------------------------------------------------------


def _run(store: KBStore, args: list[str]):
    return CliRunner().invoke(cli, args, env={"VOUCH_KB_PATH": str(store.kb_dir)})


def test_cli_chat_json(store: KBStore, claude_export: Path) -> None:
    result = _run(store, ["import", "chat-json", str(claude_export), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["format"] == "chat-json"
    assert report["imported"] == 1


def test_cli_chat_json_human_output(store: KBStore, claude_export: Path) -> None:
    result = _run(
        store, ["import", "chat-json", str(claude_export), "--max-claims", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "imported 1 new" in result.output
    assert "receipt-backed claim(s) proposed" in result.output
    assert "run `vouch review` to decide." in result.output


def test_cli_memory_export_with_a_cap(store: KBStore, tmp_path: Path) -> None:
    export = _write(tmp_path / "memories.json", MEMORIES)
    result = _run(
        store, ["import", "memory-export", str(export), "--max-proposals", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "stopped at --max-proposals 1" in result.output


def test_cli_dry_run_and_no_dedup(store: KBStore, tmp_path: Path) -> None:
    export = _write(tmp_path / "memories.json", MEMORIES)
    dry = _run(store, ["import", "memory-export", str(export), "--dry-run"])
    assert "would import 2 new" in dry.output
    assert store.list_proposals(None) == []

    _run(store, ["import", "memory-export", str(export)])
    again = _run(
        store, ["import", "memory-export", str(export), "--no-dedup", "--json"]
    )
    assert json.loads(again.output)["imported"] == 2


def test_cli_markdown_vault_alias(store: KBStore, tmp_path: Path) -> None:
    folder = tmp_path / "vault"
    folder.mkdir()
    (folder / "note.md").write_text("# Note\n\nbody\n", encoding="utf-8")
    result = _run(store, ["import", "markdown-vault", str(folder), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["imported"] == 1


def test_cli_reports_a_bad_export_cleanly(store: KBStore, tmp_path: Path) -> None:
    junk = tmp_path / "junk.json"
    junk.write_text("not json", encoding="utf-8")
    result = _run(store, ["import", "chat-json", str(junk)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "does not look like" in result.output


def test_cli_import_group_lists_every_format() -> None:
    result = CliRunner().invoke(cli, ["import", "--help"])
    assert result.exit_code == 0
    for name in ("chat-json", "memory-export", "markdown-vault"):
        assert name in result.output


# --- the tolerant paths ----------------------------------------------------
#
# An export is someone else's file, written by a tool whose schema drifts.
# Every branch below is a "skip it and carry on" or an actionable refusal.


def test_an_unreadable_export_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _write(tmp_path / "chats.json", [])

    def boom(_path: object) -> bool:
        raise OSError("permission denied")

    monkeypatch.setattr(ci.zipfile, "is_zipfile", boom)
    with pytest.raises(ConversationImportError, match="cannot read export file"):
        parse_chat_json(export)

    monkeypatch.undo()

    def boom_bytes(self: Path) -> bytes:
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "read_bytes", boom_bytes)
    with pytest.raises(ConversationImportError, match="cannot read export file"):
        parse_chat_json(export)


def test_a_zip_with_no_json_entry_is_actionable(tmp_path: Path) -> None:
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("export/readme.txt", "nothing useful")
    with pytest.raises(ConversationImportError, match=r"holds no \.json export"):
        parse_chat_json(archive)


def test_exports_over_the_byte_ceiling_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ci, "_MAX_EXPORT_BYTES", 4)
    bare = _write(tmp_path / "chats.json", [{"id": "well over four bytes"}])
    with pytest.raises(ConversationImportError, match="too large to import"):
        parse_chat_json(bare)

    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("export/chats.json", "[]  well over four bytes")
    with pytest.raises(ConversationImportError, match="too large to import"):
        parse_chat_json(archive)


def test_clip_truncates_long_text() -> None:
    assert ci._clip("x" * 50, limit=10).endswith("…")


def test_timestamps_accept_seconds_millis_and_iso() -> None:
    assert ci._iso("2026-07-31T00:00:00Z") == "2026-07-31T00:00:00Z"
    assert ci._iso(1767225600) == ci._iso(1767225600000)  # millis normalised
    assert ci._iso(None) is None
    assert ci._iso(True) is None  # a bool is not a timestamp
    assert ci._iso(1e30) is None  # out of range degrades, never raises


def test_message_content_shapes_are_all_read() -> None:
    assert ci._block_text({"text": "nested"}) == "nested"
    assert ci._block_text({"parts": ["a", "b"]}) == ""  # a dict of parts, no text
    assert ci._block_text(7) == ""
    assert ci._turns("not-a-list") == []
    assert ci._turns([7, {"role": "narrator", "content": "x"}]) == []
    # some exports nest the author as an object
    assert ci._turns([{"author": {"role": "user"}, "content": "hi there"}]) == [
        ("user", "hi there")
    ]


def test_entries_that_are_not_objects_are_skipped(tmp_path: Path) -> None:
    export = _write(tmp_path / "chats.json", [7, {"id": "c", "messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": ANSWER_A},
    ]}])
    assert [c.conversation_id for c in parse_chat_json(export)] == ["c"]


def test_memory_entries_that_are_not_records_are_skipped(tmp_path: Path) -> None:
    export = _write(tmp_path / "memories.json", [7, None, MEMORIES[0]])
    assert [m.text for m in parse_memory_export(export)] == [MEMORIES[0]]


def test_page_falls_back_to_the_conversation_id_for_a_title(
    store: KBStore, tmp_path: Path
) -> None:
    export = _write(tmp_path / "chats.json", [{"id": "c-42", "messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": ANSWER_A},
    ]}])
    import_conversations(store, export)
    page = next(p for p in store.list_proposals(None) if p.kind == ProposalKind.PAGE)
    assert page.payload["title"] == "conversation: c-42"


def test_a_long_conversation_says_how_many_exchanges_it_elided(
    store: KBStore, tmp_path: Path
) -> None:
    messages = []
    for i in range(ci._MAX_EXCHANGES_PER_PAGE + 5):
        messages.append({"role": "user", "content": f"question {i}"})
        messages.append({"role": "assistant", "content": f"answer number {i}"})
    export = _write(tmp_path / "long.json", [{
        "id": "c-long", "title": "Long one",
        "updated_at": "2026-07-31T00:00:00Z", "messages": messages,
    }])
    import_conversations(store, export)
    body = next(
        p for p in store.list_proposals(None) if p.kind == ProposalKind.PAGE
    ).payload["body"]
    assert "- last-active: 2026-07-31T00:00:00Z" in body
    assert "more exchange(s) — full conversation in the cited source" in body


def test_limit_slices_a_chat_export(store: KBStore, claude_export: Path) -> None:
    report = import_conversations(store, claude_export, limit=1)
    assert report["conversations"] == 1


def test_claim_extraction_stops_at_each_of_its_three_bounds(
    store: KBStore, tmp_path: Path
) -> None:
    """max_claims, the too-short-answer floor, and the run-wide budget each
    stop the per-conversation claim loop on their own."""
    export = _write(tmp_path / "chats.json", [{
        "id": "c-1", "title": "T",
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "ok"},          # below the floor
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": ANSWER_A},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": ANSWER_B},
        ],
    }])
    capped = import_conversations(store, export, max_claims=1)
    assert capped["claims"] == 1

    # dedup: a *different* conversation repeating an answer already filed adds
    # nothing, even though its page is new
    repeat = _write(tmp_path / "repeat.json", [{
        "id": "c-2", "title": "Asked again",
        "messages": [
            {"role": "user", "content": "remind me?"},
            {"role": "assistant", "content": ANSWER_A},
        ],
    }])
    again = import_conversations(store, repeat, max_claims=2)
    assert again["imported"] == 1
    assert again["claims"] == 0

    fresh = KBStore.init(tmp_path / "kb-budget")
    # one proposal of budget goes to the page, leaving none for the claims
    budgeted = import_conversations(fresh, export, max_claims=2, max_proposals=1)
    assert budgeted["imported"] == 1
    assert budgeted["claims"] == 0
    assert budgeted["capped"] is True
