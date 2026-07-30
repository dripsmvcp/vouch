"""Passive answer-memory: transcript extraction + capture paths.

Two extraction units, picked by `capture.answer_mode`:

* "session" (default) — `capture_session_answers`, run once from `finalize`,
  cuts claims from the full transcript history; the per-turn Stop hook defers.
* "turn" (legacy) — `capture_answer` files claims from each answer as the
  Stop hook fires.

Both self-approve only what the review gate already allows (trusted-agent or
auto_approve_on_receipt); with neither opt-in the claims stay pending. Both
are idempotent (content-addressed source bytes) and quiet (skip short
acknowledgements), so neither duplicates nor floods the KB.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vouch import capture as cap
from vouch.models import ProposalStatus
from vouch.storage import KBStore

# an answer with three clean, quotable sentences (>160 chars) so segment_source
# yields receipt-verifiable claims.
ANSWER = (
    "Vouch is pivoting from a memory store into a verified knowledge compiler. "
    "The review gate is becoming arithmetic instead of a person. "
    "Passive session capture saves a session answer and recalls it in a fresh session."
)
QUESTION = "what's vouch roadmap?"

# per-turn (legacy) mode, pinned explicitly by the capture_answer tests.
TURN = cap.CaptureConfig(answer_mode="turn")


def _transcript(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _user(text: str, *, meta: bool = False) -> dict:
    row: dict = {"type": "user", "message": _msg("user", text)}
    if meta:
        row["isMeta"] = True
    return row


def _assistant(text: str) -> dict:
    return {"type": "assistant", "message": _msg("assistant", text)}


# `capture.realtime` is off by default (issue #602) and these tests seed the
# buffer directly, so they say so explicitly rather than through config.yaml —
# several of them overwrite it to set a `review:` key.
_REALTIME = cap.CaptureConfig(realtime=True)


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    return KBStore.init(tmp_path / "kb")


def _enable_receipt_gate(store: KBStore) -> None:
    store.config_path.write_text("review:\n  auto_approve_on_receipt: true\n", encoding="utf-8")


def _enable_trusted_agent(store: KBStore) -> None:
    store.config_path.write_text("review:\n  approver_role: trusted-agent\n", encoding="utf-8")


# --- last_exchange -------------------------------------------------------


def test_last_exchange_extracts_question_and_answer(tmp_path: Path) -> None:
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    got = cap.last_exchange(tp)
    assert got is not None
    q, a = got
    assert q == QUESTION
    assert a == ANSWER


def test_last_exchange_skips_meta_and_wrapper_turns(tmp_path: Path) -> None:
    tp = _transcript(tmp_path, [
        _user("<command-name>/compact</command-name>"),
        _user("caveat: local command output"),
        _user(QUESTION, meta=True),   # meta -> ignored
        _user(QUESTION),              # the real question
        _assistant(ANSWER),
    ])
    got = cap.last_exchange(tp)
    assert got is not None
    assert got[0] == QUESTION


def test_last_exchange_pairs_the_latest_answer(tmp_path: Path) -> None:
    tp = _transcript(tmp_path, [
        _user("first question"),
        _assistant("first answer, discarded"),
        _user(QUESTION),
        _assistant(ANSWER),
    ])
    got = cap.last_exchange(tp)
    assert got == (QUESTION, ANSWER)


def test_last_exchange_none_without_assistant(tmp_path: Path) -> None:
    tp = _transcript(tmp_path, [_user(QUESTION)])
    assert cap.last_exchange(tp) is None


def test_last_exchange_missing_file(tmp_path: Path) -> None:
    assert cap.last_exchange(tmp_path / "nope.jsonl") is None


# --- capture_answer ------------------------------------------------------


def test_capture_answer_approves_under_receipt_gate(store: KBStore, tmp_path: Path) -> None:
    _enable_receipt_gate(store)
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    res = cap.capture_answer(store, "sess-1", tp, config=TURN)
    assert res["captured"] is True
    assert res["filed"] >= 3
    assert res["approved"] == res["filed"]
    # no human, and the claims are durable + queryable.
    assert cap.pending_count(store) == 0
    approved = [p for p in store.list_proposals(ProposalStatus.APPROVED)]
    assert len(approved) >= 3
    # the answer's knowledge is now durable and findable by content.
    texts = " ".join(c.text.lower() for c in store.list_claims())
    assert "knowledge compiler" in texts


def test_capture_answer_approves_under_trusted_agent(store: KBStore, tmp_path: Path) -> None:
    _enable_trusted_agent(store)
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    res = cap.capture_answer(store, "sess-1", tp, config=TURN)
    assert res["captured"] is True
    assert res["approved"] == res["filed"] >= 3


def test_capture_answer_leaves_pending_when_gate_off(store: KBStore, tmp_path: Path) -> None:
    # both opt-ins explicitly off: every capture waits for a human.
    store.config_path.write_text(
        "review:\n  auto_approve_on_receipt: false\n", encoding="utf-8"
    )
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    res = cap.capture_answer(store, "sess-1", tp, config=TURN)
    assert res["captured"] is True
    assert res["filed"] >= 3
    assert res["approved"] == 0
    # the review gate is honoured — claims wait for a human.
    pending = [p for p in store.list_proposals(ProposalStatus.PENDING)]
    assert len(pending) >= 3


def test_capture_answer_recapture_leaves_no_pending_duplicates(
    store: KBStore, tmp_path: Path
) -> None:
    # a later answer restating already-durable facts must not pile up pending
    # duplicates -- they are closed mechanically (rejected), fresh facts land.
    tp1 = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    first = cap.capture_answer(store, "sess-1", tp1, config=TURN)
    assert first["approved"] == first["filed"] >= 3
    assert cap.pending_count(store) == 0

    extra = (
        "The observation buffer feeds passive capture across every host "
        "adapter vouch ships."
    )
    tp2 = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER + " " + extra)])
    second = cap.capture_answer(store, "sess-2", tp2, config=TURN)
    assert second["captured"] is True
    # nothing waits for a human: fresh claims approved, restated ones rejected
    # as duplicates of durable claims.
    assert cap.pending_count(store) == 0
    rejected = store.list_proposals(ProposalStatus.REJECTED)
    assert any("duplicate" in (p.decision_reason or "") for p in rejected)


def test_capture_answer_skips_short_answer(store: KBStore, tmp_path: Path) -> None:
    _enable_receipt_gate(store)
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant("done.")])
    res = cap.capture_answer(store, "sess-1", tp, config=TURN)
    assert res["captured"] is False
    assert res["skipped"] == "answer-too-short"


def test_capture_answer_is_idempotent(store: KBStore, tmp_path: Path) -> None:
    _enable_receipt_gate(store)
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    first = cap.capture_answer(store, "sess-1", tp, config=TURN)
    assert first["captured"] is True
    # same answer bytes on a second Stop-hook fire -> skipped, no duplicates.
    second = cap.capture_answer(store, "sess-1", tp, config=TURN)
    assert second["captured"] is False
    assert second["skipped"] == "already-captured"
    assert cap.pending_count(store) == 0


def test_capture_answer_no_answer(store: KBStore, tmp_path: Path) -> None:
    _enable_receipt_gate(store)
    tp = _transcript(tmp_path, [_user(QUESTION)])
    res = cap.capture_answer(store, "sess-1", tp, config=TURN)
    assert res["captured"] is False
    assert res["skipped"] == "no-answer"


def test_capture_answer_disabled_by_env(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_receipt_gate(store)
    monkeypatch.setenv("VOUCH_CAPTURE_DISABLE", "1")
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    res = cap.capture_answer(store, "sess-1", tp, config=TURN)
    assert res["captured"] is False
    assert res["skipped"] == "disabled-env"


# --- session-mode answer memory -------------------------------------------

ANSWER2 = (
    "The observation buffer feeds passive capture across every host adapter "
    "vouch ships. Session-mode extraction cuts claims with the whole "
    "transcript in view instead of one answer at a time."
)


def test_capture_answer_defers_by_default(store: KBStore, tmp_path: Path) -> None:
    # the default answer_mode is "session": the Stop hook stays wired but
    # files nothing — finalize owns claim extraction.
    _enable_receipt_gate(store)
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    res = cap.capture_answer(store, "sess-1", tp)
    assert res["captured"] is False
    assert res["skipped"] == "deferred-to-session-end"
    assert store.list_claims() == []
    assert store.list_proposals(ProposalStatus.PENDING) == []


def test_session_history_collects_all_exchanges(tmp_path: Path) -> None:
    tp = _transcript(tmp_path, [
        _user("first question"),
        _assistant("First answer prose, kept in full."),
        _user(QUESTION),
        _assistant(ANSWER),
    ])
    got = cap.session_history(tp)
    assert got is not None
    questions, doc = got
    assert questions == ["first question", QUESTION]
    assert "First answer prose, kept in full." in doc
    assert "knowledge compiler" in doc
    # questions never enter the document: claims can only quote answers.
    assert QUESTION not in doc


def test_session_history_keeps_turn_final_message(tmp_path: Path) -> None:
    # two assistant rows in one turn (narration between tool calls): the
    # final row is the turn's answer, same rule as last_exchange.
    tp = _transcript(tmp_path, [
        _user(QUESTION),
        _assistant("intermediate narration, superseded"),
        _assistant(ANSWER),
    ])
    got = cap.session_history(tp)
    assert got is not None
    _, doc = got
    assert doc == ANSWER


def test_session_history_none_without_assistant(tmp_path: Path) -> None:
    tp = _transcript(tmp_path, [_user(QUESTION)])
    assert cap.session_history(tp) is None
    assert cap.session_history(tmp_path / "nope.jsonl") is None


def test_session_history_drops_oldest_over_budget(tmp_path: Path) -> None:
    tp = _transcript(tmp_path, [
        _user("q1"), _assistant("old " * 30),
        _user("q2"), _assistant("new " * 30),
    ])
    got = cap.session_history(tp, max_session_chars=150)
    assert got is not None
    questions, doc = got
    # the newest exchange survives; the oldest is dropped first.
    assert "new" in doc and "old" not in doc
    assert questions == ["q2"]


def test_capture_session_answers_files_claims_under_gate(
    store: KBStore, tmp_path: Path
) -> None:
    _enable_receipt_gate(store)
    tp = _transcript(tmp_path, [
        _user(QUESTION), _assistant(ANSWER),
        _user("what feeds capture?"), _assistant(ANSWER2),
    ])
    res = cap.capture_session_answers(store, "sess-1", tp)
    assert res["captured"] is True
    assert res["approved"] == res["filed"] >= 4
    src = store.get_source(res["source"])
    assert src.metadata["questions"] == [QUESTION, "what feeds capture?"]
    assert src.metadata["question"] == QUESTION
    assert "session-history" in src.tags
    # knowledge from BOTH turns is durable from the one session document.
    texts = " ".join(c.text.lower() for c in store.list_claims())
    assert "knowledge compiler" in texts
    assert "observation buffer" in texts


def test_capture_session_answers_leaves_pending_when_gate_off(
    store: KBStore, tmp_path: Path
) -> None:
    store.config_path.write_text(
        "review:\n  auto_approve_on_receipt: false\n", encoding="utf-8"
    )
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    res = cap.capture_session_answers(store, "sess-1", tp)
    assert res["captured"] is True
    assert res["approved"] == 0
    assert len(store.list_proposals(ProposalStatus.PENDING)) >= 3


def test_capture_session_answers_is_idempotent(store: KBStore, tmp_path: Path) -> None:
    _enable_receipt_gate(store)
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    first = cap.capture_session_answers(store, "sess-1", tp)
    assert first["captured"] is True
    # an unchanged transcript re-finalized -> same bytes, skipped.
    second = cap.capture_session_answers(store, "sess-1", tp)
    assert second["captured"] is False
    assert second["skipped"] == "already-captured"
    assert cap.pending_count(store) == 0


def test_finalize_extracts_claims_from_full_history(
    store: KBStore, tmp_path: Path
) -> None:
    _enable_receipt_gate(store)
    tp = _transcript(tmp_path, [
        _user(QUESTION), _assistant(ANSWER),
        _user("what feeds capture?"), _assistant(ANSWER2),
    ])
    res = cap.finalize(store, "sess-1", cwd=None, transcript_path=tp)
    answers = res["answers"]
    assert answers["captured"] is True
    assert answers["filed"] >= 4
    texts = " ".join(c.text.lower() for c in store.list_claims())
    assert "knowledge compiler" in texts and "observation buffer" in texts


def test_finalize_turn_mode_skips_history_extraction(
    store: KBStore, tmp_path: Path
) -> None:
    _enable_receipt_gate(store)
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    res = cap.finalize(store, "sess-1", cwd=None, transcript_path=tp, config=TURN)
    assert "answers" not in res
    assert store.list_claims() == []


def test_load_config_answer_mode(store: KBStore) -> None:
    assert cap.load_config(store).answer_mode == "session"
    store.config_path.write_text("capture:\n  answer_mode: turn\n", encoding="utf-8")
    assert cap.load_config(store).answer_mode == "turn"
    # unknown values fall back to the default rather than half-configuring.
    store.config_path.write_text("capture:\n  answer_mode: bogus\n", encoding="utf-8")
    assert cap.load_config(store).answer_mode == "session"


def test_capture_session_answers_stamps_origin(store: KBStore, tmp_path: Path) -> None:
    _enable_receipt_gate(store)
    origin = tmp_path / "no-kb-project"
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    res = cap.capture_session_answers(store, "sess-1", tp, origin=origin)
    src = store.get_source(res["source"])
    assert src.metadata["origin_path"] == str(origin)
    assert "personal-fallback" in src.tags


def test_finalize_page_cites_session_source(store: KBStore, tmp_path: Path) -> None:
    """The rollup page cites the answers source, so it clears admission."""
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    for i in range(3):
        cap.observe(store, "s1", tool="Edit", summary=f"Edit f{i}.py", now=float(i),
                    config=_REALTIME)
    res = cap.finalize(store, "s1", cwd=None, transcript_path=tp)
    assert res["answers"]["captured"] is True
    src_id = res["answers"]["source"]
    prop = store.get_proposal(res["summary_proposal_id"])
    assert prop.payload["sources"] == [src_id]
    assert prop.status is ProposalStatus.PENDING


def test_finalize_recites_source_on_refinalize(store: KBStore, tmp_path: Path) -> None:
    """already-captured answers still hand the page their source id."""
    tp = _transcript(tmp_path, [_user(QUESTION), _assistant(ANSWER)])
    cap.capture_session_answers(store, "s1", tp)
    for i in range(3):
        cap.observe(store, "s1", tool="Edit", summary=f"Edit f{i}.py", now=float(i),
                    config=_REALTIME)
    res = cap.finalize(store, "s1", cwd=None, transcript_path=tp)
    assert res["answers"]["skipped"] == "already-captured"
    prop = store.get_proposal(res["summary_proposal_id"])
    assert prop.payload["sources"] == [res["answers"]["source"]]
    assert prop.status is ProposalStatus.PENDING


# --- enrichment-driven supersession ----------------------------------------

OLD_ANSWER = (
    "For the record, the staging region is fenora-3 and every deploy targets it. "
    "The rollout script reads that value from the environment file at startup. "
    "Nothing else in the deployment pipeline depends on the region name directly, "
    "so changing it later only means updating that single configuration entry."
)
NEW_ANSWER = (
    "Heads up: the staging region moved to quvasi-8 as of this week. "
    "Everything else about the deploy flow stays exactly the same as before. "
    "The rollout script picked the change up automatically from the environment "
    "file, so no manual intervention was needed anywhere in the pipeline."
)
UPDATE_JSON = (
    '{"summary": "Staging region changed.", "subjects": [], "updates": '
    '[{"attribute": "staging region", "old": "fenora-3", "new": "quvasi-8"}]}'
)


def _enrich_stub(tmp_path: Path, output: str) -> str:
    script = tmp_path / "enrich.sh"
    script.write_text(
        f"#!/bin/sh\ncat > /dev/null\ncat <<'JSON'\n{output}\nJSON\n",
        encoding="utf-8",
    )
    return f"sh {script}"


def test_finalize_supersedes_updated_claims(store: KBStore, tmp_path: Path) -> None:
    from vouch.models import ClaimStatus

    store.config_path.write_text(
        "review:\n  auto_approve_on_receipt: true\n"
        f'capture:\n  enrich:\n    llm_cmd: "{_enrich_stub(tmp_path, UPDATE_JSON)}"\n',
        encoding="utf-8",
    )
    # session 1 states the old value; its claims become durable via receipts
    d1 = tmp_path / "s1"
    d1.mkdir()
    cap.capture_session_answers(
        store, "s1", _transcript(d1, [_user("where is staging?"), _assistant(OLD_ANSWER)])
    )
    old_claims = [c for c in store.list_claims() if "fenora-3" in c.text]
    assert len(old_claims) == 1

    # session 2 states the new value; enrichment flags the change
    d2 = tmp_path / "s2"
    d2.mkdir()
    for i in range(3):
        cap.observe(store, "s2", tool="Edit", summary=f"Edit f{i}.py", now=float(i),
                    config=_REALTIME)
    res = cap.finalize(
        store, "s2", cwd=None,
        transcript_path=_transcript(d2, [_user("region?"), _assistant(NEW_ANSWER)]),
    )
    outcomes = res["superseded"]
    assert [o["applied"] for o in outcomes] == [True]
    old = store.get_claim(outcomes[0]["old_claim"])
    new = store.get_claim(outcomes[0]["new_claim"])
    assert old.status is ClaimStatus.SUPERSEDED
    assert old.superseded_by == new.id
    assert "quvasi-8" in new.text


def test_apply_updates_gate_closed_touches_nothing(store: KBStore) -> None:
    # the starter config opts into the receipt gate; close it explicitly
    store.config_path.write_text(
        "review:\n  auto_approve_on_receipt: false\n", encoding="utf-8"
    )
    out = cap.apply_enrich_updates(
        store, [{"attribute": "a", "old": "x", "new": "y"}]
    )
    assert out == [
        {"attribute": "a", "old": "x", "new": "y",
         "applied": False, "reason": "gate-closed"}
    ]


def test_apply_updates_requires_unique_match(store: KBStore, tmp_path: Path) -> None:
    from vouch.extract import ingest_source

    store.config_path.write_text(
        "review:\n  auto_approve_on_receipt: true\n", encoding="utf-8"
    )
    # "quvasi-8" appears in two separate claims -> ambiguous -> skipped
    ingest_source(
        store,
        b"The region is quvasi-8 for staging deploys. "
        b"Note that quvasi-8 also hosts the preview environment for the team. "
        b"The old region fenora-3 is being retired at the end of the month.",
        proposed_by="test",
    )
    out = cap.apply_enrich_updates(
        store,
        [{"attribute": "staging region", "old": "fenora-3", "new": "quvasi-8"}],
    )
    assert out[0]["applied"] is False
    assert out[0]["reason"] == "no-unique-claim-match"
