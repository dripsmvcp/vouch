"""Lessons — typed procedural rules with follow-through tracking (#428).

Two invariants carry the design: an observation about *usage* must never
become an edit to reviewed knowledge, and the repeat guard must warn without
ever blocking or merging. Everything else here is behaviour around those.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vouch import audit
from vouch import lessons as lessons_mod
from vouch.capabilities import capabilities
from vouch.eval import effectiveness
from vouch.jsonl_server import HANDLERS
from vouch.lessons import LessonError
from vouch.models import Claim, ClaimStatus, ClaimType
from vouch.proposals import approve, propose_claim
from vouch.storage import KBStore

RULE = "run mypy before pushing a branch"


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    s = KBStore.init(tmp_path)
    s.config_path.write_text(
        "review:\n  approver_role: trusted-agent\n", encoding="utf-8",
    )
    return s


def _lesson(
    store: KBStore, text: str = RULE, claim_type: str = "lesson"
) -> Claim:
    src = store.put_source(text.encode("utf-8"))
    pr = propose_claim(
        store, text=text, evidence=[src.id], proposed_by="agent",
        claim_type=claim_type,
    )
    return approve(store, pr.proposal.id, approved_by="reviewer")  # type: ignore[return-value]


# --- the typed artifact ----------------------------------------------------


def test_lesson_is_a_claim_type_and_resurfaces_through_retrieval(
    store: KBStore,
) -> None:
    claim = _lesson(store)
    assert claim.type is ClaimType.LESSON
    # a lesson is an ordinary approved claim — it is in list_claims, so every
    # retrieval surface that reads claims already carries it.
    assert claim.id in {c.id for c in store.list_claims()}
    assert [c.id for c in lessons_mod.list_lessons(store)] == [claim.id]


def test_workflow_and_warning_claims_count_as_lessons(store: KBStore) -> None:
    """The procedural vocabulary predates the `lesson` type; existing KBs
    should not have to re-type their rules to get follow-through."""
    workflow = _lesson(store, "never git add -A in this repo", claim_type="workflow")
    warning = _lesson(store, "the release job silently skips arm builds", "warning")
    ids = {c.id for c in lessons_mod.list_lessons(store)}
    assert {workflow.id, warning.id} <= ids


def test_a_plain_fact_is_not_a_lesson(store: KBStore) -> None:
    fact = _lesson(store, "the api listens on port 8080", claim_type="fact")
    assert not lessons_mod.is_lesson(fact)
    assert fact.id not in {c.id for c in lessons_mod.list_lessons(store)}
    with pytest.raises(LessonError, match="only meaningful for a procedural rule"):
        lessons_mod.mark_followed(
            store, claim_id=fact.id, followed=True, actor="agent"
        )


def test_retired_lessons_drop_out_unless_asked_for(store: KBStore) -> None:
    claim = _lesson(store)
    claim.status = ClaimStatus.ARCHIVED
    store.update_claim(claim)
    assert lessons_mod.list_lessons(store) == []
    assert [c.id for c in lessons_mod.list_lessons(store, include_retired=True)] == [
        claim.id
    ]


# --- follow-through: observe, never edit -----------------------------------


def test_mark_followed_appends_an_event_and_edits_nothing(store: KBStore) -> None:
    claim = _lesson(store)
    before = store._claim_path(claim.id).read_text(encoding="utf-8")

    result = lessons_mod.mark_followed(
        store, claim_id=claim.id, followed=True, actor="agent", context="pre-push",
    )

    assert store._claim_path(claim.id).read_text(encoding="utf-8") == before
    assert result["followed"] == 1
    assert result["not_followed"] == 0
    assert result["follow_rate"] == 1.0

    events = [
        e for e in audit.read_events(store.kb_dir)
        if e.event in (lessons_mod.FOLLOWED_EVENT, lessons_mod.NOT_FOLLOWED_EVENT)
    ]
    assert len(events) == 1
    assert events[0].event == lessons_mod.FOLLOWED_EVENT
    assert events[0].object_ids == [claim.id]
    assert events[0].data == {"context": "pre-push"}
    assert events[0].reversible is False


def test_not_followed_is_recorded_as_its_own_verb(store: KBStore) -> None:
    claim = _lesson(store)
    lessons_mod.mark_followed(store, claim_id=claim.id, followed=True, actor="a")
    lessons_mod.mark_followed(store, claim_id=claim.id, followed=False, actor="a")
    lessons_mod.mark_followed(store, claim_id=claim.id, followed=False, actor="a")

    stats = lessons_mod.follow_through(store, claim_id=claim.id)
    assert stats == {
        "claim_id": claim.id,
        "followed": 1,
        "not_followed": 2,
        "observations": 3,
        "follow_rate": round(1 / 3, 4),
    }


def test_follow_through_of_an_unobserved_lesson_is_none_not_zero(
    store: KBStore,
) -> None:
    """A rate of 0.0 would read as "nobody follows it"; the truth is "nobody
    has said." """
    claim = _lesson(store)
    assert lessons_mod.follow_through(store, claim_id=claim.id)["follow_rate"] is None


def test_unknown_claim_is_refused(store: KBStore) -> None:
    with pytest.raises(LessonError, match="unknown claim id"):
        lessons_mod.mark_followed(store, claim_id="nope", followed=True, actor="a")


def test_follow_events_are_consumable_by_effectiveness() -> None:
    """#2's outcome classifier must read the follow-through verbs, or the
    signal this feature produces never reaches the measurement it exists for."""
    assert effectiveness.classify_event(lessons_mod.FOLLOWED_EVENT) == "good"
    assert effectiveness.classify_event(lessons_mod.NOT_FOLLOWED_EVENT) == "bad"


# --- the repeat guard ------------------------------------------------------


def test_proposing_a_near_restatement_warns_loudly_without_blocking(
    store: KBStore,
) -> None:
    existing = _lesson(store, "always run mypy before pushing a branch")
    src = store.put_source(b"reminder about mypy")

    result = propose_claim(
        store,
        text="run mypy before pushing any branch",
        evidence=[src.id],
        proposed_by="agent",
        claim_type="lesson",
    )

    # not blocked — the proposal is filed and pending
    assert store.get_proposal(result.proposal.id) is not None

    repeats = [w for w in result.warnings if w["code"] == "repeat_lesson"]
    assert len(repeats) == 1
    assert repeats[0]["artifact_id"] == existing.id
    assert repeats[0]["overlap"] >= 0.6
    assert existing.id in repeats[0]["message"]


def test_an_unrelated_lesson_does_not_trip_the_guard(store: KBStore) -> None:
    _lesson(store, "always run mypy before pushing a branch")
    src = store.put_source(b"deploy note")
    result = propose_claim(
        store,
        text="deploy from the release tag, never from a topic branch",
        evidence=[src.id],
        proposed_by="agent",
        claim_type="lesson",
    )
    assert not [w for w in result.warnings if w["code"] == "repeat_lesson"]


def test_the_guard_does_not_fire_for_non_lesson_claims(store: KBStore) -> None:
    _lesson(store, "always run mypy before pushing a branch")
    src = store.put_source(b"observation")
    result = propose_claim(
        store,
        text="always run mypy before pushing a branch",
        evidence=[src.id],
        proposed_by="agent",
        claim_type="observation",
    )
    assert not [w for w in result.warnings if w["code"] == "repeat_lesson"]


def test_threshold_is_configurable(store: KBStore) -> None:
    store.config_path.write_text(
        "review:\n  approver_role: trusted-agent\n  lesson_repeat_threshold: 0.95\n",
        encoding="utf-8",
    )
    _lesson(store, "always run mypy before pushing a branch")
    src = store.put_source(b"reminder about mypy")
    result = propose_claim(
        store,
        text="run mypy before pushing any branch",
        evidence=[src.id],
        proposed_by="agent",
        claim_type="lesson",
    )
    assert not [w for w in result.warnings if w["code"] == "repeat_lesson"]
    assert lessons_mod.repeat_threshold(store) == 0.95


def test_overlap_ignores_stopwords_and_case(store: KBStore) -> None:
    assert lessons_mod.overlap("Run mypy before pushing", "run MYPY before a push") > 0.4
    assert lessons_mod.overlap("", "anything") == 0.0
    assert lessons_mod.overlap("the and of", "the and of") == 0.0


def test_the_lessons_module_cannot_reach_the_review_gate() -> None:
    """Follow-through observes and the guard warns; neither may write
    knowledge. The module importing `proposals` at all would be the first
    step toward a parallel write path, so assert it does not."""
    source = Path(lessons_mod.__file__).read_text(encoding="utf-8")
    imports = [
        line for line in source.splitlines()
        if line.startswith(("import ", "from "))
    ]
    assert not [line for line in imports if "proposals" in line], imports
    assert not hasattr(lessons_mod, "approve")


# --- registration ----------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    ["kb.list_lessons", "kb.mark_lesson_followed", "kb.lesson_follow_through"],
)
def test_lesson_methods_registered_on_every_surface(method: str) -> None:
    assert method in set(capabilities().methods)
    assert method in HANDLERS
    from vouch.server import mcp

    assert mcp._tool_manager.get_tool(method.replace(".", "_")) is not None


def test_cli_exposes_the_lesson_commands() -> None:
    from vouch.cli import cli

    assert {
        "lessons", "mark-lesson-followed", "lesson-follow-through",
    } <= set(cli.commands)


# --- the surfaces, exercised rather than merely registered -----------------


def _cli(store: KBStore, args: list[str]):
    from click.testing import CliRunner

    from vouch.cli import cli

    return CliRunner().invoke(cli, args, env={"VOUCH_KB_PATH": str(store.kb_dir)})


def test_cli_lessons_listing_reports_follow_through(store: KBStore) -> None:
    assert "no lessons found" in _cli(store, ["lessons"]).output

    lesson = _lesson(store)
    fresh = _cli(store, ["lessons"])
    assert fresh.exit_code == 0, fresh.output
    assert RULE in fresh.output
    assert "no observations" in fresh.output  # nothing recorded yet

    marked = _cli(store, ["mark-lesson-followed", lesson.id, "--context", "pre-push"])
    assert marked.exit_code == 0, marked.output
    assert "followed 1, not followed 0" in marked.output
    _cli(store, ["mark-lesson-followed", lesson.id, "--not-followed"])

    listed = _cli(store, ["lessons"])
    assert "followed 50% of 2" in listed.output

    stats = _cli(store, ["lesson-follow-through", lesson.id])
    assert stats.exit_code == 0, stats.output
    assert json.loads(stats.output)["observations"] == 2


def test_cli_include_retired_shows_a_superseded_rule(store: KBStore) -> None:
    lesson = _lesson(store)
    stored = store.get_claim(lesson.id)
    stored.status = ClaimStatus.ARCHIVED
    store.update_claim(stored)
    assert "no lessons found" in _cli(store, ["lessons"]).output
    assert RULE in _cli(store, ["lessons", "--include-retired"]).output


def test_cli_marking_an_unknown_lesson_fails_cleanly(store: KBStore) -> None:
    result = _cli(store, ["mark-lesson-followed", "no-such-claim"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_mcp_lesson_tools_round_trip(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vouch import server

    monkeypatch.chdir(store.root)
    lesson = _lesson(store)
    listed = server.kb_list_lessons()
    assert [item["id"] for item in listed["items"]] == [lesson.id]
    assert server.kb_list_lessons(include_retired=True)["items"]

    marked = server.kb_mark_lesson_followed(lesson.id, context="pre-push")
    assert marked["followed"] == 1
    assert server.kb_lesson_follow_through(lesson.id)["observations"] == 1


def test_mcp_marking_an_unknown_lesson_raises_value_error(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The MCP contract: a host sees ValueError, never LessonError.
    from vouch import server

    monkeypatch.chdir(store.root)
    with pytest.raises(ValueError):
        server.kb_mark_lesson_followed("no-such-claim")


# --- config fallbacks ------------------------------------------------------


def test_repeat_threshold_falls_back_on_anything_unusable(store: KBStore) -> None:
    default = lessons_mod.DEFAULT_REPEAT_THRESHOLD
    for text in (
        "review: [unclosed\n",          # unparseable
        "just-a-string\n",              # not a mapping
        "review: not-a-mapping\n",      # review is not a mapping
        "review:\n  approver_role: x\n",  # key absent
        "review:\n  lesson_repeat_threshold: highish\n",  # not a number
    ):
        store.config_path.write_text(text, encoding="utf-8")
        assert lessons_mod.repeat_threshold(store) == default
    store.config_path.write_text(
        "review:\n  lesson_repeat_threshold: 0.9\n", encoding="utf-8"
    )
    assert lessons_mod.repeat_threshold(store) == 0.9


def test_the_repeat_guard_ignores_an_empty_proposal(store: KBStore) -> None:
    _lesson(store)
    assert lessons_mod.repeat_warnings(store, "   ") == []


def test_the_repeat_guard_can_exclude_the_claim_being_edited(store: KBStore) -> None:
    """Re-proposing the same rule as an edit of itself must not warn that it
    duplicates itself."""
    lesson = _lesson(store)
    assert lessons_mod.repeat_warnings(store, RULE)
    assert lessons_mod.repeat_warnings(store, RULE, exclude_claim_id=lesson.id) == []
