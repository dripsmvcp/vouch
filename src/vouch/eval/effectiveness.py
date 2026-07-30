"""Per-artifact effectiveness: is this claim earning its keep? (#426)

``eval/recall.py`` scores whether the right artifact *can* be retrieved. This
scores the layer above it: given that an artifact was surfaced into a session,
how did that session end? A claim can be perfectly retrievable and still be
dead weight, or actively misleading, and today a reviewer deciding what to
expire has nothing to go on.

The association is built from two things already on disk:

* ``retrieval_events`` — the append-only log of what each context pack
  contained. This is the surfacing record; no new table is introduced. The
  issue proposed a derived ``index_db`` table, but that would be a second
  copy of data this log already holds, and a parallel data path is the one
  thing the architecture rules out.
* ``audit.read_events`` — the authoritative mutation stream, which is what
  classifies a session's outcome.

Sessions, not artifacts, carry the outcome: ``Session.started_at`` /
``ended_at`` bound a window, every audit event inside it votes, and every
artifact surfaced inside it inherits the verdict. Classifying sessions rather
than artifacts also avoids a trap — ``claim.supersede`` records
``[old, new, relation]``, so per-artifact attribution would score the winning
claim as a bad outcome alongside the one it replaced.

Read-only and measurement-only: nothing here writes an artifact, an audit
event, or a proposal, and a bad verdict never expires anything. The human at
the gate decides; this only ranks what to look at.

Honesty about what this is: an *associational* signal, not a causal one. An
artifact surfaced into sessions that went badly may be the cause, a symptom,
or a bystander. That is why verdicts are power-gated — ``useful`` and
``harmful`` are withheld until a 95% Wilson interval clears the corpus
baseline and the sample meets ``min_samples``, and everything else reports
``unverified`` or ``insufficient`` rather than a confident number.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .. import audit as audit_mod
from .. import retrieval_events

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..models import Session
    from ..storage import KBStore

DEFAULT_MIN_SAMPLES = 5
DEFAULT_WINDOW = "90d"
Z_95 = 1.959963984540054

# Audit verbs that end a session well: the reviewer kept what they saw.
# `lesson.followed` (#428) is the most direct good outcome there is — a rule
# the KB surfaced actually changed what happened in the session.
GOOD_EVENTS = frozenset({"claim.confirm", "lesson.followed"})

# Verbs that end it badly: something surfaced turned out to be wrong, stale,
# or unwanted. A lesson explicitly recorded as *not* followed is the same
# signal from the other side: it was surfaced and it did not earn its place.
BAD_EVENTS = frozenset({
    "claim.contradict",
    "claim.supersede",
    "claim.archive",
    "claim.redact",
    "lesson.not_followed",
})

# Proposal decisions carry the kind in the middle of the verb
# (`proposal.claim.approve`, `proposal.page.reject`, `proposal.entity.*`), so
# they are matched by prefix + suffix. Both sides must be matched the same way:
# listing only `proposal.claim.approve` as good while every kind's reject
# counted as bad dropped page/entity/relation approvals, which both undercounts
# scored sessions and skews the baseline downward.
_PROPOSAL_PREFIX = "proposal."
_APPROVE_SUFFIX = ".approve"
_REJECT_SUFFIX = ".reject"

VERDICT_USEFUL = "useful"
VERDICT_HARMFUL = "harmful"
VERDICT_UNVERIFIED = "unverified"
VERDICT_INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class ArtifactEffect:
    """One artifact's association between being surfaced and session outcome."""

    artifact_id: str
    kind: str
    good: int
    bad: int
    rate: float
    ci_low: float
    ci_high: float
    lift: float
    verdict: str

    @property
    def samples(self) -> int:
        return self.good + self.bad

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.artifact_id,
            "kind": self.kind,
            "samples": self.samples,
            "good": self.good,
            "bad": self.bad,
            "rate": round(self.rate, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "lift": round(self.lift, 4),
            "verdict": self.verdict,
        }


def wilson_interval(successes: int, n: int, *, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because the samples here are
    small and the rates run near 0 and 1, exactly where the normal interval
    produces bounds outside [0, 1] and badly wrong coverage.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def classify_event(event: str) -> str | None:
    """``"good"`` / ``"bad"`` / ``None`` for one audit verb."""
    if event in GOOD_EVENTS:
        return "good"
    if event in BAD_EVENTS:
        return "bad"
    if event.startswith(_PROPOSAL_PREFIX):
        if event.endswith(_APPROVE_SUFFIX):
            return "good"
        if event.endswith(_REJECT_SUFFIX):
            return "bad"
    return None


def verdict_timeline(store: KBStore) -> list[tuple[datetime, str]]:
    """Every verdict-bearing audit event as ``(when, "good"|"bad")``, in order.

    Read once and reused for every session. Scanning the whole audit log inside
    ``session_outcome`` made ``compute`` O(sessions x audit_events), which is
    the wrong shape for the exact KBs this is meant to be useful on.
    """
    timeline: list[tuple[datetime, str]] = []
    for event in audit_mod.read_events(store.kb_dir):
        verdict = classify_event(event.event)
        if verdict is not None:
            timeline.append((_as_utc(event.created_at), verdict))
    timeline.sort(key=lambda row: row[0])
    return timeline


def session_outcome(
    store: KBStore,
    session: Session,
    *,
    timeline: list[tuple[datetime, str]] | None = None,
) -> str | None:
    """Classify one session from the audit events inside its window.

    ``None`` when the window holds no verdict-bearing event, or holds both —
    a session that confirmed one claim and rejected another says nothing
    clean about any single artifact in it, so it is dropped rather than
    guessed at.

    ``timeline`` is the pre-read output of :func:`verdict_timeline`; it is read
    on demand when omitted so a single-session caller stays a one-liner.
    """
    end = session.ended_at
    if end is None:
        return None
    rows = verdict_timeline(store) if timeline is None else timeline
    start_ts = _as_utc(session.started_at)
    end_ts = _as_utc(end)
    good = bad = 0
    # bisect the sorted timeline instead of rescanning it: only the slice
    # inside the session window can contribute a verdict.
    lo = bisect_left(rows, start_ts, key=lambda row: row[0])
    for when, verdict in rows[lo:]:
        if when > end_ts:
            break
        if verdict == "good":
            good += 1
        else:
            bad += 1
    if good and bad:
        return None
    if good:
        return "good"
    if bad:
        return "bad"
    return None


def _as_utc(value: datetime) -> datetime:
    """Naive timestamps are UTC — comparing them to aware ones would raise."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _surfaced_in(
    events: list[dict[str, Any]], start: datetime, end: datetime
) -> set[tuple[str, str]]:
    """``(kind, id)`` surfaced by retrievals inside a session window."""
    out: set[tuple[str, str]] = set()
    for event in events:
        raw = event.get("ts")
        if not isinstance(raw, str):
            continue
        try:
            ts = _as_utc(datetime.fromisoformat(raw))
        except ValueError:
            continue
        if ts < start or ts > end:
            continue
        for item in event.get("items", []):
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type", ""))
            artifact_id = str(item.get("id", ""))
            # both halves must be present: an item with an id but no type
            # renders as "/<id>" in the report, which is not a usable pointer.
            if kind and artifact_id:
                out.add((kind, artifact_id))
    return out


def _verdict(
    good: int, bad: int, ci_low: float, ci_high: float, baseline: float, min_samples: int
) -> str:
    """Power-gated verdict: confident labels only when the interval earns one."""
    if good + bad < min_samples:
        return VERDICT_INSUFFICIENT
    if ci_low > baseline:
        return VERDICT_USEFUL
    if ci_high < baseline:
        return VERDICT_HARMFUL
    return VERDICT_UNVERIFIED


def compute(
    store: KBStore,
    *,
    since: datetime | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict[str, Any]:
    """Rank approved artifacts by the outcome of the sessions they entered.

    ``since`` bounds the sessions considered (``None`` = all time). Returns
    ``{"baseline", "sessions_scored", "min_samples", "artifacts": [...]}``
    with artifacts ordered worst-lift first — the reviewer's work queue is the
    dead weight, not the winners.
    """
    events = retrieval_events.read_events(store)
    timeline = verdict_timeline(store)
    good_by_artifact: dict[tuple[str, str], int] = {}
    bad_by_artifact: dict[tuple[str, str], int] = {}
    sessions_scored = 0
    total_good = 0

    for session in store.list_sessions():
        if session.ended_at is None:
            continue
        start = _as_utc(session.started_at)
        end = _as_utc(session.ended_at)
        if since is not None and end < _as_utc(since):
            continue
        outcome = session_outcome(store, session, timeline=timeline)
        if outcome is None:
            continue
        sessions_scored += 1
        total_good += 1 if outcome == "good" else 0
        bucket = good_by_artifact if outcome == "good" else bad_by_artifact
        for key in _surfaced_in(events, start, end):
            bucket[key] = bucket.get(key, 0) + 1

    # The corpus rate every artifact is compared against. With no scored
    # sessions there is nothing to compare to, so every verdict degrades to
    # insufficient rather than to a comparison against a made-up baseline.
    baseline = total_good / sessions_scored if sessions_scored else 0.0

    artifacts: list[ArtifactEffect] = []
    for key in sorted(set(good_by_artifact) | set(bad_by_artifact)):
        good = good_by_artifact.get(key, 0)
        bad = bad_by_artifact.get(key, 0)
        n = good + bad
        rate = good / n if n else 0.0
        ci_low, ci_high = wilson_interval(good, n)
        artifacts.append(ArtifactEffect(
            artifact_id=key[1],
            kind=key[0],
            good=good,
            bad=bad,
            rate=rate,
            ci_low=ci_low,
            ci_high=ci_high,
            lift=rate - baseline,
            verdict=_verdict(good, bad, ci_low, ci_high, baseline, min_samples),
        ))

    artifacts.sort(key=lambda a: (a.lift, -a.samples, a.artifact_id))
    return {
        "baseline": round(baseline, 4),
        "sessions_scored": sessions_scored,
        "min_samples": min_samples,
        "artifacts": [a.to_dict() for a in artifacts],
    }


def format_report(result: dict[str, Any]) -> str:
    """Human-readable table, worst lift first."""
    lines = [
        f"sessions scored: {result['sessions_scored']}   "
        f"baseline good-rate: {result['baseline']:.2f}   "
        f"min samples: {result['min_samples']}",
    ]
    artifacts = result["artifacts"]
    if not artifacts:
        lines.append("no artifact was surfaced into a session with a clean outcome.")
        return "\n".join(lines)
    lines.append(
        f"{'verdict':<13} {'lift':>7} {'rate':>6} {'n':>4}  "
        f"{'95% ci':<15} artifact"
    )
    for a in artifacts:
        ci = f"[{a['ci_low']:.2f}, {a['ci_high']:.2f}]"
        lines.append(
            f"{a['verdict']:<13} {a['lift']:>+7.2f} {a['rate']:>6.2f} "
            f"{a['samples']:>4}  {ci:<15} {a['kind']}/{a['id']}"
        )
    return "\n".join(lines)
