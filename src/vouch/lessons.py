"""Lessons — procedural rules that have to earn their place.

`ClaimType` already carried `workflow` and `warning`, the vocabulary of a
rule like "run mypy before pushing" or "never `git add -A` here". What was
missing is the loop that keeps such a rule honest: nothing recorded whether a
surfaced rule was actually followed, so a load-bearing convention and a stale
one nobody heeds looked identical, and nothing warned at propose time when a
near-identical rule already existed, so the KB accumulated restatements.

Two additions, both deliberately narrow:

* **follow-through** — `mark_followed` appends an *observation* to the audit
  log. It never edits the lesson: not its text, not its status, not its
  confidence. The worst case of a lost or replayed observation is a noisier
  effectiveness estimate, never corrupted knowledge.
* **a repeat guard** — `repeat_warnings` surfaces an existing approved lesson
  loudly at propose time. It only warns; it never blocks and never merges.
  The reviewer decides.

Neither path can approve anything; there is no import of `proposals.approve`
here and there must never be one.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from . import audit
from .models import Claim, ClaimStatus, ClaimType
from .scoping import ViewerContext, is_visible, viewer_from
from .storage import ArtifactNotFoundError, KBStore

# The claim types that carry a procedural rule. `lesson` is the explicit one;
# `workflow` and `warning` predate it and say the same kind of thing, so
# follow-through and the repeat guard apply to them too rather than asking
# every existing KB to re-type its rules.
LESSON_TYPES: frozenset[ClaimType] = frozenset({
    ClaimType.LESSON,
    ClaimType.WORKFLOW,
    ClaimType.WARNING,
})

FOLLOWED_EVENT = "lesson.followed"
NOT_FOLLOWED_EVENT = "lesson.not_followed"

# Jaccard overlap over normalized tokens at or above which two lessons are
# "strongly overlapping". Lexical on purpose: the embedding path needs the
# [embeddings] extra, and a repeat guard that silently disappears on a base
# install is worse than a blunt one that always runs.
DEFAULT_REPEAT_THRESHOLD = 0.6

MAX_REPEAT_WARNINGS = 5

# Retired claims are not rules anyone is being asked to follow.
_RETIRED = frozenset({
    ClaimStatus.SUPERSEDED,
    ClaimStatus.ARCHIVED,
    ClaimStatus.REDACTED,
})

# Words too common to carry any signal about whether two rules are the same.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "before", "but", "by", "do",
    "dont", "for", "from", "in", "is", "it", "must", "never", "not", "of",
    "on", "or", "should", "that", "the", "then", "this", "to", "use", "we",
    "when", "with", "you",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class LessonError(RuntimeError):
    pass


def is_lesson(claim: Claim) -> bool:
    return claim.type in LESSON_TYPES


def repeat_threshold(store: KBStore) -> float:
    """`review.lesson_repeat_threshold` from config.yaml, else the default."""
    try:
        loaded = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return DEFAULT_REPEAT_THRESHOLD
    if not isinstance(loaded, dict):
        return DEFAULT_REPEAT_THRESHOLD
    review = loaded.get("review")
    if not isinstance(review, dict) or review.get("lesson_repeat_threshold") is None:
        return DEFAULT_REPEAT_THRESHOLD
    try:
        return float(review["lesson_repeat_threshold"])
    except (TypeError, ValueError):
        return DEFAULT_REPEAT_THRESHOLD


def _tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) > 2
    }


def overlap(a: str, b: str) -> float:
    """Jaccard overlap of the two texts' significant tokens, 0.0 to 1.0."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def list_lessons(
    store: KBStore,
    *,
    viewer: ViewerContext | None = None,
    include_retired: bool = False,
) -> list[Claim]:
    """Live approved lessons, viewer-scoped, newest first."""
    if viewer is None:
        viewer = viewer_from(config_path=store.config_path)
    found = [
        c for c in store.list_claims()
        if is_lesson(c)
        and (include_retired or c.status not in _RETIRED)
        and is_visible(c.scope, viewer)
    ]
    found.sort(key=lambda c: c.created_at, reverse=True)
    return found


def repeat_warnings(
    store: KBStore,
    text: str,
    *,
    exclude_claim_id: str | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Approved lessons that strongly overlap `text`, loudest first.

    Advisory only. The caller attaches these to the propose response so the
    reviewer sees the existing rule and can merge instead of duplicating;
    nothing here blocks the proposal or edits either claim.
    """
    if not text.strip():
        return []
    thresh = repeat_threshold(store) if threshold is None else threshold
    hits: list[dict[str, Any]] = []
    for existing in list_lessons(store):
        if existing.id == exclude_claim_id:
            continue
        score = overlap(text, existing.text)
        if score < thresh:
            continue
        hits.append({
            "code": "repeat_lesson",
            "artifact_kind": "claim",
            "artifact_id": existing.id,
            "overlap": round(score, 4),
            "claim_type": existing.type.value,
            "snippet": (
                existing.text if len(existing.text) <= 200
                else existing.text[:197] + "..."
            ),
            "message": (
                f"an approved lesson already says this ({score:.0%} overlap): "
                f"{existing.id}. merge into it or supersede it rather than "
                "filing a near-restatement."
            ),
        })
    hits.sort(key=lambda h: h["overlap"], reverse=True)
    return hits[:MAX_REPEAT_WARNINGS]


def mark_followed(
    store: KBStore,
    *,
    claim_id: str,
    followed: bool,
    actor: str,
    context: str | None = None,
) -> dict[str, Any]:
    """Record that a surfaced lesson was (or wasn't) applied this turn.

    Appends one append-only observation to `audit.log.jsonl` and returns the
    lesson's running follow-through. It does **not** touch the claim — an
    observation about usage is not an edit to reviewed knowledge, so this has
    no path to the review gate and needs none.
    """
    try:
        claim = store.get_claim(claim_id)
    except ArtifactNotFoundError as e:
        raise LessonError(f"unknown claim id: {claim_id}") from e
    if not is_lesson(claim):
        raise LessonError(
            f"claim {claim_id} is type {claim.type.value!r}; follow-through is "
            f"only meaningful for a procedural rule "
            f"({', '.join(sorted(t.value for t in LESSON_TYPES))})"
        )
    audit.log_event(
        store.kb_dir,
        event=FOLLOWED_EVENT if followed else NOT_FOLLOWED_EVENT,
        actor=actor,
        object_ids=[claim.id],
        # An observation is a statement about the past; there is nothing to
        # undo, which is what `reversible=False` means on this log.
        reversible=False,
        data={"context": context},
    )
    return follow_through(store, claim_id=claim.id)


def follow_through(store: KBStore, *, claim_id: str) -> dict[str, Any]:
    """Counts and rate for one lesson, derived from the audit log.

    Derived, never stored: the audit stream is the authoritative record, so
    recomputing here means there is no second copy to drift.
    """
    followed = not_followed = 0
    for event in audit.read_events(store.kb_dir):
        if claim_id not in event.object_ids:
            continue
        if event.event == FOLLOWED_EVENT:
            followed += 1
        elif event.event == NOT_FOLLOWED_EVENT:
            not_followed += 1
    total = followed + not_followed
    return {
        "claim_id": claim_id,
        "followed": followed,
        "not_followed": not_followed,
        "observations": total,
        "follow_rate": round(followed / total, 4) if total else None,
    }
