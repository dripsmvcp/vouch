"""Read helpers for goals — the in-flight objectives half of the KB.

Writes live elsewhere on purpose: `proposals.propose_goal` files one, the
reviewer approves it, and `lifecycle.set_goal_status` moves it. This module
only reads, so every surface that shows goals (digest, session-start recall,
CLI, MCP) agrees on what "open, viewer-visible, oldest first" means instead
of each re-deriving it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from .models import Goal, GoalStatus
from .scoping import ViewerContext, is_visible, viewer_from
from .storage import KBStore


def _created(goal: Goal) -> datetime:
    dt = goal.created_at
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def list_goals(
    store: KBStore,
    *,
    status: str | GoalStatus | None = GoalStatus.OPEN,
    viewer: ViewerContext | None = None,
    limit: int | None = None,
) -> list[Goal]:
    """Approved goals, viewer-scoped, oldest first.

    Defaults to open goals only — the question a returning operator is
    actually asking is "what is in flight", not "what has this project ever
    intended". Pass ``status=None`` for the whole history.

    Oldest-first (not newest-first like decisions): an objective that has been
    open longest is the one most likely to be stale or forgotten, so it earns
    the top of the list.
    """
    if viewer is None:
        viewer = viewer_from(config_path=store.config_path)
    wanted = GoalStatus(status) if status is not None else None
    goals = [
        g for g in store.list_goals()
        if (wanted is None or g.status is wanted) and is_visible(g.scope, viewer)
    ]
    goals.sort(key=_created)
    return goals[:limit] if limit is not None else goals
