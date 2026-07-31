"""Context-pack assembly — `vouch context` / `kb_context`.

A ContextPack is the bundle an agent gets back when it asks "what does the
KB know that's relevant to <task>". It's the shape AKBP defines so that
hosts can compare ranking quality and budget enforcement consistently.

This implementation:
  - runs FTS5 search if state.db has any rows, falls back to substring scan
  - resolves citations for every claim hit
  - enforces a `max_chars` budget by clipping summaries before omitting items
  - flags freshness using the source-verification cache (when available)
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any, Literal, cast

import yaml

from . import graph, hot_memory, index_db, retrieval_events
from . import pins as pins_mod
from . import strategy as strategy_mod
from .config_coerce import coerce_bool
from .embeddings.fusion import rrf_fuse
from .models import (
    ClaimStatus,
    ContextItem,
    ContextPack,
    ContextQuality,
    PageStatus,
)
from .scoping import (
    ViewerContext,
    filter_hits,
    scoped_fetch_limit,
    viewer_from,
)
from .storage import ArtifactNotFoundError, KBStore

# Claim statuses that have been explicitly retracted from active circulation.
# Any retrieval surface that hands knowledge back to an agent must exclude
# these — otherwise the archive/supersede/redact controls are decorative.
# CONTESTED is intentionally not in this set: contested claims are still
# part of the conversation, just disputed; lint / context callers can
# decide what to do with them.
_RETRACTED_CLAIM_STATUSES = frozenset({
    ClaimStatus.ARCHIVED,
    ClaimStatus.SUPERSEDED,
    ClaimStatus.REDACTED,
})

ContextItemKind = Literal["claim", "page", "entity", "relation", "source"]

# Candidate-pool sizing when a ranking strategy is active: the strategy
# ranks pool candidates and the top ``limit`` survive, so exclusion (not
# just order) is in its hands. Factor/floor keep the pool a shortlist.
_STRATEGY_POOL_FACTOR = 5
_STRATEGY_POOL_MIN = 50

# same sizing for kb.search lifecycle filtering: backends cap before status
# filtering, so without over-fetch a window full of retracted hits under-fills
# the requested limit (#581 / coderabbit).
_LIFECYCLE_POOL_FACTOR = _STRATEGY_POOL_FACTOR
_LIFECYCLE_POOL_MIN = _STRATEGY_POOL_MIN

_VALID_BACKENDS = ("auto", "hybrid", "embedding", "fts5", "substring")
_RERANKER_CACHE: Any | None = None


def _configured_backend(store: KBStore) -> str:
    """Resolve the retrieval backend from `config.yaml`, defaulting to "auto".

    Reads the singular `retrieval.backend` string. For KBs initialised
    before this knob existed, a legacy `retrieval.backends` list is honoured
    by taking its first recognised entry. Anything unreadable or unrecognised
    falls back to "auto".
    """
    try:
        loaded = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return "auto"
    if not isinstance(loaded, dict):
        return "auto"
    retrieval = loaded.get("retrieval")
    if not isinstance(retrieval, dict):
        return "auto"
    backend = retrieval.get("backend")
    if isinstance(backend, str) and backend in _VALID_BACKENDS:
        return backend
    legacy = retrieval.get("backends")
    if isinstance(legacy, list):
        for entry in legacy:
            if isinstance(entry, str) and entry in _VALID_BACKENDS:
                return entry
    return "auto"


def _configured_rerank(store: KBStore, *, limit: int) -> tuple[bool, int]:
    """Resolve the optional context rerank stage from config.yaml.

    Defaults to disabled so existing KBs keep byte-identical ordering unless
    they opt in with ``retrieval.rerank.enabled: true``. ``top_k`` is the
    window to reorder; by default it is the caller's context limit.
    """
    try:
        loaded = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False, limit
    if not isinstance(loaded, dict):
        return False, limit
    retrieval = loaded.get("retrieval")
    if not isinstance(retrieval, dict):
        return False, limit
    rerank = retrieval.get("rerank")
    if not isinstance(rerank, dict):
        return False, limit

    enabled = coerce_bool(rerank.get("enabled", False), False)

    top_k = rerank.get("top_k", limit)
    top_k = (
        top_k
        if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0
        else limit
    )
    return enabled, top_k


def _configured_recency(store: KBStore) -> tuple[bool, float]:
    """Resolve the optional recency-decay stage from config.yaml.

    Defaults to disabled so existing KBs keep byte-identical ordering unless
    they opt in with ``retrieval.recency.enabled: true`` (new KBs get it from
    the starter config). ``half_life_days`` is the age at which an artifact's
    score contribution halves; <= 0 falls back to the 90-day default.
    """
    try:
        loaded = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False, 90.0
    if not isinstance(loaded, dict):
        return False, 90.0
    retrieval = loaded.get("retrieval")
    if not isinstance(retrieval, dict):
        return False, 90.0
    recency = retrieval.get("recency")
    if not isinstance(recency, dict):
        return False, 90.0

    enabled = coerce_bool(recency.get("enabled", False), False)

    half_life = recency.get("half_life_days", 90.0)
    half_life = (
        float(half_life)
        if isinstance(half_life, (int, float)) and not isinstance(half_life, bool)
        and half_life > 0
        else 90.0
    )
    return enabled, half_life


def _artifact_timestamp(store: KBStore, kind: str, artifact_id: str) -> datetime | None:
    try:
        if kind == "claim":
            claim = store.get_claim(artifact_id)
            return claim.updated_at or claim.created_at
        if kind == "page":
            page = store.get_page(artifact_id)
            return page.updated_at or page.created_at
        if kind == "entity":
            entity = store.get_entity(artifact_id)
            return entity.updated_at or entity.created_at
    except (ArtifactNotFoundError, OSError):
        return None
    return None


def _maybe_recency(
    store: KBStore,
    *,
    hits: list[tuple[str, str, str, float]],
) -> list[tuple[str, str, str, float]]:
    """Blend a recency half-life decay into hit scores, newest-favouring.

    Rescoring-only: ``score * (0.5 + 0.5 * decay)`` keeps every hit in the
    set (an old artifact loses at most half its score, it never vanishes),
    and artifacts with no readable timestamp are left at full weight.
    """
    enabled, half_life_days = _configured_recency(store)
    if not enabled or not hits:
        return hits
    now = datetime.now(UTC)
    rescored: list[tuple[str, str, str, float]] = []
    for kind, artifact_id, summary, score in hits:
        ts = _artifact_timestamp(store, kind, artifact_id)
        if ts is None:
            rescored.append((kind, artifact_id, summary, score))
            continue
        # Whole days at half-lives of a day or more: sub-day age is noise at
        # a 90-day half-life, and quantizing keeps repeat queries
        # byte-identical within a day (fresh artifacts decay 1.0, so
        # same-day scores never drift). A sub-day half-life is an explicit
        # opt into session-scale recency, where truncation would silently
        # turn the whole stage into a no-op — there, age stays fractional.
        seconds = max((now - ts).total_seconds(), 0.0)
        age_days = (
            float(int(seconds / 86400.0))
            if half_life_days >= 1.0
            else seconds / 86400.0
        )
        decay = 0.5 ** (age_days / half_life_days)
        rescored.append((kind, artifact_id, summary, score * (0.5 + 0.5 * decay)))
    rescored.sort(key=lambda h: h[3], reverse=True)
    return rescored


_RAW_PAGE_TYPES = frozenset({"session", "log"})


def _configured_pages_first(store: KBStore) -> tuple[bool, float]:
    """Resolve the optional pages-first stage from config.yaml.

    Compiled topic pages are consolidation done at write time — when one
    answers the query it beats a pile of raw claims (the reader gets the
    reviewed synthesis, citations attached). Off by default; opt in with
    ``retrieval.pages_first.enabled: true``; ``boost`` multiplies page
    scores (default 1.25, values <= 0 fall back).
    """
    try:
        loaded = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False, 1.25
    if not isinstance(loaded, dict):
        return False, 1.25
    retrieval = loaded.get("retrieval")
    raw = retrieval.get("pages_first") if isinstance(retrieval, dict) else None
    if not isinstance(raw, dict):
        return False, 1.25
    try:
        boost = float(raw.get("boost", 1.25))
    except (TypeError, ValueError):
        boost = 1.25
    if boost <= 0:
        boost = 1.25
    return coerce_bool(raw.get("enabled", False), False), boost


def _maybe_pages_first(
    store: KBStore,
    *,
    hits: list[tuple[str, str, str, float]],
) -> list[tuple[str, str, str, float]]:
    """Boost compiled topic pages above raw claims when opted in.

    Session/log pages are raw material, not synthesis (the same
    ``_RAW_PAGE_TYPES`` line admission and compile draw) — they are never
    boosted. Unreadable pages keep their score.
    """
    enabled, boost = _configured_pages_first(store)
    if not enabled or not hits:
        return hits
    rescored: list[tuple[str, str, str, float]] = []
    for kind, artifact_id, summary, score in hits:
        if kind == "page":
            try:
                page = store.get_page(artifact_id)
            except (ArtifactNotFoundError, OSError):
                page = None
            if page is not None and page.type not in _RAW_PAGE_TYPES:
                score *= boost
        rescored.append((kind, artifact_id, summary, score))
    rescored.sort(key=lambda h: h[3], reverse=True)
    return rescored


def _default_reranker_cached() -> Any:
    global _RERANKER_CACHE
    if _RERANKER_CACHE is None:
        from .embeddings.rerank import default_reranker

        _RERANKER_CACHE = default_reranker()
    return _RERANKER_CACHE


def _maybe_rerank(
    store: KBStore,
    *,
    query: str,
    hits: list[tuple[str, str, str, float]],
    limit: int,
) -> list[tuple[str, str, str, float]]:
    enabled, top_k = _configured_rerank(store, limit=limit)
    if not enabled or not hits or top_k <= 0:
        return hits

    window_size = min(top_k, len(hits))
    window = hits[:window_size]
    try:
        from .embeddings.rerank import rerank as do_rerank

        reranked = do_rerank(
            query=query,
            hits=window,
            reranker=_default_reranker_cached(),
            top_k=window_size,
        )
    except ImportError:
        return hits

    # Keep reranking as an ordering-only stage: the configured window may move,
    # but it must not add/drop artifacts from the already-scoped result set.
    original_by_key = {(hit[0], hit[1]): hit for hit in window}
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str, str, float]] = []
    for hit in reranked:
        key = (hit[0], hit[1])
        if key in original_by_key and key not in seen:
            ordered.append(original_by_key[key])
            seen.add(key)
    for hit in window:
        key = (hit[0], hit[1])
        if key not in seen:
            ordered.append(hit)
            seen.add(key)
    return ordered + hits[window_size:]


def _retrieval_config(store: KBStore) -> dict[str, Any]:
    """The ``retrieval`` mapping from config.yaml ({} when absent/broken)."""
    try:
        loaded = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    retrieval = loaded.get("retrieval")
    return retrieval if isinstance(retrieval, dict) else {}


def _configured_strategy(store: KBStore) -> str | None:
    """Resolve ``retrieval.strategy`` - a dotted import path to a shipped,
    human-merged strategy - from config.yaml. Off (None) by default."""
    dotted = _retrieval_config(store).get("strategy")
    return dotted if isinstance(dotted, str) and dotted else None


def _configured_strategy_params(store: KBStore) -> dict[str, Any] | None:
    """Resolve ``retrieval.strategy_params`` - the champion family's knobs
    as bounded data (see ``vouch.strategies.configured``). This is the kit
    lane's hook into ranking: pure config, validated against one schema by
    both the koth gate and this runtime path."""
    params = _retrieval_config(store).get("strategy_params")
    return params if isinstance(params, dict) else None


def _maybe_strategy(
    store: KBStore,
    *,
    query: str,
    hits: list[tuple[str, str, str, float, str]],
    limit: int,
    strategy: strategy_mod.RetrievalStrategy | None = None,
) -> list[tuple[str, str, str, float, str]]:
    """Apply a pluggable ranking strategy as the final reorder stage.

    ``strategy`` is passed explicitly by the benchmark (an untrusted
    submission, wrapped in a sandbox proxy); otherwise a shipped strategy is
    resolved from ``retrieval.strategy`` config and loaded in-process. With
    neither, hits pass through byte-identical. A strategy that raises or
    returns nothing usable leaves the order untouched - retrieval never fails
    because a ranking plugin misbehaved.
    """
    strat = strategy
    if strat is None:
        # data before code: bounded params are the lane that iterates
        # without a human, so when both hooks are set the params arm is
        # the deliberate experiment. invalid params mean "no strategy",
        # never a broken retrieval.
        params = _configured_strategy_params(store)
        if params is not None:
            from .strategies import configured

            try:
                strat = configured.build(params)
            except Exception:
                return hits
        else:
            dotted = _configured_strategy(store)
            if not dotted:
                return hits
            try:
                strat = strategy_mod.load_dotted(dotted)
            except Exception:
                return hits
    if not hits:
        return hits
    # the strategy addresses hits by id; if two hits somehow share one (a
    # cross-kind slug collision), a reorder-by-id would be ambiguous, so skip
    # the stage rather than risk attaching an order to the wrong artifact.
    ids = [h[1] for h in hits]
    if len(set(ids)) != len(ids):
        return hits
    candidates = [
        strategy_mod.Candidate(kind=k, id=i, summary=s, score=sc)
        for k, i, s, sc, _b in hits
    ]
    try:
        ordered_ids = strat.rank(query, candidates, limit=limit)
    except Exception:
        return hits
    by_id = {h[1]: h for h in hits}
    reordered = strategy_mod.apply_ordering(
        list(ordered_ids), [(k, i, s, sc) for k, i, s, sc, _b in hits]
    )
    return [by_id[h4[1]] for h4 in reordered]


def _retrieve(
    store: KBStore,
    query: str,
    limit: int,
    viewer: ViewerContext,
) -> list[tuple[str, str, str, float, str]]:
    """Return list of (kind, id, summary, score, backend).

    The backend is chosen by `retrieval.backend` in config.yaml:
      - "auto" (default) / "hybrid": fuse embedding + FTS5 via RRF, falling
        back to a substring scan only if both retrievers are empty
      - "embedding": semantic search only
      - "fts5": lexical FTS5 only
      - "substring": substring scan only
    """
    backend = _configured_backend(store)
    fetch_limit = scoped_fetch_limit(limit, viewer)

    if backend in ("auto", "hybrid"):
        sem = index_db.search_semantic(store.kb_dir, query, limit=fetch_limit)
        try:
            lex = index_db.search(store.kb_dir, query, limit=fetch_limit)
        except sqlite3.Error:
            lex = []
        fused = rrf_fuse(sem, lex, limit=fetch_limit)
        if fused:
            filtered = filter_hits(store, fused, viewer, limit=limit)
            filtered = _maybe_recency(store, hits=filtered)
            filtered = _maybe_pages_first(store, hits=filtered)
            filtered = _maybe_rerank(store, query=query, hits=filtered, limit=limit)
            return [(k, i, s, sc, "hybrid") for k, i, s, sc in filtered]
        # both retrievers empty -> fall through to the substring scan below.

    if backend == "embedding":
        raw = index_db.search_semantic(store.kb_dir, query, limit=fetch_limit)
        if raw:
            filtered = filter_hits(store, raw, viewer, limit=limit)
            # Parity with the hybrid path: an operator who opted into
            # recency or rerank gets it regardless of which backend serves
            # the query — both are configured globally, not per backend.
            filtered = _maybe_recency(store, hits=filtered)
            filtered = _maybe_pages_first(store, hits=filtered)
            filtered = _maybe_rerank(store, query=query, hits=filtered, limit=limit)
            return [(k, i, s, sc, "embedding") for k, i, s, sc in filtered]
        return []

    if backend == "fts5":
        try:
            hits = index_db.search(store.kb_dir, query, limit=fetch_limit)
            if hits:
                filtered = filter_hits(store, hits, viewer, limit=limit)
                filtered = _maybe_recency(store, hits=filtered)
                filtered = _maybe_pages_first(store, hits=filtered)
                filtered = _maybe_rerank(store, query=query, hits=filtered, limit=limit)
                return [(k, i, s, sc, "fts5") for k, i, s, sc in filtered]
        except sqlite3.Error:
            pass
        return []

    substring_hits = store.search_substring(query, limit=fetch_limit)
    filtered = filter_hits(store, substring_hits, viewer, limit=limit)
    return [(k, i, s, sc, "substring") for k, i, s, sc in filtered]


def search_kb(
    store: KBStore,
    *,
    query: str,
    limit: int = 10,
    backend: str | None = None,
    min_score: float = 0.0,
    project: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    """The one `kb.search` implementation every surface delegates to.

    MCP, JSONL, and the CLI used to carry three copies of the backend
    waterfall and drifted (fusion landed in one, not the others). Keep the
    logic here only.

    ``backend=None`` defers to ``retrieval.backend`` in config.yaml; "auto"
    then fuses embedding + FTS5 via RRF and falls back to a substring scan
    only when both are empty. The ``retrieval`` block reports what actually
    served the query — a base install degrades to "fts5" and says so.
    """
    backend_arg = backend or _configured_backend(store)
    viewer = viewer_from(
        config_path=store.config_path,
        project=project,
        agent=agent,
    )
    fetch_limit = scoped_fetch_limit(limit, viewer)
    # over-fetch before lifecycle filtering so retracted/archived hits that
    # consume the backend window can be replaced by later live candidates.
    candidate_limit = max(
        fetch_limit * _LIFECYCLE_POOL_FACTOR, _LIFECYCLE_POOL_MIN,
    )
    hits: list[tuple[str, str, str, float]] = []
    used = backend_arg

    valid_backends = {"auto", "embedding", "fts5", "substring", "hybrid"}
    if backend_arg not in valid_backends:
        raise ValueError(
            f"unknown backend: {backend_arg!r} "
            f"(expected one of {sorted(valid_backends)})"
        )

    if backend_arg in ("auto", "hybrid"):
        emb = index_db.search_semantic(
            store.kb_dir, query, limit=candidate_limit * 2, min_score=min_score,
        )
        try:
            fts = index_db.search(store.kb_dir, query, limit=candidate_limit * 2)
        except sqlite3.Error:
            fts = []
        hits = rrf_fuse(emb, fts, limit=candidate_limit)
        if emb and fts:
            used = "hybrid"
        elif emb:
            used = "embedding"
        elif fts:
            used = "fts5"
        if not hits and backend_arg == "auto":
            hits = store.search_substring(query, limit=candidate_limit)
            used = "substring"
    elif backend_arg == "embedding":
        hits = index_db.search_semantic(
            store.kb_dir, query, limit=candidate_limit, min_score=min_score,
        )
        used = "embedding"
    elif backend_arg == "fts5":
        try:
            hits = index_db.search(store.kb_dir, query, limit=candidate_limit)
        except sqlite3.Error:
            hits = []
        used = "fts5"
    else:  # substring
        hits = store.search_substring(query, limit=candidate_limit)
        used = "substring"

    semantic_ok = index_db.semantic_search_available()
    # scope first without a limit so status filtering can refill the window —
    # otherwise a page of retracted hits would leave search under-filled.
    scoped = filter_hits(store, hits, viewer, limit=None)
    live = _filter_live_hits(store, scoped, limit=limit)
    hits_list = [
        {"kind": k, "id": i, "snippet": sn, "score": sc, "backend": used}
        for k, i, sn, sc in live
    ]
    result: dict[str, Any] = {
        "backend": used,
        "retrieval": {
            "configured": backend_arg,
            "used": used,
            "semantic_available": semantic_ok,
            "degraded": (
                backend_arg in ("auto", "hybrid", "embedding")
                and not semantic_ok
            ),
        },
        "viewer": {"project": viewer.project, "agent": viewer.agent},
        "hits": hits_list,
    }
    # The single search path serves both agent-facing surfaces (MCP + JSONL),
    # so the hot-memory sidebar (#261) is attached here rather than duplicated
    # at each call site.
    return hot_memory.attach_hot_memory(  # type: ignore[no-any-return]
        result, store, query=query,
        exclude_ids=[str(hit["id"]) for hit in hits_list],
    )


def _page_is_live(store: KBStore, page_id: str) -> bool:
    """False for an archived page, or one whose yaml is gone.

    Shared by ``kb.search``'s hit filter and both context-pack builders. The
    claim half of this predicate is inlined at each call site because those
    callers need the fetched claim anyway (citations, origin tags); pages are
    only ever tested, so the check lives here once — keeping it in three
    places is what let ``kb.context`` keep serving archived pages after #581
    fixed ``kb.search``.
    """
    try:
        return store.get_page(page_id).status is not PageStatus.ARCHIVED
    except ArtifactNotFoundError:
        return False


def _filter_live_hits(
    store: KBStore,
    hits: list[tuple[str, str, str, float]],
    *,
    limit: int | None = None,
) -> list[tuple[str, str, str, float]]:
    """Drop retracted claims and archived pages from search hits.

    ``kb.context`` already applies ``_RETRACTED_CLAIM_STATUSES``; ``kb.search``
    must do the same or archive/supersede/redact become decorative on the
    surface agents use for detail after recall (#581).
    """
    kept: list[tuple[str, str, str, float]] = []
    for kind, artifact_id, summary, score in hits:
        if kind == "claim":
            try:
                claim = store.get_claim(artifact_id)
            except ArtifactNotFoundError:
                continue
            if claim.status in _RETRACTED_CLAIM_STATUSES:
                continue
        elif kind == "page" and not _page_is_live(store, artifact_id):
            continue
        kept.append((kind, artifact_id, summary, score))
        if limit is not None and len(kept) >= limit:
            break
    return kept


def _enrich_summary(store: KBStore, kind: str, artifact_id: str, summary: str) -> str:
    """Return a non-empty summary, falling back to the stored artifact text."""
    if summary:
        return summary
    try:
        if kind == "claim":
            return store.get_claim(artifact_id).text
        if kind == "page":
            p = store.get_page(artifact_id)
            return p.title or p.body[:200]
        if kind == "entity":
            e = store.get_entity(artifact_id)
            return e.name or (e.description or "")[:200]
    except Exception:
        pass
    return summary


def _append_graph_neighbors(
    store: KBStore,
    items: list[ContextItem],
    *,
    depth: int,
    limit: int,
    rel_types: list[str] | None,
) -> list[str]:
    """Expand `items` with 1-hop (or deeper) graph neighbors. Returns warnings."""
    warnings: list[str] = []
    if not items:
        return warnings
    seed_scores = {it.id: it.score for it in items}
    neighbors = graph.graph_neighbors_for_seeds(
        store,
        [it.id for it in items],
        depth=depth,
        rel_types=rel_types,
        max_nodes=limit,
    )
    existing = {it.id for it in items}
    added = 0
    for node in neighbors:
        nid = node["id"]
        if nid in existing:
            continue
        kind = node["kind"]
        cites: list[str] = []
        if kind == "claim":
            try:
                claim = store.get_claim(nid)
            except ArtifactNotFoundError:
                continue
            if claim.status in _RETRACTED_CLAIM_STATUSES:
                continue
            cites = list(claim.evidence)
        elif kind == "page" and not _page_is_live(store, nid):
            continue
        via = node.get("via", "")
        parent_score = seed_scores.get(via, 0.5)
        distance = int(node.get("distance", 1))
        score = parent_score * (0.8 ** distance)
        summary = node.get("summary") or _enrich_summary(store, kind, nid, "")
        items.append(
            ContextItem(
                id=nid,
                type=cast(ContextItemKind, kind),
                summary=summary,
                score=score,
                backend="graph",
                citations=cites,
                freshness="unknown",
            )
        )
        existing.add(nid)
        added += 1
    if added:
        warnings.append(f"graph expansion added {added} neighbor(s)")
    return warnings


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# The near-duplicate heuristic, shared by the retrieval pass and the pin
# injection below it so the two cannot drift to different notions of "same".
_NEAR_DUP_THRESHOLD = 0.85
_NEAR_DUP_TOKENS = 40


def _near_dup_tokens(summary: str) -> set[str]:
    return set(summary.lower().split()[:_NEAR_DUP_TOKENS])


def _dedupe_near_duplicates(items: list[ContextItem]) -> list[ContextItem]:
    """Drop items whose summary is near-identical to a higher-scored one.

    The *keep* decision runs in descending-score order so the highest-scored
    member of a near-duplicate cluster survives; survivors are returned in the
    caller's original order. build_context_pack appends lower-priority items
    (graph-expansion neighbours) after the ranked hits and relies on that tail
    ordering for budget eviction, so this pass must not re-rank the pack.

    Cheap greedy heuristic (token-set Jaccard >= 0.85 over the first 40 tokens);
    it can over-merge long near-templated claims that differ by a single token.
    """
    dropped: set[int] = set()
    kept_tokens: list[set[str]] = []
    order = sorted(range(len(items)), key=lambda i: items[i].score, reverse=True)
    for idx in order:
        toks = _near_dup_tokens(items[idx].summary)
        if any(_jaccard(toks, seen) >= _NEAR_DUP_THRESHOLD for seen in kept_tokens):
            dropped.add(idx)
            continue
        kept_tokens.append(toks)
    return [it for i, it in enumerate(items) if i not in dropped]


def _origin_from_tags(tags: list[str]) -> str | None:
    """The origin-KB label a gated import stamped on a claim (`origin:<kb>`), so a
    federated result can name which KB vouched for it. None for local claims."""
    for tag in tags:
        if tag.startswith("origin:"):
            return tag[len("origin:") :]
    return None


def build_context_pack(
    store: KBStore,
    *,
    query: str,
    limit: int = 10,
    max_chars: int | None = None,
    min_items: int = 0,
    require_citations: bool = False,
    fail_on_warnings: bool = False,
    fail_on_budget_truncation: bool = False,
    explain: bool = False,
    project: str | None = None,
    agent: str | None = None,
    expand_graph: bool = False,
    graph_depth: int = 1,
    graph_limit: int = 20,
    graph_rel_types: list[str] | None = None,
    strategy: strategy_mod.RetrievalStrategy | None = None,
) -> ContextPack | dict[str, Any]:
    viewer = viewer_from(
        config_path=store.config_path,
        project=project,
        agent=agent,
    )
    # with a ranking strategy active, retrieval over-fetches a bounded pool
    # and the strategy's order decides which ``limit`` survive the cut —
    # de-prioritising a candidate below the window excludes it from the
    # pack. without one, the pool IS the limit and nothing changes. the
    # pool is bounded so a strategy ranks a shortlist, never the whole kb.
    strategy_active = strategy is not None or _configured_strategy(store)
    pool = max(limit * _STRATEGY_POOL_FACTOR, _STRATEGY_POOL_MIN) if strategy_active else limit
    hits = _retrieve(store, query, pool, viewer)
    hits = _maybe_strategy(
        store, query=query, hits=hits, limit=limit, strategy=strategy
    )[:limit]
    items: list[ContextItem] = []
    for kind, hid, summary, score, backend in hits:
        cites: list[str] = []
        origin: str | None = None
        if kind == "claim":
            # Exclude retracted claims even if the underlying index still
            # matches them (the FTS5 row's status column can lag — see #78
            # and the companion update_claim reindex). A missing claim is
            # also treated as retracted: the YAML may have been deleted
            # while the index row survived.
            try:
                claim = store.get_claim(hid)
            except ArtifactNotFoundError:
                continue
            if claim.status in _RETRACTED_CLAIM_STATUSES:
                continue
            cites = list(claim.evidence)
            origin = _origin_from_tags(claim.tags)
        elif kind == "page" and not _page_is_live(store, hid):
            # Archiving a page must remove it from recall, not just from
            # kb.search — this is the surface that seeds agent context.
            continue
        summary = _enrich_summary(store, kind, hid, summary)
        items.append(
            ContextItem(
                id=hid, type=cast(ContextItemKind, kind), summary=summary, score=score,
                backend=backend, citations=cites,
                freshness="unknown", origin=origin,
            )
        )

    warnings: list[str] = []
    if expand_graph:
        warnings.extend(
            _append_graph_neighbors(
                store, items, depth=graph_depth, limit=graph_limit,
                rel_types=graph_rel_types,
            )
        )

    items = _dedupe_near_duplicates(items)

    # Pins go in front of everything retrieval chose (#615): the working set is
    # a standing instruction, so it must not have to win the query every turn.
    # `pinned_items` re-checks lifecycle and viewer scope on every build, so a
    # pin can reorder the pack but never widen what it may contain. The budget
    # share caps them, and de-duplication keeps a pinned artifact that also
    # ranked from occupying two slots.
    pinned = pins_mod.pinned_items(store, viewer=viewer, max_chars=max_chars)
    if pinned:
        # Exact `(type, id)` de-duplication is not enough: the same knowledge
        # can be stored under a second id, and `_dedupe_near_duplicates` ran
        # before the pins existed, so a retrieved near-duplicate of a pin has
        # never been compared against it.
        #
        # Deliberately not solved by moving the pass below this injection: it
        # keeps the *highest-scored* member of a cluster rather than the first,
        # and `pages_first` multiplies a page's score by `boost` (1.25 by
        # default), so a retrieved page can outscore a pin's flat 1.0 and evict
        # it. That is precisely the outcome pinning exists to prevent, so the
        # comparison runs here instead, where the pin always wins.
        pinned_keys = {(p.type, p.id) for p in pinned}
        pinned_tokens = [_near_dup_tokens(p.summary) for p in pinned]
        items = pinned + [
            i for i in items
            if (i.type, i.id) not in pinned_keys
            and not any(
                _jaccard(_near_dup_tokens(i.summary), pt) >= _NEAR_DUP_THRESHOLD
                for pt in pinned_tokens
            )
        ]

    failed: list[str] = []
    uncited: list[str] = []
    budget_truncated = False
    budget_clipped = 0
    budget_omitted = 0

    if max_chars is not None:
        total = sum(len(i.summary) for i in items)
        if total > max_chars:
            budget_truncated = True
            # First clip each summary uniformly, then drop tail items if still over.
            for it in items:
                if len(it.summary) > 200:
                    it.summary = it.summary[:200] + "…"
                    budget_clipped += 1
            while items and sum(len(i.summary) for i in items) > max_chars:
                items.pop()
                budget_omitted += 1

    # Compute the citation gate over the items actually returned — after the
    # max_chars budget has dropped tail items — so the gate never fails on (or
    # reports in uncited_items) claims the consumer did not receive.
    if require_citations:
        uncited = [
            it.id for it in items if it.type == "claim" and not it.citations
        ]

    if len(items) < min_items:
        warnings.append(f"only {len(items)} items, minimum {min_items}")
        failed.append("min_items")
    if uncited:
        warnings.append(f"{len(uncited)} uncited claims")
        if require_citations:
            failed.append("require_citations")
    if fail_on_budget_truncation and budget_truncated:
        failed.append("budget_truncated")
    if fail_on_warnings and warnings:
        failed.append("fail_on_warnings")

    quality = ContextQuality(
        ok=len(failed) == 0,
        minimum_items=min_items,
        require_citations=require_citations,
        fail_on_warnings=fail_on_warnings,
        budget_truncated=budget_truncated,
        budget_omitted_items=budget_omitted,
        budget_clipped_items=budget_clipped,
        items=len(items),
        uncited_items=uncited,
        warnings=len(warnings),
        failed=failed,
    )

    pack = ContextPack(query=query, items=items, quality=quality, warnings=warnings)
    result: dict[str, Any] = pack.model_dump()
    result["viewer"] = {
        "project": viewer.project,
        "agent": viewer.agent,
    }
    # Determine the backend used (all hits share the same backend in _retrieve).
    result["backend"] = hits[0][4] if hits else "none"
    # Federated provenance: name every KB that vouched for a returned item, so a
    # reader can see which knowledge came from elsewhere (roadmap step 10).
    origins = sorted({it.origin for it in items if it.origin})
    if origins:
        result["origins"] = origins
    # Honesty block: say when a semantic-capable backend actually served
    # lexical-only results (embeddings extra absent / no embedder registered)
    # instead of letting "hybrid" imply semantic coverage that never happened.
    configured = _configured_backend(store)
    semantic_ok = index_db.semantic_search_available()
    recency_enabled, _ = _configured_recency(store)
    result["retrieval"] = {
        "configured": configured,
        "used": result["backend"],
        "semantic_available": semantic_ok,
        "degraded": (
            configured in ("auto", "hybrid", "embedding") and not semantic_ok
        ),
        "recency": recency_enabled,
    }
    if explain:
        result["explain"] = [
            {"kind": k, "id": i, "score": sc, "backend": hits[0][4] if hits else "none"}
            for k, i, _sn, sc, _be in hits
        ]
    # Telemetry, never load-bearing: the flywheel record of what was asked
    # and what came back (see retrieval_events module docstring).
    retrieval_events.log_event(
        store, query=query, backend=str(result["backend"]), limit=limit,
        budget_chars=max_chars, items=result["items"],
    )
    return result
