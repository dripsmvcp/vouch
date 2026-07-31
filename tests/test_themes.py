"""Cross-session pattern detection — detect_themes + propose_theme."""

from __future__ import annotations

from pathlib import Path

import pytest

from vouch import sessions as sess_mod
from vouch import themes
from vouch.models import ArtifactScope, Visibility
from vouch.proposals import ProposalError, approve, propose_claim, propose_entity
from vouch.scoping import ViewerContext, is_visible, viewer_from
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    return KBStore.init(tmp_path)


def _seed_multi_session_claims(store: KBStore) -> dict:
    """Create two sessions with overlapping entity claims."""
    src = store.put_source(b"evidence-material")

    # Register entities so propose_page can reference them.
    e1 = propose_entity(
        store, name="auth", entity_type="concept",
        proposed_by="setup", slug_hint="auth",
    )
    e2 = propose_entity(
        store, name="jwt", entity_type="concept",
        proposed_by="setup", slug_hint="jwt",
    )
    e3 = propose_entity(
        store, name="session-mgmt", entity_type="concept",
        proposed_by="setup", slug_hint="session-mgmt",
    )
    approve(store, e1.id, approved_by="human")
    approve(store, e2.id, approved_by="human")
    approve(store, e3.id, approved_by="human")

    # Session 1: claims mentioning auth + jwt.
    s1 = sess_mod.session_start(store, agent="agent-a", task="review auth")
    c1 = propose_claim(
        store, text="auth uses jwt for token validation",
        evidence=[src.id], proposed_by="agent-a",
        entities=["auth", "jwt"], session_id=s1.id,
    )
    c2 = propose_claim(
        store, text="jwt tokens expire after 1 hour",
        evidence=[src.id], proposed_by="agent-a",
        entities=["auth", "jwt"], session_id=s1.id,
        slug_hint="jwt-expiry",
    )
    approve(store, c1.id, approved_by="human")
    approve(store, c2.id, approved_by="human")
    sess_mod.session_end(store, s1.id)

    # Session 2: claims also mentioning auth + jwt + session-mgmt.
    s2 = sess_mod.session_start(store, agent="agent-b", task="review sessions")
    c3 = propose_claim(
        store, text="auth middleware validates jwt on every request",
        evidence=[src.id], proposed_by="agent-b",
        entities=["auth", "jwt"], session_id=s2.id,
        slug_hint="auth-middleware-jwt",
    )
    c4 = propose_claim(
        store, text="session management depends on auth and jwt",
        evidence=[src.id], proposed_by="agent-b",
        entities=["auth", "jwt", "session-mgmt"], session_id=s2.id,
        slug_hint="session-depends-auth",
    )
    approve(store, c3.id, approved_by="human")
    approve(store, c4.id, approved_by="human")
    sess_mod.session_end(store, s2.id)

    return {"sessions": [s1.id, s2.id], "source": src.id}


def test_detect_themes_finds_clusters(store: KBStore) -> None:
    _seed_multi_session_claims(store)
    result = themes.detect_themes(store, min_sessions=2, min_claims=2)
    assert len(result.clusters) > 0
    # The auth+jwt pair should be the strongest cluster.
    top = result.clusters[0]
    assert "auth" in top.entities
    assert "jwt" in top.entities
    assert top.session_count >= 2
    assert top.claim_count >= 2
    assert top.score > 0


def test_detect_themes_respects_min_sessions(store: KBStore) -> None:
    _seed_multi_session_claims(store)
    result = themes.detect_themes(store, min_sessions=10, min_claims=1)
    assert len(result.clusters) == 0


def test_detect_themes_respects_min_claims(store: KBStore) -> None:
    _seed_multi_session_claims(store)
    result = themes.detect_themes(store, min_sessions=1, min_claims=100)
    assert len(result.clusters) == 0


def test_detect_themes_read_only(store: KBStore) -> None:
    """detect_themes must not create any proposals or pages."""
    _seed_multi_session_claims(store)
    pages_before = len(store.list_pages())
    proposals_before = len(store.list_proposals())
    themes.detect_themes(store, min_sessions=2, min_claims=2)
    assert len(store.list_pages()) == pages_before
    assert len(store.list_proposals()) == proposals_before


def test_detect_themes_excludes_archived(store: KBStore) -> None:
    """Archived claims should not contribute to theme detection."""
    from vouch import lifecycle as life

    _seed_multi_session_claims(store)
    # Archive all claims — themes should vanish.
    for claim in store.list_claims():
        life.archive(store, claim_id=claim.id, actor="human")
    result = themes.detect_themes(store, min_sessions=1, min_claims=1)
    assert len(result.clusters) == 0


def test_detect_themes_excludes_claims_the_viewer_cannot_see(
    store: KBStore,
) -> None:
    """A claim outside the viewer's scope must not reach a cluster."""
    _seed_multi_session_claims(store)
    # Re-stamp every claim private to an agent nobody is reading as.
    for claim in store.list_claims():
        claim.scope = ArtifactScope(visibility=Visibility.PRIVATE, agent="alice")
        store.update_claim(claim)

    viewer = viewer_from(config_path=store.config_path)
    assert not [c for c in store.list_claims() if is_visible(c.scope, viewer)]

    result = themes.detect_themes(store, min_sessions=1, min_claims=1)
    assert result.clusters == []


def test_detect_themes_honors_explicit_viewer(store: KBStore) -> None:
    """The claim's own agent still sees it; a different agent does not."""
    _seed_multi_session_claims(store)
    for claim in store.list_claims():
        claim.scope = ArtifactScope(visibility=Visibility.PRIVATE, agent="alice")
        store.update_claim(claim)

    alice = themes.detect_themes(
        store, min_sessions=2, min_claims=2,
        viewer=ViewerContext(agent="alice"),
    )
    assert [c.entities for c in alice.clusters], "alice owns these claims"

    bob = themes.detect_themes(
        store, min_sessions=1, min_claims=1, viewer=ViewerContext(agent="bob"),
    )
    assert bob.clusters == []


def test_detect_themes_scope_leak_does_not_reach_a_theme_page(
    store: KBStore,
) -> None:
    """Regression: the detector fed private claim ids into a durable page.

    `propose_theme` writes `cluster.claim_ids` and `cluster.session_ids` into
    the page body, so an unfiltered scan leaked them past the review gate as
    committed yaml rather than as an advisory read.
    """
    _seed_multi_session_claims(store)
    private_ids = set()
    for claim in store.list_claims():
        if "session-mgmt" in claim.entities:
            continue
        claim.scope = ArtifactScope(visibility=Visibility.PRIVATE, agent="alice")
        store.update_claim(claim)
        private_ids.add(claim.id)
    assert private_ids

    result = themes.detect_themes(store, min_sessions=1, min_claims=1)
    for cluster in result.clusters:
        assert not private_ids & set(cluster.claim_ids)


def test_detect_themes_disabled_config(store: KBStore) -> None:
    """When themes.enabled=false in config, returns empty."""
    import yaml

    _seed_multi_session_claims(store)
    cfg = yaml.safe_load(store.config_path.read_text()) or {}
    cfg["themes"] = {"enabled": False}
    store.config_path.write_text(yaml.dump(cfg))
    result = themes.detect_themes(store, min_sessions=1, min_claims=1)
    assert len(result.clusters) == 0
    assert result.config_used.get("enabled") is False


def test_detect_themes_quoted_false_does_not_enable(store: KBStore) -> None:
    """Regression: bool(\"false\") is True, so a quoted themes.enabled
    previously silently kept theme detection on (#648)."""
    import yaml

    _seed_multi_session_claims(store)
    store.config_path.write_text(
        (store.config_path.read_text(encoding="utf-8") or "")
        + '\nthemes:\n  enabled: "false"\n',
        encoding="utf-8",
    )
    result = themes.detect_themes(store, min_sessions=1, min_claims=1)
    assert len(result.clusters) == 0
    assert result.config_used.get("enabled") is False
    # quoted true still enables
    cfg = yaml.safe_load(store.config_path.read_text(encoding="utf-8")) or {}
    cfg["themes"] = {"enabled": "true"}
    store.config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert themes._load_theme_config(store)["enabled"] is True


def test_propose_theme(store: KBStore) -> None:
    _seed_multi_session_claims(store)
    result = themes.detect_themes(store, min_sessions=2, min_claims=2)
    assert len(result.clusters) > 0
    cluster = result.clusters[0]
    proposal_result = themes.propose_theme(
        store, cluster, proposed_by="theme-agent",
    )
    assert "proposal_id" in proposal_result
    assert proposal_result["claim_count"] >= 2
    # The proposal should appear in pending.
    pending = store.list_proposals()
    theme_proposals = [
        p for p in pending
        if p.kind.value == "page" and p.payload.get("type") == "theme"
    ]
    assert len(theme_proposals) == 1


def test_propose_theme_dedup(store: KBStore) -> None:
    """Proposing the same cluster twice should deduplicate on detect."""
    _seed_multi_session_claims(store)
    result = themes.detect_themes(store, min_sessions=2, min_claims=2)
    cluster = result.clusters[0]
    themes.propose_theme(store, cluster, proposed_by="agent")

    # Detect again — already-proposed themes should be excluded.
    result2 = themes.detect_themes(store, min_sessions=2, min_claims=2)
    matching = [
        c for c in result2.clusters
        if set(c.entities) == set(cluster.entities)
    ]
    assert len(matching) == 0


def test_propose_theme_validates_claims(store: KBStore) -> None:
    """Proposing with no valid claims should raise."""
    cluster = themes.ThemeCluster(
        entities=["nonexistent-entity"],
        claim_ids=["nonexistent-claim"],
        session_ids=["sess-fake"],
        score=1.0,
        session_count=1,
        claim_count=1,
    )
    with pytest.raises(ProposalError):
        themes.propose_theme(store, cluster, proposed_by="agent")


def test_detect_themes_deterministic(store: KBStore) -> None:
    """Running detect_themes twice should produce identical results."""
    _seed_multi_session_claims(store)
    r1 = themes.detect_themes(store, min_sessions=2, min_claims=2)
    r2 = themes.detect_themes(store, min_sessions=2, min_claims=2)
    assert len(r1.clusters) == len(r2.clusters)
    for c1, c2 in zip(r1.clusters, r2.clusters, strict=True):
        assert c1.entities == c2.entities
        assert c1.score == c2.score
        assert c1.claim_ids == c2.claim_ids


def test_detect_themes_top_k(store: KBStore) -> None:
    _seed_multi_session_claims(store)
    result = themes.detect_themes(store, min_sessions=1, min_claims=1, top_k=1)
    assert len(result.clusters) <= 1
