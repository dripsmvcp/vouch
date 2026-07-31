"""Advisory triage scoring over the pending-review queue — issue #322."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from vouch import triage
from vouch.cli import cli
from vouch.jsonl_server import HANDLERS, handle_request
from vouch.models import Claim, Entity, EntityType, Proposal, ProposalKind, ProposalStatus
from vouch.proposals import propose_claim, propose_entity
from vouch.storage import KBStore

SIGNAL_NAMES = {"fit", "citation_quality", "duplication_risk", "contradiction_risk"}


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    return KBStore.init(tmp_path)


def _enable_triage(store: KBStore, **overrides: object) -> None:
    cfg = {"triage": {"enabled": True, **overrides}}
    store.config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _no_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(name: str | None = None) -> None:
        raise KeyError("no embedder registered")

    monkeypatch.setattr("vouch.embeddings.get_embedder", _raise)


def _assert_block_shape(block: dict) -> None:
    assert set(block) == {"recommendation", "score", "signals", "rationale"}
    assert block["recommendation"] in {"approve", "reject", "needs-human"}
    assert 0.0 <= block["score"] <= 1.0
    assert set(block["signals"]) == SIGNAL_NAMES
    for sig in block["signals"].values():
        assert 0.0 <= sig["score"] <= 1.0
        assert isinstance(sig["reason"], str) and sig["reason"]
    assert isinstance(block["rationale"], str) and block["rationale"]


# --- opt-in gate -------------------------------------------------------------


def test_disabled_by_default_raises(store: KBStore) -> None:
    with pytest.raises(triage.TriageError, match="disabled"):
        triage.triage_pending(store)


def test_enabled_scores_pending_proposals(store: KBStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    propose_claim(store, text="vouch requires citations", evidence=[src.id], proposed_by="agent")
    results = triage.triage_pending(store)
    assert len(results) == 1


# --- output shape --------------------------------------------------------------


def test_triage_block_shape(store: KBStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    propose_claim(store, text="vouch requires citations", evidence=[src.id], proposed_by="agent")
    [result] = triage.triage_pending(store)
    assert result["kind"] == "claim"
    _assert_block_shape(result["_meta"]["vouch_triage"])


# --- no-write invariant ---------------------------------------------------------


def test_never_mutates_pending_queue(store: KBStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    p1 = propose_claim(store, text="a claim", evidence=[src.id], proposed_by="agent").id
    p2 = propose_entity(store, name="widget", entity_type="concept", proposed_by="agent").id

    before = {p.id for p in store.list_proposals(ProposalStatus.PENDING)}
    triage.triage_pending(store)
    after = {p.id for p in store.list_proposals(ProposalStatus.PENDING)}

    assert before == after == {p1, p2}
    assert store.list_proposals(ProposalStatus.APPROVED) == []
    assert store.list_proposals(ProposalStatus.REJECTED) == []
    assert store.list_claims() == []
    assert store.list_entities() == []


# --- citation_quality: reuses proposals._payload_block_reason -----------------


def test_citation_quality_flags_dangling_ref_and_forces_reject(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    # Bypass propose_claim's own ref validation (store.put_proposal is raw
    # I/O) to simulate a dangling reference slipping into the queue —
    # the same shape proposals._payload_block_reason guards at approve time.
    bad = Proposal(
        id="bad-1", kind=ProposalKind.CLAIM, proposed_by="agent",
        payload={
            "id": "c-bad", "text": "x", "type": "observation", "confidence": 0.7,
            "evidence": ["missing-source"], "entities": [], "tags": [],
        },
    )
    store.put_proposal(bad)
    [result] = triage.triage_pending(store)
    block = result["_meta"]["vouch_triage"]
    assert block["signals"]["citation_quality"]["score"] == 0.0
    assert "missing-source" in block["signals"]["citation_quality"]["reason"]
    assert block["recommendation"] == "reject"


def test_citation_quality_scores_clean_claim_positively(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    propose_claim(store, text="a well cited claim", evidence=[src.id], proposed_by="agent")
    [result] = triage.triage_pending(store)
    assert result["_meta"]["vouch_triage"]["signals"]["citation_quality"]["score"] > 0.0


# --- duplication_risk: heuristic fallback (default in this dev env) ----------


def test_duplication_risk_heuristic_fallback_flags_near_duplicate(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    text = "auth uses jwts in the authorization header for every request"
    store.put_claim(Claim(id="c1", text=text, evidence=[src.id]))
    propose_claim(store, text=text, evidence=[src.id], proposed_by="agent")
    [result] = triage.triage_pending(store)
    dup = result["_meta"]["vouch_triage"]["signals"]["duplication_risk"]
    assert dup["score"] > 0.9
    assert "heuristic backend" in dup["reason"]
    assert "c1" in dup["reason"]


def test_duplication_risk_heuristic_no_match_for_unrelated_text(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    store.put_claim(Claim(id="c1", text="apples and oranges", evidence=[src.id]))
    propose_claim(
        store, text="zebras run fast in the savanna", evidence=[src.id], proposed_by="agent",
    )
    [result] = triage.triage_pending(store)
    dup = result["_meta"]["vouch_triage"]["signals"]["duplication_risk"]
    assert dup["score"] == 0.0
    assert "heuristic backend" in dup["reason"]


def test_duplication_risk_ignores_archived_claim_twin(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # archived twin must not auto-reject a re-filed claim as a near-duplicate
    from vouch.models import ClaimStatus

    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    text = "prefer postgres over mysql for this service"
    store.put_claim(
        Claim(id="old", text=text, evidence=[src.id], status=ClaimStatus.ARCHIVED),
    )
    propose_claim(store, text=text, evidence=[src.id], proposed_by="agent")
    [result] = triage.triage_pending(store)
    block = result["_meta"]["vouch_triage"]
    assert block["signals"]["duplication_risk"]["score"] == 0.0
    assert block["recommendation"] != "reject"


def test_duplication_risk_ignores_archived_page_title(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vouch.models import Page, PageStatus
    from vouch.proposals import propose_page

    _no_embedder(monkeypatch)
    _enable_triage(store)
    title = "deploy runbook for the edge fleet"
    store.put_page(
        Page(id="old-page", title=title, body="retired", status=PageStatus.ARCHIVED),
    )
    propose_page(store, title=title, body="fresh copy", proposed_by="agent")
    [result] = triage.triage_pending(store)
    dup = result["_meta"]["vouch_triage"]["signals"]["duplication_risk"]
    assert dup["score"] == 0.0
    assert "old-page" not in dup["reason"]


def test_live_embedding_hits_drops_retracted_keeps_missing(
    store: KBStore,
) -> None:
    from vouch.models import ClaimStatus

    src = store.put_source(b"evidence")
    store.put_claim(Claim(id="live", text="alive", evidence=[src.id]))
    store.put_claim(
        Claim(id="dead", text="gone", evidence=[src.id], status=ClaimStatus.ARCHIVED),
    )
    hits = [
        {"artifact_kind": "claim", "artifact_id": "live", "cosine": 0.99},
        {"artifact_kind": "claim", "artifact_id": "dead", "cosine": 0.99},
        {"artifact_kind": "claim", "artifact_id": "ghost", "cosine": 0.99},
        {"artifact_kind": "proposal", "artifact_id": "pending-1", "cosine": 0.99},
    ]
    kept = {h["artifact_id"] for h in triage._live_embedding_hits(store, hits)}
    assert kept == {"live", "ghost", "pending-1"}


def test_topical_fit_scores_skips_retracted_artifacts(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vouch.models import ClaimStatus, Page, PageStatus

    class _Emb:
        def encode(self, text: str) -> list[float]:
            return [0.1] * 4

    src = store.put_source(b"evidence")
    store.put_claim(
        Claim(id="arch", text="a", evidence=[src.id], status=ClaimStatus.ARCHIVED),
    )
    store.put_claim(Claim(id="live", text="b", evidence=[src.id]))
    store.put_page(Page(id="parch", title="p", body="x", status=PageStatus.ARCHIVED))
    store.put_page(Page(id="plive", title="q", body="y"))

    def _fake_search(*_a: object, **_k: object) -> list[tuple]:
        return [
            ("claim", "new-claim", "", 0.5),  # exclude_id — skipped
            ("claim", "near-dup", "", 0.96),  # at/above dup threshold — skipped
            ("claim", "arch", "", 0.5),
            ("claim", "live", "", 0.4),
            ("claim", "missing-claim", "", 0.45),
            ("page", "parch", "", 0.5),
            ("page", "plive", "", 0.41),
            ("page", "missing-page", "", 0.42),
        ]

    monkeypatch.setattr("vouch.index_db.search_embedding", _fake_search)
    monkeypatch.setattr(
        "vouch.embeddings.similarity.similarity_threshold", lambda _store: 0.95,
    )
    proposal = Proposal(
        id="p1", kind=ProposalKind.CLAIM, proposed_by="agent",
        payload={"text": "query text", "id": "new-claim"},
    )
    assert triage._topical_fit_scores(store, proposal, _Emb()) == [
        0.4, 0.45, 0.41, 0.42,
    ]


def test_contradiction_risk_skips_archived_embedding_hit(
    store: KBStore,
) -> None:
    from vouch.models import ClaimStatus

    src = store.put_source(b"evidence")
    store.put_entity(Entity(id="api", name="API", type=EntityType.CONCEPT))
    store.put_claim(Claim(
        id="arch", text="the api requires an auth token for every request",
        evidence=[src.id], entities=["api"], status=ClaimStatus.ARCHIVED,
    ))
    proposal = Proposal(
        id="p1", kind=ProposalKind.CLAIM, proposed_by="agent",
        payload={
            "text": "the api does not require an auth token for every request",
            "entities": ["api"], "evidence": [src.id],
        },
    )
    hits = [{"artifact_kind": "claim", "artifact_id": "arch", "cosine": 0.92}]
    out = triage._signal_contradiction_risk(store, proposal, hits)
    assert out["score"] == 0.0


def test_duplication_risk_relation_exact_match(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    from vouch.models import Relation

    store.put_entity(Entity(id="a", name="A", type=EntityType.CONCEPT))
    store.put_entity(Entity(id="b", name="B", type=EntityType.CONCEPT))
    store.put_relation(
        Relation(id="a--relates_to--b", source="a", relation="relates_to", target="b")
    )
    dup_proposal = Proposal(
        id="rel-1", kind=ProposalKind.RELATION, proposed_by="agent",
        payload={
            "id": "a--relates_to--b-2", "source": "a", "relation": "relates_to",
            "target": "b", "confidence": 0.7, "evidence": [],
        },
    )
    store.put_proposal(dup_proposal)
    [result] = triage.triage_pending(store)
    dup = result["_meta"]["vouch_triage"]["signals"]["duplication_risk"]
    assert dup["score"] == 1.0
    assert "already approved" in dup["reason"]


# --- fit: entity-overlap heuristic (no embeddings needed) ---------------------


def test_fit_scores_high_when_entities_already_known(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    store.put_entity(Entity(id="jwt", name="JWT", type=EntityType.CONCEPT))
    src = store.put_source(b"evidence")
    propose_claim(
        store, text="jwt tokens expire after an hour", evidence=[src.id],
        entities=["jwt"], proposed_by="agent",
    )
    [result] = triage.triage_pending(store)
    fit = result["_meta"]["vouch_triage"]["signals"]["fit"]
    assert fit["score"] == 1.0


def test_fit_neutral_when_no_entities_referenced(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    propose_claim(store, text="an unrelated observation", evidence=[src.id], proposed_by="agent")
    [result] = triage.triage_pending(store)
    fit = result["_meta"]["vouch_triage"]["signals"]["fit"]
    assert fit["score"] == 0.5


# --- contradiction_risk --------------------------------------------------------


def test_contradiction_risk_flags_polarity_conflict(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    store.put_entity(Entity(id="api", name="API", type=EntityType.CONCEPT))
    src = store.put_source(b"evidence")
    store.put_claim(Claim(
        id="c1", text="the api requires an auth token for every request",
        evidence=[src.id], entities=["api"],
    ))
    propose_claim(
        store, text="the api does not require an auth token for every request",
        evidence=[src.id], entities=["api"], proposed_by="agent",
    )
    [result] = triage.triage_pending(store)
    conflict = result["_meta"]["vouch_triage"]["signals"]["contradiction_risk"]
    assert conflict["score"] > 0.0
    assert "c1" in conflict["reason"]


def test_contradiction_risk_no_conflict_without_shared_entity(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    store.put_claim(Claim(
        id="c1", text="the api requires an auth token for every request",
        evidence=[src.id],
    ))
    propose_claim(
        store, text="the api does not require an auth token for every request",
        evidence=[src.id], proposed_by="agent",
    )
    [result] = triage.triage_pending(store)
    conflict = result["_meta"]["vouch_triage"]["signals"]["contradiction_risk"]
    assert conflict["score"] == 0.0


def test_contradiction_risk_not_applicable_to_non_claim_kind(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    propose_entity(store, name="widget", entity_type="concept", proposed_by="agent")
    [result] = triage.triage_pending(store)
    conflict = result["_meta"]["vouch_triage"]["signals"]["contradiction_risk"]
    assert conflict["score"] == 0.0
    assert "only assessed for claim proposals" in conflict["reason"]


# --- embeddings-present path (requires numpy; skipped without it) ------------


@pytest.fixture
def _mock_embedder() -> None:
    pytest.importorskip("numpy")
    from tests.embeddings._fakes import MockEmbedder
    from vouch.embeddings import register
    from vouch.embeddings.base import DEFAULT_MODEL_NAME

    register(DEFAULT_MODEL_NAME, lambda: MockEmbedder(dim=8))


def test_duplication_risk_embedding_backend_flags_exact_duplicate(
    store: KBStore, _mock_embedder: None,
) -> None:
    _enable_triage(store)
    src = store.put_source(b"evidence")
    text = "auth uses jwts in the authorization header"
    store.put_claim(Claim(id="c1", text=text, evidence=[src.id]))
    propose_claim(store, text=text, evidence=[src.id], proposed_by="agent")
    [result] = triage.triage_pending(store)
    block = result["_meta"]["vouch_triage"]
    dup = block["signals"]["duplication_risk"]
    assert dup["score"] >= 0.95
    assert "embedding backend" in dup["reason"]
    # A near-duplicate hit is penalized by duplication_risk and must not
    # also inflate fit via the same signal (see _topical_fit_scores).
    assert block["recommendation"] != "approve"


def test_backend_heuristic_config_forces_fallback_even_with_embedder(
    store: KBStore, _mock_embedder: None,
) -> None:
    _enable_triage(store, backend="heuristic")
    src = store.put_source(b"evidence")
    text = "auth uses jwts in the authorization header"
    store.put_claim(Claim(id="c1", text=text, evidence=[src.id]))
    propose_claim(store, text=text, evidence=[src.id], proposed_by="agent")
    [result] = triage.triage_pending(store)
    dup = result["_meta"]["vouch_triage"]["signals"]["duplication_risk"]
    assert "heuristic backend" in dup["reason"]


# --- proposal_ids filter / config plumbing ------------------------------------


def test_proposal_ids_filters_to_subset(store: KBStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    p1 = propose_claim(store, text="first claim", evidence=[src.id], proposed_by="agent").id
    propose_claim(store, text="second claim", evidence=[src.id], proposed_by="agent")
    results = triage.triage_pending(store, proposal_ids=[p1])
    assert [r["id"] for r in results] == [p1]


def test_custom_weights_read_from_config(store: KBStore) -> None:
    custom_weights = {
        "fit": 1.0, "citation_quality": 0.0, "duplication_risk": 0.0, "contradiction_risk": 0.0,
    }
    _enable_triage(store, weights=custom_weights)
    cfg = triage.triage_cfg(store)
    assert cfg.weights == custom_weights


def test_disabled_config_value_keeps_default_false(store: KBStore) -> None:
    raw = yaml.safe_dump({"triage": {"weights": {"fit": 0.9}}})
    store.config_path.write_text(raw, encoding="utf-8")
    cfg = triage.triage_cfg(store)
    assert cfg.enabled is False


def test_quoted_enabled_is_parsed(store: KBStore) -> None:
    """Regression (#658): `triage.enabled` accepted only a real YAML bool, so
    a quoted `enabled: "true"` left kb.triage_pending answering "disabled" for
    a config that plainly says otherwise."""
    store.config_path.write_text(
        'triage:\n  enabled: "true"\n', encoding="utf-8",
    )
    assert triage.triage_cfg(store).enabled is True

    store.config_path.write_text(
        'triage:\n  enabled: "false"\n', encoding="utf-8",
    )
    assert triage.triage_cfg(store).enabled is False

    # unrecognized spelling degrades to the opt-in default (off)
    store.config_path.write_text(
        "triage:\n  enabled: perhaps\n", encoding="utf-8",
    )
    assert triage.triage_cfg(store).enabled is False


# --- registration sites --------------------------------------------------------


def test_jsonl_handler_registered() -> None:
    assert "kb.triage_pending" in HANDLERS


def test_jsonl_triage_pending_disabled_returns_invalid_request(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vouch.jsonl_server as jsonl_server

    monkeypatch.setattr(jsonl_server, "_store", lambda: store)
    resp = handle_request({"id": "1", "method": "kb.triage_pending", "params": {}})
    assert resp["ok"] is False
    assert resp["error"]["code"] == "invalid_request"


def test_jsonl_triage_pending_enabled_returns_blocks(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vouch.jsonl_server as jsonl_server

    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    propose_claim(store, text="a claim", evidence=[src.id], proposed_by="agent")
    monkeypatch.setattr(jsonl_server, "_store", lambda: store)
    resp = handle_request({"id": "1", "method": "kb.triage_pending", "params": {}})
    assert resp["ok"] is True
    [item] = resp["result"]
    _assert_block_shape(item["_meta"]["vouch_triage"])


def test_cli_triage_disabled_shows_clean_error(
    store: KBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(store.root)
    result = CliRunner().invoke(cli, ["triage"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "Error:" in result.output
    assert "disabled" in result.output


def test_cli_triage_json_output(store: KBStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    propose_claim(store, text="a claim", evidence=[src.id], proposed_by="agent")
    monkeypatch.chdir(store.root)
    result = CliRunner().invoke(cli, ["triage", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    _assert_block_shape(data[0]["_meta"]["vouch_triage"])


def test_cli_triage_sorts_ranked_table(store: KBStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_embedder(monkeypatch)
    _enable_triage(store)
    src = store.put_source(b"evidence")
    propose_claim(store, text="a well cited unique claim", evidence=[src.id], proposed_by="agent")
    bad = Proposal(
        id="bad-1", kind=ProposalKind.CLAIM, proposed_by="agent",
        payload={
            "id": "c-bad", "text": "y", "type": "observation", "confidence": 0.7,
            "evidence": ["missing-source"], "entities": [], "tags": [],
        },
    )
    store.put_proposal(bad)
    monkeypatch.chdir(store.root)
    result = CliRunner().invoke(cli, ["triage"])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln and ln[0].isdigit()]
    scores = [float(ln.split()[0]) for ln in lines]
    assert scores == sorted(scores, reverse=True)
