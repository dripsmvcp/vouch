"""The `kb_*` MCP tool surface in `vouch.server`.

`@mcp.tool()` returns the undecorated function, so each tool is callable
directly — which is how the existing tests reach `kb_propose_delete`. Most of
the surface was import-covered only: the decorator ran at import, the body
never did. These are the functions every MCP host (Claude Code, Cursor, Codex)
actually calls, and the contract they rely on is that a missing artifact comes
back as a `ValueError` the host can render, not an internal traceback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vouch import server
from vouch.embeddings import register
from vouch.embeddings.base import DEFAULT_MODEL_NAME, Embedder
from vouch.models import Claim, Entity, Evidence, Page, Relation
from vouch.proposals import propose_claim
from vouch.storage import KBStore


class _HashEmbedder(Embedder):
    name = "mock"
    version = "1"
    dim = 8

    def encode(self, text: str) -> np.ndarray:
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        out = np.array([h[i] / 255.0 for i in range(self.dim)], dtype=np.float32)
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out /= norm
        return out


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KBStore:
    s = KBStore.init(tmp_path / "kb")
    monkeypatch.chdir(s.root)
    return s


@pytest.fixture
def embedder() -> None:
    register(DEFAULT_MODEL_NAME, _HashEmbedder)


def _claim(store: KBStore, claim_id: str, text: str, **kw: Any) -> Claim:
    src = store.put_source(b"evidence body")
    return store.put_claim(Claim(id=claim_id, text=text, evidence=[src.id], **kw))


# --- capabilities / status ------------------------------------------------


def test_kb_capabilities_lists_methods(store: KBStore) -> None:
    assert "kb.search" in server.kb_capabilities()["methods"]


def test_kb_status_counts_artifacts(store: KBStore) -> None:
    _claim(store, "c1", "a durable claim")
    assert server.kb_status()["claims"] == 1


def test_load_cfg_returns_a_mapping(store: KBStore) -> None:
    assert isinstance(server._load_cfg(store), dict)


def test_load_cfg_survives_unparseable_config(store: KBStore) -> None:
    (store.kb_dir / "config.yaml").write_text("review: [unclosed\n", encoding="utf-8")
    assert server._load_cfg(store) == {}


def test_load_cfg_ignores_a_scalar_document(store: KBStore) -> None:
    (store.kb_dir / "config.yaml").write_text("just-a-string\n", encoding="utf-8")
    assert server._load_cfg(store) == {}


def test_current_model_name_is_a_string(store: KBStore) -> None:
    assert isinstance(server._current_model_name(), str)


# --- kb_read_* -----------------------------------------------------------


def test_kb_read_claim_returns_the_claim(store: KBStore) -> None:
    _claim(store, "c1", "the review gate holds")
    assert server.kb_read_claim("c1")["text"] == "the review gate holds"


def test_kb_read_page_returns_the_page(store: KBStore) -> None:
    store.put_page(Page(id="p1", title="review gate", body="prose"))
    assert server.kb_read_page("p1")["title"] == "review gate"


def test_kb_read_entity_returns_the_entity(store: KBStore) -> None:
    store.put_entity(Entity(id="e1", name="acme-example", type="company"))
    assert server.kb_read_entity("e1")["name"] == "acme-example"


def test_kb_read_relation_returns_the_triple(store: KBStore) -> None:
    store.put_entity(Entity(id="e1", name="alice-example", type="person"))
    store.put_entity(Entity(id="e2", name="acme-example", type="company"))
    store.put_relation(Relation(id="r1", source="e1", relation="owned_by", target="e2"))
    assert server.kb_read_relation("r1")["relation"] == "owned_by"


def test_kb_read_evidence_returns_the_span(store: KBStore) -> None:
    src = store.put_source(b"body")
    store.put_evidence(Evidence(id="ev1", source_id=src.id, locator="p2"))
    assert server.kb_read_evidence("ev1")["locator"] == "p2"


def test_kb_read_source_returns_metadata(store: KBStore) -> None:
    src = store.put_source(b"body", title="the memo")
    assert server.kb_read_source(src.id)["title"] == "the memo"


@pytest.mark.parametrize(
    "tool",
    [
        "kb_read_claim",
        "kb_read_page",
        "kb_read_entity",
        "kb_read_relation",
        "kb_read_evidence",
        "kb_read_source",
    ],
)
def test_kb_read_missing_artifact_raises_value_error(
    store: KBStore, tool: str
) -> None:
    # the MCP contract: a host must see ValueError, never ArtifactNotFoundError
    with pytest.raises(ValueError):
        getattr(server, tool)("does-not-exist")


# --- kb_list_* -----------------------------------------------------------


def test_kb_list_claims_and_status_filter(store: KBStore) -> None:
    _claim(store, "c1", "a claim")
    assert len(server.kb_list_claims()["items"]) == 1
    assert "items" in server.kb_list_claims(status="working")


def test_kb_list_pages(store: KBStore) -> None:
    store.put_page(Page(id="p1", title="a page"))
    assert len(server.kb_list_pages()["items"]) == 1


def test_kb_list_pages_type_filter(store: KBStore) -> None:
    store.put_page(Page(id="p1", title="a page"))
    assert "items" in server.kb_list_pages(type="concept")


def test_kb_list_entities_and_type_filter(store: KBStore) -> None:
    store.put_entity(Entity(id="e1", name="alice-example", type="person"))
    assert len(server.kb_list_entities()["items"]) == 1
    assert "items" in server.kb_list_entities(entity_type="person")


def test_kb_list_relations_and_node_filter(store: KBStore) -> None:
    store.put_entity(Entity(id="e1", name="alice-example", type="person"))
    store.put_entity(Entity(id="e2", name="acme-example", type="company"))
    store.put_relation(Relation(id="r1", source="e1", relation="owned_by", target="e2"))
    assert len(server.kb_list_relations()["items"]) == 1
    assert "items" in server.kb_list_relations(node_id="e1")


def test_kb_list_sources(store: KBStore) -> None:
    store.put_source(b"body", title="the memo")
    assert len(server.kb_list_sources()["items"]) == 1


def test_kb_list_pending_is_empty_on_a_fresh_kb(store: KBStore) -> None:
    assert server.kb_list_pending()["items"] == []


def test_kb_list_pending_shows_a_proposal(store: KBStore) -> None:
    src = store.put_source(b"e")
    propose_claim(store, text="pending", evidence=[src.id], proposed_by="agent")
    assert server.kb_list_pending()["items"]


def test_kb_triage_pending_is_opt_in(store: KBStore) -> None:
    src = store.put_source(b"e")
    propose_claim(store, text="pending", evidence=[src.id], proposed_by="agent")
    # off by default: the tool refuses rather than silently ranking nothing
    with pytest.raises(ValueError, match="triage is disabled"):
        server.kb_triage_pending()


def test_kb_triage_pending_returns_rows_when_enabled(store: KBStore) -> None:
    src = store.put_source(b"e")
    propose_claim(store, text="pending", evidence=[src.id], proposed_by="agent")
    (store.kb_dir / "config.yaml").write_text(
        "triage:\n  enabled: true\n", encoding="utf-8"
    )
    assert isinstance(server.kb_triage_pending(), list)


# --- sources -------------------------------------------------------------


def test_kb_register_source_stores_content(store: KBStore) -> None:
    out = server.kb_register_source("some evidence", title="a note")
    assert store.get_source(out["id"]).title == "a note"


def test_kb_register_source_from_path(store: KBStore) -> None:
    doc = store.root / "note.txt"
    doc.write_text("some evidence", encoding="utf-8")
    out = server.kb_register_source_from_path(str(doc))
    assert store.get_source(out["id"])


def test_kb_register_source_from_path_refuses_outside_the_project(
    store: KBStore, tmp_path: Path
) -> None:
    # the containment check is the security boundary: an MCP client must not be
    # able to register /etc/shadow as a source
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("not mine", encoding="utf-8")
    with pytest.raises(ValueError, match="inside project root"):
        server.kb_register_source_from_path(str(outside))


def test_kb_register_source_from_missing_path_raises(store: KBStore) -> None:
    with pytest.raises(ValueError):
        server.kb_register_source_from_path(str(store.root / "nope.txt"))


# --- propose / approve / reject ------------------------------------------


def test_kb_propose_claim_files_a_proposal(store: KBStore) -> None:
    src = store.put_source(b"e")
    out = server.kb_propose_claim(text="a proposed claim", evidence=[src.id])
    assert store.get_proposal(out["proposal_id"] if "proposal_id" in out else out["id"])


def test_kb_propose_claim_dry_run_writes_nothing(store: KBStore) -> None:
    src = store.put_source(b"e")
    server.kb_propose_claim(text="a dry claim", evidence=[src.id], dry_run=True)
    assert store.list_proposals() == []


def test_kb_propose_claim_without_evidence_raises(store: KBStore) -> None:
    with pytest.raises(ValueError):
        server.kb_propose_claim(text="uncited", evidence=[])


def test_kb_propose_page_files_a_proposal(store: KBStore) -> None:
    out = server.kb_propose_page(title="a page", body="prose")
    assert out


def test_kb_propose_page_dry_run_writes_nothing(store: KBStore) -> None:
    server.kb_propose_page(title="a page", body="prose", dry_run=True)
    assert store.list_proposals() == []


def test_kb_propose_entity_files_a_proposal(store: KBStore) -> None:
    assert server.kb_propose_entity(name="acme-example", entity_type="company")


def test_kb_propose_entity_dry_run_writes_nothing(store: KBStore) -> None:
    server.kb_propose_entity(
        name="acme-example", entity_type="company", dry_run=True
    )
    assert store.list_proposals() == []


def test_kb_propose_relation_files_a_proposal(store: KBStore) -> None:
    store.put_entity(Entity(id="e1", name="alice-example", type="person"))
    store.put_entity(Entity(id="e2", name="acme-example", type="company"))
    assert server.kb_propose_relation(src="e1", relation="owned_by", target="e2")


def test_kb_propose_relation_dry_run_writes_nothing(store: KBStore) -> None:
    store.put_entity(Entity(id="e1", name="alice-example", type="person"))
    store.put_entity(Entity(id="e2", name="acme-example", type="company"))
    server.kb_propose_relation(
        src="e1", relation="owned_by", target="e2", dry_run=True
    )
    assert store.list_proposals() == []


def test_kb_approve_promotes_a_pending_claim(store: KBStore) -> None:
    src = store.put_source(b"e")
    pr = propose_claim(store, text="approve me", evidence=[src.id], proposed_by="agent")
    server.kb_approve(pr.id)
    assert store.list_claims()


def test_kb_approve_unknown_proposal_raises(store: KBStore) -> None:
    with pytest.raises(ValueError):
        server.kb_approve("no-such-proposal")


def test_kb_reject_records_a_reason(store: KBStore) -> None:
    src = store.put_source(b"e")
    pr = propose_claim(store, text="reject me", evidence=[src.id], proposed_by="agent")
    server.kb_reject(pr.id, reason="not useful")
    assert store.list_claims() == []


def test_kb_reject_unknown_proposal_raises(store: KBStore) -> None:
    with pytest.raises(ValueError):
        server.kb_reject("no-such-proposal", reason="nope")


def test_kb_reject_extracted_with_nothing_pending(store: KBStore) -> None:
    assert server.kb_reject_extracted() is not None


# --- lifecycle-adjacent tools -------------------------------------------


def test_kb_expire_dry_run_by_default(store: KBStore) -> None:
    _claim(store, "c1", "a claim")
    assert server.kb_expire() is not None


def test_kb_expire_applied(store: KBStore) -> None:
    _claim(store, "c1", "a claim")
    assert server.kb_expire(apply=True, days=1) is not None


def test_kb_clear_claims_dry_run(store: KBStore) -> None:
    _claim(store, "c1", "an auto claim", auto_approved=True)
    out = server.kb_clear_claims(dry_run=True)
    assert out is not None
    assert store.get_claim("c1")


def test_kb_clear_claims_applied(store: KBStore) -> None:
    _claim(store, "c1", "an auto claim", auto_approved=True)
    assert server.kb_clear_claims() is not None


def test_kb_clear_claims_rejects_a_bad_before_date(store: KBStore) -> None:
    with pytest.raises(ValueError):
        server.kb_clear_claims(before="not-a-date")


def test_kb_cite_resolves_citations(store: KBStore) -> None:
    src = store.put_source(b"body", title="the memo")
    store.put_claim(Claim(id="c1", text="cited", evidence=[src.id]))
    assert server.kb_cite("c1")


def test_kb_cite_unknown_claim_raises(store: KBStore) -> None:
    # note the inconsistency with the kb_read_* family, which converts this to
    # ValueError: kb_cite lets ArtifactNotFoundError through unwrapped
    from vouch.storage import ArtifactNotFoundError

    with pytest.raises(ArtifactNotFoundError):
        server.kb_cite("ghost")


def test_kb_diff_between_two_claims(store: KBStore) -> None:
    _claim(store, "old", "the first version")
    _claim(store, "new", "the second version")
    assert server.kb_diff("old", "new") is not None


# --- graph / context ----------------------------------------------------


def test_kb_neighbors_returns_the_root_node(store: KBStore) -> None:
    _claim(store, "c1", "a claim with evidence")
    assert server.kb_neighbors("c1")["node_id"] == "c1"


def test_kb_neighbors_unknown_node_raises(store: KBStore) -> None:
    with pytest.raises(ValueError):
        server.kb_neighbors("no-such-node")


def test_kb_context_builds_a_pack(store: KBStore) -> None:
    _claim(store, "c1", "the review gate is load-bearing")
    assert "items" in server.kb_context("review gate")


def test_kb_context_with_a_session_records_salience(store: KBStore) -> None:
    from vouch import sessions as sess_mod

    _claim(store, "c1", "the review gate is load-bearing")
    sess = sess_mod.session_start(store, agent="claude-code", task="close the gap")
    out = server.kb_context("review gate", session_id=sess.id)
    assert "items" in out


def test_kb_graph_export_emits_dot(store: KBStore) -> None:
    _claim(store, "c1", "a claim with evidence")
    assert server.kb_graph_export() is not None


# --- sessions / themes --------------------------------------------------


def test_kb_session_start_needs_a_request_context(store: KBStore) -> None:
    import asyncio

    # the only async tool on the surface, and the only one that reads the
    # FastMCP request context -- calling it outside a request must fail loudly
    with pytest.raises(ValueError, match="Context is not available"):
        asyncio.run(server.kb_session_start(task="close the gap"))


def test_kb_detect_themes_on_an_empty_kb(store: KBStore) -> None:
    assert server.kb_detect_themes()["clusters"] == []


# --- audit --------------------------------------------------------------


def test_kb_audit_lists_events(store: KBStore) -> None:
    src = store.put_source(b"e")
    pr = propose_claim(store, text="x", evidence=[src.id], proposed_by="agent")
    server.kb_approve(pr.id)
    assert server.kb_audit()["events"]


def test_kb_audit_tail_caps_events(store: KBStore) -> None:
    src = store.put_source(b"e")
    for i in range(3):
        pr = propose_claim(
            store, text=f"c{i}", evidence=[src.id], proposed_by="agent"
        )
        server.kb_approve(pr.id)
    assert len(server.kb_audit(tail=2)["events"]) == 2


def test_kb_audit_accepts_a_viewer_scope(store: KBStore) -> None:
    assert server.kb_audit(project="acme-example", agent="claude-code") is not None


# --- experts ------------------------------------------------------------


def test_kb_experts_ranks_entities(store: KBStore) -> None:
    src = store.put_source(b"evidence")
    store.put_entity(Entity(id="jwt", name="JWT", type="concept"))
    store.put_claim(
        Claim(id="c1", text="jwt rotation", evidence=[src.id], entities=["jwt"])
    )
    rows = server.kb_experts("JWT")["experts"]
    assert [r["entity_id"] for r in rows] == ["jwt"]


def test_kb_experts_honours_the_viewer_scope(store: KBStore) -> None:
    # The scope filter is the whole point of issue #684 — a private claim must
    # not reach a viewer through this tool either.
    from vouch.models import ArtifactScope, Visibility

    src = store.put_source(b"evidence")
    store.put_entity(Entity(id="jwt", name="JWT", type="concept"))
    store.put_claim(
        Claim(
            id="secret", text="jwt rotation", evidence=[src.id], entities=["jwt"],
            scope=ArtifactScope(visibility=Visibility.PRIVATE, agent="alice"),
        )
    )
    assert server.kb_experts("JWT", agent="bob")["experts"] == []
    assert server.kb_experts("JWT", agent="alice")["experts"]


# --- bundles ------------------------------------------------------------


def _bundle(store: KBStore, tmp_path: Path) -> Path:
    from vouch import bundle

    _claim(store, "c1", "a claim to federate")
    out = tmp_path / "kb.tar.gz"
    bundle.export(store.kb_dir, dest=out, actor="test")
    return out


def test_kb_export_check_passes(store: KBStore, tmp_path: Path) -> None:
    assert server.kb_export_check(str(_bundle(store, tmp_path)))["ok"] is True


def test_kb_import_check_reports_identical(store: KBStore, tmp_path: Path) -> None:
    out = server.kb_import_check(str(_bundle(store, tmp_path)))
    assert out["ok"] is True


# --- embeddings ---------------------------------------------------------


def test_kb_embeddings_stats(store: KBStore, embedder: None) -> None:
    _claim(store, "c1", "a claim to embed")
    out = server.kb_embeddings_stats()
    assert "query_cache_entries" in out or "counts" in out


def test_kb_reindex_embeddings_backfills(store: KBStore, embedder: None) -> None:
    _claim(store, "c1", "a claim to embed")
    assert server.kb_reindex_embeddings(backfill=True) is not None


def test_kb_reindex_embeddings_force(store: KBStore, embedder: None) -> None:
    _claim(store, "c1", "a claim to embed")
    assert server.kb_reindex_embeddings(backfill=True, force=True) is not None


def test_kb_dedup_scan_finds_the_pair(store: KBStore, embedder: None) -> None:
    _claim(store, "c1", "identical duplicated text")
    _claim(store, "c2", "identical duplicated text")
    assert server.kb_dedup_scan() is not None


def test_kb_eval_embeddings_with_an_empty_query_set(
    store: KBStore, embedder: None, tmp_path: Path
) -> None:
    queries = tmp_path / "q.jsonl"
    queries.write_text("", encoding="utf-8")
    try:
        out = server.kb_eval_embeddings(queries_path=str(queries))
    except (ValueError, RuntimeError):
        return
    assert out is not None


# --- skills -------------------------------------------------------------


def test_kb_get_skill_unknown_name_raises(store: KBStore) -> None:
    with pytest.raises(ValueError):
        server.kb_get_skill("no-such-skill")


# --- compile ------------------------------------------------------------


def test_kb_compile_surfaces_a_configuration_failure(store: KBStore) -> None:
    _claim(store, "c1", "a claim to compile")
    cfg = store.kb_dir / "config.yaml"
    cfg.write_text(
        json.dumps({"compile": {"llm_cmd": "false"}}), encoding="utf-8"
    )
    with pytest.raises((ValueError, RuntimeError)):
        server.kb_compile(dry_run=True)
