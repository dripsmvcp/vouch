"""Note-vault importers: obsidian, joplin, apple notes, keep, markdown (#612)."""

from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from vouch import note_import
from vouch.cli import cli
from vouch.models import ProposalKind, ProposalStatus
from vouch.note_import import NoteImportError, import_vault, load_vault, session_key
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    return KBStore.init(tmp_path / "kb")


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """A small obsidian vault: frontmatter, a wikilink, a subfolder, machinery."""
    root = tmp_path / "vault"
    (root / "notes").mkdir(parents=True)
    (root / ".obsidian").mkdir()
    (root / ".obsidian" / "app.json").write_text("{}", encoding="utf-8")
    (root / "Postgres.md").write_text(
        "---\ntitle: Postgres tuning\ntags: [db, ops]\nowner: alice-example\n---\n"
        "The connection pool must be sized to the worker count, never higher.\n"
        "See [[Deploys]] for how this rolls out.\n",
        encoding="utf-8",
    )
    (root / "notes" / "Deploys.md").write_text(
        "# Deploys\n\nDeploys run every second Tuesday from the release branch.\n",
        encoding="utf-8",
    )
    (root / "notes" / "Orphan.md").write_text(
        "A note linking to [[Nothing At All]] which was never written.\n",
        encoding="utf-8",
    )
    return root


# --- parsing ---------------------------------------------------------------


def test_split_frontmatter_reads_a_mapping() -> None:
    fm, body = note_import.split_frontmatter("---\na: 1\nb: two\n---\nbody here\n")
    assert fm == {"a": 1, "b": "two"}
    assert body.strip() == "body here"


def test_split_frontmatter_leaves_malformed_blocks_alone() -> None:
    # A vault is other people's files: one bad note must not fail the import.
    text = "---\n: : :\n  - broken\n---\nbody\n"
    fm, body = note_import.split_frontmatter(text)
    assert fm == {}
    assert body == text


def test_split_frontmatter_ignores_a_non_mapping_block() -> None:
    fm, body = note_import.split_frontmatter("---\n- a\n- b\n---\nbody\n")
    assert fm == {}
    assert "- a" in body


def test_obsidian_loader_reads_titles_tags_and_links(vault: Path) -> None:
    notes = {n.key: n for n in load_vault("obsidian", vault)}
    assert set(notes) == {"Postgres", "notes/Deploys", "notes/Orphan"}
    pg = notes["Postgres"]
    assert pg.title == "Postgres tuning"  # frontmatter wins
    assert pg.tags == ["db", "ops"]
    assert pg.links == ["Deploys"]
    assert pg.locator == "obsidian:Postgres.md"
    assert notes["notes/Deploys"].title == "Deploys"  # falls back to the heading
    # `.obsidian/` is machinery, not notes
    assert not any(k.startswith(".obsidian") for k in notes)


def test_wikilink_variants_are_normalised() -> None:
    links = note_import._wikilinks(
        "[[Plain]] [[Aliased|shown as this]] [[Deep#heading]] [[Block^ref]] [[Plain]]"
    )
    assert links == ["Plain", "Aliased", "Deep", "Block"]


def test_loader_rejects_an_unknown_kind(vault: Path) -> None:
    with pytest.raises(NoteImportError, match="unknown vault kind"):
        load_vault("evernote", vault)


def test_loader_rejects_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(NoteImportError, match="no such path"):
        load_vault("obsidian", tmp_path / "nope")


def test_empty_folder_is_an_actionable_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(NoteImportError, match="no markdown notes"):
        load_vault("md", empty)


# --- import ----------------------------------------------------------------


def test_import_files_one_pending_page_per_note(store: KBStore, vault: Path) -> None:
    report = import_vault(store, "obsidian", vault, generated_at="2026-07-31T00:00:00Z")
    assert report["notes"] == 3
    assert report["imported"] == 3
    pages = [p for p in store.list_proposals(None) if p.kind == ProposalKind.PAGE]
    assert len(pages) == 3
    assert all(p.status is ProposalStatus.PENDING for p in pages)
    titles = {p.payload["title"] for p in pages}
    assert "Postgres tuning" in titles


def test_page_cites_a_source_holding_the_note_verbatim(
    store: KBStore, vault: Path
) -> None:
    # The whole advantage over an embedding importer: the source is the note's
    # own bytes, so a claim extracted from it quotes real offsets.
    import_vault(store, "obsidian", vault)
    page = next(
        p for p in store.list_proposals(None)
        if p.kind == ProposalKind.PAGE and p.payload["title"] == "Postgres tuning"
    )
    (source_id,) = page.payload["sources"]
    content = store.read_source_content(source_id).decode("utf-8")
    assert "The connection pool must be sized to the worker count" in content
    source = store.get_source(source_id)
    assert source.locator == "obsidian:Postgres.md"
    assert source.metadata["note_key"] == "Postgres"
    assert source.metadata["frontmatter"]["owner"] == "alice-example"


def test_reimport_of_an_unchanged_vault_is_a_no_op(store: KBStore, vault: Path) -> None:
    first = import_vault(store, "obsidian", vault, generated_at="2026-07-31T00:00:00Z")
    assert first["imported"] == 3
    second = import_vault(store, "obsidian", vault, generated_at="2026-08-01T00:00:00Z")
    assert second["imported"] == 0
    assert second["skipped"] == 3
    assert all(r["reason"] == "unchanged" for r in second["rows"])
    pages = [p for p in store.list_proposals(None) if p.kind == ProposalKind.PAGE]
    assert len(pages) == 3  # nothing duplicated


def test_a_changed_note_refreshes_its_pending_proposal_in_place(
    store: KBStore, vault: Path
) -> None:
    import_vault(store, "obsidian", vault)
    before = [p for p in store.list_proposals(None) if p.kind == ProposalKind.PAGE]
    (vault / "notes" / "Deploys.md").write_text(
        "# Deploys\n\nDeploys now run weekly, on Thursdays.\n", encoding="utf-8"
    )
    report = import_vault(store, "obsidian", vault)
    assert report["updated"] == 1
    assert report["skipped"] == 2
    after = [p for p in store.list_proposals(None) if p.kind == ProposalKind.PAGE]
    assert len(after) == len(before)  # refreshed, not re-filed
    refreshed = next(p for p in after if p.payload["title"] == "Deploys")
    assert "weekly, on Thursdays" in refreshed.payload["body"]


def test_a_decided_proposal_blocks_reimport(store: KBStore, vault: Path) -> None:
    import_vault(store, "obsidian", vault)
    page = next(
        p for p in store.list_proposals(None)
        if p.kind == ProposalKind.PAGE and p.payload["title"] == "Deploys"
    )
    from vouch.proposals import reject

    reject(store, page.id, rejected_by="reviewer-example", reason="not wanted")
    (vault / "notes" / "Deploys.md").write_text("# Deploys\n\nchanged\n", encoding="utf-8")
    report = import_vault(store, "obsidian", vault)
    row = next(r for r in report["rows"] if r["note"] == "notes/Deploys")
    assert row["action"] == "skipped"
    assert row["reason"] == "already-imported"


def test_limit_slices_the_vault_deterministically(store: KBStore, vault: Path) -> None:
    first = import_vault(store, "obsidian", vault, limit=2)
    assert first["imported"] == 2
    # The remaining note lands on the next run, and the first two are no-ops —
    # which is what makes a big vault resumable.
    second = import_vault(store, "obsidian", vault)
    assert second["imported"] == 1
    assert second["skipped"] == 2


def test_dry_run_writes_nothing(store: KBStore, vault: Path) -> None:
    report = import_vault(store, "obsidian", vault, dry_run=True)
    assert report["imported"] == 3
    assert report["dry_run"] is True
    assert store.list_proposals(None) == []
    assert not list(store.list_sources())


def test_wikilinks_become_relation_proposals_only_where_they_resolve(
    store: KBStore, vault: Path
) -> None:
    report = import_vault(store, "obsidian", vault)
    assert report["relations"] == 1  # Postgres -> Deploys; the orphan link is dropped
    rels = [p for p in store.list_proposals(None) if p.kind == ProposalKind.RELATION]
    assert len(rels) == 1
    payload = rels[0].payload
    src = store.get_source(payload["source"])
    target = store.get_source(payload["target"])
    assert src.locator == "obsidian:Postgres.md"
    assert target.locator == "obsidian:notes/Deploys.md"
    assert payload["relation"] == "references"


def test_claims_are_off_by_default_and_bounded_when_on(
    store: KBStore, vault: Path
) -> None:
    off = import_vault(store, "obsidian", vault)
    assert off["claims"] == 0
    assert not [p for p in store.list_proposals(None) if p.kind == ProposalKind.CLAIM]

    fresh = KBStore.init(vault.parent / "kb2")
    on = import_vault(fresh, "obsidian", vault, max_claims=1)
    assert on["claims"] > 0
    claims = [p for p in fresh.list_proposals(None) if p.kind == ProposalKind.CLAIM]
    assert claims
    assert all(p.status is ProposalStatus.PENDING for p in claims)  # gate intact
    # `--max-claims` is a per-note ceiling: 3 notes, at most 1 claim each.
    assert len(claims) <= 3


def test_session_key_is_stable_and_kind_scoped() -> None:
    assert session_key("obsidian", "a/b") == session_key("obsidian", "a/b")
    assert session_key("obsidian", "a/b") != session_key("md", "a/b")


# --- the other four formats ------------------------------------------------


def _joplin_note(note_id: str, title: str, body: str, extra: str = "") -> str:
    return (
        f"{title}\n\n{body}\n\n"
        f"id: {note_id}\n"
        f"parent_id: 0123456789abcdef0123456789abcdef\n"
        f"created_time: 2026-01-01T00:00:00.000Z\n"
        f"{extra}"
        f"type_: 1"
    )


def test_joplin_folder_import(store: KBStore, tmp_path: Path) -> None:
    export = tmp_path / "joplin"
    export.mkdir()
    a_id = "a" * 32
    b_id = "b" * 32
    (export / f"{a_id}.md").write_text(
        _joplin_note(a_id, "Runbook", f"Restart order matters. See [link](:/{b_id})."),
        encoding="utf-8",
    )
    (export / f"{b_id}.md").write_text(
        _joplin_note(b_id, "Escalation", "Page the on-call after ten minutes."),
        encoding="utf-8",
    )
    # A folder record must not be imported as a note.
    (export / "folder.md").write_text(
        "Notebook\n\nid: c0ffee00000000000000000000000000\ntype_: 2", encoding="utf-8"
    )
    report = import_vault(store, "joplin", export)
    assert report["notes"] == 2
    assert report["imported"] == 2
    assert report["relations"] == 1  # the `:/id` link resolves
    titles = {
        p.payload["title"]
        for p in store.list_proposals(None)
        if p.kind == ProposalKind.PAGE
    }
    assert titles == {"Runbook", "Escalation"}


def test_joplin_jex_archive_import(store: KBStore, tmp_path: Path) -> None:
    note_path = tmp_path / f"{'d' * 32}.md"
    note_path.write_text(_joplin_note("d" * 32, "From a jex", "Body text."), "utf-8")
    jex = tmp_path / "export.jex"
    with tarfile.open(jex, "w") as tar:
        tar.add(note_path, arcname=note_path.name)
    report = import_vault(store, "joplin", jex)
    assert report["imported"] == 1


def test_joplin_rejects_a_non_archive_file(store: KBStore, tmp_path: Path) -> None:
    bogus = tmp_path / "notes.jex"
    bogus.write_text("not a tar", encoding="utf-8")
    with pytest.raises(NoteImportError, match=r"not a joplin \.jex archive"):
        import_vault(store, "joplin", bogus)


def test_apple_notes_html_import(store: KBStore, tmp_path: Path) -> None:
    export = tmp_path / "notes-export"
    export.mkdir()
    (export / "Grocery.html").write_text(
        "<html><head><style>b{}</style></head><body>"
        "<h1>Grocery</h1><div>Oat milk</div><div>Coffee beans</div>"
        "<script>ignored()</script></body></html>",
        encoding="utf-8",
    )
    report = import_vault(store, "notes", export)
    assert report["imported"] == 1
    page = next(
        p for p in store.list_proposals(None) if p.kind == ProposalKind.PAGE
    )
    assert page.payload["title"] == "Grocery"
    body = page.payload["body"]
    assert "Oat milk" in body and "Coffee beans" in body
    assert "ignored()" not in body  # script/style are dropped, not read as text


def test_html_to_text_survives_malformed_markup() -> None:
    assert "hello" in note_import.html_to_text("<p>hello<<</p")


def test_google_keep_folder_import(store: KBStore, tmp_path: Path) -> None:
    export = tmp_path / "Keep"
    export.mkdir()
    (export / "Shopping.json").write_text(json.dumps({
        "title": "Shopping",
        "textContent": "for the weekend",
        "listContent": [
            {"text": "bread", "isChecked": False},
            {"text": "milk", "isChecked": True},
        ],
        "labels": [{"name": "errands"}],
        "userEditedTimestampUsec": 1767225600000000,
    }), encoding="utf-8")
    (export / "Trashed.json").write_text(
        json.dumps({"title": "Old", "textContent": "x", "isTrashed": True}),
        encoding="utf-8",
    )
    report = import_vault(store, "keep", export)
    assert report["imported"] == 1  # the trashed note is skipped
    page = next(p for p in store.list_proposals(None) if p.kind == ProposalKind.PAGE)
    assert page.payload["title"] == "Shopping"
    assert "- [ ] bread" in page.payload["body"]
    assert "- [x] milk" in page.payload["body"]


def test_google_keep_takeout_zip_import(store: KBStore, tmp_path: Path) -> None:
    archive = tmp_path / "takeout.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "Takeout/Keep/Idea.json",
            json.dumps({"title": "Idea", "textContent": "ship the importer"}),
        )
        zf.writestr("Takeout/Mail/ignored.json", json.dumps({"title": "no"}))
    report = import_vault(store, "keep", archive)
    assert report["imported"] == 1


def test_markdown_folder_import(store: KBStore, tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "one.md").write_text("# One\n\nfirst\n", encoding="utf-8")
    (folder / "two.txt").write_text("plain text note\n", encoding="utf-8")
    report = import_vault(store, "md", folder)
    assert report["notes"] == 2
    assert report["kind"] == "md"


# --- cli -------------------------------------------------------------------


def _run(store: KBStore, args: list[str]):
    return CliRunner().invoke(cli, args, env={"VOUCH_KB_PATH": str(store.kb_dir)})


def test_cli_import_obsidian(store: KBStore, vault: Path) -> None:
    result = _run(store, ["import", "obsidian", str(vault), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["imported"] == 3
    assert report["kind"] == "obsidian"


def test_cli_import_dry_run_reports_without_writing(
    store: KBStore, vault: Path
) -> None:
    result = _run(store, ["import", "obsidian", str(vault), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would import 3 new" in result.output
    assert store.list_proposals(None) == []


def test_cli_import_max_claims_is_forwarded(store: KBStore, vault: Path) -> None:
    result = _run(
        store, ["import", "md", str(vault), "--max-claims", "1", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["claims"] > 0


def test_cli_import_group_lists_every_kind() -> None:
    result = CliRunner().invoke(cli, ["import", "--help"])
    assert result.exit_code == 0
    for kind in ("obsidian", "joplin", "notes", "keep", "md", "chatgpt"):
        assert kind in result.output


def test_cli_import_reports_a_bad_vault_cleanly(store: KBStore, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _run(store, ["import", "obsidian", str(empty)])
    assert result.exit_code != 0
    assert "no markdown notes" in result.output
    assert "Traceback" not in result.output
