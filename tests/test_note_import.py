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


def test_cli_human_output_reports_claims_and_links(
    store: KBStore, vault: Path
) -> None:
    result = _run(store, ["import", "obsidian", str(vault), "--max-claims", "1"])
    assert result.exit_code == 0, result.output
    assert "claim(s)" in result.output
    assert "link relation(s) proposed" in result.output
    assert "run `vouch review` to decide." in result.output
    # skipped rows are counted, not listed
    again = _run(store, ["import", "obsidian", str(vault)])
    assert "3 skipped" in again.output
    assert "•" not in again.output


def test_cli_each_kind_reaches_its_loader(store: KBStore, tmp_path: Path) -> None:
    """One CLI test per subcommand — the group is the surface #612 asks for,
    and a subcommand wired to the wrong kind would be invisible otherwise."""
    joplin = tmp_path / "joplin"
    joplin.mkdir()
    (joplin / f"{'e' * 32}.md").write_text(
        _joplin_note("e" * 32, "Joplin note", "body"), encoding="utf-8"
    )
    notes = tmp_path / "apple"
    notes.mkdir()
    (notes / "Note.txt").write_text("apple note body", encoding="utf-8")
    keep = tmp_path / "Keep"
    keep.mkdir()
    (keep / "Keep note.json").write_text(
        json.dumps({"title": "Keep note", "textContent": "keep body"}), encoding="utf-8"
    )
    md = tmp_path / "docs"
    md.mkdir()
    (md / "doc.md").write_text("# Doc\n\nbody\n", encoding="utf-8")

    for kind, path in (
        ("joplin", joplin), ("notes", notes), ("keep", keep), ("md", md)
    ):
        result = _run(store, ["import", kind, str(path), "--json"])
        assert result.exit_code == 0, (kind, result.output)
        report = json.loads(result.output)
        assert report["kind"] == kind
        assert report["imported"] == 1


def test_cli_import_chatgpt_alias(store: KBStore, tmp_path: Path) -> None:
    """`vouch import chatgpt` is the same importer as the flat command — one
    `vouch import <kind>` surface over every source."""
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([{
        "conversation_id": "conv-1",
        "title": "About deploys",
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["u1"]},
            "u1": {
                "id": "u1", "parent": "root", "children": ["a1"],
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["when do we deploy?"]},
                },
            },
            "a1": {
                "id": "a1", "parent": "u1", "children": [],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {
                        "content_type": "text",
                        "parts": ["Every second Tuesday from the release branch."],
                    },
                },
            },
        },
    }]), encoding="utf-8")
    result = _run(store, ["import", "chatgpt", str(export), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["imported"] == 1


# --- the tolerant paths ----------------------------------------------------
#
# A vault is other people's files. Every branch below is a "skip it and carry
# on" that exists so one malformed note cannot fail an import of ten thousand.


def test_clip_truncates_a_long_body(store: KBStore, tmp_path: Path) -> None:
    folder = tmp_path / "long"
    folder.mkdir()
    (folder / "big.md").write_text("x" * 10_000, encoding="utf-8")
    import_vault(store, "md", folder)
    page = next(p for p in store.list_proposals(None) if p.kind == ProposalKind.PAGE)
    assert page.payload["body"].rstrip().endswith("…")


def test_frontmatter_needs_a_delimiter_line_and_a_terminator() -> None:
    # `---title: x` starts with `---` but is not a fence
    assert note_import.split_frontmatter("---title: x\nbody\n")[0] == {}
    # opened and never closed
    assert note_import.split_frontmatter("---\na: 1\nbody with no terminator\n")[0] == {}


def test_frontmatter_tags_accept_a_string_list() -> None:
    fm, _ = note_import.split_frontmatter('---\ntags: "db, ops #infra"\n---\nbody\n')
    assert note_import._tags_from_frontmatter(fm) == ["db", "ops", "infra"]
    assert note_import._tags_from_frontmatter({"tags": 7}) == []


def test_wikilinks_stop_at_the_per_note_ceiling() -> None:
    body = " ".join(f"[[Note{i}]]" for i in range(note_import._MAX_LINKS_PER_NOTE + 10))
    assert len(note_import._wikilinks(body)) == note_import._MAX_LINKS_PER_NOTE


def test_walk_skips_machinery_and_dotfiles(store: KBStore, tmp_path: Path) -> None:
    root = tmp_path / "vault2"
    (root / "_resources").mkdir(parents=True)
    (root / "_resources" / "attached.md").write_text("attachment", encoding="utf-8")
    (root / ".hidden.md").write_text("hidden", encoding="utf-8")
    (root / "Real.md").write_text("# Real\n\nkept\n", encoding="utf-8")
    keys = {n.key for n in load_vault("md", root)}
    assert keys == {"Real"}


def test_walk_skips_a_file_it_cannot_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "racy"
    root.mkdir()
    (root / "gone.md").write_text("body", encoding="utf-8")
    (root / "kept.md").write_text("# Kept\n\nbody\n", encoding="utf-8")
    real_stat = Path.stat

    def flaky(self: Path, *a: object, **kw: object):
        if self.name == "gone.md":
            raise OSError("vanished mid-walk")
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", flaky)
    assert {n.key for n in load_vault("md", root)} == {"kept"}


def test_vault_over_the_byte_ceiling_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(note_import, "_MAX_VAULT_BYTES", 4)
    root = tmp_path / "huge"
    root.mkdir()
    (root / "a.md").write_text("well over four bytes", encoding="utf-8")
    with pytest.raises(NoteImportError, match="import ceiling"):
        load_vault("md", root)


def test_markdown_loader_rejects_a_file(tmp_path: Path) -> None:
    target = tmp_path / "one.md"
    target.write_text("body", encoding="utf-8")
    with pytest.raises(NoteImportError, match="not a directory"):
        load_vault("md", target)


def test_timestamp_helpers_degrade_instead_of_raising() -> None:
    assert note_import._iso("not-a-number") is None
    assert note_import._iso_ms(object()) is None


def test_joplin_skips_notes_whose_footer_is_not_metadata(tmp_path: Path) -> None:
    export = tmp_path / "joplin"
    export.mkdir()
    # trailing block with a line that carries no colon
    (export / "a.md").write_text("Title\n\nbody\n\njust prose\n", encoding="utf-8")
    # trailing block whose "key" has a space in it — prose, not metadata
    (export / "b.md").write_text("Title\n\nbody\n\nnot a key: value\n", encoding="utf-8")
    with pytest.raises(NoteImportError, match="no joplin notes"):
        load_vault("joplin", export)


def test_joplin_missing_path_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(NoteImportError, match="no such path"):
        load_vault("joplin", tmp_path / "nope")
    # the loader guards for itself too — it is importable on its own
    with pytest.raises(NoteImportError, match="no such joplin export"):
        note_import.load_joplin(tmp_path / "nope")


def test_joplin_jex_skips_non_note_members(tmp_path: Path) -> None:
    note = tmp_path / f"{'f' * 32}.md"
    note.write_text(_joplin_note("f" * 32, "Kept", "body"), encoding="utf-8")
    resource = tmp_path / "resource.bin"
    resource.write_bytes(b"attachment")
    inner = tmp_path / "subdir"
    inner.mkdir()
    jex = tmp_path / "mixed.jex"
    with tarfile.open(jex, "w") as tar:
        tar.add(note, arcname=note.name)
        tar.add(resource, arcname=resource.name)
        # a *directory* whose name ends in .md clears the suffix filter but has
        # no readable stream — extractfile returns None
        tar.add(inner, arcname="looks-like-a-note.md")
    notes = load_vault("joplin", jex)
    assert [n.title for n in notes] == ["Kept"]


def test_joplin_jex_over_the_ceiling_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(note_import, "_MAX_VAULT_BYTES", 4)
    note = tmp_path / f"{'g' * 32}.md"
    note.write_text(_joplin_note("g" * 32, "Big", "body"), encoding="utf-8")
    jex = tmp_path / "big.jex"
    with tarfile.open(jex, "w") as tar:
        tar.add(note, arcname=note.name)
    with pytest.raises(NoteImportError, match="import ceiling"):
        load_vault("joplin", jex)


def test_html_to_text_falls_back_when_the_parser_blows_up(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: object, data: str) -> None:
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(note_import._TextExtractor, "feed", boom)
    assert note_import.html_to_text("<p>still readable</p>") == "still readable"


def test_apple_notes_loader_rejects_a_file_and_an_empty_folder(
    tmp_path: Path
) -> None:
    target = tmp_path / "one.html"
    target.write_text("<p>x</p>", encoding="utf-8")
    with pytest.raises(NoteImportError, match="not a directory"):
        load_vault("notes", target)
    empty = tmp_path / "no-notes"
    empty.mkdir()
    with pytest.raises(NoteImportError, match="no apple-notes files"):
        load_vault("notes", empty)


def test_keep_skips_an_entirely_empty_note(tmp_path: Path) -> None:
    export = tmp_path / "Keep"
    export.mkdir()
    (export / "blank.json").write_text(json.dumps({"color": "WHITE"}), encoding="utf-8")
    (export / "real.json").write_text(
        json.dumps({"title": "Real", "textContent": "body"}), encoding="utf-8"
    )
    assert [n.title for n in load_vault("keep", export)] == ["Real"]


def test_keep_skips_undecodable_json(tmp_path: Path) -> None:
    export = tmp_path / "Keep"
    export.mkdir()
    (export / "broken.json").write_text("{not json", encoding="utf-8")
    (export / "real.json").write_text(
        json.dumps({"title": "Real", "textContent": "body"}), encoding="utf-8"
    )
    assert [n.title for n in load_vault("keep", export)] == ["Real"]


def test_keep_zip_skips_undecodable_json_and_non_notes(tmp_path: Path) -> None:
    archive = tmp_path / "takeout.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Takeout/Keep/broken.json", "{not json")
        zf.writestr("Takeout/Keep/attachment.png", b"\x89PNG")  # not json at all
        zf.writestr("Takeout/Keep/", b"")  # a directory entry
        zf.writestr(
            "Takeout/Keep/real.json", json.dumps({"title": "Real", "textContent": "b"})
        )
    assert [n.title for n in load_vault("keep", archive)] == ["Real"]


def test_keep_zip_over_the_ceiling_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(note_import, "_MAX_VAULT_BYTES", 4)
    archive = tmp_path / "takeout.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Takeout/Keep/a.json", json.dumps({"title": "A", "textContent": "b"}))
    with pytest.raises(NoteImportError, match="import ceiling"):
        load_vault("keep", archive)


def test_keep_missing_path_and_empty_export_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(NoteImportError, match="no such path"):
        load_vault("keep", tmp_path / "nope")
    with pytest.raises(NoteImportError, match="no such keep export"):
        note_import.load_google_keep(tmp_path / "nope")
    empty = tmp_path / "Keep"
    empty.mkdir()
    with pytest.raises(NoteImportError, match="no google keep notes"):
        load_vault("keep", empty)


def test_a_link_written_with_its_extension_still_resolves(
    store: KBStore, tmp_path: Path
) -> None:
    root = tmp_path / "vault3"
    root.mkdir()
    (root / "A.md").write_text("links to [[B.md]]\n", encoding="utf-8")
    (root / "B.md").write_text("target\n", encoding="utf-8")
    assert import_vault(store, "obsidian", root)["relations"] == 1


def test_links_are_not_refiled_for_notes_that_were_skipped(
    store: KBStore, vault: Path
) -> None:
    # Second run: every note is unchanged, so no source is registered and the
    # link pass has nothing to file — the relation must not duplicate.
    assert import_vault(store, "obsidian", vault)["relations"] == 1
    assert import_vault(store, "obsidian", vault)["relations"] == 0
    rels = [p for p in store.list_proposals(None) if p.kind == ProposalKind.RELATION]
    assert len(rels) == 1


def test_a_link_into_an_unchanged_note_is_not_refiled(
    store: KBStore, vault: Path
) -> None:
    """The partial case: the linking note changed, its target did not. The
    target has no source this run, so the edge is left alone rather than
    re-proposed against a stale id."""
    assert import_vault(store, "obsidian", vault)["relations"] == 1
    (vault / "Postgres.md").write_text(
        "---\ntitle: Postgres tuning\n---\nrewritten. still see [[Deploys]].\n",
        encoding="utf-8",
    )
    report = import_vault(store, "obsidian", vault)
    assert report["updated"] == 1
    assert report["relations"] == 0
    rels = [p for p in store.list_proposals(None) if p.kind == ProposalKind.RELATION]
    assert len(rels) == 1


def test_a_rejected_relation_does_not_fail_the_import(
    store: KBStore, vault: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        from vouch.proposals import ProposalError

        raise ProposalError("nope")

    monkeypatch.setattr(note_import, "propose_relation", refuse)
    report = import_vault(store, "obsidian", vault)
    assert report["imported"] == 3  # the pages still land
    assert report["relations"] == 0
