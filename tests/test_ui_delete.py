"""Quick delete of transcripts: library tile, selection bar and detail page, with an in-page dialog."""

import re

from mgtranscriber.library import Metadata
from mgtranscriber.models import Segment

from .conftest import PROJECT_ROOT
from .test_ui_watch import page, server  # noqa: F401  (fixtures)


def make(library, name):
    folder = library.create(name, "file:/" + name)
    library.save_segments(folder.name, [Segment(0, 5, " Hello there.", "SPEAKER_00")])
    library.save_metadata(folder.name, Metadata(title=name, language="en", duration=5))
    return folder.name


def listing(page):  # noqa: F811
    return page.evaluate("fetch('/api/transcripts').then(r => r.json())")


def test_delete_from_library_tile_and_selection(page, server, library):  # noqa: F811
    for n in ("alpha", "beta", "gamma", "delta"):
        make(library, n)
    page.goto(f"{server}/#/")
    page.wait_for_selector(".tile")
    assert len(listing(page)) == 4

    # one tile: the dialog names it, cancel keeps it
    page.click('.tile:has-text("alpha") .tile-del')
    dialog = page.locator(".fx-dialog")
    dialog.wait_for()
    assert "alpha" in dialog.inner_text()
    dialog.locator(".fx-cancel").click()
    assert len(listing(page)) == 4
    page.click('.tile:has-text("alpha") .tile-del')
    page.locator(".fx-dialog .fx-ok").click()
    page.wait_for_function("document.querySelectorAll('.tile').length === 3")
    assert {i["title"] for i in listing(page)} == {"beta", "gamma", "delta"}

    # two selected tiles
    page.check('.tile:has-text("beta") input[data-select]')
    page.check('.tile:has-text("gamma") input[data-select]')
    page.click('[data-sel="delete"]')
    text = page.locator(".fx-dialog").inner_text()
    assert "beta" in text and "gamma" in text and "delta" not in text
    page.locator(".fx-dialog .fx-ok").click()
    page.wait_for_function("document.querySelectorAll('.tile').length === 1")
    assert [i["title"] for i in listing(page)] == ["delta"]


def test_delete_from_detail_goes_to_library(page, server, library):  # noqa: F811
    keep = make(library, "keep me")
    gone = make(library, "delete me")
    page.goto(f"{server}/#/t/{gone}")
    page.wait_for_selector(".d-delete")
    page.wait_for_selector(".d-speakers .speaker")
    page.click(".d-delete")
    assert "delete me" in page.locator(".fx-dialog").inner_text()
    page.locator(".fx-dialog .fx-ok").click()
    page.wait_for_selector(".tile")
    assert page.evaluate("location.hash") in ("#/", "")
    assert [i["id"] for i in listing(page)] == [keep]


def test_i18n_keys_exist_in_both_languages():
    text = (PROJECT_ROOT / "static" / "js" / "i18n" / "remove.js").read_text(encoding="utf-8")
    en, cs = text.split("const cs = {")
    for part in (en, cs):
        for key in ("action", "aria", "title", "text", "more", "confirm", "done", "partial", "failed"):
            assert re.search(rf"^\s+{key}: ", part, re.M), key
    used = (PROJECT_ROOT / "static" / "js" / "components" / "remove.js").read_text(encoding="utf-8")
    for key in re.findall(r't\("remove\.(\w+)"', used):
        assert re.search(rf"^\s+{key}: ", en, re.M), key
