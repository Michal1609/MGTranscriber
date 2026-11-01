"""Speaker counts and panel in the browser: UNKNOWN (words without a speaker) is not a person."""


from mgtranscriber.library import Metadata
from mgtranscriber.models import Segment

from .test_ui_watch import page, server  # noqa: F401  (fixtures)


def make(library, name, unknown_seconds):
    folder = library.create(name, "file:/" + name)
    segments = [Segment(0, 30, " Hello there.", "SPEAKER_00"), Segment(30, 60, " Hi.", "SPEAKER_01")]
    if unknown_seconds:
        segments.append(Segment(60, 60 + unknown_seconds, " Mumble.", "UNKNOWN"))
    library.save_segments(folder.name, segments)
    library.save_metadata(folder.name, Metadata(title=name, language="en", duration=70))
    return folder.name


def test_unknown_is_not_counted_and_shown_only_when_it_has_time(page, server, library):  # noqa: F811
    blink = make(library, "blink", 0.1)
    some = make(library, "some", 5)
    page.goto(f"{server}/#/t/{blink}?lang=en")
    page.wait_for_selector(".d-speakers .speaker")
    assert page.locator(".d-speakers .speaker").count() == 2
    assert "2 speakers" in page.inner_text(".d-meta") or "2 mluvčí" in page.inner_text(".d-meta")
    assert page.locator(".d-speakers .speaker.unassigned").count() == 0

    page.goto(f"{server}/#/t/{some}")
    page.wait_for_selector(".d-speakers .speaker.unassigned")
    rows = page.locator(".d-speakers .speaker")
    assert rows.count() == 3 and "unassigned" in rows.nth(2).get_attribute("class")  # last, separate
    assert "2 speakers" in page.inner_text(".d-meta") or "2 mluvčí" in page.inner_text(".d-meta")
    assert "UNKNOWN" not in page.inner_text(".d-speakers .unassigned small")

    page.goto(f"{server}/#/")
    page.wait_for_selector(".card .meta")
    texts = [page.locator(".card .meta").nth(i).inner_text() for i in range(page.locator(".card .meta").count())]
    assert texts and all(("2 mluvčí" in t or "2 speakers" in t) for t in texts if "mluv" in t or "speak" in t)
