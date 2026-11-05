"""The "Source" line of the detail page: GET /api/transcripts/{id} returns ``origin``."""

import asyncio

import httpx
import pytest

from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.library import Metadata
from mgtranscriber.models import Segment
from mgtranscriber.server.app import create_app


def make(app, key: str, **fields) -> str:
    lib = app.state.library
    path = lib.create(fields.get("title", "T"), key)
    meta = Metadata(title=fields.pop("title", "T"), **fields)
    lib.save_segments(path.name, [Segment.from_dict({"start": 0.0, "end": 1.0, "text": "hi"})])
    lib.save_metadata(path.name, meta)
    return path.name


def get(app, tid: str, peer: str = "127.0.0.1") -> dict:
    async def run():
        transport = httpx.ASGITransport(app=app, client=(peer, 1))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            r = await c.get(f"/api/transcripts/{tid}")
            assert r.status_code == 200, r.text
            return r.json()
    return asyncio.run(run())


@pytest.fixture
def app(settings):
    return create_app(settings, runner=InProcessRunner())


def test_youtube_url_with_channel(app):
    tid = make(app, "k1", source="https://youtu.be/abc", extra={"source_info": {
        "webpage_url": "https://www.youtube.com/watch?v=abc", "channel": "Chan", "channel_url": "https://www.youtube.com/@chan"}})
    origin = get(app, tid)["origin"]
    assert origin == {"url": "https://www.youtube.com/watch?v=abc", "channel": "Chan", "channel_url": "https://www.youtube.com/@chan"}


def test_uploader_fallback_and_bad_channel_url(app):
    tid = make(app, "k2", source="https://example.com/v", extra={"source_info": {
        "uploader": "Up", "uploader_url": "javascript:alert(1)"}})
    origin = get(app, tid)["origin"]
    assert origin == {"url": "https://example.com/v", "channel": "Up"}


def test_javascript_url_refused(app):
    tid = make(app, "k3", source="javascript:alert(1)", extra={"source_info": {"webpage_url": "javascript:alert(2)"}})
    origin = get(app, tid)["origin"]
    assert not origin or "url" not in origin
    assert "javascript" not in str(origin)


def test_watch_origin_from_metadata_and_from_items(app):
    tid = make(app, "k4", source="https://example.com/a", extra={"watch": {"id": "w1", "name": "My feed", "kind": "rss"}})
    assert get(app, tid)["origin"]["watch"] == {"id": "w1", "name": "My feed", "kind": "rss"}

    # an older transcript: found through the watch source's items
    old = make(app, "k5", source="https://example.com/b")
    manager = app.state.watch
    from mgtranscriber.watch.models import Source  # noqa: PLC0415
    src = Source.from_dict({"id": "w2", "name": "Chan feed", "kind": "youtube", "target": "https://youtube.com/@x",
                            "items": [{"key": "b", "source": "https://example.com/b", "title": "b", "result": old}]})
    manager.sources["w2"] = src
    assert get(app, old)["origin"]["watch"] == {"id": "w2", "name": "Chan feed", "kind": "youtube"}


def test_local_file_name_only_for_remote_peer(app):
    tid = make(app, "k6", source=r"C:\Users\me\Videos\talk.mp4")
    local = get(app, tid, "127.0.0.1")["origin"]
    remote = get(app, tid, "9.9.9.9")["origin"]
    assert local == {"file": "talk.mp4", "path": r"C:\Users\me\Videos\talk.mp4"}
    assert remote == {"file": "talk.mp4"}
    assert "Users" not in str(remote)


def test_upload_shows_nothing(app):
    tid = make(app, "k7", source=r"C:\srv\uploads\x\rec.webm", extra={"uploaded_via": "web-upload"})
    assert get(app, tid)["origin"] is None


# --- the watch manager stores the origin when a job finishes ------------------------------------

def _finish(settings, library, status="done", result="tid", with_library=True):
    from types import SimpleNamespace  # noqa: PLC0415

    from mgtranscriber.watch.manager import WatchManager  # noqa: PLC0415
    from mgtranscriber.watch.models import Source  # noqa: PLC0415
    manager = WatchManager(settings, SimpleNamespace(listing=lambda: []), library=library if with_library else None)
    src = Source.from_dict({"id": "w9", "name": "Podcast", "kind": "rss", "target": "https://example.com/feed",
                            "items": [{"key": "k", "source": "https://example.com/e", "title": "e", "job": "7"}]})
    src.pending["7"] = "k"
    manager.sources["w9"] = src
    job = SimpleNamespace(id=7, status=status, result=result, error="boom" if status != "done" else None,
                          title="e", add_log=lambda line: None, request=SimpleNamespace(source="https://example.com/e"))
    manager.on_job_finished(job)
    return manager


def _transcript(library, **extra):
    path = library.create("T", "wk")
    library.save_segments(path.name, [Segment.from_dict({"start": 0.0, "end": 1.0, "text": "hi"})])
    library.save_metadata(path.name, Metadata(title="T", source="https://example.com/e", extra=extra))
    return path.name


def test_finished_job_stores_watch_origin_and_keeps_extra(settings, library):
    tid = _transcript(library, source_info={"channel": "C"}, custody={"x": 1})
    _finish(settings, library, result=tid)
    extra = library.metadata(tid).extra
    assert extra["watch"] == {"id": "w9", "name": "Podcast", "kind": "rss"}
    assert extra["source_info"] == {"channel": "C"} and extra["custody"] == {"x": 1}


def test_failed_job_stores_nothing(settings, library):
    tid = _transcript(library)
    _finish(settings, library, status="error", result=None)
    assert "watch" not in library.metadata(tid).extra
    tid2 = _transcript(library)
    _finish(settings, library, status="error", result=tid2)  # even with a result id
    assert "watch" not in library.metadata(tid2).extra


def test_missing_library_is_a_noop(settings, library):
    tid = _transcript(library)
    manager = _finish(settings, library, result=tid, with_library=False)
    assert manager.library is None
    assert "watch" not in library.metadata(tid).extra
