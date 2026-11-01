"""Pre-release performance fixes: paging of the library list, cached names/voices, lazy watch.json saves, LRU."""

import json

from fastapi.testclient import TestClient

from mgtranscriber.library import Library
from mgtranscriber.server import create_app
from mgtranscriber.watch import WatchManager
from tests.test_watch import Clock, FakeQueue, make_file, wsettings  # noqa: F401 - fixture


def seed(output, n):
    for i in range(n):
        folder = output / f"talk-{i:02d}"
        folder.mkdir(parents=True)
        (folder / "prepis.json").write_text(json.dumps([{"start": 0, "end": 1, "text": "hello", "speaker": "SPEAKER_00"}]),
                                            encoding="utf-8")
        (folder / "metadata.json").write_text(json.dumps({"title": f"Talk {i}", "source": "x", "date": f"2026-01-{i + 1:02d}T00:00:00+00:00"}),
                                              encoding="utf-8")


def test_transcript_list_pages_and_stays_a_plain_array(settings):
    seed(settings.output_dir, 5)
    client = TestClient(create_app(settings))
    full = client.get("/api/transcripts")
    assert full.status_code == 200 and len(full.json()) == 5 and full.headers["X-Total-Count"] == "5"
    page = client.get("/api/transcripts?limit=2&offset=1")
    assert page.json() == full.json()[1:3] and page.headers["X-Total-Count"] == "5"
    assert client.get("/api/transcripts?offset=4").json() == full.json()[4:]
    assert client.get("/api/transcripts?limit=0").status_code == 422


def test_summary_notices_renamed_speakers_audio_and_deleted_transcripts(settings):
    seed(settings.output_dir, 1)
    lib = Library(settings.output_dir)
    tid = "talk-00"
    first = lib.summary(tid)
    assert first["names"] == {} and first["audio"] is None and first["has_audio"] is False
    lib.save_names(tid, {"SPEAKER_00": "Eva"})
    assert lib.summary(tid)["names"] == {"SPEAKER_00": "Eva"} == lib.names_cached(tid)
    lib.save_names(tid, {"SPEAKER_00": "Zoe"})
    assert lib.summary(tid)["names"] == {"SPEAKER_00": "Zoe"} == lib.names_cached(tid)
    (settings.output_dir / tid / "audio.mp3").write_bytes(b"x")
    assert lib.summary(tid)["audio"] == "audio.mp3"
    assert lib.audio_file(tid).name == "audio.mp3"


def test_path_still_refuses_traversal_and_missing(settings):
    import pytest
    from mgtranscriber.library import InvalidTranscriptId, TranscriptNotFound

    seed(settings.output_dir, 1)
    lib = Library(settings.output_dir)
    for bad in ("..", ".", "a/b", "a" + chr(92) + "b", "c:x", ""):
        with pytest.raises(InvalidTranscriptId):
            lib.path(bad)
    with pytest.raises(TranscriptNotFound):
        lib.path("nope")
    assert lib.path("talk-00") == (settings.output_dir / "talk-00").resolve()


def test_quiet_watch_checks_do_not_rewrite_the_file_but_changes_are_saved_at_once(wsettings, tmp_path):  # noqa: F811
    settings = wsettings
    clock = Clock()
    manager = WatchManager(settings, FakeQueue(), clock=clock)
    watched = tmp_path / "in"
    make_file(watched, "old.mp3", mtime=clock.now - 100)
    src = manager.create({"kind": "folder", "target": str(watched), "interval_s": 5})
    manager.check(src.id)  # the baseline: saved
    saved = manager.path.read_text(encoding="utf-8")
    for _ in range(3):  # nothing new: bookkeeping stays in memory
        clock.now += 10
        manager.check(src.id)
    assert manager.path.read_text(encoding="utf-8") == saved
    assert src.last_check == clock.now and manager._dirty_since is not None
    manager.stop()  # a clean stop writes it
    assert manager.path.read_text(encoding="utf-8") != saved
    # an error is saved immediately
    watched.rename(tmp_path / "gone")
    clock.now += 10
    manager.check(src.id)
    assert json.loads(manager.path.read_text(encoding="utf-8"))["sources"][0]["last_error"]


def test_retriever_vectors_does_not_deadlock_on_lazy_preparation():
    """vectors() holds the lock while the lazily prepared texts are read (a deadlock hung every retrieval-mode question)."""
    import threading

    from mgtranscriber.llm.qa import Retriever
    from mgtranscriber.llm.transcript import build_turns
    from mgtranscriber.models import Segment

    class Provider:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    turns = build_turns([Segment(start=float(i), end=i + 0.9, text=f"sentence {i}", speaker="SPEAKER_00") for i in range(5)])
    retriever = Retriever(turns, {"SPEAKER_00": "Eva"}, Provider())
    out = {}
    worker = threading.Thread(target=lambda: out.update(vectors=retriever.vectors(), texts=retriever.texts), daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "Retriever.vectors() deadlocked"
    assert len(out["vectors"]) == len(out["texts"]) >= 1
