"""POST /api/jobs/upload: the web UI's file upload (a queued job; the file becomes the transcript's audio)."""

import pytest
from fastapi.testclient import TestClient

from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.server import create_app

from .conftest import wait_for


def make_client(settings, **overrides):
    return TestClient(create_app(settings.with_overrides(**overrides), runner=InProcessRunner()))


def post(c, name="talk.wav", content=b"RIFF fake", mime="audio/wav", **data):
    return c.post("/api/jobs/upload", files={"file": (name, content, mime)}, data=data)


def job_state(c, job_id):
    return c.get(f"/api/jobs/{job_id}").json()


def test_upload_becomes_a_job_and_the_audio_of_the_transcript(settings):
    with make_client(settings) as c:
        r = post(c, "My talk.wav", language="en", speakers="2")
        assert r.status_code == 201
        job = r.json()
        assert job["title"] == "My talk.wav" or job["status"] != "queued"
        assert job["language"] == "en" and job["speakers"] == 2
        wait_for(lambda: job_state(c, job["id"])["status"] in ("done", "error"))
        done = job_state(c, job["id"])
        assert done["status"] == "done", done
        tid = done["result"]
        # The end-of-job hook runs on its own thread just after the job: it moves the file in (audio_file), then
        # saves the metadata, then removes the upload folder. Wait for its last step, not for the first.
        uploads = settings.data_dir / "uploads"
        wait_for(lambda: c.app.state.library.audio_file(tid)
                 and c.app.state.library.metadata(tid).extra.get("uploaded_via")
                 and not list(uploads.glob("*")))
        meta = c.app.state.library.metadata(tid)
        assert meta.source == "My talk.wav" and meta.extra["uploaded_via"] == "web-upload"
        assert c.app.state.library.audio_file(tid).name == "audio.wav"
        assert c.get(f"/api/transcripts/{tid}/files/audio.wav").status_code == 200
        assert not list((settings.data_dir / "uploads").glob("*/*"))  # nothing left over


def test_works_without_local_paths_and_offline(settings, monkeypatch):
    with make_client(settings, allow_local_paths="false") as c:
        assert c.post("/api/jobs", json={"source": "C:/x.wav"}).status_code == 403  # a path is refused ...
        assert post(c).status_code == 201  # ... an upload is not a path
    offline = settings.with_overrides(extra={**settings.extra, "offline_mode": True})
    with TestClient(create_app(offline, runner=InProcessRunner())) as c:
        assert c.post("/api/jobs", json={"source": "https://example.com/v"}).status_code == 403
        r = post(c, language="en")
        assert r.status_code == 201
        wait_for(lambda: job_state(c, r.json()["id"])["status"] == "done")


def test_internal_upload_paths_are_not_valid_sources(settings):
    with make_client(settings, allow_local_paths="true") as c:
        folder = settings.data_dir / "uploads" / "abc"
        folder.mkdir(parents=True)
        (folder / "x.wav").write_bytes(b"RIFF fake")
        assert c.post("/api/jobs", json={"source": str(folder / "x.wav")}).status_code == 400
        assert (folder / "x.wav").exists()


def test_too_large_is_413(settings):
    with make_client(settings, max_upload_mb=1) as c:
        r = post(c, content=b"\0" * (1024 * 1024 + 10))
        assert r.status_code == 413 and r.json()["code"] == "request_too_large"
        assert not list((settings.data_dir / "uploads").glob("*/*"))


@pytest.mark.parametrize("name,mime", [("notes.txt", "text/plain"), ("talk.wav", "text/html"), ("run.exe", "audio/wav"),
                                       ("noext", "audio/wav")])
def test_wrong_type_is_415(settings, name, mime):
    with make_client(settings) as c:
        r = post(c, name, mime=mime)
        assert r.status_code == 415 and r.json()["code"] == "unsupported_media_type"
        assert not (settings.data_dir / "uploads").exists() or not list((settings.data_dir / "uploads").glob("*/*"))


def test_unknown_mime_with_media_extension_is_accepted(settings):
    with make_client(settings) as c:
        assert post(c, "talk.m4a", mime="application/octet-stream").status_code == 201
        assert post(c, "clip.mkv", mime="video/x-matroska").status_code == 201


def test_empty_file_and_bad_options(settings):
    with make_client(settings) as c:
        assert post(c, content=b"").status_code == 400
        assert post(c, language="english").status_code == 422
        assert post(c, speakers="99").status_code == 422
        r = post(c, engine="nope")
        assert r.status_code == 400 and r.json()["code"] == "unknown_engine"
        assert not list((settings.data_dir / "uploads").glob("*/*"))


def test_cross_origin_browser_request_is_rejected(settings):
    with make_client(settings) as c:
        r = c.post("/api/jobs/upload", files={"file": ("a.wav", b"RIFF fake", "audio/wav")},
                   headers={"Origin": "http://evil.example"})
        assert r.status_code == 403 and r.json()["code"] == "cross_origin"
        r = c.post("/api/jobs/upload", files={"file": ("a.wav", b"RIFF fake", "audio/wav")},
                   headers={"Origin": "http://testserver"})
        assert r.status_code == 201  # the page's own origin


def test_api_key_is_required(settings):
    with make_client(settings, api_key="secret") as c:
        assert post(c).status_code == 401
        r = c.post("/api/jobs/upload", files={"file": ("a.wav", b"RIFF fake", "audio/wav")},
                   headers={"X-API-Key": "secret"})
        assert r.status_code == 201


def test_failed_and_cancelled_uploads_are_deleted(settings):
    with make_client(settings) as c:
        r = post(c, "fail.wav")
        wait_for(lambda: job_state(c, r.json()["id"])["status"] == "error")
        wait_for(lambda: not list((settings.data_dir / "uploads").glob("*/*")))
        slow = post(c, "slow.wav").json()  # runs ~10 s; the second one waits in the queue
        waiting = post(c, "queued.wav").json()
        assert c.delete(f"/api/jobs/{waiting['id']}").json()["status"] == "cancelled"
        assert len(list((settings.data_dir / "uploads").glob("*/*"))) == 1
        c.delete(f"/api/jobs/{slow['id']}")
        wait_for(lambda: not list((settings.data_dir / "uploads").glob("*/*")))


def test_info_reports_the_limits(settings):
    with make_client(settings, max_upload_mb=7) as c:
        upload = c.get("/api/info").json()["upload"]
        assert upload["max_mb"] == 7 and ".mp3" in upload["extensions"]


def test_leftover_upload_folders_are_swept_at_start(settings):
    stale = settings.data_dir / "uploads" / "deadbeef"
    stale.mkdir(parents=True)
    (stale / "old.wav").write_bytes(b"RIFF fake")
    with make_client(settings):
        pass
    assert not stale.exists()
