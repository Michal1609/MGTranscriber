"""Recording sessions: chunks uploaded while the browser records, then one queued job (API /api/recordings)."""

import os
import time

import pytest
from fastapi.testclient import TestClient

from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.server import create_app, recordings

from .conftest import wait_for

WEBM = "audio/webm;codecs=opus"


def make_client(settings, **overrides):
    return TestClient(create_app(settings.with_overrides(**overrides), runner=InProcessRunner()))


def start(c, **body):
    r = c.post("/api/recordings", json={"mime": WEBM, "source": "microphone", **body})
    assert r.status_code == 201, r.text
    return r.json()


def put(c, rid, n, data=b"\x1a\x45\xdf\xa3chunk", t=None):
    url = f"/api/recordings/{rid}/chunks/{n}" + (f"?t={t}" if t is not None else "")
    return c.put(url, content=data, headers={"Content-Type": "audio/webm"})


def offline_extra(settings):
    return {"extra": {**settings.extra, "offline_mode": True}}


def test_lifecycle_start_chunks_finish_makes_a_job_and_audio(settings):
    with make_client(settings) as c:
        rec = start(c, slice_ms=2000)
        rid = rec["id"]
        assert len(rid) == 32 and rec["chunks"] == 0
        for n in range(3):
            r = put(c, rid, n, b"chunk%d" % n, t=(n + 1) * 2000)
            assert r.status_code == 200, r.text
        info = c.get(f"/api/recordings/{rid}").json()
        assert info["chunks"] == 3 and info["bytes"] == 18 and info["duration"] == 6.0 and info["contiguous"]
        r = c.post(f"/api/recordings/{rid}/finish", json={"chunks": 3, "language": "en", "speakers": 2,
                                                          "title": "Weekly sync"})
        assert r.status_code == 201, r.text
        job = r.json()
        assert job["language"] == "en" and job["speakers"] == 2
        assert c.get("/api/recordings").json() == []  # the session is gone
        assert c.get(f"/api/recordings/{rid}").status_code == 404
        wait_for(lambda: c.get(f"/api/jobs/{job['id']}").json()["status"] in ("done", "error"))
        done = c.get(f"/api/jobs/{job['id']}").json()
        assert done["status"] == "done", done
        tid = done["result"]
        lib = c.app.state.library
        wait_for(lambda: lib.audio_file(tid) and lib.metadata(tid).extra.get("uploaded_via")
                 and not list((settings.data_dir / "uploads").glob("*")))
        assert lib.metadata(tid).source == "Weekly sync.webm"
        assert lib.audio_file(tid).suffix == ".webm"


def test_chunks_are_joined_in_order_not_arrival_order(settings):
    with make_client(settings) as c:
        rid = start(c)["id"]
        for n in (2, 0, 1):
            assert put(c, rid, n, b"<%d>" % n).status_code == 200
        upload, name = recordings.assemble(settings, rid)
        assert name.endswith(".webm") and name.startswith("Recording ")
        # ffmpeg (when installed) cannot remux this fake stream, so the plain concatenation is kept
        assert upload.read_bytes() == b"<0><1><2>"


def test_retry_of_a_chunk_replaces_it(settings):
    with make_client(settings) as c:
        rid = start(c)["id"]
        put(c, rid, 0, b"aaaa")
        put(c, rid, 0, b"bb")
        assert c.get(f"/api/recordings/{rid}").json()["bytes"] == 2
        assert not list((settings.data_dir / "recordings" / rid).glob("*.tmp"))


def test_missing_chunk_is_409_and_the_session_survives(settings):
    with make_client(settings) as c:
        rid = start(c)["id"]
        put(c, rid, 0)
        put(c, rid, 2)
        r = c.post(f"/api/recordings/{rid}/finish", json={"chunks": 3})
        assert r.status_code == 409 and r.json()["code"] == "recording_chunks_missing"
        assert c.get(f"/api/recordings/{rid}").json()["contiguous"] is False
        r = c.post(f"/api/recordings/{rid}/finish", json={})  # recovery without a count: still refused
        assert r.status_code == 409
        r = c.post(f"/api/recordings/{rid}/finish", json={"allow_gaps": True})  # unless the user accepts the gap
        assert r.status_code == 201
        assert c.get("/api/recordings").json() == []


def test_first_chunk_missing_is_never_allowed(settings):
    with make_client(settings) as c:
        rid = start(c)["id"]
        put(c, rid, 1)
        r = c.post(f"/api/recordings/{rid}/finish", json={"allow_gaps": True})
        assert r.status_code == 409 and r.json()["code"] == "recording_chunks_missing"


def test_finish_without_data_is_409(settings):
    with make_client(settings) as c:
        rid = start(c)["id"]
        r = c.post(f"/api/recordings/{rid}/finish", json={})
        assert r.status_code == 409 and r.json()["code"] == "recording_empty"


def test_discard(settings):
    with make_client(settings) as c:
        rid = start(c)["id"]
        put(c, rid, 0)
        assert c.delete(f"/api/recordings/{rid}").status_code == 200
        assert not (settings.data_dir / "recordings" / rid).exists()
        assert c.delete(f"/api/recordings/{rid}").status_code == 404
        assert put(c, rid, 1).status_code == 404  # a late chunk of a discarded recording


def test_listing_recovers_unfinished_sessions_after_a_restart(settings):
    with make_client(settings) as c:
        a = start(c, source="tab", label="Teams")["id"]
        put(c, a, 0, t=5000)
        put(c, a, 1, t=10000)
        b = start(c)["id"]
    with make_client(settings) as c:  # a new process: the sessions are on disk
        items = c.get("/api/recordings").json()
        assert {i["id"] for i in items} == {a, b}
        first = next(i for i in items if i["id"] == a)
        assert first["chunks"] == 2 and first["duration"] == 10.0 and first["source"] == "tab" and first["label"] == "Teams"
        assert first["created"] and first["updated"]


def test_sweep_removes_old_sessions_only(settings):
    with make_client(settings) as c:
        old = start(c)["id"]
        new = start(c)["id"]
        put(c, old, 0)
        folder = settings.data_dir / "recordings" / old
        past = time.time() - 8 * 86400
        for p in folder.iterdir():
            os.utime(p, (past, past))
        assert recordings.sweep(settings) == 1
        assert not folder.exists()
        assert {i["id"] for i in c.get("/api/recordings").json()} == {new}


def test_limits(settings):
    with make_client(settings, max_upload_mb=1) as c:
        rid = start(c)["id"]
        assert put(c, rid, 0, b"\0" * 600_000).status_code == 200
        r = put(c, rid, 1, b"\0" * 600_000)  # the whole recording counts against max_upload_mb
        assert r.status_code == 413 and r.json()["code"] == "recording_too_large"
        assert c.get(f"/api/recordings/{rid}").json()["chunks"] == 1
        assert put(c, rid, 1, b"\0" * (1024 * 1024 + 10)).status_code == 413  # body limit of the middleware
    with make_client(settings) as c:
        rid = start(c)["id"]
        assert put(c, rid, 0, b"").status_code == 400  # empty chunk
        assert put(c, rid, 100_000).status_code == 422  # index out of range


def test_session_cap(settings, monkeypatch):
    monkeypatch.setattr(recordings, "MAX_SESSIONS", 2)
    with make_client(settings) as c:
        start(c)
        start(c)
        r = c.post("/api/recordings", json={"mime": WEBM})
        assert r.status_code == 409 and r.json()["code"] == "too_many_recordings"


@pytest.mark.parametrize("body", [{"mime": "text/html"}, {"mime": WEBM, "source": "screen"}, {"mime": WEBM, "slice_ms": 5},
                                  {}])
def test_start_validation(settings, body):
    with make_client(settings) as c:
        assert c.post("/api/recordings", json=body).status_code in (400, 415, 422)


@pytest.mark.parametrize("bad", ["..", "ABCDEF", "0" * 31, "g" * 32, "a" * 33, "%2e%2e", "..%2f..%2fetc"])
def test_ids_are_validated(settings, bad):
    with make_client(settings) as c:
        for r in (c.get(f"/api/recordings/{bad}"), c.delete(f"/api/recordings/{bad}"),
                  c.put(f"/api/recordings/{bad}/chunks/0", content=b"x"),
                  c.post(f"/api/recordings/{bad}/finish", json={})):
            assert r.status_code in (404, 405, 422), (bad, r.status_code)


def test_works_offline_and_without_local_paths(settings):
    with make_client(settings, allow_local_paths="false", **offline_extra(settings)) as c:
        rid = start(c)["id"]
        put(c, rid, 0)
        r = c.post(f"/api/recordings/{rid}/finish", json={"language": "en"})
        assert r.status_code == 201
        wait_for(lambda: c.get(f"/api/jobs/{r.json()['id']}").json()["status"] == "done")


def test_cloud_engine_refused_in_offline_mode_keeps_the_session(settings):
    with make_client(settings, **offline_extra(settings)) as c:
        rid = start(c)["id"]
        put(c, rid, 0)
        r = c.post(f"/api/recordings/{rid}/finish", json={"engine": "deepgram"})
        assert r.status_code in (400, 403)
        assert c.get(f"/api/recordings/{rid}").status_code == 200  # nothing lost


def test_api_key_and_cross_site_writes(settings):
    with make_client(settings, api_key="s3cret") as c:
        assert c.post("/api/recordings", json={"mime": WEBM}).status_code == 401
        h = {"Authorization": "Bearer s3cret"}
        r = c.post("/api/recordings", json={"mime": WEBM}, headers=h)
        assert r.status_code == 201
        rid = r.json()["id"]
        assert c.put(f"/api/recordings/{rid}/chunks/0", content=b"x").status_code == 401
        assert c.put(f"/api/recordings/{rid}/chunks/0", content=b"x", headers=h).status_code == 200
        evil = {**h, "Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}
        assert c.put(f"/api/recordings/{rid}/chunks/1", content=b"x", headers=evil).status_code == 403
        assert c.delete(f"/api/recordings/{rid}", headers=evil).status_code == 403
        assert c.get("/api/recordings").status_code == 401
