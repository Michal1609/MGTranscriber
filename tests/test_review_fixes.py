"""Hardening of the API, MCP and search: SSRF guard, upload limit, ephemeral /v1 jobs, MCP job routing."""

import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient
from mcp import Client

from mgtranscriber import netguard
from mgtranscriber.jobs import InProcessRunner, JobQueue
from mgtranscriber.mcp_server import AutoJobs, LocalJobs, RemoteJobs, ServerRejectedKey, ServerUnreachable, build_server
from mgtranscriber.pipeline import TranscriptionRequest
from mgtranscriber.search import SearchIndex
from mgtranscriber.server import create_app
from tests.conftest import wait_for


def fake_dns(monkeypatch, address: str):
    monkeypatch.setattr(netguard.socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", (address, 0))])


# --- netguard ---------------------------------------------------------------------------------


@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.2.3", "192.168.0.10", "172.16.5.4", "169.254.169.254",
                                     "::1", "fe80::1%12", "fd00::1", "::ffff:192.168.1.1", "0.0.0.0", "224.0.0.1"])
def test_private_addresses_are_blocked(address):
    assert not netguard.is_public_ip(address)


@pytest.mark.parametrize("address", ["8.8.8.8", "142.250.74.14", "2a00:1450:4014:80d::200e"])
def test_public_addresses_pass(address):
    assert netguard.is_public_ip(address)


def test_require_public_url(monkeypatch):
    fake_dns(monkeypatch, "192.168.1.1")
    with pytest.raises(netguard.BlockedAddress) as err:
        netguard.require_public_url("http://router.lan/admin")
    assert err.value.host == "router.lan" and isinstance(err.value, ValueError)
    fake_dns(monkeypatch, "8.8.8.8")
    netguard.require_public_url("https://example.com/video")
    with pytest.raises(netguard.BlockedAddress):
        netguard.require_public_url("http:///nohost")


def test_jobs_api_refuses_lan_urls_when_exposed(settings, monkeypatch):
    exposed = settings.with_overrides(host="0.0.0.0", allowed_hosts="testserver")
    with TestClient(create_app(exposed, runner=InProcessRunner())) as c:
        fake_dns(monkeypatch, "10.0.0.5")
        r = c.post("/api/jobs", json={"source": "http://nas.local/video.mp4"})
        assert r.status_code == 403 and r.json()["code"] == "blocked_address"
        fake_dns(monkeypatch, "8.8.8.8")
        assert c.post("/api/jobs", json={"source": "https://example.com/v"}).status_code == 201
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:  # loopback: the user's own computer
        fake_dns(monkeypatch, "10.0.0.5")
        assert c.post("/api/jobs", json={"source": "http://nas.local/video.mp4"}).status_code == 201


# --- upload limit and ephemeral /v1 jobs -------------------------------------------------------


def test_upload_limit(settings, media_file):
    app = create_app(settings.with_overrides(max_upload_mb=1), runner=InProcessRunner())
    with TestClient(app) as c:
        big = b"\0" * (1024 * 1024 + 10)
        r = c.post("/v1/audio/transcriptions", files={"file": ("a.wav", big)}, data={"model": "whisper-1"})
        assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"

        def chunks():  # no Content-Length: counted while it arrives
            yield b"--x\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n"
            for _ in range(20):
                yield b"\0" * 100_000
            yield b"\r\n--x--\r\n"

        r = c.post("/v1/audio/transcriptions", content=chunks(),
                   headers={"Content-Type": "multipart/form-data; boundary=x"})
        assert r.status_code == 413
        assert not any((settings.data_dir / "uploads").glob("*/*"))
        with open(media_file(), "rb") as f:  # small uploads still work
            assert c.post("/v1/audio/transcriptions", files={"file": ("a.wav", f)}).status_code == 200


def test_ephemeral_jobs_are_never_resumed(settings, media_file):
    queue = JobQueue(settings, runner=InProcessRunner(), autostart=False)
    upload = queue.add(TranscriptionRequest(str(media_file("up.wav"))), ephemeral=True)
    normal = queue.add(TranscriptionRequest(str(media_file("n.wav"))))
    queue.stop()
    assert upload.status == "cancelled" and upload.done.is_set()  # a waiting /v1 request gets its answer
    assert normal.status == "queued"
    # a crash while an upload job ran: after the restart it is interrupted, not queued again
    data = json.loads(settings.jobs_file.read_text(encoding="utf-8"))
    for job in data["jobs"]:
        job["status"] = "running" if job["id"] == normal.id else job["status"]
    data["jobs"].append({**data["jobs"][0], "id": 9, "status": "running", "ephemeral": True})
    settings.jobs_file.write_text(json.dumps(data), encoding="utf-8")
    again = JobQueue(settings, runner=InProcessRunner(), autostart=False)
    assert again.get(9).status == "interrupted" and again.get(9).done.is_set()
    assert again.get(normal.id).status == "queued"


# --- MCP over stdio: where jobs go -------------------------------------------------------------


def test_auto_jobs_follows_the_web_server(settings, library, media_file, live_server):
    made = []

    def make_local():
        queue = JobQueue(settings, runner=InProcessRunner(), store=settings.data_dir / "mcp-jobs.json")
        made.append(queue)
        return LocalJobs(queue)

    jobs = AutoJobs("http://127.0.0.1:1", None, make_local)  # nothing listens: own queue
    first = jobs.submit(TranscriptionRequest(str(media_file("a.wav"))))
    assert len(made) == 1 and jobs.status(first["id"])["source"].endswith("a.wav")

    url = live_server(api_key="right")  # a web server shows up later: new jobs go there
    jobs.base_url, jobs.api_key = url, "right"
    second = jobs.submit(TranscriptionRequest(str(media_file("b.wav"))))
    assert len(made) == 1 and jobs.owner[second["id"]].__class__ is RemoteJobs

    jobs.api_key = "wrong"  # it runs but refuses the key: an error, not a second GPU queue
    with pytest.raises(ServerRejectedKey, match="rejected"):
        jobs.submit(TranscriptionRequest(str(media_file("c.wav"))))
    assert len(made) == 1
    for queue in made:
        queue.stop()


def test_remote_server_going_away_is_a_tool_error():
    with pytest.raises(ServerUnreachable, match="not reachable"):
        RemoteJobs("http://127.0.0.1:1", timeout=1).status(1)


def test_mcp_language_rule_and_url_guard(settings, library, legacy_transcript, monkeypatch):
    queue = JobQueue(settings, runner=InProcessRunner(), autostart=False)
    index = SearchIndex(library, settings.search_file)
    server = build_server(settings, library, index, LocalJobs(queue), allow_local_paths=False, public_urls_only=True)

    async def scenario():
        async with Client(server) as client:
            for bad in ("EN", "e1", "xxxx"):
                result = await client.call_tool("transcribe", {"url_or_path": "https://example.com/v", "language": bad})
                assert result.is_error and "language" in result.content[0].text
            fake_dns(monkeypatch, "192.168.1.20")
            result = await client.call_tool("transcribe", {"url_or_path": "http://camera.lan/rec.mp4"})
            assert result.is_error and "local network" in result.content[0].text

    asyncio.run(scenario())
    queue.stop()
    index.close()


# --- search ------------------------------------------------------------------------------------


def write_transcript(library, folder: str, text: str) -> None:
    path = library.root / folder
    path.mkdir(parents=True, exist_ok=True)
    (path / "prepis.json").write_text(json.dumps([{"start": 0, "end": 1, "text": text}]), encoding="utf-8")


def test_deleted_transcript_disappears_from_search(settings, library):
    write_transcript(library, "keep", "zebra one")
    write_transcript(library, "gone", "zebra two")
    index = SearchIndex(library, settings.search_file)
    library.listeners.append(index.reindex)
    index.sync()
    assert index.search("zebra")["total"] == 2
    import shutil
    shutil.rmtree(library.root / "gone")
    result = index.search("zebra")  # within the sync interval: still dropped at once
    assert result["total"] == 1 and [h["transcript_id"] for h in result["results"]] == ["keep"]
    # the hook a delete feature calls: Library.changed(id) -> listeners -> reindex removes it
    shutil.rmtree(library.root / "keep")
    library.changed("keep")
    assert index._db().execute("SELECT count(*) FROM transcripts").fetchone()[0] == 0
    index.close()


def test_search_does_not_wait_for_a_running_sync(settings, library, monkeypatch):
    write_transcript(library, "a", "kolo")
    index = SearchIndex(library, settings.search_file)
    index.sync()
    release, entered = threading.Event(), threading.Event()
    real_ids = library.ids

    def slow_ids():
        entered.set()
        release.wait(10)
        return real_ids()

    monkeypatch.setattr(library, "ids", slow_ids)
    index._last_sync = 0
    sync = threading.Thread(target=index.sync)
    sync.start()
    assert entered.wait(5)
    done = []
    threading.Thread(target=lambda: done.append(index.search("kolo")["total"]), daemon=True).start()
    wait_for(lambda: done, timeout=5)  # answered from the index while the sync still runs
    assert done == [1]
    release.set()
    sync.join(10)
    index.close()
