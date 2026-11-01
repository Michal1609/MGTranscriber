"""Pre-release audit fixes: UNC paths, browser headers, cross-site GETs, server paths, 422 echo."""

import asyncio
import dataclasses
import json
import pathlib

import pytest
from fastapi.testclient import TestClient
from mcp import Client

from mgtranscriber.jobs import InProcessRunner, JobQueue
from mgtranscriber.mcp_server import LocalJobs, build_server
from mgtranscriber.pathguard import is_unc_path
from mgtranscriber.search import SearchIndex
from mgtranscriber.server import create_app
from mgtranscriber.server.errors import ApiError
from mgtranscriber.server.routers.jobs import check_source
from mgtranscriber.watch import WatchManager
from mgtranscriber.watch.manager import WatchError

UNC = [
    r"\\attacker.example\share\a.wav", "//attacker.example/share/a.wav", r"\\?\UNC\attacker.example\share\a.wav",
    "//?/UNC/attacker.example/share/a.wav", r"\/attacker.example/share\a.wav", r"/\attacker.example\share",
    r"\\.\pipe\x", r"\\?\GLOBALROOT\Device\x", r"\??\UNC\attacker.example\share\a.wav",
    r'  "\\attacker.example\s\a.wav"', r"\\attacker.example@80\DavWWWRoot\a.wav",
]
LOCAL = [r"C:\Users\me\a.wav", "C:/Users/me/a.wav", r"\\?\C:\very\long\a.wav", "//?/C:/long/a.wav", "a.wav",
         "/home/me/a.wav", "relative/dir/a.wav", "https://example.com/a.mp3", r"\single\slash.wav"]


@pytest.mark.parametrize("path", UNC)
def test_unc_spellings_are_detected(path):
    assert is_unc_path(path)


@pytest.mark.parametrize("path", LOCAL)
def test_local_paths_are_not_unc(path):
    assert not is_unc_path(path)


@pytest.fixture
def no_unc_touch(monkeypatch):
    """Fails the test when the file system is asked about a network path (that is what sends credentials)."""
    for name in ("is_file", "is_dir", "exists", "resolve", "stat", "expanduser"):
        real = getattr(pathlib.Path, name)

        def guard(self, *a, _real=real, _name=name, **k):
            assert not is_unc_path(str(self)), f"Path.{_name} touched {self}"
            return _real(self, *a, **k)

        monkeypatch.setattr(pathlib.Path, name, guard)


@pytest.mark.parametrize("path", UNC)
def test_api_refuses_unc_sources_before_touching_them(settings, no_unc_touch, path):
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        r = c.post("/api/jobs", json={"source": path})
        assert r.status_code == 403 and r.json()["code"] == "unc_path_disabled", r.text
        assert c.get("/api/jobs").json() == []


def test_unc_can_be_allowed(settings):
    with pytest.raises(ApiError):
        check_source(settings, r"\\nas\share\a.wav")
    check_source(dataclasses.replace(settings, allow_unc_paths=True), r"\\nas\share\a.wav")  # opt-in: accepted


def test_unc_env_variable_opts_in(tmp_path, monkeypatch):
    from mgtranscriber.config import load_settings
    monkeypatch.setenv("MGTRANSCRIBER_ALLOW_UNC_PATHS", "true")
    assert load_settings(root=tmp_path).allow_unc_paths is True
    monkeypatch.delenv("MGTRANSCRIBER_ALLOW_UNC_PATHS")
    assert load_settings(root=tmp_path).allow_unc_paths is False


@pytest.mark.parametrize("path", UNC)
def test_mcp_transcribe_refuses_unc(settings, library, no_unc_touch, path):
    queue = JobQueue(settings, runner=InProcessRunner(), autostart=False)
    index = SearchIndex(library, settings.search_file)
    server = build_server(settings, library, index, LocalJobs(queue), allow_local_paths=True)

    async def scenario():
        async with Client(server) as client:
            result = await client.call_tool("transcribe", {"url_or_path": path})
            assert result.is_error and "Network paths" in result.content[0].text

    asyncio.run(scenario())
    assert not queue.jobs
    index.close()


@pytest.mark.parametrize("path", UNC)
def test_watch_folder_refuses_unc(settings, no_unc_touch, path):
    manager = WatchManager(settings, type("Q", (), {"add": lambda *a, **k: None})())
    with pytest.raises(WatchError) as e:
        manager.create({"kind": "folder", "target": path})
    assert e.value.code == "unc_path_disabled" and e.value.status == 403


def test_pipeline_refuses_unc(settings, no_unc_touch):
    from mgtranscriber.engines.base import EngineError
    from mgtranscriber.library import Library
    from mgtranscriber.pipeline import TranscriptionRequest, run

    settings.output_dir.mkdir(parents=True)
    with pytest.raises(EngineError, match="Network paths"):
        run(TranscriptionRequest(source=r"\\nas\share\a.wav"), settings, library=Library(settings.output_dir))


# --- browser headers ------------------------------------------------------------------------------------

@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, runner=InProcessRunner()), client=("127.0.0.1", 50000)) as c:
        yield c


def test_security_headers_everywhere(client):
    for path in ("/", "/api/info", "/api/transcripts/nope", "/nothing-here"):
        h = client.get(path).headers
        assert h["x-frame-options"] == "DENY" and h["x-content-type-options"] == "nosniff", path
        assert h["referrer-policy"] == "no-referrer"
        assert "frame-ancestors 'none'" in h["content-security-policy"]
    h = client.get("/api/info", headers={"host": "evil.com"}).headers  # refused requests carry them too
    assert h["x-frame-options"] == "DENY"


def test_csp_fits_the_frontend(client):
    csp = client.get("/").headers["content-security-policy"]
    directives = dict(d.strip().split(" ", 1) for d in csp.split(";") if d.strip())
    assert directives["script-src"] == "'self'"  # one module script, no inline code
    assert "'unsafe-inline'" in directives["style-src"]  # style="" attributes
    assert "blob:" in directives["media-src"] and "blob:" in directives["img-src"] and directives["connect-src"] == "'self'"
    html = client.get("/").text
    assert "<script>" not in html and "onclick=" not in html


def test_swagger_page_keeps_working_but_cannot_be_framed(client):
    r = client.get("/api/docs")
    assert r.status_code == 200 and r.headers["content-security-policy"] == "frame-ancestors 'none'"
    assert r.headers["x-frame-options"] == "DENY"


# --- cross-site GET -------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/api/info", "/api/transcripts", "/v1/models", "/mcp"])
@pytest.mark.parametrize("headers", [{"sec-fetch-site": "cross-site"}, {"origin": "http://evil.com"},
                                     {"origin": "null"}, {"origin": "http://testserver:9999"}])
def test_cross_site_gets_are_refused(client, path, headers):
    r = client.get(path, headers={"host": "testserver", **headers})
    assert r.status_code == 403 and "cross_origin" in r.text


@pytest.mark.parametrize("headers", [{}, {"sec-fetch-site": "same-origin"}, {"sec-fetch-site": "none"},
                                     {"origin": "http://testserver"}])
def test_same_origin_and_plain_clients_still_work(client, headers):
    assert client.get("/api/info", headers={"host": "testserver", **headers}).status_code == 200


def test_audio_of_the_app_itself_still_plays(client, library, legacy_transcript):
    r = client.get("/api/transcripts/Old video/files/audio.webm",
                   headers={"sec-fetch-site": "same-origin", "sec-fetch-mode": "no-cors", "sec-fetch-dest": "audio"})
    assert r.status_code == 200
    r = client.get("/api/transcripts/Old video/files/audio.webm", headers={"sec-fetch-site": "cross-site"})
    assert r.status_code == 403


# --- server paths -----------------------------------------------------------------------------------------

def test_upload_does_not_reveal_the_server_path(settings):
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        r = c.post("/api/jobs/upload", files={"file": ("evil.mp3", b"ID3fake", "audio/mpeg")})
        assert r.status_code == 201
        job = r.json()
        assert job["upload"] is True and job["source"] == job["url"] == "uploads/evil.mp3"
        assert str(settings.data_dir) not in json.dumps(job)
        for listed in c.get("/api/jobs").json():
            assert str(settings.data_dir) not in json.dumps(listed)
        assert str(settings.data_dir) not in json.dumps(c.get(f"/api/jobs/{job['id']}").json())


def test_voices_and_system_paths_only_for_this_computer(settings):
    with TestClient(create_app(settings, runner=InProcessRunner())) as remote:  # peer "testclient": not loopback
        voices = remote.get("/api/voices").json()
        assert str(settings.data_dir) not in json.dumps(voices)
        assert "/" not in voices["file"] and "\\" not in voices["file"]
        system = remote.get("/api/system").json()
        assert str(settings.output_dir) not in json.dumps(system, ensure_ascii=False)
    with TestClient(create_app(settings, runner=InProcessRunner()), client=("127.0.0.1", 50000)) as local:
        assert str(settings.data_dir) in local.get("/api/voices").json()["file"]
        storage = next(c for c in local.get("/api/system").json()["checks"] if c["id"] == "storage")
        assert storage["params"]["path"] == str(settings.output_dir)


# --- 422 echo -----------------------------------------------------------------------------------------------

def test_validation_errors_do_not_echo_values(client):
    secret = "hf_SECRETSECRETSECRET"
    for method, path, body in [("put", "/api/system/hf-token", {"token": [secret]}),
                               ("post", "/api/jobs", {"source": {"x": secret}})]:
        r = getattr(client, method)(path, json=body)
        assert r.status_code == 422, (path, r.text)
        assert secret not in r.text and all("input" not in e for e in r.json()["detail"])
        assert r.json()["detail"][0]["loc"]


# --- review fixes ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/api/info", "/api/transcripts", "/v1/models"])
def test_same_site_gets_are_refused_but_the_app_itself_works(client, path):
    # a page on another port of localhost is same-site, not same-origin
    assert client.get(path, headers={"host": "testserver", "sec-fetch-site": "same-site"}).status_code == 403
    assert client.get(path, headers={"host": "testserver", "sec-fetch-site": "same-origin"}).status_code == 200
    assert client.get(path, headers={"host": "testserver", "sec-fetch-site": "none"}).status_code == 200


def test_job_log_does_not_carry_the_upload_path(settings):
    from .conftest import wait_for
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        job = c.post("/api/jobs/upload", files={"file": ("talk.wav", b"RIFF fake", "audio/wav")}).json()
        wait_for(lambda: c.get(f"/api/jobs/{job['id']}").json()["status"] in ("done", "error"))
        detail = c.get(f"/api/jobs/{job['id']}").json()
        assert any(line.startswith("> uploads/talk.wav") for line in detail["log"]), detail["log"]
        assert str(settings.data_dir) not in json.dumps(detail)
        events = c.get(f"/api/jobs/{job['id']}/events").text
        assert str(settings.data_dir) not in events


def test_only_real_uploads_are_flagged(settings, tmp_path):
    queue = JobQueue(settings, runner=InProcessRunner(), autostart=False)
    from mgtranscriber.pipeline import TranscriptionRequest
    lookalike = tmp_path / "media" / "uploads" / "2024" / "talk.mp3"  # a user's own folder named "uploads"
    real = settings.data_dir / "uploads" / "abc123" / "talk.mp3"
    a = queue.add(TranscriptionRequest(source=str(lookalike)))
    b = queue.add(TranscriptionRequest(source=str(real)))
    assert a.public()["upload"] is False and a.public()["source"] == str(lookalike)
    assert b.public()["upload"] is True and b.public()["source"] == "uploads/talk.mp3"
    # survives a restart, and an old store without the flag is classified on load
    reloaded = JobQueue(settings, runner=InProcessRunner(), autostart=False)
    assert reloaded.jobs[a.id].upload is False and reloaded.jobs[b.id].upload is True
    data = json.loads(settings.jobs_file.read_text(encoding="utf-8"))
    for item in data["jobs"]:
        item.pop("upload", None)
    settings.jobs_file.write_text(json.dumps(data), encoding="utf-8")
    old = JobQueue(settings, runner=InProcessRunner(), autostart=False)
    assert old.jobs[a.id].upload is False and old.jobs[b.id].upload is True
