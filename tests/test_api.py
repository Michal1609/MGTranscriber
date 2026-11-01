import pytest
from fastapi.testclient import TestClient

from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.server import create_app
from tests.conftest import wait_for


@pytest.fixture
def client(settings, legacy_transcript):
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        yield c


def test_info_lists_engines_and_exporters(client):
    info = client.get("/api/info").json()
    assert info["name"] == "mgtranscriber"
    assert {"whisperx", "gemini", "youtube"} <= {e["name"] for e in info["engines"]}
    assert {"txt", "srt", "json"} <= {e["name"] for e in info["exporters"]}


def test_list_and_detail(client):
    [item] = client.get("/api/transcripts").json()
    assert item["id"] == "Old video" and item["has_audio"]
    detail = client.get("/api/transcripts/Old video").json()
    assert detail["audio"] == "audio.webm"
    assert detail["segments"][1] == {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01", "text": "Nazdar,"}
    assert detail["metadata"]["title"] == "Old video"


def test_errors_have_codes(client):
    r = client.get("/api/transcripts/missing")
    assert r.status_code == 404 and r.json()["code"] == "transcript_not_found"
    r = client.get("/api/transcripts/..%5Cx")
    assert r.status_code in {400, 404}


def test_speakers_and_exports(client):
    assert client.put("/api/transcripts/Old video/speakers", json={"SPEAKER_00": "Eva"}).json() == {"SPEAKER_00": "Eva"}
    txt = client.get("/api/transcripts/Old video/export/txt")
    assert txt.status_code == 200 and "Eva:" in txt.text
    assert "prepis.txt" in txt.headers["content-disposition"]
    raw = client.get("/api/transcripts/Old video/export/txt?names=false")
    assert "SPEAKER_00:" in raw.text
    # original download URLs keep working
    legacy = client.get("/api/transcripts/Old video/files/titulky.srt?names=1")
    assert "[Eva] Ahoj." in legacy.text
    assert client.get("/api/transcripts/Old video/files/audio.webm").status_code == 200
    assert client.get("/api/transcripts/Old video/files/secret.txt").status_code == 404
    assert client.get("/api/transcripts/Old video/files/prepis.json").json()[0]["text"] == " Ahoj."
    assert client.get("/api/transcripts/Old video/export/pdf").json()["code"] == "unknown_format"


def test_job_lifecycle(client, media_file):
    r = client.post("/api/jobs", json={"url": str(media_file("new talk.wav")), "language": "en", "speakers": 2})
    assert r.status_code == 201
    job_id = r.json()["id"]
    wait_for(lambda: client.get("/api/jobs").json()[0]["status"] == "done")
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["result"].startswith("new-talk-") and job["title"] == "new talk"
    assert any("Mock transcribing" in line for line in job["log"])

    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        body = "".join(stream.iter_text())
    assert "event: state" in body and "event: log" in body and body.rstrip().endswith("data: {}")

    ids = [t["id"] for t in client.get("/api/transcripts").json()]
    assert job["result"] in ids


def test_job_validation(client):
    assert client.post("/api/jobs", json={"url": "-x"}).json()["code"] == "invalid_source"
    assert client.post("/api/jobs", json={"url": "a", "engine": "nonexistent"}).json()["code"] == "unknown_engine"
    assert client.post("/api/jobs", json={"language": "cs"}).status_code == 422
    assert client.delete("/api/jobs/999").status_code == 404


def test_static_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text


def test_static_files_are_revalidated(client):
    """A code update must reach the browser without Ctrl+F5: static files say no-cache (304 via ETag)."""
    for path in ("/", "/js/main.js", "/app.css"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache", path
    etag = client.get("/js/main.js").headers["etag"]
    again = client.get("/js/main.js", headers={"If-None-Match": etag})
    assert again.status_code == 304
    # API responses and the audio route keep their own caching
    assert "cache-control" not in client.get("/api/info").headers
    audio = client.get("/api/transcripts/Old video/files/audio.webm")
    assert audio.status_code in (200, 206) and audio.headers.get("cache-control") != "no-cache"


def test_cancelled_job_has_no_error(client, media_file):
    job = client.post("/api/jobs", json={"source": str(media_file("slow.wav"))}).json()
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] == "running")
    client.delete(f"/api/jobs/{job['id']}")
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] == "cancelled")
    assert client.get(f"/api/jobs/{job['id']}").json()["error"] is None


def test_foreign_host_is_rejected(client):
    r = client.get("/api/transcripts", headers={"Host": "evil.example"})
    assert r.status_code == 400 and r.json()["code"] == "invalid_host"
    assert client.get("/api/transcripts", headers={"Host": "127.0.0.1:8000"}).status_code == 200


def test_cross_origin_writes_are_rejected(client):
    names = {"SPEAKER_00": "Eva"}
    r = client.put("/api/transcripts/Old video/speakers", json=names, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and r.json()["code"] == "cross_origin"
    r = client.put("/api/transcripts/Old video/speakers", json=names, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    assert client.put("/api/transcripts/Old video/speakers", json=names,
                      headers={"Origin": "http://testserver"}).status_code == 200
    # a cross-site GET is refused too (an <img src> must not trigger exports); plain clients are unaffected
    assert client.get("/api/transcripts", headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.get("/api/transcripts").status_code == 200


def test_local_paths_can_be_disabled(settings, media_file):
    app = create_app(settings.with_overrides(allow_local_paths="false"), runner=InProcessRunner())
    with TestClient(app) as c:
        r = c.post("/api/jobs", json={"source": str(media_file("x.wav"))})
        assert r.status_code == 403 and r.json()["code"] == "local_paths_disabled"


def test_local_paths_default_depends_on_bind_address(settings):
    assert settings.local_paths_allowed  # 127.0.0.1
    assert not settings.with_overrides(host="0.0.0.0").local_paths_allowed
    assert settings.with_overrides(host="0.0.0.0", allow_local_paths="true").local_paths_allowed
