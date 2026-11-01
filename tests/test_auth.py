import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.server import create_app
from mgtranscriber.server import security
from mgtranscriber.server.security import SESSION_COOKIE, Sessions

KEY = "s3cret-key"


@pytest.fixture
def secured(settings, legacy_transcript, monkeypatch):
    with TestClient(create_app(settings.with_overrides(api_key=KEY), runner=InProcessRunner())) as c:
        yield c


def test_open_server_needs_no_key(settings, legacy_transcript):
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        assert c.get("/api/auth/status").json() == {"required": False, "authenticated": True}
        assert c.get("/api/transcripts").status_code == 200


@pytest.mark.parametrize("path", ["/api/transcripts", "/api/info", "/api/search?q=a", "/api/jobs",
                                  "/api/transcripts/Old video/files/audio.webm", "/api/docs"])
def test_api_requires_key(secured, path):
    r = secured.get(path)
    assert r.status_code == 401 and r.json()["code"] == "auth_required"
    assert r.headers["www-authenticate"] == "Bearer"


def test_bearer_and_header_keys(secured):
    assert secured.get("/api/transcripts", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200
    assert secured.get("/api/transcripts", headers={"X-API-Key": KEY}).status_code == 200
    assert secured.get("/api/transcripts", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert secured.get("/api/transcripts", headers={"Authorization": KEY}).status_code == 401


def test_frontend_and_auth_endpoints_stay_public(secured):
    assert secured.get("/").status_code == 200
    assert secured.get("/js/main.js").status_code == 200
    assert secured.get("/api/auth/status").json() == {"required": True, "authenticated": False}


def test_login_cookie_flow(secured):
    assert secured.post("/api/auth/login", json={"key": "nope"}).status_code == 401
    r = secured.post("/api/auth/login", json={"key": KEY})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"]
    assert "httponly" in cookie.lower() and "samesite=strict" in cookie.lower()
    assert KEY not in cookie
    token = secured.cookies[SESSION_COOKIE]
    assert secured.get("/api/auth/status").json()["authenticated"] is True
    assert secured.get("/api/transcripts").status_code == 200  # cookie sent automatically
    # SSE and the audio element only have the cookie
    assert secured.get("/api/transcripts/Old video/files/audio.webm").status_code == 200
    secured.post("/api/auth/logout")
    assert secured.get("/api/transcripts").status_code == 401
    # logout ends the session on the server: a copy of the old cookie no longer works
    secured.cookies.set(SESSION_COOKIE, token)
    assert secured.get("/api/transcripts").status_code == 401


def test_sessions_expire_survive_restarts_and_die_with_the_key(tmp_path, monkeypatch):
    path = tmp_path / "sessions.json"
    token = Sessions(KEY, path).create()
    assert token not in path.read_text(encoding="utf-8")  # only an HMAC of the token is stored
    assert Sessions(KEY, path).valid(token)  # a server restart keeps the login
    assert not Sessions("another-key", path).valid(token)  # a new key ends every session
    assert not Sessions(KEY, path).valid("forged") and not Sessions(KEY, path).valid(None)
    later = time.time() + security.SESSION_TTL + 1
    monkeypatch.setattr(security.time, "time", lambda: later)
    assert not Sessions(KEY, path).valid(token)


def test_wrong_keys_in_headers_are_throttled(secured):
    wrong = {"Authorization": "Bearer wrong"}
    assert [secured.get("/api/transcripts", headers=wrong).status_code for _ in range(3)] == [401, 401, 401]
    r = secured.get("/api/transcripts", headers={"X-API-Key": "other"})  # the third miss paused the address
    assert r.status_code == 429 and r.json()["code"] == "too_many_attempts" and int(r.headers["retry-after"]) >= 1
    # even the right key waits, otherwise the pause would still tell a guess from a hit
    assert secured.get("/api/transcripts", headers={"X-API-Key": KEY}).status_code == 429
    assert secured.post("/api/auth/login", json={"key": KEY}).status_code == 429
    assert secured.get("/api/auth/status", headers={"X-API-Key": KEY}).json()["authenticated"] is False
    # no key at all (the UI before login) is plain 401
    assert secured.get("/api/transcripts").status_code == 401


def test_login_misses_also_pause_the_header_check(secured):
    for _ in range(3):
        assert secured.post("/api/auth/login", json={"key": "nope"}).status_code == 401
    assert secured.get("/api/transcripts", headers={"X-API-Key": KEY}).status_code == 429


def test_throttle_pause_grows_ends_and_resets(monkeypatch):
    throttle = security.AuthThrottle()
    clock = [1000.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: clock[0])
    for _ in range(3):
        throttle.failed("1.2.3.4")
    assert throttle.retry_after("1.2.3.4") == 2 and throttle.retry_after("5.6.7.8") == 0  # per address
    clock[0] += 2
    assert throttle.retry_after("1.2.3.4") == 0
    throttle.failed("1.2.3.4")
    assert throttle.retry_after("1.2.3.4") == 4
    throttle.succeeded("1.2.3.4")
    assert throttle.retry_after("1.2.3.4") == 0
    for _ in range(20):
        throttle.failed("1.2.3.4")
    assert throttle.retry_after("1.2.3.4") == 300  # capped
    clock[0] += throttle.FORGET_AFTER + 301
    throttle.failed("1.2.3.4")
    assert throttle.retry_after("1.2.3.4") == 0  # old misses forgotten


def test_failed_logins_hold_nothing(settings):
    """Concurrent wrong keys are answered at once; another address is not affected."""
    app = create_app(settings.with_overrides(api_key=KEY), runner=InProcessRunner())

    async def scenario():
        bad = httpx.ASGITransport(app=app, client=("9.9.9.9", 1))
        async with httpx.AsyncClient(transport=bad, base_url="http://testserver") as client:
            start = time.monotonic()
            results = await asyncio.gather(*(client.post("/api/auth/login", json={"key": f"guess{i}"})
                                             for i in range(8)))
            codes = [r.status_code for r in results]
        good = httpx.ASGITransport(app=app, client=("1.1.1.1", 1))
        async with httpx.AsyncClient(transport=good, base_url="http://testserver") as user:
            ok = await user.post("/api/auth/login", json={"key": KEY})
        return time.monotonic() - start, codes, ok.status_code

    elapsed, codes, ok = asyncio.run(scenario())
    assert codes[:3] == [401] * 3 and set(codes) <= {401, 429} and ok == 200 and elapsed < 2


def test_forged_cookie_is_rejected(secured):
    secured.cookies.set(SESSION_COOKIE, "0" * 64)
    assert secured.get("/api/transcripts").status_code == 401


def test_openai_and_mcp_paths_need_the_key(secured):
    r = secured.get("/v1/models")
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"
    assert secured.get("/v1/models", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200
    assert secured.post("/mcp", json={}).status_code == 401


def test_startup_warning_for_exposed_server_without_key(settings, capsys, monkeypatch):
    from mgtranscriber import cli

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    cli.main(["serve", "--host", "0.0.0.0", "--port", "8799"])
    assert "no API key is set" in capsys.readouterr().out
    monkeypatch.setattr(cli, "load_settings", lambda: settings.with_overrides(api_key=KEY))
    cli.main(["serve", "--host", "0.0.0.0", "--port", "8799"])
    assert "no API key" not in capsys.readouterr().out
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    cli.main(["serve"])
    assert "WARNING" not in capsys.readouterr().out
