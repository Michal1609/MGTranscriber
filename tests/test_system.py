"""/api/system and /api/settings: diagnostics, the settings whitelist and the rule that secrets never leave."""

import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from mgtranscriber import diagnostics, settings_store
from mgtranscriber.server import create_app
from tests.conftest import wait_for

SECRET = "hf_SECRETSECRETSECRET1234"
OTHER_SECRET = "sk-ant-OTHERSECRET987654"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("HF_TOKEN", "MGTRANSCRIBER_MODEL", "MGTRANSCRIBER_LANGUAGE", "MGTRANSCRIBER_ENGINE", "MGTRANSCRIBER_DIARIZE",
                 "MGTRANSCRIBER_LLM_PROVIDER", "MGTRANSCRIBER_LLM_MODEL", "MGTRANSCRIBER_VOICE_MEMORY", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", OTHER_SECRET)
    diagnostics.clear_cache()


@pytest.fixture
def hf_ok(monkeypatch):
    calls = []

    def fake(token, refresh=False):
        calls.append(token)
        if token == "invalid-token-value":
            return {"id": "hf", "group": "models", "status": "error", "code": "invalid_token", "params": {}, "detail": "rejected"}
        if not token:
            return {"id": "hf", "group": "models", "status": "error", "code": "no_token", "params": {}, "detail": "none"}
        return {"id": "hf", "group": "models", "status": "ok", "code": "ok", "params": {"user": "tester"}, "detail": "fine"}

    monkeypatch.setattr(diagnostics, "hf_status", fake)
    monkeypatch.setattr(diagnostics, "check_ollama", lambda s: {"id": "ollama", "group": "services", "status": "na",
                                                                "code": "not_running", "params": {}, "detail": "off"})
    return calls


@pytest.fixture
def ssettings(settings, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    return dataclasses.replace(settings, root=root, extra={}, hf_token=SECRET)


@pytest.fixture
def client(ssettings, hf_ok):
    with TestClient(create_app(ssettings), base_url="http://testserver", client=("127.0.0.1", 50000)) as c:
        yield c


# --- /api/system ------------------------------------------------------------------------------------

def test_system_report_shape_and_no_secrets(client):
    r = client.get("/api/system")
    assert r.status_code == 200
    body = r.json()
    ids = {c["id"] for c in body["checks"]}
    assert {"gpu", "ffmpeg", "jsruntime", "hf", "ollama", "storage", "network"} <= ids
    assert all(c["status"] in ("ok", "warn", "error", "na") and c["detail"] for c in body["checks"])
    assert body["ready"].keys() == {"transcribe", "diarize"} and body["version"]
    keys = {k["env"]: k["set"] for k in body["keys"]}
    assert keys["ANTHROPIC_API_KEY"] is True and keys["DEEPGRAM_API_KEY"] is False
    text = r.text
    assert SECRET not in text and OTHER_SECRET not in text
    assert body["hf_token_source"] in (".env", "environment") and "whisperx" in body["pyannote_engines"]
    assert "elevenlabs" not in body["pyannote_engines"]


def test_system_reports_missing_token_as_not_ready(client, ssettings, hf_ok):
    object.__setattr__(ssettings, "hf_token", None)
    body = client.get("/api/system").json()
    assert body["ready"]["diarize"] is False
    assert next(c for c in body["checks"] if c["id"] == "hf")["code"] == "no_token"
    assert body["hf_token_source"] == "none"


# --- diagnostics units ------------------------------------------------------------------------------

@pytest.mark.parametrize("gpus, torch, status, code", [
    (None, {"installed": True, "version": "2.8.0+cu128", "cuda": True}, "warn", "no_nvidia"),
    ([{"name": "RTX", "total_mb": 12000, "used_mb": 500, "driver": "576.80"}], {"installed": True, "version": "2.8.0+cpu", "cuda": False}, "error", "cpu_torch"),
    ([{"name": "RTX", "total_mb": 12000, "used_mb": 500, "driver": "531.10"}], {"installed": True, "version": "2.8.0+cu128", "cuda": True}, "error", "old_driver"),
    ([{"name": "GTX", "total_mb": 4096, "used_mb": 100, "driver": "576.80"}], {"installed": True, "version": "2.8.0+cu128", "cuda": True}, "warn", "low_vram"),
    ([{"name": "RTX", "total_mb": 12000, "used_mb": 11000, "driver": "576.80"}], {"installed": True, "version": "2.8.0+cu128", "cuda": True}, "warn", "vram_busy"),
    ([{"name": "RTX", "total_mb": 12000, "used_mb": 500, "driver": "576.80"}], {"installed": True, "version": "2.8.0+cu128", "cuda": True}, "ok", "ok"),
])
def test_gpu_check(monkeypatch, gpus, torch, status, code):
    monkeypatch.setattr(diagnostics, "nvidia_smi", lambda: gpus)
    monkeypatch.setattr(diagnostics, "torch_build", lambda: torch)
    result = diagnostics.check_gpu()
    assert (result["status"], result["code"]) == (status, code)


def test_missing_ffmpeg_is_an_error(monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: None)
    result = diagnostics.check_ffmpeg()
    assert result["status"] == "error" and "winget" in result["detail"]


def test_a_crashing_check_does_not_break_the_report(ssettings, hf_ok, monkeypatch):
    def boom():
        raise RuntimeError("kaboom")

    monkeypatch.setattr(diagnostics, "check_ffmpeg", boom)
    report = diagnostics.run_checks(ssettings)
    ffmpeg = next(c for c in report["checks"] if c["id"] == "ffmpeg")
    assert ffmpeg["code"] == "check_failed" and report["ready"]["transcribe"] is False


def test_cli_exit_code_follows_the_essentials(monkeypatch, capsys, ssettings, hf_ok):
    monkeypatch.setattr("mgtranscriber.config.load_settings", lambda *a, **k: ssettings)
    monkeypatch.setattr(diagnostics, "check_ffmpeg", lambda: diagnostics._check("ffmpeg", "tools", "ok", "ok", "ffmpeg"))
    monkeypatch.setattr(diagnostics, "check_gpu", lambda: diagnostics._check("gpu", "hardware", "warn", "no_nvidia", "cpu"))
    assert diagnostics.main([]) == 0  # CPU mode is not an error
    assert "[ OK ] ffmpeg" in capsys.readouterr().out
    monkeypatch.setattr(diagnostics, "check_ffmpeg", lambda: diagnostics._check("ffmpeg", "tools", "error", "missing", "none"))
    assert diagnostics.main([]) == 1


# --- /api/settings ----------------------------------------------------------------------------------

def test_settings_lists_only_the_whitelist(client):
    body = client.get("/api/settings").json()
    keys = {s["key"] for s in body["settings"]}
    assert keys == set(settings_store.SPECS)
    assert not keys & {"api_key", "hf_token", "output_dir", "data_dir", "host", "port", "allowed_hosts",
                       "allow_local_paths", "root", "extra"}
    assert SECRET not in json.dumps(body) and OTHER_SECRET not in json.dumps(body)
    engine = next(s for s in body["settings"] if s["key"] == "engine")
    assert "whisperx" in engine["choices"] and engine["source"] == "default"


def test_settings_save_to_file_and_apply_live(client, ssettings):
    r = client.put("/api/settings", json={"values": {"language": "en", "diarize": False, "llm_provider": "gemini",
                                                      "llm_auto": True, "voice_memory": False, "model": "small"}})
    assert r.status_code == 200, r.text
    saved = json.loads((ssettings.data_dir / "settings.json").read_text(encoding="utf-8"))
    assert saved == {"language": "en", "diarize": False, "llm_provider": "gemini", "llm_auto": True,
                     "voice_memory": False, "model": "small"}
    by_key = {s["key"]: s for s in r.json()["settings"]}
    assert by_key["language"]["source"] == "settings.json" and by_key["llm_provider"]["value"] == "gemini"
    assert by_key["voice_memory"]["value"] is False and by_key["diarize"]["value"] is False
    # live: the shared Settings object changed, /api/info shows it, no restart needed
    assert ssettings.model == "small" and ssettings.diarize is False
    assert client.get("/api/info").json()["defaults"]["model"] == "small"
    from mgtranscriber import voices
    from mgtranscriber.llm import load_config
    assert load_config(ssettings).provider == "gemini" and load_config(ssettings).auto is True
    assert voices.enabled(ssettings) is False


def test_settings_keep_other_keys_and_reset_to_default(client, ssettings):
    path = ssettings.data_dir / "settings.json"
    path.write_text(json.dumps({"watch_min_rss_s": 60, "language": "cs"}), encoding="utf-8")
    client.put("/api/settings", json={"values": {"llm_model": "gemma3:4b"}})
    assert json.loads(path.read_text(encoding="utf-8")) == {"watch_min_rss_s": 60, "language": "cs", "llm_model": "gemma3:4b"}
    r = client.put("/api/settings", json={"values": {"llm_model": None, "language": None}})
    assert json.loads(path.read_text(encoding="utf-8")) == {"watch_min_rss_s": 60}
    assert {s["key"]: s["source"] for s in r.json()["settings"]}["language"] == "default"
    assert ssettings.language == "auto"  # the dataclass default


@pytest.mark.parametrize("key", ["api_key", "hf_token", "output_dir", "data_dir", "host", "port", "allowed_hosts",
                                 "allow_local_paths", "root", "ANTHROPIC_API_KEY", "gemini_api_key", "llm_base_url"])
def test_secrets_and_dangerous_keys_cannot_be_written(client, ssettings, key):
    r = client.put("/api/settings", json={"values": {key: "x"}})
    assert r.status_code == 400 and r.json()["code"] == "setting_unknown"
    assert not (ssettings.data_dir / "settings.json").exists()


@pytest.mark.parametrize("values", [
    {"language": "Czech!"}, {"engine": "nope"}, {"diarize": "yes"}, {"model": ""}, {"model": "a b; rm -rf"},
    {"llm_provider": "openai"}, {"llm_auto": 1}, {"watch_enabled": "true"},
])
def test_invalid_values_are_rejected_without_writing(client, ssettings, values):
    r = client.put("/api/settings", json={"values": {**values, "voice_memory": False}})
    assert r.status_code == 400 and r.json()["code"] in ("setting_invalid", "setting_unknown")
    assert not (ssettings.data_dir / "settings.json").exists()  # all or nothing
    assert client.get("/api/settings").json()["settings"]


def test_environment_overrides_are_locked(client, ssettings, monkeypatch):
    monkeypatch.setenv("MGTRANSCRIBER_MODEL", "medium")
    monkeypatch.setenv("MGTRANSCRIBER_LLM_PROVIDER", "anthropic")
    by_key = {s["key"]: s for s in client.get("/api/settings").json()["settings"]}
    assert by_key["model"]["locked"] and by_key["model"]["source"] == "env"
    assert by_key["llm_provider"]["locked"] and not by_key["language"]["locked"]
    r = client.put("/api/settings", json={"values": {"model": "small"}})
    assert r.status_code == 409 and r.json()["code"] == "setting_overridden"
    assert not (ssettings.data_dir / "settings.json").exists()


def test_a_broken_settings_file_is_not_overwritten(client, ssettings):
    path = ssettings.data_dir / "settings.json"
    path.write_text("{ not json", encoding="utf-8")
    r = client.put("/api/settings", json={"values": {"language": "en"}})
    assert r.status_code == 409 and r.json()["code"] == "settings_file_invalid"
    assert path.read_text(encoding="utf-8") == "{ not json"


def test_toggling_watch_enabled_starts_and_stops_the_scheduler(client, ssettings):
    watch = client.app.state.watch
    client.put("/api/settings", json={"values": {"watch_enabled": False}})
    assert not (watch._thread and watch._thread.is_alive())
    client.put("/api/settings", json={"values": {"watch_enabled": True}})
    wait_for(lambda: watch._thread and watch._thread.is_alive())
    client.put("/api/settings", json={"values": {"watch_enabled": False}})


# --- HuggingFace token ------------------------------------------------------------------------------

def test_hf_token_is_checked_saved_to_env_and_never_returned(client, ssettings, hf_ok):
    (ssettings.root / ".env").write_text("# my notes\nOTHER=1\nHF_TOKEN=old\n", encoding="utf-8")
    new = "hf_" + "n" * 30
    r = client.put("/api/system/hf-token", json={"token": new})
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert new not in r.text
    assert (ssettings.root / ".env").read_text(encoding="utf-8").splitlines() == ["# my notes", "OTHER=1", f"HF_TOKEN={new}"]
    assert ssettings.hf_token == new
    for url in ("/api/system", "/api/settings", "/api/info"):
        assert new not in client.get(url).text
    assert client.delete("/api/system/hf-token").json() == {"removed": True}
    assert "HF_TOKEN" not in (ssettings.root / ".env").read_text(encoding="utf-8") and ssettings.hf_token is None


def test_rejected_token_is_not_saved(client, ssettings):
    r = client.put("/api/system/hf-token", json={"token": "invalid-token-value"})
    assert r.status_code == 400 and r.json()["code"] == "hf_token_invalid"
    assert not (ssettings.root / ".env").exists() and ssettings.hf_token == SECRET


def test_token_is_only_saved_on_loopback(ssettings, hf_ok):
    open_settings = dataclasses.replace(ssettings, host="0.0.0.0")
    with TestClient(create_app(open_settings), base_url="http://testserver", client=("127.0.0.1", 50000)) as c:
        r = c.put("/api/system/hf-token", json={"token": "hf_" + "x" * 30})
    assert r.status_code == 403 and r.json()["code"] == "local_only"
    assert not (ssettings.root / ".env").exists()


def test_token_from_the_environment_cannot_be_replaced(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fromenvironment12345")
    r = client.put("/api/system/hf-token", json={"token": "hf_" + "x" * 30})
    assert r.status_code == 409 and r.json()["code"] == "token_from_environment"


# --- cloud API keys ---------------------------------------------------------------------------------

KEY = "AIzaSyKEYKEYKEYKEY1234567890"


def _engine(client, name):
    return next(e for e in client.get("/api/info").json()["engines"] if e["name"] == name)


def test_api_key_is_written_applied_live_and_never_returned(client, ssettings, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    env = ssettings.root / ".env"
    env.write_text("# notes\nOTHER=1\nELEVENLABS_API_KEY=old-old-old-old\n", encoding="utf-8")
    r = client.put("/api/system/keys/ELEVENLABS_API_KEY", json={"value": KEY})
    assert r.status_code == 200 and r.json() == {"env": "ELEVENLABS_API_KEY", "set": True} and KEY not in r.text
    assert env.read_text(encoding="utf-8").splitlines() == ["# notes", "OTHER=1", f"ELEVENLABS_API_KEY={KEY}"]
    assert _engine(client, "elevenlabs")["missing"] is None  # live, no restart
    for url in ("/api/system", "/api/settings", "/api/info", "/api/llm"):
        assert KEY not in client.get(url).text
    row = next(k for k in client.get("/api/system").json()["keys"] if k["env"] == "ELEVENLABS_API_KEY")
    assert row["set"] and row["source"] == ".env"
    assert client.delete("/api/system/keys/ELEVENLABS_API_KEY").json() == {"env": "ELEVENLABS_API_KEY", "set": False}
    assert env.read_text(encoding="utf-8").splitlines() == ["# notes", "OTHER=1"]
    assert _engine(client, "elevenlabs")["missing"]["code"] == "missing_key"


def test_api_key_from_the_environment_is_locked(client, ssettings):
    r = client.put("/api/system/keys/ANTHROPIC_API_KEY", json={"value": KEY})  # set by the autouse fixture
    assert r.status_code == 409 and r.json()["code"] == "key_from_environment"
    assert client.delete("/api/system/keys/ANTHROPIC_API_KEY").status_code == 409
    assert not (ssettings.root / ".env").exists()
    row = next(k for k in client.get("/api/system").json()["keys"] if k["env"] == "ANTHROPIC_API_KEY")
    assert row["source"] == "environment" and OTHER_SECRET not in client.get("/api/system").text


def test_api_key_validation_and_unknown_name(client, ssettings, monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    assert client.put("/api/system/keys/DEEPGRAM_API_KEY", json={"value": "has a space inside"}).json()["code"] == "key_invalid"
    assert client.put("/api/system/keys/HF_TOKEN", json={"value": KEY}).status_code == 404
    assert client.put("/api/system/keys/PATH", json={"value": KEY}).status_code == 404
    assert not (ssettings.root / ".env").exists()


def test_overlong_secrets_are_refused_without_echo(client, ssettings, monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    long = "S3CR3T" * 100
    r = client.put("/api/system/keys/DEEPGRAM_API_KEY", json={"value": long})
    assert r.status_code == 400 and r.json()["code"] == "key_invalid" and "S3CR3T" not in r.text
    r = client.put("/api/system/hf-token", json={"token": long})
    assert r.status_code == 400 and "S3CR3T" not in r.text
    assert not (ssettings.root / ".env").exists()


def test_api_key_only_on_loopback(ssettings, monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    open_settings = dataclasses.replace(ssettings, host="0.0.0.0")
    with TestClient(create_app(open_settings), base_url="http://testserver", client=("127.0.0.1", 50000)) as c:
        r = c.put("/api/system/keys/DEEPGRAM_API_KEY", json={"value": KEY})
    assert r.status_code == 403 and r.json()["code"] == "local_only"
    assert not (ssettings.root / ".env").exists()


def test_api_key_test_reports_status_only_and_not_offline(client, ssettings, monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    assert client.post("/api/system/keys/DEEPGRAM_API_KEY/test").json()["code"] == "no_key"
    client.put("/api/system/keys/DEEPGRAM_API_KEY", json={"value": KEY})
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=0):
        seen["auth"] = req.get_header("Authorization")
        seen["url"] = req.full_url
        return Resp()

    monkeypatch.setattr(diagnostics.urllib.request, "urlopen", fake_open)
    r = client.post("/api/system/keys/DEEPGRAM_API_KEY/test")
    assert r.json() == {"env": "DEEPGRAM_API_KEY", "ok": True, "code": "ok"} and seen["auth"] == f"Token {KEY}"
    assert seen["url"].startswith("https://api.eu.deepgram.com/")
    monkeypatch.setenv("DEEPGRAM_API_URL", "https://api.deepgram.com")
    client.post("/api/system/keys/DEEPGRAM_API_KEY/test")
    assert seen["url"].startswith("https://api.deepgram.com/")
    monkeypatch.delenv("DEEPGRAM_API_URL")
    for status, code in ((401, "rejected"), (403, "limited"), (500, "unreachable")):
        def fail(req, timeout=0, status=status):
            raise diagnostics.urllib.error.HTTPError(req.full_url, status, "x", {}, None)
        monkeypatch.setattr(diagnostics.urllib.request, "urlopen", fail)
        assert client.post("/api/system/keys/DEEPGRAM_API_KEY/test").json()["code"] == code
    client.put("/api/settings", json={"values": {"offline_mode": True}})
    r = client.post("/api/system/keys/DEEPGRAM_API_KEY/test")
    assert r.status_code == 403 and r.json()["code"] == "offline_test"
    assert client.put("/api/system/keys/DEEPGRAM_API_KEY", json={"value": KEY + "2"}).status_code == 200  # saving works offline


# --- Ollama model download --------------------------------------------------------------------------

@pytest.fixture
def fake_ollama():
    state = {"pulled": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["pulled"].append(body["model"])
            self.send_response(200)
            self.end_headers()
            for line in ({"status": "pulling manifest"}, {"status": "pulling a", "digest": "a", "total": 100, "completed": 40},
                         {"status": "pulling a", "digest": "a", "total": 100, "completed": 100}, {"status": "success"}):
                self.wfile.write((json.dumps(line) + "\n").encode())
                self.wfile.flush()

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


def test_ollama_pull_reports_progress(client, ssettings, fake_ollama):
    ssettings.extra["llm_base_url"] = fake_ollama["url"]
    assert client.get("/api/system/ollama/pull").json()["status"] == "idle"
    r = client.post("/api/system/ollama/pull", json={"model": "gemma3:4b"})
    assert r.status_code == 202 and r.json()["model"] == "gemma3:4b"
    wait_for(lambda: client.get("/api/system/ollama/pull").json()["status"] == "done")
    assert fake_ollama["pulled"] == ["gemma3:4b"] and client.get("/api/system/ollama/pull").json()["percent"] == 100.0


def test_ollama_pull_rejects_bad_names_and_unreachable_servers(client, ssettings):
    assert client.post("/api/system/ollama/pull", json={"model": "x; rm -rf /"}).status_code == 400
    ssettings.extra["llm_base_url"] = "http://127.0.0.1:9"
    client.post("/api/system/ollama/pull", json={"model": "gemma3:4b"})
    wait_for(lambda: client.get("/api/system/ollama/pull").json()["status"] == "error")
    assert client.get("/api/system/ollama/pull").json()["error"]


# --- offline mode -----------------------------------------------------------------------------------

def test_offline_mode_is_a_setting_and_changes_the_report(client, ssettings, hf_ok, monkeypatch):
    from mgtranscriber import privacy

    r = client.put("/api/settings", json={"values": {"offline_mode": True}})
    assert r.status_code == 200 and privacy.offline(ssettings)
    assert json.loads((ssettings.data_dir / "settings.json").read_text(encoding="utf-8")) == {"offline_mode": True}
    monkeypatch.setattr(diagnostics, "_hf_cached_locally", lambda: True)
    hf_ok.clear()
    body = client.get("/api/system").json()
    by_id = {c["id"]: c for c in body["checks"]}
    assert by_id["offline"]["code"] == "mode"
    assert by_id["hf"]["code"] == "offline_cached" and by_id["hf"]["status"] == "ok"
    assert hf_ok == []  # HuggingFace is not contacted
    assert by_id["jsruntime"]["status"] == "na"
    engines = {e["name"]: e["missing"] for e in body["engines"]}
    assert engines["youtube"] == {"code": "offline", "detail": ""} and engines["whisperx"] is None
    client.put("/api/settings", json={"values": {"offline_mode": None}})
    assert not privacy.offline(ssettings)


def test_offline_mode_locked_by_environment(client, monkeypatch):
    monkeypatch.setenv("MGTRANSCRIBER_OFFLINE_MODE", "true")
    item = next(s for s in client.get("/api/settings").json()["settings"] if s["key"] == "offline_mode")
    assert item["locked"] and item["value"] is True
    assert client.put("/api/settings", json={"values": {"offline_mode": False}}).status_code == 409


# --- review findings of round 4 ---------------------------------------------------------------------

def test_feed_items_pointing_at_the_lan_are_blocked_and_remembered(ssettings):
    from mgtranscriber.watch import WatchManager, fetchers
    from tests.test_watch import Clock, FakeQueue

    queue, clock = FakeQueue(), Clock()
    open_settings = dataclasses.replace(ssettings, host="0.0.0.0", extra={"watch_stable_s": 0, "watch_min_rss_s": 1})
    m = WatchManager(open_settings, queue, clock=clock)
    src = m.create({"kind": "rss", "target": "http://93.184.216.34/feed.xml", "mode": "last_n", "last_n": 10})
    items = [fetchers.Item(key=f"guid:{i}", source=url, title=str(i), published=clock.now - 10 + i)
             for i, url in enumerate(["http://192.168.1.1/a.mp3", "http://169.254.169.254/latest/b.mp3",
                                      "http://[::ffff:10.0.0.1]/c.mp3", "http://93.184.216.34/ok.mp3"])]
    m._list_items = lambda *a, **k: (items, None)
    m.check(src.id)
    assert queue.sources == ["http://93.184.216.34/ok.mp3"]
    assert all(src.is_seen(i.keys) for i in items)  # the blocked ones are not retried at every check


@pytest.mark.parametrize("address", ["::127.0.0.1", "::10.0.0.1", "64:ff9b::7f00:1", "64:ff9b::a9fe:a9fe",
                                     "2002:7f00:1::1", "2002:c0a8:101::1", "::"])
def test_ipv6_forms_of_private_ipv4_are_blocked(address):
    from mgtranscriber import netguard

    assert not netguard.is_public_ip(address)


@pytest.mark.parametrize("address", ["64:ff9b::808:808", "2002:808:808::1", "::8.8.8.8", "2a00:1450:4014:80d::200e"])
def test_ipv6_forms_of_public_ipv4_pass(address):
    from mgtranscriber import netguard

    assert netguard.is_public_ip(address)


@pytest.fixture
def fresh_limits():
    diagnostics._last_refresh.clear()
    diagnostics.clear_cache()
    yield
    diagnostics._last_refresh.clear()
    diagnostics.clear_cache()


def test_gpu_test_is_an_action_run_one_at_a_time_and_rate_limited(client, monkeypatch, fresh_limits):
    runs = []

    class Done:
        returncode, stdout, stderr = 0, "True 12.8 FakeGPU", ""

    monkeypatch.setattr(diagnostics.subprocess, "run", lambda *a, **k: runs.append(a) or Done())
    assert client.get("/api/system/gpu-test").json()["code"] == "not_run" and not runs  # GET never starts torch
    assert client.get("/api/system/gpu-test?refresh=true").json()["code"] == "not_run" and not runs
    assert client.post("/api/system/gpu-test").json()["status"] == "ok"
    client.post("/api/system/gpu-test")
    client.post("/api/system/gpu-test")
    assert len(runs) == 1  # the rest was served from the cache
    assert client.get("/api/system/gpu-test").json()["code"] == "ok"


def test_gpu_test_is_refused_while_a_job_runs(client, monkeypatch, fresh_limits):
    monkeypatch.setattr(client.app.state.queue, "listing", lambda: [{"id": 1, "status": "running"}])
    r = client.post("/api/system/gpu-test")
    assert r.status_code == 409 and r.json()["code"] == "gpu_busy"


def test_a_concurrent_check_does_not_start_a_second_run(fresh_limits):
    started, release, results = threading.Event(), threading.Event(), []

    def slow():
        started.set()
        release.wait(5)
        return {"id": "x", "group": "g", "status": "ok", "code": "ok", "params": {}, "detail": ""}

    busy = {"id": "x", "group": "g", "status": "warn", "code": "timeout", "params": {}, "detail": ""}
    first = threading.Thread(target=lambda: results.append(diagnostics._cached("k", False, slow, busy)))
    first.start()
    started.wait(5)
    assert diagnostics._cached("k", True, slow, busy) == busy  # a second caller gets "busy", slow() ran once
    release.set()
    first.join()
    assert results[0]["status"] == "ok"


def test_refresh_of_the_report_is_rate_limited(client, monkeypatch, fresh_limits):
    cleared = []
    real = diagnostics.clear_cache
    monkeypatch.setattr(diagnostics, "clear_cache", lambda: cleared.append(1) or real())
    for _ in range(3):
        assert client.get("/api/system?refresh=true").status_code == 200
    assert len(cleared) == 1


def test_token_format_is_checked_before_any_network_access(client, hf_ok):
    for bad in ("hf_ünïcode_token_123456", "short", "has space in it 12345", "hf_" + "x" * 300):
        r = client.put("/api/system/hf-token", json={"token": bad})
        assert r.status_code in (400, 422), bad
    assert hf_ok == []


def test_token_is_saved_unverified_in_offline_mode_and_when_huggingface_is_unreachable(client, ssettings, hf_ok, monkeypatch):
    token = "hf_" + "o" * 30
    client.put("/api/settings", json={"values": {"offline_mode": True}})
    monkeypatch.setattr(diagnostics, "_hf_cached_locally", lambda: False)
    r = client.put("/api/system/hf-token", json={"token": token})
    assert r.status_code == 200 and r.json()["verified"] is False and r.json()["saved"] is True
    assert hf_ok == [] and token not in r.text  # HuggingFace was not contacted
    assert f"HF_TOKEN={token}" in (ssettings.root / ".env").read_text(encoding="utf-8")
    client.put("/api/settings", json={"values": {"offline_mode": None}})
    monkeypatch.setattr(diagnostics, "hf_status", lambda t, refresh=False: diagnostics._check("hf", "models", "warn", "unreachable", "x"))
    other = "hf_" + "u" * 30
    r = client.put("/api/system/hf-token", json={"token": other})
    assert r.status_code == 200 and r.json()["verified"] is False
    monkeypatch.setattr(diagnostics, "hf_status", lambda t, refresh=False: diagnostics._check("hf", "models", "ok", "ok", "x"))
    assert client.put("/api/system/hf-token", json={"token": "hf_" + "v" * 30}).json()["verified"] is True


def test_token_needs_a_loopback_client_not_only_a_loopback_bind(ssettings, hf_ok):
    with TestClient(create_app(ssettings), base_url="http://testserver", client=("203.0.113.9", 4444)) as c:
        r = c.put("/api/system/hf-token", json={"token": "hf_" + "x" * 30})
        assert r.status_code == 403 and r.json()["code"] == "local_only"
        assert c.delete("/api/system/hf-token").status_code == 403
    assert not (ssettings.root / ".env").exists()


def test_env_file_rewrite_keeps_encoding_line_endings_and_creates_nothing(client, ssettings):
    env = ssettings.root / ".env"
    assert client.delete("/api/system/hf-token").json() == {"removed": True}
    assert not env.exists()  # nothing to remove: no empty file appears
    env.write_bytes(b"OTHER=1\r\nNAME=caf\xe9\r\n")  # cp1250/latin-1, not UTF-8
    r = client.put("/api/system/hf-token", json={"token": "hf_" + "n" * 30})
    assert r.status_code == 409 and r.json()["code"] == "env_file_unreadable"
    assert env.read_bytes() == b"OTHER=1\r\nNAME=caf\xe9\r\n" and ssettings.hf_token is None  # the failed save changed nothing
    env.write_bytes("OTHER=1\r\nNAME=café\r\n".encode("utf-8"))
    client.put("/api/system/hf-token", json={"token": "hf_" + "n" * 30})
    assert env.read_bytes() == ("OTHER=1\r\nNAME=café\r\nHF_TOKEN=hf_" + "n" * 30 + "\r\n").encode("utf-8")  # CRLF kept
    env.write_bytes(b"OTHER=1\n")
    before = env.stat().st_mtime_ns
    client.delete("/api/system/hf-token")
    assert env.read_bytes() == b"OTHER=1\n" and env.stat().st_mtime_ns == before  # no token there: not rewritten


@pytest.mark.skipif(__import__("os").name != "posix", reason="POSIX permissions")
def test_env_file_permissions_are_kept(client, ssettings):
    import os

    env = ssettings.root / ".env"
    env.write_text("OTHER=1\n", encoding="utf-8")
    os.chmod(env, 0o600)
    client.put("/api/system/hf-token", json={"token": "hf_" + "n" * 30})
    assert env.stat().st_mode & 0o777 == 0o600
    env.unlink()
    client.put("/api/system/hf-token", json={"token": "hf_" + "m" * 30})
    assert env.stat().st_mode & 0o777 == 0o600  # a new file with a secret is owner-only


def test_settings_apply_swaps_fields_and_extra_in_one_step(ssettings):
    old_extra = ssettings.extra
    settings_store.apply(ssettings, {"engine": "parakeet", "model": "parakeet-tdt-0.6b-v3", "llm_auto": True})
    assert (ssettings.engine, ssettings.model) == ("parakeet", "parakeet-tdt-0.6b-v3")
    assert ssettings.extra == {"llm_auto": True} and old_extra == {}  # a modified copy replaced the dict
    settings_store.apply(ssettings, {"engine": None, "model": None, "llm_auto": None})
    assert ssettings.engine == "whisperx" and ssettings.model == "large-v3" and ssettings.extra == {}


def test_model_download_is_refused_in_offline_mode_and_for_remote_servers(client, ssettings):
    ssettings.extra["offline_mode"] = True
    r = client.post("/api/system/ollama/pull", json={"model": "gemma3:4b"})
    assert r.status_code == 403 and r.json()["code"] == "offline_download"
    ssettings.extra.pop("offline_mode")
    ssettings.extra["llm_base_url"] = "http://192.168.1.50:11434"
    ssettings.extra["offline_mode"] = False
    assert client.post("/api/system/ollama/pull", json={"model": "gemma3:4b"}).status_code in (202, 409)  # allowed online


def test_dotenv_values_reach_features_that_read_the_environment(tmp_path, monkeypatch):
    import os

    from mgtranscriber import privacy
    from mgtranscriber.config import load_settings
    from mgtranscriber.llm import load_config

    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in ("MGTRANSCRIBER_OFFLINE_MODE", "MGTRANSCRIBER_LLM_PROVIDER", "MGTRANSCRIBER_MODEL"):
        os.environ.pop(name, None)
    os.environ["MGTRANSCRIBER_LLM_MODEL"] = "from-the-real-environment"
    (tmp_path / ".env").write_text("MGTRANSCRIBER_OFFLINE_MODE=true\nMGTRANSCRIBER_LLM_PROVIDER=gemini\nMGTRANSCRIBER_LLM_MODEL=from-dotenv\n"
                                   "MGTRANSCRIBER_MODEL=small\nHF_TOKEN=hf_notexported12345\n", encoding="utf-8")
    settings = load_settings(tmp_path)
    assert privacy.offline(settings) and load_config(settings).provider == "gemini" and settings.model == "small"
    assert load_config(settings).model == "from-the-real-environment"  # a real variable still wins over .env
    assert "HF_TOKEN" not in os.environ and settings.hf_token == "hf_notexported12345"


def test_demo_data_never_touches_a_real_folder(tmp_path):
    from tools import make_demo_data

    real = tmp_path / "data"
    real.mkdir()
    (real / "voiceprints.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        make_demo_data.claim(real)
    assert (real / "voiceprints.json").read_text(encoding="utf-8") == "{}"
    fresh = tmp_path / "demo"
    make_demo_data.claim(fresh)
    make_demo_data.claim(fresh)  # again: it is ours now
    assert (fresh / make_demo_data.MARKER).exists()


# --- where settings.json lives -----------------------------------------------------------------------

def test_settings_file_lives_in_the_data_dir_and_an_old_one_is_moved_there_once(tmp_path):
    from mgtranscriber.config import load_settings

    (tmp_path / "settings.json").write_text(json.dumps({"model": "small", "feature_x": 1}), encoding="utf-8")
    first = load_settings(tmp_path, env={})
    new = tmp_path / "data" / "settings.json"
    assert first.model == "small" and first.extra == {"feature_x": 1} and first.settings_path == new
    assert new.is_file() and not (tmp_path / "settings.json").exists()
    (tmp_path / "settings.json").write_text(json.dumps({"model": "tiny"}), encoding="utf-8")  # never overwrites
    assert load_settings(tmp_path, env={}).model == "small"
    assert json.loads((tmp_path / "settings.json").read_text(encoding="utf-8")) == {"model": "tiny"}


def test_settings_file_follows_data_dir_and_can_be_set_explicitly(tmp_path):
    from mgtranscriber.config import load_settings

    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "settings.json").write_text(json.dumps({"language": "en"}), encoding="utf-8")
    assert load_settings(tmp_path, env={"MGTRANSCRIBER_DATA_DIR": "state"}).language == "en"
    elsewhere = tmp_path / "cfg" / "my.json"
    elsewhere.parent.mkdir()
    elsewhere.write_text(json.dumps({"language": "de"}), encoding="utf-8")
    s = load_settings(tmp_path, env={"MGTRANSCRIBER_SETTINGS_FILE": str(elsewhere)})
    assert s.language == "de" and s.settings_path == elsewhere
    assert not (tmp_path / "data").exists()  # nothing was created for a file that is elsewhere


def test_saving_settings_creates_the_data_dir_and_survives_a_round_trip(tmp_path):
    from mgtranscriber.config import Settings, load_settings

    settings = load_settings(tmp_path, env={})
    settings_store.update(settings, {"language": "en"})
    assert json.loads((tmp_path / "data" / "settings.json").read_text(encoding="utf-8")) == {"language": "en"}
    assert load_settings(tmp_path, env={}).language == "en"
    assert Settings.from_dict(settings.to_dict()).settings_path == settings.settings_path


def test_legacy_env_prefix_is_migrated(capsys):
    from mgtranscriber.config import migrate_legacy_env
    env = {"LOCALSCRIBE_API_KEY": "old", "LOCALSCRIBE_PORT": "9000", "MGTRANSCRIBER_PORT": "9100"}
    used = migrate_legacy_env(env)
    assert env["MGTRANSCRIBER_API_KEY"] == "old"
    assert env["MGTRANSCRIBER_PORT"] == "9100"  # the new name wins
    assert used == ["LOCALSCRIBE_API_KEY"]


def test_legacy_env_reaches_load_settings(tmp_path):
    from mgtranscriber.config import load_settings
    settings = load_settings(tmp_path, env={"LOCALSCRIBE_OUTPUT_DIR": str(tmp_path / "out")})
    assert settings.output_dir == tmp_path / "out"
