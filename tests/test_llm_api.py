import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.llm import PROVIDERS, load_config, make_provider
from mgtranscriber.llm.base import LLMError, MissingApiKey, ModelMissing, ProviderUnavailable
from mgtranscriber.llm.ollama import OllamaProvider
from mgtranscriber.llm.store import load_analysis
from mgtranscriber.server import create_app
from tests.conftest import wait_for
from tests.llm_fakes import FakeProvider, make_segments
from tests.test_llm import script

SECRET = "sk-test-secret-value-12345"


@pytest.fixture
def transcript(library, settings):
    """A 48-turn transcript in the library; returns its id."""
    from mgtranscriber.library import Metadata
    folder = library.create("Talk", "src")  # create() derives the id from title + source
    library.save_segments(folder.name, make_segments())
    library.save_metadata(folder.name, Metadata(title="Talk"))
    return folder.name


@pytest.fixture
def client(settings, transcript):
    settings = dataclasses.replace(settings, extra={"llm_language": "cs"})
    app = create_app(settings, runner=InProcessRunner())
    fake = FakeProvider(**script(answer=lambda u: {"found": True, "answer": "Yes.", "citations": [
        {"time": "00:00", "text": make_segments()[0].text.strip()}]}), max_context=100_000)
    app.state.llm.provider = lambda provider=None, model=None: fake
    with TestClient(app) as c:
        c.app_state = app.state
        c.fake = fake
        yield c


def run_analysis(client, tid, **body):
    r = client.post(f"/api/transcripts/{tid}/analysis", json=body)
    assert r.status_code == 202, r.text
    wait_for(lambda: client.get(f"/api/transcripts/{tid}/analysis").json()["run"]["status"] != "running")
    return client.get(f"/api/transcripts/{tid}/analysis").json()


def test_analysis_runs_in_background_and_is_stored(client, transcript, library):
    assert client.get(f"/api/transcripts/{transcript}/analysis").json() == {"run": None, "analysis": None}
    state = run_analysis(client, transcript)
    assert state["run"]["status"] == "done" and state["run"]["percent"] == 100.0
    result = state["analysis"]
    assert result["language"] == "cs" and result["provider"] == "fake" and result["model"] == "fake-1"
    assert result["summary"] and result["chapters"] and result["created"]
    assert load_analysis(library, transcript) == result  # persisted next to the transcript
    assert (library.path(transcript) / "analysis.json").is_file()
    # the transcript list/detail routes are unaffected
    assert client.get(f"/api/transcripts/{transcript}").status_code == 200


def test_analysis_errors_are_reported_and_second_start_is_refused(client, transcript):
    gate = threading.Event()

    def slow(user):
        gate.wait(5)
        return {"summary": "", "key_points": [], "topics": [], "quotes": [], "action_items": []}

    client.fake.script["map"] = slow
    assert client.post(f"/api/transcripts/{transcript}/analysis", json={}).status_code == 202
    again = client.post(f"/api/transcripts/{transcript}/analysis", json={})
    assert again.status_code == 409 and again.json()["code"] == "analysis_running"
    assert client.post(f"/api/transcripts/{transcript}/analysis/cancel").json() == {"cancelled": True}
    gate.set()
    wait_for(lambda: client.get(f"/api/transcripts/{transcript}/analysis").json()["run"]["status"] == "cancelled")

    client.fake.script["map"] = lambda u: (_ for _ in ()).throw(LLMError("boom", "llm_bad_output"))
    state = run_analysis(client, transcript)
    assert state["run"]["status"] == "error" and state["run"]["error"] == {"code": "llm_bad_output", "message": "boom"}
    assert state["analysis"] is None


def test_delete_removes_stored_analysis(client, transcript):
    run_analysis(client, transcript)
    assert client.delete(f"/api/transcripts/{transcript}/analysis").json() == {"deleted": True}
    assert client.get(f"/api/transcripts/{transcript}/analysis").json()["analysis"] is None


def test_ask_endpoint(client, transcript):
    r = client.post(f"/api/transcripts/{transcript}/ask", json={"question": "What is said first?"})
    body = r.json()
    assert r.status_code == 200 and body["found"] and body["citations"][0]["time"] == 0.0
    assert body["provider"] == "fake"
    assert client.post(f"/api/transcripts/{transcript}/ask", json={"question": "x"}).status_code == 422
    assert client.post("/api/transcripts/missing/ask", json={"question": "hello?"}).status_code == 404


def test_markdown_export_contains_analysis_only_when_present(client, transcript):
    plain = client.get(f"/api/transcripts/{transcript}/export/md").text
    assert plain.startswith("# Talk") and "## Summary" not in plain and "Sentence number 0" in plain
    run_analysis(client, transcript, language="cs")
    md = client.get(f"/api/transcripts/{transcript}/export/md")
    assert "attachment" in md.headers["content-disposition"]
    assert "## Shrnutí" in md.text and "Whole summary." in md.text and "## Kapitoly" in md.text
    assert md.text.index("## Kapitoly") < md.text.index("## Přepis") < md.text.index("Sentence number 0")
    assert "md" in {e["name"] for e in client.get("/api/info").json()["exporters"]}


def test_auto_analysis_after_transcription_is_opt_in(settings, transcript):
    fake = FakeProvider(**script())

    def finished(extra):
        app = create_app(dataclasses.replace(settings, extra=extra), runner=InProcessRunner())
        app.state.llm.provider = lambda provider=None, model=None: fake
        with TestClient(app):
            job = SimpleNamespace(status="done", result=transcript, add_log=lambda line: None,
                                  request=SimpleNamespace(task="transcribe"))
            voiceprints = SimpleNamespace(status="done", result=transcript, add_log=lambda line: None,
                                          request=SimpleNamespace(task="voiceprints"))
            app.state.llm.on_job_finished(voiceprints)  # other job types never trigger an analysis
            assert transcript not in app.state.llm.runs
            app.state.llm.on_job_finished(job)
            run = app.state.llm.runs.get(transcript)
            if run:
                wait_for(lambda: run.status != "running")
            return run

    assert finished({}) is None  # default: off
    run = finished({"llm_auto": True})
    assert run is not None and run.status == "done"


# --- configuration and secrets ------------------------------------------------------------

def test_config_comes_from_settings_and_environment(settings):
    cfg = load_config(dataclasses.replace(settings, extra={"llm_provider": "gemini", "llm_model": "m1", "llm_auto": "yes"}),
                      env={})
    assert (cfg.provider, cfg.model, cfg.auto) == ("gemini", "m1", True)
    cfg = load_config(dataclasses.replace(settings, extra={"llm_provider": "gemini"}),
                      env={"MGTRANSCRIBER_LLM_PROVIDER": "anthropic", "MGTRANSCRIBER_LLM_NUM_CTX": "16384"})
    assert cfg.provider == "anthropic" and cfg.num_ctx == 16384
    assert load_config(dataclasses.replace(settings, extra={"llm_provider": "nope"}), env={}).provider == "ollama"
    assert cfg.public()["model"] == PROVIDERS["anthropic"]["model"]


def test_keys_come_from_the_environment_only_and_are_never_returned(settings, transcript, tmp_path):
    # a key in settings.json (-> settings.extra) is ignored
    # (root=tmp_path: a developer's real .env must not satisfy the check)
    leaky = dataclasses.replace(settings, root=tmp_path, extra={"gemini_api_key": SECRET, "llm_gemini_api_key": SECRET})
    with pytest.raises(MissingApiKey) as err:
        make_provider(load_config(leaky, env={}), leaky, "gemini", env={}).check()
    assert "GEMINI_API_KEY" in err.value.message
    # .env in the project root is accepted
    (tmp_path / ".env").write_text(f"ANTHROPIC_API_KEY={SECRET}\n", encoding="utf-8")
    rooted = dataclasses.replace(settings, root=tmp_path)
    assert make_provider(load_config(rooted, env={}), rooted, "anthropic", env={})._api_key == SECRET
    # nothing the API returns contains a key
    app = create_app(dataclasses.replace(settings, extra={"gemini_api_key": SECRET}), runner=InProcessRunner())
    app.state.llm.env = {"GEMINI_API_KEY": SECRET, "ANTHROPIC_API_KEY": SECRET}
    with TestClient(app) as c:
        info = c.get("/api/llm")
        assert SECRET not in info.text and info.status_code == 200
        providers = {p["name"]: p for p in info.json()["providers"]}
        assert providers["gemini"]["available"] and providers["gemini"]["key_env"] == "GEMINI_API_KEY"
        assert SECRET not in c.get("/api/info").text


def test_missing_key_and_unreachable_ollama_give_helpful_errors(settings, transcript, tmp_path):
    # root=tmp_path: a developer's real .env must not provide the "missing" key
    app = create_app(dataclasses.replace(settings, root=tmp_path, extra={"llm_base_url": "http://127.0.0.1:1"}), runner=InProcessRunner())
    app.state.llm.env = {}
    with TestClient(app) as c:
        r = c.post(f"/api/transcripts/{transcript}/analysis", json={})
        assert r.status_code == 503 and r.json()["code"] == "llm_unreachable"
        assert "ollama serve" in r.json()["detail"]
        r = c.post(f"/api/transcripts/{transcript}/analysis", json={"provider": "gemini"})
        assert r.status_code == 409 and r.json()["code"] == "llm_missing_key" and "GEMINI_API_KEY" in r.json()["detail"]
        r = c.post(f"/api/transcripts/{transcript}/analysis", json={"provider": "nope"})
        assert r.json()["code"] == "unknown_llm_provider"
        status = {p["name"]: p for p in c.get("/api/llm").json()["providers"]}
        assert status["ollama"]["running"] is False and status["gemini"]["available"] is False


# --- Ollama provider against a fake server --------------------------------------------------

class FakeOllama(BaseHTTPRequestHandler):
    requests: list = []
    models = ["gemma3:4b"]

    def log_message(self, *args):
        pass

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, {"models": [{"name": m} for m in self.models]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append((self.path, body))
        if body["model"] not in self.models and body["model"] != "bge-m3":
            return self.reply(404, {"error": f"model '{body['model']}' not found"})
        if self.path == "/api/embed":
            return self.reply(200, {"embeddings": [[1.0, 0.0] for _ in body["input"]]})
        self.reply(200, {"message": {"content": '{"summary": "ok", "key_points": []}'},
                         "prompt_eval_count": 7, "eval_count": 3})


@pytest.fixture
def fake_ollama():
    FakeOllama.requests = []
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_ollama_provider_sends_context_schema_and_reports_usage(fake_ollama):
    p = OllamaProvider("gemma3:4b", fake_ollama, num_ctx=8192)
    p.check()
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    assert p.generate_json("sys", "usr", schema) == {"summary": "ok", "key_points": []}
    path, body = FakeOllama.requests[-1]
    assert path == "/api/chat" and body["format"] == schema and body["options"]["num_ctx"] == 8192
    assert body["options"]["temperature"] == 0.0 and body["stream"] is False
    assert p.usage == {"calls": 1, "prompt_tokens": 7, "output_tokens": 3}
    assert p.embed(["a", "b", "c"]) == [[1.0, 0.0]] * 3 and FakeOllama.requests[-1][1]["model"] == "bge-m3"


def test_ollama_missing_model_tells_how_to_install_it(fake_ollama):
    p = OllamaProvider("llama9:70b", fake_ollama)
    with pytest.raises(ModelMissing) as err:
        p.check()
    assert "ollama pull llama9:70b" in err.value.message
    with pytest.raises(ModelMissing):
        p.generate_json("s", "u", {"type": "object", "properties": {}})
    with pytest.raises(ProviderUnavailable):
        OllamaProvider("x", "http://127.0.0.1:1").check()


# --- cloud providers with mocked SDK clients ----------------------------------------------

def test_gemini_provider_with_mock_client():
    pytest.importorskip("google.genai")
    from mgtranscriber.llm.gemini import GeminiProvider

    seen = {}

    class Models:
        def generate_content(self, model, contents, config):
            seen.update(model=model, contents=contents, config=config)
            return SimpleNamespace(text='{"summary": "hi"}', usage_metadata=SimpleNamespace(
                prompt_token_count=11, candidates_token_count=4))

        def embed_content(self, model, contents):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5, 0.5]) for _ in contents])

    p = GeminiProvider("gemini-3.5-flash-lite", "key", client=SimpleNamespace(models=Models()))
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    assert p.generate_json("system text", "user text", schema) == {"summary": "hi"}
    cfg = seen["config"]
    assert seen["model"] == "gemini-3.5-flash-lite" and seen["contents"] == "user text"
    assert cfg.system_instruction == "system text" and cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == schema and cfg.temperature == 0.0
    assert p.usage["prompt_tokens"] == 11 and p.embed(["a"]) == [[0.5, 0.5]]

    class Boom:
        class models:
            @staticmethod
            def generate_content(**kw):
                raise type("ClientError", (Exception,), {"code": 429, "message": "quota"})("quota")

    with pytest.raises(LLMError) as err:
        GeminiProvider("m", "key", client=Boom()).generate_json("s", "u", schema)
    assert err.value.code == "llm_rate_limited"


def test_anthropic_provider_with_mock_client():
    pytest.importorskip("anthropic")
    from mgtranscriber.llm.claude import AnthropicProvider

    seen = {}

    class Messages:
        def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text='{"summary": "yo"}')],
                                   usage=SimpleNamespace(input_tokens=20, output_tokens=6))

    p = AnthropicProvider(None, "key", client=SimpleNamespace(messages=Messages()))
    assert p.model == "claude-haiku-5-5"
    schema = {"type": "object", "properties": {"summary": {"type": "string"}, "x": {"type": "object", "properties": {}}}}
    assert p.generate_json("sys", "usr", schema) == {"summary": "yo"}
    assert seen["model"] == "claude-haiku-5-5" and seen["system"] == "sys"
    assert seen["messages"] == [{"role": "user", "content": "usr"}]
    fmt = seen["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
    assert fmt["schema"]["properties"]["x"]["additionalProperties"] is False
    assert "tool_choice" not in seen and "temperature" not in seen  # both rejected by current models
    assert p.usage == {"calls": 1, "prompt_tokens": 20, "output_tokens": 6}
    assert p.with_model("claude-sonnet-5-5").model == "claude-sonnet-5-5"

    class Refusing:
        class messages:
            @staticmethod
            def create(**kw):
                return SimpleNamespace(stop_reason="refusal", content=[], usage=None)

    with pytest.raises(LLMError) as err:
        AnthropicProvider("m", "key", client=Refusing()).generate_json("s", "u", schema)
    assert err.value.code == "llm_refused"


def test_cancel_endpoint_never_deletes_the_stored_analysis(client, transcript):
    run_analysis(client, transcript)
    assert client.post(f"/api/transcripts/{transcript}/analysis/cancel").json() == {"cancelled": False}
    assert client.get(f"/api/transcripts/{transcript}/analysis").json()["analysis"] is not None


def test_analysis_is_marked_stale_after_the_transcript_changes(client, transcript, library):
    state = run_analysis(client, transcript)
    assert state["analysis"]["stale"] is False
    library.save_segments(transcript, make_segments()[:5])  # transcribed again
    assert client.get(f"/api/transcripts/{transcript}/analysis").json()["analysis"]["stale"] is True


def test_start_after_shutdown_is_refused_cleanly(client, transcript):
    client.app_state.llm.shutdown()
    r = client.post(f"/api/transcripts/{transcript}/analysis", json={})
    assert r.status_code >= 400 and r.json()["code"] == "shutting_down"
    assert client.app_state.llm.runs.get(transcript) is None


def test_invalid_num_ctx_falls_back_to_the_default(settings):
    for bad in ("abc", "-5", 3):
        cfg = load_config(dataclasses.replace(settings, extra={"llm_num_ctx": bad}), env={})
        assert cfg.num_ctx == 8192
    assert load_config(dataclasses.replace(settings, extra={"llm_num_ctx": "16384"}), env={}).num_ctx == 16384


def test_auto_output_language_follows_the_transcript(settings, library, transcript):
    from mgtranscriber.library import Metadata
    settings = dataclasses.replace(settings, language="auto")
    assert load_config(settings, env={}).language == "auto"
    assert load_config(dataclasses.replace(settings, language="cs"), env={}).language == "cs"
    app = create_app(settings, runner=InProcessRunner())
    app.state.llm.provider = lambda provider=None, model=None: FakeProvider(**script(), max_context=100_000)
    with TestClient(app) as c:
        library.save_metadata(transcript, Metadata(title="Talk", language="de"))
        assert run_analysis(c, transcript)["analysis"]["language"] == "de"
        assert run_analysis(c, transcript, language="en")["analysis"]["language"] == "en"  # an explicit choice wins
        library.save_metadata(transcript, Metadata(title="Talk"))  # language unknown
        assert run_analysis(c, transcript)["analysis"]["language"] == "en"
