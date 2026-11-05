"""A recording without speech: the job succeeds and says so, and no follow-up work is queued for the empty transcript."""

import dataclasses
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mgtranscriber import tasks
from mgtranscriber.jobs import InProcessRunner, runners
from mgtranscriber.library import Metadata
from mgtranscriber.llm import auto_translate
from mgtranscriber.llm.base import EmptyTranscript
from mgtranscriber.llm.service import AnalysisManager
from mgtranscriber.llm.translate import translate_task
from mgtranscriber.models import Segment
from mgtranscriber.server import create_app
from mgtranscriber.watch.manager import WatchManager
from mgtranscriber.watch.models import Source
from tests.conftest import wait_for
from tests.llm_fakes import FakeProvider


def make_transcript(library, text: str | None, key: str = "ns") -> str:
    path = library.create("Music", key)
    library.save_segments(path.name, [Segment(0.0, 1.0, text)] if text is not None else [])
    library.save_metadata(path.name, Metadata(title="Music", language="en"))
    return path.name


@pytest.fixture
def client(settings):
    settings = dataclasses.replace(settings, extra={"llm_language": "cs"})
    app = create_app(settings, runner=InProcessRunner())
    fake = FakeProvider(max_context=100_000)
    app.state.llm.provider = lambda p=None, m=None: fake
    with TestClient(app) as c:
        c.library = app.state.library
        yield c


def run_job(client, media_file, name):
    job = client.post("/api/jobs", json={"source": str(media_file(name)), "language": "en"}).json()
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] == "done")
    return client.get(f"/api/jobs/{job['id']}").json()


# --- 1. the job and the transcript say so ------------------------------------------------------------

def test_job_without_speech_succeeds_and_is_flagged(client, media_file):
    job = run_job(client, media_file, "nospeech.wav")
    assert job["status"] == "done" and job["no_speech"] is True and job["result"]
    assert any("no speech" in line.lower() for line in job["log"])
    detail = client.get(f"/api/transcripts/{job['result']}").json()
    assert detail["no_speech"] is True and detail["segments"] == []
    assert client.get("/api/transcripts").json()[0]["no_speech"] is True


def test_job_with_speech_is_not_flagged(client, media_file):
    job = run_job(client, media_file, "talk.wav")
    assert job["no_speech"] is False
    assert client.get(f"/api/transcripts/{job['result']}").json()["no_speech"] is False


def test_blank_text_counts_as_no_speech(client):
    assert client.get(f"/api/transcripts/{make_transcript(client.library, '  ')}").json()["no_speech"] is True


# --- 2. no follow-up work --------------------------------------------------------------------------

def forbidden(*a, **k):
    raise AssertionError("follow-up work must not be queued for an empty transcript")


def test_form_translate_to_is_skipped(library):
    queue = SimpleNamespace(add=forbidden, listing=forbidden, save=lambda: None)
    log = []
    job = SimpleNamespace(id=1, status="done", result="x", no_speech=True, translation=None, add_log=log.append,
                          request=SimpleNamespace(translate_to="cs", task="transcribe"))
    auto_translate.on_job_finished(queue, SimpleNamespace(provider=forbidden), library, job)
    assert job.translation == {"language": "cs", "skipped": "no_speech"}
    assert "no speech" in log[0]


def test_llm_auto_analysis_is_skipped():
    log = []
    job = SimpleNamespace(status="done", result="x", no_speech=True, request=SimpleNamespace(task="transcribe"),
                          add_log=log.append)
    AnalysisManager.on_job_finished(SimpleNamespace(config=SimpleNamespace(auto=True), start=forbidden), job)
    assert "no speech" in log[0]


def test_watch_source_skips_analysis_and_translation(settings, library):
    llm = SimpleNamespace(library=library, start=forbidden, provider=forbidden)
    manager = WatchManager(settings, SimpleNamespace(listing=forbidden, add=forbidden), llm, library=library)
    src = Source.from_dict({"id": "w1", "name": "OpenAi", "kind": "youtube", "target": "https://youtube.com/@x",
                            "analyze": True, "translate_to": "cs",
                            "items": [{"key": "k", "source": "https://example.com/e", "title": "e", "job": "7"}]})
    src.pending["7"] = "k"
    manager.sources["w1"] = src
    log = []
    job = SimpleNamespace(id=7, status="done", result=make_transcript(library, None), error=None, title="e",
                          no_speech=True, add_log=log.append, request=SimpleNamespace(source="https://example.com/e"))
    manager.on_job_finished(job)
    item = src.items[0]
    assert item["status"] == "done" and item["no_speech"] is True and not item.get("error")
    assert item["translation"] == {"language": "cs", "skipped": "no_speech"}
    assert not src.last_error and any("no speech" in line for line in log)


# --- 3. started by hand: a user-facing error with a code, no traceback ------------------------------

def test_manual_translation_and_analysis_are_refused(client):
    tid = make_transcript(client.library, None)
    for url, body in ((f"/api/transcripts/{tid}/translations", {"language": "cs"}),
                      (f"/api/transcripts/{tid}/analysis", {})):
        r = client.post(url, json=body)
        assert r.status_code == 409 and r.json()["code"] == "empty_transcript", r.text


def test_translate_task_on_an_empty_transcript_raises_the_coded_error(settings, library, monkeypatch):
    tid = make_transcript(library, None, "late")
    provider = FakeProvider(max_context=100_000)
    monkeypatch.setattr("mgtranscriber.llm.make_provider", lambda *a, **k: provider)
    request = SimpleNamespace(source=tid, language="cs", engine=None, model=None)
    with pytest.raises(EmptyTranscript) as e:
        translate_task(request, settings, lambda event: None, library)
    assert e.value.code == "empty_transcript"


def test_runner_reports_the_code_without_a_traceback(settings, monkeypatch):
    def boom(*a, **k):
        raise EmptyTranscript()

    monkeypatch.setattr(tasks, "execute", boom)
    events = []
    runners.InProcessExecution(SimpleNamespace(), settings, events.append).wait()
    assert events[-1]["event"] == "error" and events[-1]["code"] == "empty_transcript"
    assert "Traceback" not in events[-1]["message"] and "LLMError" not in events[-1]["message"]
