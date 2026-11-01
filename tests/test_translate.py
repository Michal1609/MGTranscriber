"""Translation of a transcript: batching and validation, storage, stale stamp, export, search, delete."""

import dataclasses
import re

import pytest
from fastapi.testclient import TestClient

from mgtranscriber import privacy
from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.library import Metadata
from mgtranscriber.llm.base import BadModelOutput, LLMError
from mgtranscriber.llm.translate import build_translation, translate_segments
from mgtranscriber.llm.translations import (delete_translation, list_translations, load_translation, normalize_lang,
                                            save_translation)
from mgtranscriber.models import Segment, Word
from mgtranscriber.search import SearchIndex
from mgtranscriber.server import create_app
from tests.conftest import wait_for
from tests.llm_fakes import FakeProvider

LINE = re.compile(r"^(\d+) \[(.*?)\] (.*)$")


def wanted(user: str) -> list[tuple[int, str]]:
    """The (id, text) lines of the 'Translate these' block of a prompt."""
    block = user.split("Translate these", 1)[1].split("\n\n")[0].splitlines()[1:]
    return [(int(m.group(1)), m.group(3)) for m in map(LINE.match, block) if m]


def cs(text: str) -> str:
    return "CZ: " + text.strip()


def good(user: str) -> dict:
    return {"items": [{"id": i, "text": cs(t)} for i, t in wanted(user)]}


def segs(n: int = 10) -> list[Segment]:
    return [Segment(i * 5.0, i * 5.0 + 4, f" Line number {i} says something.", f"SPEAKER_0{i % 2}",
                    [Word("Line", i * 5.0, i * 5.0 + 1)]) for i in range(n)]


def fake(translate, **kw) -> FakeProvider:
    provider = FakeProvider(translate=translate, **kw)
    provider.local = True  # small batches (8 segments), like Ollama
    return provider


# --- the algorithm ---------------------------------------------------------------------------

def test_translation_keeps_alignment_and_drops_words():
    source = segs()
    out = build_translation(source, {"SPEAKER_00": "Alice"}, fake(good), "cs", source="en")
    assert len(out["segments"]) == len(source)
    for seg, tr in zip(source, out["segments"]):
        assert (tr["start"], tr["end"], tr["speaker"]) == (seg.start, seg.end, seg.speaker)
        assert tr["text"] == cs(seg.text) and "words" not in tr
    assert out["untranslated"] == [] and out["provider"] == "fake" and out["source_lang"] == "en"


def test_prompt_has_speaker_names_and_context():
    f = fake(good)
    translate_segments(segs(), {"SPEAKER_00": "Alice"}, f, "cs", source="en", title="Demo")
    kind, system, user = f.calls[1]  # second batch: context lines before it
    assert "Alice" in system and "Demo" in system and "Czech" in system and "English" in system
    assert "already translated" in user and "CZ: Line number" in user
    assert "[Alice]" in user


def test_wrong_count_is_retried_then_split_and_never_shifts_text():
    sizes = []

    def flaky(user):
        items = wanted(user)
        sizes.append(len(items))
        if len(items) > 2:  # too long a list: the model drops the last item
            return {"items": [{"id": i, "text": cs(t)} for i, t in items[:-1]]}
        return good(user)

    source = segs(8)
    out = translate_segments(source, {}, fake(flaky), "cs")
    assert out["untranslated"] == []
    assert out["texts"] == [cs(s.text) for s in source]
    assert max(sizes) > 2 and min(sizes) <= 2  # it did split


def test_wrong_ids_are_rejected_and_reordering_is_accepted():
    calls = {"n": 0}

    def reply(user):
        calls["n"] += 1
        items = wanted(user)
        if calls["n"] == 1:  # ids shifted by one: must not be accepted
            return {"items": [{"id": i + 1, "text": cs(t)} for i, t in items]}
        return {"items": [{"id": i, "text": cs(t)} for i, t in reversed(items)]}  # reordered but right

    source = segs(4)
    out = translate_segments(source, {}, fake(reply), "cs")
    assert out["texts"] == [cs(s.text) for s in source]
    assert calls["n"] >= 2


def test_a_segment_that_never_works_keeps_its_original():
    def reply(user):
        items = [(i, t) for i, t in wanted(user)]
        return {"items": [{"id": i, "text": ("" if i == 3 else cs(t))} for i, t in items]}  # empty text is invalid

    source = segs(6)
    out = translate_segments(source, {}, fake(reply), "cs")
    assert out["untranslated"] == [3]
    assert out["texts"][3] == source[3].text.strip()
    assert [out["texts"][i] for i in (0, 1, 2, 4, 5)] == [cs(source[i].text) for i in (0, 1, 2, 4, 5)]


def test_bad_json_counts_as_a_failed_attempt():
    state = {"n": 0}

    def reply(user):
        state["n"] += 1
        if state["n"] == 1:
            raise BadModelOutput("not json")
        return good(user)

    out = translate_segments(segs(3), {}, fake(reply), "cs")
    assert out["untranslated"] == [] and all(t.startswith("CZ:") for t in out["texts"])


def test_empty_segments_stay_empty_and_progress_is_monotonic():
    source = segs(5)
    source[2].text = "  "
    seen = []
    out = translate_segments(source, {}, fake(good), "cs", progress=lambda p, m: seen.append(p))
    assert out["texts"][2] == "" and out["untranslated"] == []
    assert seen == sorted(seen) and 0 < seen[0] and seen[-1] < 100


# --- storage -----------------------------------------------------------------------------------

@pytest.fixture
def transcript(library):
    folder = library.create("Talk", "src")
    library.save_segments(folder.name, segs())
    library.save_metadata(folder.name, Metadata(title="Talk", language="en"))
    return folder.name


def stored(library, tid, lang="cs"):
    data = build_translation(library.segments(tid), {}, fake(good), lang, source="en")
    save_translation(library, tid, lang, data)
    return data


def test_translation_file_and_stale_stamp(library, transcript):
    data = stored(library, transcript)
    path = library.path(transcript) / "translations" / "cs.json"
    assert path.is_file()
    loaded = load_translation(library, transcript, "cs")
    assert loaded["stale"] is False and loaded["target_lang"] == "cs" and loaded["source_lang"] == "en"
    assert loaded["segments_stamp"] == library.segments_stamp(transcript) and loaded["created"]
    assert [s["text"] for s in loaded["segments"]] == [s["text"] for s in data["segments"]]
    # the original is edited: the translation is outdated, not lost
    edited = library.segments(transcript)
    edited[0].text = " Changed."
    library.save_segments(transcript, edited)
    assert load_translation(library, transcript, "cs")["stale"] is True
    assert list_translations(library, transcript)[0]["stale"] is True
    assert delete_translation(library, transcript, "cs") is True
    assert load_translation(library, transcript, "cs") is None
    assert not (library.path(transcript) / "translations").exists()


def test_language_codes_are_validated():
    assert normalize_lang(" CS ") == "cs" and normalize_lang("pt_BR") == "pt-br"
    for bad in ("", "../x", "czech language", "c", "cs/../../a"):
        with pytest.raises(LLMError):
            normalize_lang(bad)


def test_translations_survive_a_retranscription_as_stale(library, transcript):
    stored(library, transcript)
    library.save_segments(transcript, segs(4))  # what a re-transcription does to prepis.json
    assert load_translation(library, transcript, "cs")["stale"] is True


# --- search ------------------------------------------------------------------------------------

def test_search_finds_the_translation_with_its_language(settings, library, transcript):
    index = SearchIndex(library, settings.search_file)
    library.listeners.append(index.reindex)
    index.sync()
    assert index.search("CZ")["total"] == 0
    stored(library, transcript)  # save_translation notifies the listeners
    found = index.search("CZ line number 3")
    assert found["total"] == 1
    hit = found["results"][0]
    assert hit["lang"] == "cs" and hit["segment"] == 3 and hit["start"] == 15.0
    # the original still matches without a tag, and a Czech hit does not double the speaker listing
    assert index.search("Line number 3")["results"][0]["lang"] in ("", "cs")
    assert {h["lang"] for h in index.search("number 3")["results"]} == {"", "cs"}
    assert all(h["lang"] == "" for h in index.search("", speaker="SPEAKER_00", limit=50)["results"])
    delete_translation(library, transcript, "cs")
    assert index.search("CZ")["total"] == 0
    index.close()


# --- API: job, export, delete ------------------------------------------------------------------

@pytest.fixture
def client(settings, library, transcript, monkeypatch):
    settings = dataclasses.replace(settings, extra={})
    app = create_app(settings, runner=InProcessRunner())
    provider = fake(good, max_context=100_000)
    app.state.llm.provider = lambda p=None, m=None: provider
    monkeypatch.setattr("mgtranscriber.llm.make_provider", lambda *a, **k: provider)
    with TestClient(app) as c:
        c.provider = provider
        c.tid = transcript
        yield c


def translate(client, lang="cs"):
    r = client.post(f"/api/transcripts/{client.tid}/translations", json={"language": lang})
    assert r.status_code == 202, r.text
    job = r.json()
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] in ("done", "error"))
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "done", client.get(f"/api/jobs/{job['id']}").text
    return job


def test_translate_job_end_to_end(client):
    tid = client.tid
    assert client.get(f"/api/transcripts/{tid}/translations").json()["translations"] == []
    job = translate(client)
    assert job["task"] == "translate" and job["source"] == tid and job["language"] == "cs"
    listing = client.get(f"/api/transcripts/{tid}/translations").json()
    assert [t["lang"] for t in listing["translations"]] == ["cs"] and listing["translations"][0]["stale"] is False
    one = client.get(f"/api/transcripts/{tid}/translations/cs").json()
    assert len(one["segments"]) == 10 and one["segments"][0]["text"].startswith("CZ:") and one["stale"] is False
    # the audit log records it (matters when a cloud provider received the text)
    events = [r for r in privacy.read_audit(client.app.state.library, tid) if r["event"] == "translated"]
    assert events and events[-1]["lang"] == "cs" and events[-1]["provider"] == "fake"


def test_export_with_lang_srt_vtt_txt_docx_md(client):
    translate(client)
    tid = client.tid
    srt = client.get(f"/api/transcripts/{tid}/export/srt?lang=cs")
    assert srt.status_code == 200 and "CZ: Line number 0" in srt.text and "00:00:00,000 --> 00:00:04,000" in srt.text
    assert "transcript.cs" in srt.headers["content-disposition"] or "titulky.cs" in srt.headers["content-disposition"]
    assert "CZ: Line number 1" in client.get(f"/api/transcripts/{tid}/export/vtt?lang=cs").text
    assert "CZ: Line number 2" in client.get(f"/api/transcripts/{tid}/export/txt?lang=cs").text
    assert "CZ: Line number 2" in client.get(f"/api/transcripts/{tid}/export/md?lang=cs").text
    assert client.get(f"/api/transcripts/{tid}/export/docx?lang=cs").status_code == 200
    # the original is untouched, and unknown / invalid languages are clear errors
    assert "CZ:" not in client.get(f"/api/transcripts/{tid}/export/srt").text
    assert client.get(f"/api/transcripts/{tid}/export/srt?lang=de").status_code == 404
    assert client.get(f"/api/transcripts/{tid}/export/srt?lang=../x").status_code == 400
    # a translation has no word marks, so a redacted export of it is refused rather than leaking words
    r = client.get(f"/api/transcripts/{tid}/export/srt?lang=cs&redact=true")
    assert r.status_code == 409 and r.json()["code"] == "translation_not_redactable"


def test_same_language_invalid_language_and_duplicate_job(client):
    tid = client.tid
    r = client.post(f"/api/transcripts/{tid}/translations", json={"language": "en"})
    assert r.status_code == 409 and r.json()["code"] == "translation_same_language"
    r = client.post(f"/api/transcripts/{tid}/translations", json={"language": "not a lang"})
    assert r.status_code == 400 and r.json()["code"] == "invalid_language"
    assert client.get(f"/api/transcripts/{tid}/translations/de").status_code == 404
    assert client.post("/api/transcripts/nope/translations", json={"language": "cs"}).status_code == 404


def test_delete_translation_and_stale_flag_via_api(client, library):
    translate(client)
    tid = client.tid
    edited = library.segments(tid)
    edited[1].text = " Edited."
    library.save_segments(tid, edited)
    assert client.get(f"/api/transcripts/{tid}/translations/cs").json()["stale"] is True
    assert client.delete(f"/api/transcripts/{tid}/translations/cs").json()["deleted"] is True
    assert client.delete(f"/api/transcripts/{tid}/translations/cs").json()["deleted"] is False
    assert client.get(f"/api/transcripts/{tid}/translations/cs").status_code == 404


def test_secure_delete_removes_translations(client, library, settings):
    translate(client)
    tid = client.tid
    folder = library.path(tid)
    assert (folder / "translations" / "cs.json").is_file()
    result = privacy.delete_transcript(library, tid, settings)
    assert result["deleted"] is True and not result.get("incomplete")
    assert not folder.exists()
    assert not list(library.root.glob("**/cs.json"))  # no tombstone with the file left behind


# --- "Translate to" of the new-transcription form ------------------------------------------------

def translations_of(client, tid):
    return [j for j in client.get("/api/jobs").json() if j["task"] == "translate" and j["source"] == tid]


def run_upload(client, **data):
    r = client.post("/api/jobs/upload", files={"file": ("talk.wav", b"RIFF fake", "audio/wav")}, data=data)
    assert r.status_code == 201, r.text
    job = r.json()
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] in ("done", "error"))
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "done", done
    return job, done


def test_option_is_carried_for_link_and_upload_and_survives_a_restart(settings, media_file):
    from mgtranscriber.jobs import JobQueue

    queue = JobQueue(settings, autostart=False)
    with TestClient(create_app(settings, queue=queue)) as c:
        link = c.post("/api/jobs", json={"source": "https://example.com/v", "translate_to": "CS"}).json()
        assert link["translate_to"] == "cs"
        assert c.post("/api/jobs", json={"source": "https://example.com/w"}).json()["translate_to"] is None
        up = c.post("/api/jobs/upload", files={"file": ("a.wav", b"RIFF fake", "audio/wav")},
                    data={"translate_to": "pt-BR"}).json()
        assert up["translate_to"] == "pt"  # stored as exactly a LANGUAGES key
        blank = c.post("/api/jobs/upload", files={"file": ("a.wav", b"RIFF fake", "audio/wav")},
                       data={"translate_to": ""}).json()
        assert blank["translate_to"] is None
    restarted = JobQueue(settings, autostart=False)
    assert restarted.get(link["id"]).request.translate_to == "cs"
    assert restarted.get(up["id"]).request.translate_to == "pt"


def test_invalid_language_is_refused_at_submit_time(settings, media_file):
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        for bad in ("not a lang", "../x", "xx", "klingon"):
            r = c.post("/api/jobs", json={"source": str(media_file()), "translate_to": bad})
            assert r.status_code in (400, 422), (bad, r.text)
        r = c.post("/api/jobs", json={"source": str(media_file()), "translate_to": "xx"})
        assert r.status_code == 400 and r.json()["code"] == "invalid_language"
        r = c.post("/api/jobs/upload", files={"file": ("a.wav", b"RIFF fake", "audio/wav")}, data={"translate_to": "xx"})
        assert r.status_code == 400 and r.json()["code"] == "invalid_language"
        assert c.get("/api/jobs").json() == []  # nothing was queued


def test_finished_transcription_queues_exactly_one_translation(client):
    job, done = run_upload(client, language="de", translate_to="cs")
    tid = done["result"]
    wait_for(lambda: any(j["status"] == "done" for j in translations_of(client, tid)))
    jobs = translations_of(client, tid)
    assert len(jobs) == 1 and jobs[0]["language"] == "cs" and jobs[0]["translate_to"] is None  # no loop
    assert client.get(f"/api/transcripts/{tid}/translations/cs").status_code == 200
    log = client.get(f"/api/jobs/{job['id']}").json()["log"]
    assert any("translation to cs queued" in line for line in log)


def test_without_the_option_nothing_is_translated(client):
    _, done = run_upload(client, language="de")
    assert translations_of(client, done["result"]) == []


def test_same_language_is_skipped_without_an_error(client):
    job, done = run_upload(client, language="en", translate_to="en")
    assert translations_of(client, done["result"]) == []
    log = client.get(f"/api/jobs/{job['id']}").json()["log"]
    assert not any("not queued" in line or "hook failed" in line for line in log)


def test_provider_failure_leaves_the_transcription_done_and_is_logged(client):
    from mgtranscriber.llm.base import LLMError

    def unavailable(provider=None, model=None):
        raise LLMError("No LLM provider is configured.", "llm_unavailable")

    client.app.state.llm.provider = unavailable
    job, done = run_upload(client, language="de", translate_to="cs")
    assert done["status"] == "done" and done["result"]
    assert translations_of(client, done["result"]) == []
    wait_for(lambda: any("not queued" in line for line in client.get(f"/api/jobs/{job['id']}").json()["log"]))
    log = client.get(f"/api/jobs/{job['id']}").json()["log"]
    assert any("translation to cs not queued: No LLM provider is configured." in line for line in log)


def test_watch_jobs_are_not_translated_twice(client):
    """A watched source translates by itself (its own translate_to) and leaves the request's field empty."""
    from mgtranscriber.llm import auto_translate

    class Job:
        id = 1
        status, result = "done", client.tid
        request = type("R", (), {"translate_to": None, "task": "transcribe"})()

        def add_log(self, line):
            raise AssertionError(line)

    auto_translate.on_job_finished(client.app.state.queue, client.app.state.llm, client.app.state.library, Job())
    assert translations_of(client, client.tid) == []


def test_recording_finish_takes_the_option(client):
    r = client.post("/api/recordings", json={"mime": "audio/webm"})
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    assert client.post(f"/api/recordings/{sid}/finish", json={"translate_to": "xx"}).json()["code"] == "invalid_language"
    assert client.get("/api/info").json()["translate_languages"]["cs"] == "Czech"


# --- the reason a translation was not queued is visible --------------------------------------------

@pytest.fixture
def no_provider(settings, tmp_path):
    """A real app without any provider: no keys, and the Ollama address leads nowhere (nothing is faked)."""
    s = dataclasses.replace(settings, root=tmp_path, extra={"llm_base_url": "http://127.0.0.1:1"})
    app = create_app(s, runner=InProcessRunner())
    app.state.llm.env = {}
    with TestClient(app) as c:
        yield c


def test_no_provider_warns_at_submit_and_shows_the_reason_on_the_finished_job(no_provider, settings):
    c = no_provider
    r = c.post("/api/jobs/upload", files={"file": ("talk.wav", b"RIFF fake", "audio/wav")},
               data={"language": "de", "translate_to": "cs"})
    assert r.status_code == 201
    job = r.json()
    assert "ollama serve" in job["translation_warning"]  # early: the response of the submit itself
    wait_for(lambda: c.get(f"/api/jobs/{job['id']}").json()["status"] == "done")
    wait_for(lambda: c.get(f"/api/jobs/{job['id']}").json()["translation"] is not None)
    done = c.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "done" and done["result"]  # the transcription is not affected
    assert done["translation"]["language"] == "cs" and "ollama serve" in done["translation"]["error"]
    assert "translation_warning" not in done and translations_of(c, done["result"]) == []
    assert any(j["translation"] and j["translation"]["error"] for j in c.get("/api/jobs").json() if j["id"] == job["id"])
    # persisted form round-trips
    stored = c.app.state.queue.jobs[job["id"]].to_dict()
    from mgtranscriber.jobs.models import Job
    assert Job.from_dict(stored).translation == done["translation"]


def test_link_warns_too_and_no_option_means_no_warning(no_provider, media_file):
    c = no_provider
    link = c.post("/api/jobs", json={"source": str(media_file()), "translate_to": "cs"}).json()
    assert "translation_warning" in link
    plain = c.post("/api/jobs", json={"source": str(media_file())}).json()
    assert "translation_warning" not in plain


def test_a_queued_translation_is_recorded_on_the_job(client):
    job, _ = run_upload(client, language="de", translate_to="cs")
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["translation"] is not None)
    tr = client.get(f"/api/jobs/{job['id']}").json()["translation"]
    assert tr["language"] == "cs" and "error" not in tr and isinstance(tr["job"], int)


def test_same_language_leaves_no_trace_on_the_job(client):
    job, _ = run_upload(client, language="en", translate_to="en")
    assert "translation_warning" not in job
    assert client.get(f"/api/jobs/{job['id']}").json()["translation"] is None
