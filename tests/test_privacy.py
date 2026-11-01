"""Sensitive-data mode: offline mode in the API, chain of custody, audit log, redaction, exports, secure delete."""

import dataclasses
import io
import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from mgtranscriber import editor, pipeline, privacy, voices
from mgtranscriber.engines import EngineError, get_engine, list_engines
from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.llm import LLMError, load_config, make_provider, provider_status
from mgtranscriber.server import create_app


@pytest.fixture
def offline_settings(settings):
    return dataclasses.replace(settings, extra={**settings.extra, "offline_mode": True})


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        yield c


@pytest.fixture
def offline_client(offline_settings):
    with TestClient(create_app(offline_settings, runner=InProcessRunner())) as c:
        yield c


@pytest.fixture
def tid(settings, library, media_file):
    return pipeline.run(pipeline.TranscriptionRequest(str(media_file("talk.wav"))), settings, library=library)


def redact(library, tid, seg_index, word_indexes):
    segs = [editor.normalize(s) for s in library.segments(tid)]
    for i in word_indexes:
        segs[seg_index].words[i].redacted = True
    editor.save(library, tid, segs, base=None)


# --- offline mode ---------------------------------------------------------------------------


def test_offline_flag_from_settings_and_env(settings, offline_settings, monkeypatch):
    assert not privacy.offline(settings) and privacy.offline(offline_settings)
    monkeypatch.setenv("MGTRANSCRIBER_OFFLINE_MODE", "true")
    assert privacy.offline(settings)
    monkeypatch.setenv("MGTRANSCRIBER_OFFLINE_MODE", "0")
    assert not privacy.offline(offline_settings)  # the environment beats settings.json


def test_offline_refuses_cloud_and_network_engines(offline_settings):
    for name in ("elevenlabs", "deepgram", "gemini", "youtube"):
        with pytest.raises(EngineError, match="Offline mode"):
            get_engine(name, offline_settings)
    get_engine("mock", offline_settings)  # local engines still work
    public = {e.name: e.public(offline_settings) for e in list_engines()}
    assert public["gemini"]["missing"]["code"] == "offline" and public["youtube"]["missing"]["code"] == "offline"
    assert public["whisperx"]["missing"] is None or public["whisperx"]["missing"]["code"] != "offline"


def test_offline_refuses_cloud_llms_and_remote_ollama(offline_settings):
    cfg = load_config(offline_settings, env={})
    for provider in ("gemini", "anthropic"):
        with pytest.raises(LLMError) as e:
            make_provider(cfg, offline_settings, provider, env={"GEMINI_API_KEY": "x", "ANTHROPIC_API_KEY": "x"})
        assert e.value.code == "offline_llm" and e.value.status == 403
    make_provider(cfg, offline_settings, "ollama", env={})  # local Ollama is fine
    remote = dataclasses.replace(offline_settings, extra={**offline_settings.extra, "llm_base_url": "http://10.0.0.5:11434"})
    with pytest.raises(LLMError):
        make_provider(load_config(remote, env={}), remote, "ollama", env={})
    status = {p["name"]: p for p in provider_status(cfg, offline_settings, env={"GEMINI_API_KEY": "x"})}
    assert status["gemini"]["available"] is False and status["gemini"]["offline"] is True


def test_offline_refuses_url_sources_in_the_pipeline(offline_settings, library):
    with pytest.raises(privacy.PrivacyError) as e:
        pipeline.run(pipeline.TranscriptionRequest("https://www.youtube.com/watch?v=x"), offline_settings, library=library)
    assert e.value.code == "offline_url"


def test_offline_api(offline_client, offline_settings, library, media_file):
    info = offline_client.get("/api/info").json()
    assert info["privacy"] == {"offline": True}
    assert offline_client.get("/api/privacy").json() == {"offline": True}
    # URL sources and cloud engines are refused when the job is created
    r = offline_client.post("/api/jobs", json={"source": "https://example.com/v"})
    assert r.status_code == 403 and r.json()["code"] == "offline_url"
    r = offline_client.post("/api/jobs", json={"source": str(media_file("x.wav")), "engine": "deepgram"})
    assert r.status_code == 403 and r.json()["code"] == "offline_engine"
    r = offline_client.post("/v1/audio/transcriptions", data={"model": "gemini"},
                            files={"file": ("a.wav", b"RIFF fake", "audio/wav")})
    assert r.status_code == 403 and r.json()["error"]["code"] == "offline_engine"
    # analysis with a cloud provider is refused by the API
    tid = pipeline.run(pipeline.TranscriptionRequest(str(media_file("a.wav"))), offline_settings, library=library)
    r = offline_client.post(f"/api/transcripts/{tid}/analysis", json={"provider": "gemini"})
    assert r.status_code == 403 and r.json()["code"] == "offline_llm"
    r = offline_client.post(f"/api/transcripts/{tid}/ask", json={"question": "Who?", "provider": "anthropic"})
    assert r.status_code == 403 and r.json()["code"] == "offline_llm"
    # watching feeds is refused, webhooks are not sent
    r = offline_client.post("/api/watch/preview", json={"kind": "rss", "target": "https://example.com/feed.xml"})
    assert r.status_code in (403, 422) and "offline" in json.dumps(r.json()).lower()


# --- chain of custody and audit log ---------------------------------------------------------


def test_custody_and_audit_after_transcription(client, tid, library, media_file):
    meta = library.metadata(tid)
    custody = meta.extra["custody"]
    assert custody["algorithm"] == "sha256"
    assert custody["source"]["sha256"] == privacy.sha256_file(media_file("talk.wav"))
    assert custody["outputs"]["prepis.json"] == privacy.sha256_file(library.path(tid) / "prepis.json")
    body = client.get(f"/api/transcripts/{tid}/custody").json()
    assert body["audit_intact"] and set(body["check"].values()) == {"ok"}
    assert body["audit"][0]["event"] == "created" and body["audit"][0]["prev"] is None

    # tampering with prepis.json outside the app is detected
    path = library.path(tid) / "prepis.json"
    path.write_text(path.read_text(encoding="utf-8").replace("Hello", "Bye"), encoding="utf-8")
    assert client.get(f"/api/transcripts/{tid}/custody").json()["check"]["prepis.json"] == "changed"


def test_audit_log_is_a_hash_chain(client, tid, library):
    client.get(f"/api/transcripts/{tid}/export/txt")
    client.get(f"/api/transcripts/{tid}/export/srt?redact=true")
    log = privacy.read_audit(library, tid)
    assert [e["event"] for e in log] == ["created", "exported", "exported"]
    assert log[2]["redacted"] is True and log[1]["format"] == "txt"
    path = library.path(tid) / privacy.AUDIT_FILE
    assert privacy.verify_audit(path)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + lines[2])  # a line removed
    assert not privacy.verify_audit(path)


def test_record_custody_for_older_transcripts(client, legacy_transcript, library):
    r = client.post(f"/api/transcripts/{legacy_transcript}/custody")
    assert r.status_code == 200
    stored = r.json()["stored"]
    assert stored["source"]["file"] == "audio.webm" and stored["outputs"]["prepis.json"]
    assert r.json()["audit"][-1]["event"] == "hashed"


# --- redaction and exports ------------------------------------------------------------------


def test_redacted_exports(client, tid, library):
    redact(library, tid, 0, [1])  # "there."
    for fmt in ("txt", "srt", "vtt", "json", "md"):
        plain = client.get(f"/api/transcripts/{tid}/export/{fmt}").text
        red = client.get(f"/api/transcripts/{tid}/export/{fmt}?redact=true")
        assert "there." in plain, fmt
        assert "there." not in red.text and "[REDACTED]" in red.text, fmt
        assert ".redacted." in red.headers["content-disposition"]
    docx = client.get(f"/api/transcripts/{tid}/export/docx?redact=true")
    assert "there." not in docx_text(docx.content) and "[REDACTED]" in docx_text(docx.content)


def test_redaction_label_follows_language_and_merges_runs(library, tid):
    segs = [editor.normalize(s) for s in library.segments(tid)]
    segs[1].words[0].redacted = segs[1].words[1].redacted = True
    out = privacy.redact_segments(segs, privacy.redacted_label("cs"))
    assert out[1].text.split()[0] == "[REDIGOVÁNO]" and out[1].text.count("[REDIGOVÁNO]") == 1
    assert segs[1].words[0].word != "[REDIGOVÁNO]"  # the originals are untouched


def test_redacted_ranges():
    from mgtranscriber.models import Segment, Word

    seg = Segment(0, 10, " a b c d", None, [Word("a", 0, 1), Word("b", 1, 2, redacted=True),
                                             Word("c", None, None, redacted=True), Word("d", 6, 7)])
    assert privacy.redacted_ranges([seg], pad=0) == [(1.0, 6.0)]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")
def test_redacted_audio_is_silenced(client, library, tmp_path, settings, media_file):
    wav = tmp_path / "in" / "tone.wav"
    wav.parent.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=6",
                    "-ac", "1", str(wav)], check=True)
    tid = pipeline.run(pipeline.TranscriptionRequest(str(wav)), settings, library=library)
    redact(library, tid, 1, [0])  # "Hi," 2.5-2.8 s
    r = client.get(f"/api/transcripts/{tid}/redacted/audio?beep=false")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"
    out = tmp_path / "red.mp3"
    out.write_bytes(r.content)

    def level(start, length=0.2):
        proc = subprocess.run(["ffmpeg", "-hide_banner", "-ss", str(start), "-t", str(length), "-i", str(out),
                               "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True)
        return float(proc.stderr.split("max_volume:")[1].split("dB")[0])

    assert level(2.55) < -60  # silenced
    assert level(1.0) > -20  # untouched
    assert privacy.read_audit(library, tid)[-1]["format"] == "audio"
    out.write_bytes(client.get(f"/api/transcripts/{tid}/redacted/audio").content)  # with the beep
    assert -20 < level(2.55) < -6


def docx_text(content: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(content))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        parts += [c.text for row in table.rows for c in row.cells]
    return "\n".join(parts)


def test_docx_export(client, tid, library):
    library.save_names(tid, {"SPEAKER_00": "Alice"})
    r = client.get(f"/api/transcripts/{tid}/export/docx")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/vnd.openxmlformats")
    from docx import Document

    doc = Document(io.BytesIO(r.content))
    assert doc.paragraphs[0].text == library.metadata(tid).title
    text = docx_text(r.content)
    assert "Source:" in text and "talk.wav" in text and "Alice" in text and "[00:00:00]" in text
    bold = [run.text for p in doc.paragraphs for run in p.runs if run.bold]
    assert "Alice" in bold
    table = client.get(f"/api/transcripts/{tid}/export/docx?layout=table")
    tdoc = Document(io.BytesIO(table.content))
    assert len(tdoc.tables) == 1 and tdoc.tables[0].rows[1].cells[1].text == "Alice"
    assert "docx" in [e["name"] for e in client.get("/api/info").json()["exporters"]]


def test_vtt_uses_speaker_names(client, tid, library):
    library.save_names(tid, {"SPEAKER_01": "Bob"})
    text = client.get(f"/api/transcripts/{tid}/export/vtt").text
    assert text.startswith("WEBVTT") and "<v Bob>Hi, how are you?" in text


# --- secure delete --------------------------------------------------------------------------


def test_secure_delete(client, tid, library, settings):
    import numpy as np

    vec = list(np.eye(voices.DIM)[0])
    voices.store_embeddings(library, tid, {"SPEAKER_00": vec}, {"SPEAKER_00": 60})
    voices.enroll_label(library, tid, "SPEAKER_00", voices.open_db(settings), name="Alice")
    other = voices.open_db(settings).enroll("Bob", list(np.eye(voices.DIM)[1]), speech_s=60)
    redact(library, tid, 0, [0])
    assert client.get("/api/search", params={"q": "Hello"}).json()["total"] == 1

    r = client.delete(f"/api/transcripts/{tid}")
    assert r.status_code == 400 and r.json()["code"] == "confirm_required"
    r = client.delete(f"/api/transcripts/{tid}", params={"confirm": tid})
    assert r.status_code == 200, r.text
    assert r.json()["voice_samples"] == 1
    assert not (library.root / tid).exists()
    assert client.get(f"/api/transcripts/{tid}").status_code == 404
    assert client.get("/api/search", params={"q": "Hello"}).json()["total"] == 0
    assert [p["name"] for p in voices.open_db(settings).list()] == ["Bob"] and other["name"] == "Bob"
    record = json.loads((settings.data_dir / privacy.GLOBAL_AUDIT_FILE).read_text(encoding="utf-8").splitlines()[-1])
    assert record["event"] == "deleted" and tid not in json.dumps(record)


def test_overwrite_before_delete(tmp_path):
    path = tmp_path / "secret.bin"
    path.write_bytes(b"secret" * 1000)
    privacy._overwrite(path)
    assert path.read_bytes() == b"\0" * 6000


# --- audit log: interrupted writes and truncation ---------------------------------------------


def test_audit_survives_an_interrupted_write(client, tid, library):
    client.get(f"/api/transcripts/{tid}/export/txt")
    path = library.path(tid) / privacy.AUDIT_FILE
    with open(path, "ab") as f:
        f.write(b'{"time":"2026-10-10T10:00:00Z","event":"expo')  # power cut in the middle of a record
    assert privacy.read_audit(library, tid)[-1] == {"event": "incomplete"}
    status = privacy.audit_status(path)
    assert status["chain"] and status["partial_tail"] and status["anchor"] == "ok"
    assert client.get(f"/api/transcripts/{tid}/custody").json()["audit_intact"]

    client.get(f"/api/transcripts/{tid}/export/srt")  # the next record cuts the tail off and notes it
    log = privacy.read_audit(library, tid)
    assert [e["event"] for e in log] == ["created", "exported", "exported"] and log[-1]["recovered_bytes"] > 0
    assert privacy.audit_status(path) == {"chain": True, "records": 3, "partial_tail": False, "anchor": "ok"}


def test_audit_anchor_shows_a_cut_off_end(client, tid, library):
    client.get(f"/api/transcripts/{tid}/export/txt")
    client.get(f"/api/transcripts/{tid}/export/srt")
    path = library.path(tid) / privacy.AUDIT_FILE
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))  # the last record removed: the chain alone cannot tell
    assert privacy.verify_audit(path)
    assert privacy.audit_status(path)["anchor"] == "truncated"
    body = client.get(f"/api/transcripts/{tid}/custody").json()
    assert not body["audit_intact"] and body["audit_state"]["anchor"] == "truncated"
    path.write_bytes(b"".join(lines[:-1]) + lines[-1].replace(b"exported", b"exportXX"))
    assert privacy.audit_status(path)["anchor"] == "mismatch"


def test_older_log_without_an_anchor_is_still_intact(client, tid, library):
    path = library.path(tid) / privacy.AUDIT_FILE
    path.with_name(path.name + privacy.ANCHOR_SUFFIX).unlink()
    assert privacy.audit_status(path)["anchor"] == "none"
    assert client.get(f"/api/transcripts/{tid}/custody").json()["audit_intact"]


# --- redacted exports: title, audio ---------------------------------------------------------


def test_redacted_markdown_leaves_out_the_title_unless_asked(client, tid, library):
    title = library.metadata(tid).title
    assert f"# {title}" in client.get(f"/api/transcripts/{tid}/export/md").text
    red = client.get(f"/api/transcripts/{tid}/export/md?redact=true").text
    assert title not in red and not red.lstrip().startswith("#")
    assert f"# {title}" in client.get(f"/api/transcripts/{tid}/export/md?redact=true&title=true").text


def test_redacted_audio_one_at_a_time_and_no_leftovers(client, library, tmp_path, settings):
    from mgtranscriber.server.routers import privacy as privacy_router

    wav = tmp_path / "in" / "tone.wav"
    wav.parent.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=3",
                    "-ac", "1", str(wav)], check=True)
    tid = pipeline.run(pipeline.TranscriptionRequest(str(wav)), settings, library=library)

    assert privacy_router._audio_slot.acquire(blocking=False)
    try:
        r = client.get(f"/api/transcripts/{tid}/redacted/audio")
        assert r.status_code == 409 and r.json()["code"] == "redaction_running"
    finally:
        privacy_router._audio_slot.release()
    r = client.get(f"/api/transcripts/{tid}/redacted/audio")
    assert r.status_code == 200
    assert list((settings.data_dir / "tmp").glob("redacted-*.mp3")) == []  # removed after sending


def test_redacted_audio_failure_removes_the_temp_file_and_frees_the_slot(client, tid, settings, monkeypatch):
    from mgtranscriber.server.routers import privacy as privacy_router

    def boom(*args, **kwargs):
        raise privacy.PrivacyError("ffmpeg_failed", "ffmpeg failed", 500)

    monkeypatch.setattr(privacy, "redacted_audio", boom)
    assert client.get(f"/api/transcripts/{tid}/redacted/audio").status_code == 500
    assert list((settings.data_dir / "tmp").glob("redacted-*.mp3")) == []
    assert privacy_router._audio_slot.acquire(blocking=False)
    privacy_router._audio_slot.release()


# --- job_targets, tombstone, loopback, JSON limit -------------------------------------------


def test_job_targets_matches_metadata_source_and_source_key(library, tid, media_file, tmp_path):
    meta = library.metadata(tid)
    assert privacy.job_targets(library, tid, {"source": meta.source})  # metadata.source
    assert privacy.job_targets(library, tid, {"source": "x", "result": tid})
    assert not privacy.job_targets(library, tid, {"source": str(tmp_path / "other.wav"), "title": "other"})
    # the same local file named differently (relative path, other case on Windows): same source key
    import os

    from pathlib import Path

    indirect = str(Path(meta.source).parent / ".." / Path(meta.source).parent.name / Path(meta.source).name)
    assert privacy.job_targets(library, tid, {"source": indirect})
    assert privacy.job_targets(library, tid, {"source": meta.source.upper()}) == (os.name == "nt")


def test_failed_delete_leaves_a_tombstone_not_a_half_deleted_transcript(client, tid, library, settings, monkeypatch):
    def broken_rmtree(*args, **kwargs):
        raise OSError("disk says no")

    monkeypatch.setattr(privacy.shutil, "rmtree", broken_rmtree)
    r = client.delete(f"/api/transcripts/{tid}", params={"confirm": tid})
    assert r.status_code == 200 and r.json()["incomplete"] is True
    assert client.get(f"/api/transcripts/{tid}").status_code == 404  # gone from the library at once
    assert tid not in [t["id"] for t in client.get("/api/transcripts").json()]
    tombstones = list(library.root.glob(privacy.TOMBSTONE_PREFIX + "*"))
    assert len(tombstones) == 1
    record = json.loads((settings.data_dir / privacy.GLOBAL_AUDIT_FILE).read_text(encoding="utf-8").splitlines()[-1])
    assert record["event"] == "deleted" and record["incomplete"] is True
    monkeypatch.undo()
    assert privacy.purge_tombstones(library) == 1 and not tombstones[0].exists()


def test_voice_samples_and_index_are_cleaned_even_when_files_fail(client, tid, library, settings, monkeypatch):
    import numpy as np

    voices.store_embeddings(library, tid, {"SPEAKER_00": list(np.eye(voices.DIM)[0])}, {"SPEAKER_00": 60})
    voices.enroll_label(library, tid, "SPEAKER_00", voices.open_db(settings), name="Alice")
    monkeypatch.setattr(privacy, "_overwrite", lambda path: (_ for _ in ()).throw(RuntimeError("boom")))
    r = client.delete(f"/api/transcripts/{tid}", params={"confirm": tid})
    assert r.json()["incomplete"] is True and r.json()["voice_samples"] == 1
    assert client.get("/api/search", params={"q": "Hello"}).json()["total"] == 0


@pytest.mark.parametrize("host, loopback", [
    ("127.0.0.1", True), ("127.5.4.3", True), ("::1", True), ("[::1]", True), ("localhost", True),
    ("LOCALHOST.", True), ("127.evil.com", False), ("evil.localhost", False), ("localhost.evil.com", False),
    ("0.0.0.0", False), ("192.168.1.5", False), ("", False),
])
def test_is_loopback_host(host, loopback):
    from mgtranscriber.config import is_loopback_host

    assert is_loopback_host(host) is loopback


def test_json_bodies_are_limited_to_50_mb(client, tid, monkeypatch):
    from mgtranscriber.server import limits

    assert limits.JSON_MAX_BYTES == 50 * 1024 * 1024
    monkeypatch.setattr(limits, "JSON_MAX_BYTES", 2000)
    big = {"segments": [{"start": 0, "end": 1, "text": "x" * 3000, "speaker": "A"}]}
    r = client.put(f"/api/transcripts/{tid}/segments", json=big)
    assert r.status_code == 413 and r.json()["code"] == "request_too_large"
    r = client.put(f"/api/transcripts/{tid}/segments", content=b"x" * 3000,  # chunk-less body of a wrong type
                   headers={"content-type": "text/plain"})
    assert r.status_code != 413 or "JSON" not in r.text
