"""Voice memory with synthetic vectors: database, thresholds, 1:1 assignment, model versioning, API."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mgtranscriber import voices
from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.server import create_app
from mgtranscriber.voices import AUTO_THRESHOLD, DIM, MIN_ENROLL_S, SUGGEST_THRESHOLD, VoiceDB, VoiceError

rng = np.random.default_rng(42)


def unit(v):
    return v / np.linalg.norm(v)


def random_voice():
    return unit(rng.standard_normal(DIM))


def near(v, cos):
    """A vector with cosine ``cos`` to ``v``."""
    noise = rng.standard_normal(DIM)
    noise -= (noise @ v) * v
    return unit(cos * v + np.sqrt(1 - cos**2) * unit(noise))


@pytest.fixture
def db(tmp_path):
    return VoiceDB(tmp_path / "voiceprints.json")


def test_vectors_roundtrip_normalized():
    v = rng.standard_normal(DIM) * 3.7
    assert np.allclose(voices.decode(voices.encode(v)), unit(v), atol=1e-6)
    with pytest.raises(VoiceError):
        voices.normalize(np.zeros(DIM))
    with pytest.raises(VoiceError):
        voices.normalize(np.ones(10))


def test_enroll_list_rename_delete(db):
    steven = random_voice()
    r = db.enroll("Steven", steven, speech_s=60, source={"transcript": "a", "label": "SPEAKER_00"})
    db.enroll("steven", near(steven, 0.9), speech_s=40)  # same person, case-insensitive name
    [person] = db.list()
    assert person["name"] == "Steven" and len(person["samples"]) == 2 and person["speech_s"] == 100
    assert "vector" not in person["samples"][0]
    db.rename(r["person_id"], "Steven Bartlett")
    assert db.person_id("steven bartlett") == r["person_id"]
    db.delete(r["person_id"], person["samples"][0]["id"])
    assert len(db.list()[0]["samples"]) == 1
    db.delete(r["person_id"], db.list()[0]["samples"][0]["id"])  # the last sample takes the person with it
    assert db.list() == []


def test_enroll_guards(db):
    with pytest.raises(VoiceError) as e:
        db.enroll("Short", random_voice(), speech_s=MIN_ENROLL_S - 1)
    assert e.value.code == "voice_too_short"
    v = random_voice()
    db.enroll("Eva", v, speech_s=30)
    with pytest.raises(VoiceError) as e:
        db.enroll("Eva", random_voice(), speech_s=30)  # a different voice under the same name
    assert e.value.code == "voice_mismatch"
    db.enroll("Eva", random_voice(), speech_s=30, force=True)
    other = db.enroll("Jan", random_voice(), speech_s=30)
    with pytest.raises(VoiceError) as e:
        db.rename(other["person_id"], "EVA")
    assert e.value.code == "voice_name_taken"


def test_thresholds(db):
    known = random_voice()
    db.enroll("Known", known, speech_s=60)
    db.enroll("Other", random_voice(), speech_s=60)
    m = db.identify({
        "A": (near(known, 0.85), 100),  # clearly the same person
        "B": (random_voice(), 100),  # a stranger
        "C": (near(known, 0.52), 100),  # between the thresholds: only a suggestion
        "D": (near(known, 0.9), 5),  # too little speech
    })
    assert m["A"].status == "auto" and m["A"].name == "Known" and m["A"].score >= AUTO_THRESHOLD
    assert m["B"].status == "unknown" and m["B"].person_id is None
    assert m["C"].status == "suggest" and SUGGEST_THRESHOLD <= m["C"].score < AUTO_THRESHOLD
    assert m["D"].status == "unknown"


def test_margin_required(db):
    a = random_voice()
    b = near(a, 0.95)  # two enrolled people with almost the same voice
    db.enroll("A", a, speech_s=60)
    db.enroll("B", b, speech_s=60)
    m = db.identify({"X": (near(a, 0.9), 60)})
    assert m["X"].status == "suggest" and m["X"].margin < voices.MARGIN


def test_one_to_one_assignment(db):
    """Two unknown labels must never both become the same known person."""
    steven = random_voice()
    db.enroll("Steven", steven, speech_s=60)
    m = db.identify({"S0": (near(steven, 0.8), 100), "S1": (near(steven, 0.9), 100)})
    assert m["S1"].status == "auto"
    assert m["S0"].status == "suggest" and m["S0"].name == "Steven"


def test_rejected_person_is_not_suggested_again(db):
    v = random_voice()
    db.enroll("Eva", v, speech_s=60)
    m = db.identify({"S0": (near(v, 0.9), 60)}, rejected={"S0": [db.person_id("Eva")]})
    assert m["S0"].status == "unknown"


def test_other_model_samples_are_ignored(tmp_path):
    path = tmp_path / "voiceprints.json"
    v = random_voice()
    VoiceDB(path, model="old/model").enroll("Eva", v, speech_s=60)
    db = VoiceDB(path)  # current model
    assert db.identify({"S0": (v, 60)})["S0"].status == "unknown"
    assert db.list()[0]["samples"][0]["outdated"] is True
    db.enroll("Eva", v, speech_s=60)  # a new sample of the current model is accepted
    assert db.identify({"S0": (v, 60)})["S0"].status == "auto"


def test_newer_or_damaged_database_is_refused(tmp_path):
    path = tmp_path / "voiceprints.json"
    path.write_text(json.dumps({"version": voices.DB_VERSION + 1, "people": {}}), encoding="utf-8")
    with pytest.raises(VoiceError) as e:
        VoiceDB(path).enroll("X", random_voice(), speech_s=60)
    assert e.value.code == "voice_db_newer"
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == voices.DB_VERSION + 1  # not overwritten
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(VoiceError) as e:
        VoiceDB(path).list()
    assert e.value.code == "voice_db_corrupt"


def test_label_spans_follow_word_speakers():
    from mgtranscriber.models import Segment

    seg = Segment.from_dict({"start": 0, "end": 9, "text": "x", "speaker": "A", "words": [
        {"word": "a", "start": 0.0, "end": 4.0, "speaker": "A"},
        {"word": "b", "start": 4.1, "end": 9.0, "speaker": "B"},
    ]})
    spans = voices.label_spans([seg])
    assert spans["A"] == [(0.0, 4.0, 4.0)] and spans["B"][0][:2] == (4.1, 9.0)


# --- pipeline hook and API ------------------------------------------------------------------


def make_transcript(library, tid, speakers: dict[str, float]):
    """A transcript whose labels speak for the given seconds."""
    folder = library.root / tid
    folder.mkdir(parents=True)
    segs, t = [], 0.0
    for label, seconds in speakers.items():
        segs.append({"start": t, "end": t + seconds, "text": f" {label} talks.", "speaker": label, "words": []})
        t += seconds
    (folder / "prepis.json").write_text(json.dumps(segs), encoding="utf-8")
    return tid


def test_after_transcription_names_known_voices(settings, library):
    db = voices.open_db(settings)
    steven = random_voice()
    db.enroll("Steven", steven, speech_s=60)
    tid = make_transcript(library, "ep2", {"SPEAKER_00": 120, "SPEAKER_01": 90})
    voices.after_transcription(library, tid, {"SPEAKER_00": list(near(steven, 0.85)),
                                              "SPEAKER_01": list(random_voice())}, settings)
    assert library.names(tid) == {"SPEAKER_00": "Steven"}
    state = voices.public_state(voices.read_transcript(library, tid))
    assert state["speakers"]["SPEAKER_00"]["match"]["status"] == "auto"
    assert state["speakers"]["SPEAKER_01"]["match"]["status"] == "unknown"
    assert state["speakers"]["SPEAKER_00"]["speech_s"] == 120
    # a re-run without embeddings (e.g. a cloud engine) removes the old voiceprints
    voices.after_transcription(library, tid, None, settings)
    assert not (library.path(tid) / voices.TRANSCRIPT_FILE).exists()


def test_after_transcription_never_raises(settings, library):
    tid = make_transcript(library, "bad", {"SPEAKER_00": 60})
    voices.after_transcription(library, tid, {"SPEAKER_00": [1.0, 2.0]}, settings)  # wrong dimension
    voices.after_transcription(library, "missing", {"SPEAKER_00": list(random_voice())}, settings)


def test_user_names_are_not_overwritten(settings, library):
    db = voices.open_db(settings)
    v = random_voice()
    db.enroll("Steven", v, speech_s=60)
    tid = make_transcript(library, "ep", {"SPEAKER_00": 60})
    library.save_names(tid, {"SPEAKER_00": "Host"})
    voices.after_transcription(library, tid, {"SPEAKER_00": list(near(v, 0.9))}, settings)
    assert library.names(tid) == {"SPEAKER_00": "Host"}


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        yield c


def test_api_enroll_then_recognise_in_another_transcript(client, settings, library):
    steven, roman = random_voice(), random_voice()
    a = make_transcript(library, "ep-a", {"SPEAKER_00": 300, "SPEAKER_01": 100})
    voices.store_embeddings(library, a, {"SPEAKER_00": list(roman), "SPEAKER_01": list(steven)},
                            {"SPEAKER_00": 300, "SPEAKER_01": 100})
    state = client.get(f"/api/voices/transcripts/{a}").json()
    assert state["available"] and state["speakers"]["SPEAKER_01"]["match"]["status"] == "unknown"

    r = client.post(f"/api/voices/transcripts/{a}/enroll", json={"label": "SPEAKER_01", "name": "Steven"})
    assert r.status_code == 200, r.text
    assert r.json()["names"] == {"SPEAKER_01": "Steven"}
    r = client.post(f"/api/voices/transcripts/{a}/enroll", json={"label": "SPEAKER_00", "name": "Roman"})
    [p1, p2] = client.get("/api/voices").json()["people"]
    assert [p1["name"], p2["name"]] == ["Roman", "Steven"] and p2["samples"][0]["source"]["transcript"] == a

    # a new transcript: the pipeline hook recognises Steven automatically, Roman's voice is only similar
    b = make_transcript(library, "ep-b", {"SPEAKER_00": 200, "SPEAKER_01": 200})
    voices.after_transcription(library, b, {"SPEAKER_00": list(near(steven, 0.88)),
                                            "SPEAKER_01": list(near(roman, 0.5))}, settings)
    assert library.names(b) == {"SPEAKER_00": "Steven"}
    state = client.get(f"/api/voices/transcripts/{b}").json()
    assert state["speakers"]["SPEAKER_01"]["match"]["status"] == "suggest"

    # accepting the suggestion names the label and adds a sample
    r = client.post(f"/api/voices/transcripts/{b}/accept", json={"label": "SPEAKER_01"}).json()
    assert r["names"] == {"SPEAKER_00": "Steven", "SPEAKER_01": "Roman"}
    roman_person = next(p for p in client.get("/api/voices").json()["people"] if p["name"] == "Roman")
    assert len(roman_person["samples"]) == 2

    # rejecting an automatic match removes the automatic name
    r = client.post(f"/api/voices/transcripts/{b}/reject", json={"label": "SPEAKER_00"}).json()
    assert r["names"] == {"SPEAKER_01": "Roman"}
    assert r["state"]["speakers"]["SPEAKER_00"]["match"]["status"] == "unknown"


def test_api_errors_and_delete_all(client, settings, library):
    a = make_transcript(library, "ep", {"SPEAKER_00": 5})
    r = client.post(f"/api/voices/transcripts/{a}/enroll", json={"label": "SPEAKER_00", "name": "X"})
    assert r.status_code == 409 and r.json()["code"] == "voice_not_available"
    voices.store_embeddings(library, a, {"SPEAKER_00": list(random_voice())}, {"SPEAKER_00": 5})
    r = client.post(f"/api/voices/transcripts/{a}/enroll", json={"label": "SPEAKER_00", "name": "X"})
    assert r.status_code == 400 and r.json()["code"] == "voice_too_short"
    r = client.post(f"/api/voices/transcripts/{a}/accept", json={"label": "SPEAKER_00"})
    assert r.status_code == 409 and r.json()["code"] == "voice_no_suggestion"
    assert client.post(f"/api/voices/transcripts/{a}/enroll", json={"label": "../x", "name": "X"}).status_code == 422
    assert client.get("/api/voices/transcripts/nope").status_code == 404
    assert client.delete("/api/voices/people/nope").json()["code"] == "voice_person_not_found"

    voices.open_db(settings).enroll("Eva", random_voice(), speech_s=60)
    export = client.get("/api/voices/export")
    assert "attachment" in export.headers["content-disposition"] and "Eva" in export.text
    assert client.delete("/api/voices").json() == {"deleted": True, "transcripts": 1, "jobs_cancelled": 0}
    assert client.get("/api/voices").json()["people"] == []
    assert not (library.path(a) / voices.TRANSCRIPT_FILE).exists()


def test_api_compute_queues_a_voiceprints_job(client, settings, library, monkeypatch):
    a = make_transcript(library, "ep", {"SPEAKER_00": 60})
    r = client.post(f"/api/voices/transcripts/{a}/compute")
    assert r.status_code == 409  # no HF token in the test settings
    monkeypatch.setattr(client.app.state, "settings", settings.with_overrides(hf_token="x"))
    assert client.post(f"/api/voices/transcripts/{a}/compute").json()["code"] == "voice_no_audio"


# --- review fixes: concurrency, damaged files, confirmations --------------------------------


def test_concurrent_enrolls_from_threads_keep_the_database_valid(tmp_path):
    import threading

    path = tmp_path / "voiceprints.json"
    errors = []

    def worker(n):
        try:
            for i in range(25):
                VoiceDB(path).enroll(f"P{n}-{i}", random_voice(), speech_s=60)  # a new instance per call, like the router
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(VoiceDB(path).list()) == 100
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_enrolls_from_processes(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "voiceprints.json"
    code = ("import sys, numpy as np; from mgtranscriber.voices import VoiceDB, DIM\n"
            "rng = np.random.default_rng(int(sys.argv[2]))\n"
            "for i in range(15): VoiceDB(sys.argv[1]).enroll(f'{sys.argv[2]}-{i}', rng.standard_normal(DIM), speech_s=60)\n")
    procs = [subprocess.Popen([sys.executable, "-c", code, str(path), str(n)]) for n in range(3)]
    assert [p.wait(120) for p in procs] == [0, 0, 0]
    assert len(VoiceDB(path).list()) == 45


@pytest.mark.parametrize("content", [
    "{not json", b"\xff\xfe\x00garbage", '{"version": "1", "people": {}}', '{"version": 1, "people": {"a": {}}}',
    '{"version": 1, "people": {"a": {"name": "X", "samples": [1]}}}', "[]",
])
def test_damaged_database_is_a_voice_error(tmp_path, content):
    path = tmp_path / "voiceprints.json"
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    with pytest.raises(VoiceError) as e:
        VoiceDB(path).list()
    assert e.value.code == "voice_db_corrupt"


def test_damaged_database_api_answers_with_a_code(client, settings, library):
    voices.db_path(settings).parent.mkdir(parents=True, exist_ok=True)
    voices.db_path(settings).write_text("{oops", encoding="utf-8")
    r = client.get("/api/voices")
    assert r.status_code == 500 and r.json()["code"] == "voice_db_corrupt"


def test_damaged_transcript_voices_count_as_missing(client, library):
    a = make_transcript(library, "ep", {"SPEAKER_00": 60})
    (library.path(a) / voices.TRANSCRIPT_FILE).write_text("{broken", encoding="utf-8")
    assert client.get(f"/api/voices/transcripts/{a}").json()["available"] is False
    (library.path(a) / voices.TRANSCRIPT_FILE).write_text('{"speakers": {"SPEAKER_00": {"no": "vector"}}}', encoding="utf-8")
    assert client.get(f"/api/voices/transcripts/{a}").json()["available"] is False


def test_accept_with_little_speech_names_without_a_sample(client, settings, library):
    v = random_voice()
    voices.open_db(settings).enroll("Eva", v, speech_s=60)
    a = make_transcript(library, "ep", {"SPEAKER_00": 15})
    voices.store_embeddings(library, a, {"SPEAKER_00": list(near(v, 0.5))}, {"SPEAKER_00": 15})
    assert client.get(f"/api/voices/transcripts/{a}").json()["speakers"]["SPEAKER_00"]["match"]["status"] == "suggest"
    r = client.post(f"/api/voices/transcripts/{a}/accept", json={"label": "SPEAKER_00"})
    assert r.status_code == 200, r.text
    assert r.json()["names"] == {"SPEAKER_00": "Eva"}
    assert r.json()["state"]["speakers"]["SPEAKER_00"]["match"]["confirmed"] is True
    [eva] = client.get("/api/voices").json()["people"]
    assert len(eva["samples"]) == 1
    # identifying again keeps the confirmation
    r = client.post(f"/api/voices/transcripts/{a}/identify").json()
    assert r["state"]["speakers"]["SPEAKER_00"]["match"]["confirmed"] is True


def test_enrolled_disappears_when_the_person_is_deleted(client, library):
    a = make_transcript(library, "ep", {"SPEAKER_00": 60})
    voices.store_embeddings(library, a, {"SPEAKER_00": list(random_voice())}, {"SPEAKER_00": 60})
    r = client.post(f"/api/voices/transcripts/{a}/enroll", json={"label": "SPEAKER_00", "name": "Eva"}).json()
    assert r["state"]["speakers"]["SPEAKER_00"]["enrolled"]
    # the same label cannot be saved twice
    again = client.post(f"/api/voices/transcripts/{a}/enroll", json={"label": "SPEAKER_00", "name": "Eva"})
    assert again.status_code == 409 and again.json()["code"] == "voice_already_enrolled"
    client.delete(f"/api/voices/people/{r['person_id']}")
    state = client.get(f"/api/voices/transcripts/{a}").json()
    assert state["speakers"]["SPEAKER_00"]["enrolled"] is None
    r = client.post(f"/api/voices/transcripts/{a}/identify").json()
    assert not r["state"]["speakers"]["SPEAKER_00"]["match"].get("confirmed")
    # and the voice can be saved again
    assert client.post(f"/api/voices/transcripts/{a}/enroll", json={"label": "SPEAKER_00", "name": "Eva"}).status_code == 200


def test_reject_keeps_a_name_the_user_typed(client, settings, library):
    v = random_voice()
    voices.open_db(settings).enroll("Steven", v, speech_s=60)
    a = make_transcript(library, "ep", {"SPEAKER_00": 60})
    library.save_names(a, {"SPEAKER_00": "Steven"})  # typed by the user before identification
    voices.after_transcription(library, a, {"SPEAKER_00": list(near(v, 0.9))}, settings)
    r = client.post(f"/api/voices/transcripts/{a}/reject", json={"label": "SPEAKER_00"}).json()
    assert r["names"] == {"SPEAKER_00": "Steven"}


def test_delete_all_removes_leftovers_and_cancels_voiceprint_jobs(client, settings, library):
    a = make_transcript(library, "ep", {"SPEAKER_00": 60})
    voices.store_embeddings(library, a, {"SPEAKER_00": list(random_voice())}, {"SPEAKER_00": 60})
    (library.path(a) / "voices.json.123.tmp").write_text("{}", encoding="utf-8")
    orphan = library.root / "half-written"
    orphan.mkdir()
    (orphan / voices.TRANSCRIPT_FILE).write_text("{}", encoding="utf-8")
    db = voices.open_db(settings)
    db.enroll("Eva", random_voice(), speech_s=60)
    (db.path.parent / "voiceprints.json.999.tmp").write_text("{}", encoding="utf-8")

    from mgtranscriber.pipeline import TranscriptionRequest
    queue = client.app.state.queue
    job = queue.add(TranscriptionRequest(a, task="voiceprints"))
    r = client.delete("/api/voices").json()
    assert r["transcripts"] == 2
    assert queue.get(job.id).status in ("cancelled", "error")
    assert not list(library.path(a).glob("voices.json*[!k]"))  # only the lock file may stay
    assert not (orphan / voices.TRANSCRIPT_FILE).exists()
    assert not list(db.path.parent.glob("voiceprints.json*.tmp")) and not db.path.exists()
