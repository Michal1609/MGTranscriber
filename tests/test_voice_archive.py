"""Voice memory across the archive: scan, person page (appearances, suggestions), merge, search by speaker.

Voices are synthetic vectors (``near(v, cos)`` has a known cosine to ``v``), so no GPU is needed.
"""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from mgtranscriber import voices
from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.jobs.runners import Cancelled
from mgtranscriber.search import SearchIndex
from mgtranscriber.server import create_app

from .conftest import wait_for
from .test_voices import near, random_voice


def make(library, tid, speakers, *, date=None, texts=None, embeddings=None, speech=None, names=None):
    """A transcript with one long segment per label and, when ``embeddings`` is given, voiceprints
    (stored without identifying, like a transcription made before anybody was enrolled)."""
    folder = library.root / tid
    folder.mkdir(parents=True)
    segs, t = [], 0.0
    for label, seconds in speakers.items():
        text = (texts or {}).get(label, f"{label} says hello in {tid}")
        segs.append({"start": t, "end": t + seconds, "text": " " + text, "speaker": label, "words": []})
        t += seconds
    (folder / "prepis.json").write_text(json.dumps(segs), encoding="utf-8")
    if date:
        (folder / "metadata.json").write_text(json.dumps({"title": tid, "date": date, "source": "x"}), encoding="utf-8")
    if names:
        library.save_names(tid, names)
    if embeddings:
        voices.store_embeddings(library, tid, {k: list(v) for k, v in embeddings.items()},
                                speech or dict(speakers))
    return tid


@pytest.fixture
def world(settings, library):
    """Steven and Roman are known (one sample each); six transcripts with voiceprints, none identified yet."""
    db = voices.open_db(settings)
    steven, roman = random_voice(), random_voice()
    ids = {"steven": db.enroll("Steven", steven, speech_s=60)["person_id"],
           "roman": db.enroll("Roman", roman, speech_s=60)["person_id"]}
    return {
        "db": db, "ids": ids, "steven": steven, "roman": roman,
        "library": library, "settings": settings,
    }


def populate(w):
    lib, steven, roman = w["library"], w["steven"], w["roman"]
    make(lib, "t1", {"SPEAKER_00": 100, "SPEAKER_01": 100}, date="2026-01-01T00:00:00+00:00",
         embeddings={"SPEAKER_00": near(steven, 0.9), "SPEAKER_01": random_voice()})  # Steven auto, other unknown
    make(lib, "t2", {"SPEAKER_00": 100}, date="2026-02-01T00:00:00+00:00",
         embeddings={"SPEAKER_00": near(steven, 0.52)})  # Steven only suggested
    make(lib, "t3", {"SPEAKER_00": 100, "SPEAKER_01": 50}, date="2026-03-01T00:00:00+00:00",
         embeddings={"SPEAKER_00": near(roman, 0.85), "SPEAKER_01": near(steven, 0.8)},
         names={"SPEAKER_01": "Host"})  # Roman auto; the user called Steven's voice "Host"
    make(lib, "t4", {"SPEAKER_00": 5}, embeddings={"SPEAKER_00": near(steven, 0.95)})  # too little speech
    make(lib, "t5", {"SPEAKER_00": 100}, date="2026-05-01T00:00:00+00:00")  # no voiceprints at all
    make(lib, "t6", {"SPEAKER_00": 100}, date="2026-06-01T00:00:00+00:00", names={"SPEAKER_00": "steven"})
    return ["t1", "t2", "t3", "t4", "t5", "t6"]


def test_scan_archive_identifies_every_transcript(world):
    populate(world)
    lib, db = world["library"], world["db"]
    seen = []
    result = voices.scan_archive(lib, db, progress=lambda n, total, tid, counts: seen.append((n, total)))
    assert result == {"transcripts": 6, "scanned": 4, "without_voiceprints": 2, "auto": 3, "suggest": 1, "errors": 0,
                      "cancelled": False, "people": 2}
    assert seen[0] == (0, 6) and seen[-1] == (6, 6)
    assert lib.names("t1") == {"SPEAKER_00": "Steven"}
    assert lib.names("t2") == {}  # a suggestion never becomes a name
    assert lib.names("t3") == {"SPEAKER_00": "Roman", "SPEAKER_01": "Host"}  # the typed name stays
    match = voices.read_transcript(lib, "t3")["speakers"]["SPEAKER_01"]["match"]
    assert match["status"] == "auto" and match["name"] == "Steven"  # stored, but the name is the user's
    assert voices.read_transcript(lib, "t4")["speakers"]["SPEAKER_00"]["match"]["status"] == "unknown"
    # running it again changes nothing
    assert voices.scan_archive(lib, db) == result
    assert lib.names("t1") == {"SPEAKER_00": "Steven"}


def test_scan_with_empty_voice_memory_says_so(settings, library, tmp_path):
    """Nobody remembered yet: the summary names the cause instead of reporting bare zero counts."""
    from types import SimpleNamespace

    make(library, "e1", {"SPEAKER_00": 30.0}, embeddings={"SPEAKER_00": random_voice()})
    db = voices.open_db(settings)
    assert db.list() == []
    result = voices.scan_archive(library, db)
    assert result["people"] == 0 and result["scanned"] == 1 and result["auto"] == 0 and result["suggest"] == 0

    messages = []
    voices.scan_task(SimpleNamespace(source="all"), settings, lambda p: messages.append(p.message), library)
    final = [m for m in messages if m][-1]
    assert "Voice memory is empty" in final and "remember a voice first" in final
    assert "0 automatic" not in final

    db.enroll("Steven", random_voice(), speech_s=60)  # with a person the normal counts come back
    messages.clear()
    voices.scan_task(SimpleNamespace(source="all"), settings, lambda p: messages.append(p.message), library)
    assert "automatic" in [m for m in messages if m][-1]


def test_scan_for_one_person_leaves_other_transcripts_alone(world):
    populate(world)
    lib, db, roman = world["library"], world["db"], world["ids"]["roman"]
    result = voices.scan_archive(lib, db, person_id=roman)
    assert result["scanned"] == 1  # only t3 has a voice like Roman's
    assert lib.names("t3") == {"SPEAKER_00": "Roman", "SPEAKER_01": "Host"}
    assert voices.read_transcript(lib, "t1")["speakers"]["SPEAKER_00"]["match"] is None  # never touched
    with pytest.raises(voices.VoiceError) as e:
        voices.scan_archive(lib, db, person_id="nope")
    assert e.value.code == "voice_person_not_found"


def test_scan_cancelled_leaves_consistent_files(world):
    populate(world)
    lib, db = world["library"], world["db"]

    def stop_after_two(n, total, tid, counts):
        if n == 2:
            raise Cancelled()

    partial = voices.scan_archive(lib, db, progress=stop_after_two)  # the summary of what was done, not an error
    assert partial["cancelled"] is True and partial["transcripts"] == 2 and partial["scanned"] == 2
    assert partial["auto"] == 1 and partial["suggest"] == 1  # t1 (Steven) and t2 (suggestion)
    assert lib.names("t1") == {"SPEAKER_00": "Steven"}  # done before the cancel
    assert voices.read_transcript(lib, "t3")["speakers"]["SPEAKER_00"]["match"] is None  # not reached, intact
    voices.scan_archive(lib, db)  # and a new scan finishes the rest
    assert lib.names("t3")["SPEAKER_00"] == "Roman"


def test_appearances_and_suggestions(world):
    populate(world)
    lib, db, sid = world["library"], world["db"], world["ids"]["steven"]
    voices.scan_archive(lib, db)
    rows = voices.appearances(lib, db, sid)
    assert [(r["transcript_id"], r["label"], r["status"]) for r in rows] == [
        ("t6", "SPEAKER_00", "manual"),  # newest first; a typed name needs no voiceprints
        ("t3", "SPEAKER_01", "suggest"),  # the automatic match is contradicted by the user's own name "Host"
        ("t2", "SPEAKER_00", "suggest"),
        ("t1", "SPEAKER_00", "auto"),
    ]
    assert voices.appearances(lib, db, sid, include_suggest=False) == [r for r in rows if r["status"] in ("manual", "auto")]
    assert rows[0]["speech_s"] == 100 and rows[3]["speech_s"] == 100 and rows[3]["title"] == "t1"
    assert voices.appearance_counts(lib, db) == {sid: 2, world["ids"]["roman"]: 1}  # suggestions do not count

    got = voices.suggestions(lib, db, sid)  # not yet confirmed: auto and suggest, best score first
    assert [(s["transcript_id"], s["label"]) for s in got] == [("t1", "SPEAKER_00"), ("t3", "SPEAKER_01"), ("t2", "SPEAKER_00")]
    assert all(s["snippet"] and s["end"] - s["start"] == 5.0 for s in got)
    assert voices.suggestions(lib, db, sid, limit=1)[0]["transcript_id"] == "t1"
    with pytest.raises(voices.VoiceError):
        voices.appearances(lib, db, "nope")


# --- bulk confirmation ----------------------------------------------------------------------


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        yield c


def test_bulk_accept_above_a_score_and_reject(world, client):
    populate(world)
    lib, sid = world["library"], world["ids"]["steven"]
    voices.scan_archive(lib, world["db"])
    r = client.post(f"/api/voices/people/{sid}/suggestions/accept", json={"min_score": 0.7}).json()
    assert r["accepted"] == 1 and r["failed"] == []  # t1 (0.9); t3 (0.8) is called "Host" by the user; t2 is 0.52
    assert lib.names("t1") == {"SPEAKER_00": "Steven"}
    assert lib.names("t3")["SPEAKER_01"] == "Host"  # a bulk "accept all" never replaces a typed name
    # choosing that very row explicitly is the user's decision and does replace it
    r = client.post(f"/api/voices/people/{sid}/suggestions/accept",
                    json={"items": [{"transcript": "t3", "label": "SPEAKER_01"}]}).json()
    assert r["accepted"] == 1
    assert lib.names("t3")["SPEAKER_01"] == "Steven"
    match = voices.read_transcript(lib, "t3")["speakers"]["SPEAKER_01"]["match"]
    assert match["confirmed"] is True
    statuses = {(a["transcript_id"], a["status"]) for a in
                client.get(f"/api/voices/people/{sid}/appearances").json()["appearances"]}
    assert statuses == {("t1", "confirmed"), ("t3", "confirmed"), ("t2", "suggest"), ("t6", "manual")}

    r = client.post(f"/api/voices/people/{sid}/suggestions/reject",
                    json={"items": [{"transcript": "t2", "label": "SPEAKER_00"}]}).json()
    assert r["rejected"] == 1
    assert client.get(f"/api/voices/people/{sid}/suggestions").json()["suggestions"] == []
    assert client.post(f"/api/voices/people/{sid}/suggestions/reject", json={}).json()["code"] == "voice_items_required"


def test_bulk_accept_never_confirms_someone_else(world, client):
    populate(world)
    lib, roman = world["library"], world["ids"]["roman"]
    voices.scan_archive(lib, world["db"])
    # t1/SPEAKER_00 belongs to Steven: confirming it "for Roman" must be refused
    r = client.post(f"/api/voices/people/{roman}/suggestions/accept",
                    json={"items": [{"transcript": "t1", "label": "SPEAKER_00"}]}).json()
    assert r["accepted"] == 0 and r["failed"][0]["code"] == "voice_changed"
    assert lib.names("t1") == {"SPEAKER_00": "Steven"}  # unchanged
    assert not voices.read_transcript(lib, "t1")["speakers"]["SPEAKER_00"]["match"].get("confirmed")
    # gone transcripts are reported, not fatal
    r = client.post(f"/api/voices/people/{roman}/suggestions/accept",
                    json={"items": [{"transcript": "zzz", "label": "SPEAKER_00"}]}).json()
    assert r["failed"][0]["code"] in {"voice_not_available", "voice_no_suggestion"}


def test_bulk_accept_stops_adding_samples_at_the_cap(world, client, monkeypatch):
    populate(world)
    lib, db, sid = world["library"], world["db"], world["ids"]["steven"]
    voices.scan_archive(lib, db)
    monkeypatch.setattr(voices, "BULK_SAMPLE_CAP", 1)  # Steven already has one sample
    r = client.post(f"/api/voices/people/{sid}/suggestions/accept", json={"min_score": 0.7}).json()
    assert r["accepted"] == 1  # t1; t3 carries the name "Host"
    steven = next(p for p in db.list() if p["id"] == sid)
    assert len(steven["samples"]) == 1  # named and confirmed only, no new samples


# --- merge ----------------------------------------------------------------------------------


def test_merge_people_moves_samples_and_names(world):
    lib, db, steven = world["library"], world["db"], world["steven"]
    # "Stevie" is the same voice enrolled under another name
    stevie = db.enroll("Stevie", near(steven, 0.9), speech_s=50)["person_id"]
    sid, roman = world["ids"]["steven"], world["ids"]["roman"]
    make(lib, "m1", {"SPEAKER_00": 100, "SPEAKER_01": 100},
         embeddings={"SPEAKER_00": near(steven, 0.9), "SPEAKER_01": random_voice()})
    voices.enroll_label(lib, "m1", "SPEAKER_00", db, name="Stevie", person_id=stevie, force=True)  # confirmed + sample
    make(lib, "m2", {"SPEAKER_00": 100, "SPEAKER_01": 100},
         embeddings={"SPEAKER_00": random_voice(), "SPEAKER_01": random_voice()})
    # a stored automatic match to Stevie with its automatic name, and a name the user typed on another label
    info = voices.read_transcript(lib, "m2")
    spk = info["speakers"]["SPEAKER_00"]
    spk["match"] = {"status": "auto", "person_id": stevie, "name": "Stevie", "score": 0.7, "margin": 0.3}
    spk["auto_name"] = "Stevie"
    spk["rejected"] = [stevie, roman]
    voices.write_transcript(lib, "m2", info)
    lib.save_names("m2", {"SPEAKER_00": "Stevie", "SPEAKER_01": "Stevie"})

    result = voices.merge_people(lib, db, sid, stevie)
    assert result["name"] == "Steven" and result["samples"] == 3 and result["transcripts"] == 2
    people = db.list()
    assert [p["name"] for p in people] == ["Roman", "Steven"]
    assert len(people[1]["samples"]) == 3
    m1 = voices.read_transcript(lib, "m1")["speakers"]["SPEAKER_00"]
    assert m1["match"]["person_id"] == sid and m1["match"]["name"] == "Steven" and m1["enrolled"]["person_id"] == sid
    assert lib.names("m1") == {"SPEAKER_00": "Steven"}
    m2 = voices.read_transcript(lib, "m2")["speakers"]["SPEAKER_00"]
    assert m2["match"]["person_id"] == sid and m2["auto_name"] == "Steven" and m2["rejected"] == [sid, roman]
    # SPEAKER_00 belonged to the merged person: renamed. SPEAKER_01 was typed by the user: stays "Stevie"
    assert lib.names("m2") == {"SPEAKER_00": "Steven", "SPEAKER_01": "Stevie"}
    # the enrolled sample of m1 moved with the person, so it is still reported as saved
    assert voices.public_state(voices.read_transcript(lib, "m1"), db)["speakers"]["SPEAKER_00"]["enrolled"]
    with pytest.raises(voices.VoiceError) as e:  # merging again is an error, not a second merge
        voices.merge_people(lib, db, sid, stevie)
    assert e.value.code == "voice_person_not_found"


def test_merge_validation_and_rename(world):
    lib, db = world["library"], world["db"]
    sid, roman = world["ids"]["steven"], world["ids"]["roman"]
    third = db.enroll("Cecil", random_voice(), speech_s=60)["person_id"]
    for args, code in [((sid, sid), "voice_merge_same"), ((sid, "zz"), "voice_person_not_found"),
                       ((sid, roman, "cecil"), "voice_name_taken")]:
        with pytest.raises(voices.VoiceError) as e:
            voices.merge_people(lib, db, *args)
        assert e.value.code == code
    assert len(db.list()) == 3  # nothing changed
    out = voices.merge_people(lib, db, sid, roman, "Steven B.")  # the merged person gets a new name
    assert out["name"] == "Steven B." and sorted(p["name"] for p in db.list()) == ["Cecil", "Steven B."]
    assert third in {p["id"] for p in db.list()}


def test_merge_api(world, client):
    sid, roman = world["ids"]["steven"], world["ids"]["roman"]
    populate(world)
    voices.scan_archive(world["library"], world["db"])
    r = client.post(f"/api/voices/people/{sid}/merge", json={"other": roman})
    assert r.status_code == 200 and r.json()["samples"] == 2
    assert world["library"].names("t3")["SPEAKER_00"] == "Steven"  # Roman's automatic name
    people = client.get("/api/voices").json()["people"]
    assert [p["name"] for p in people] == ["Steven"] and people[0]["appearances"] == 3  # t1, t3 (Roman's), t6
    assert client.post(f"/api/voices/people/{sid}/merge", json={"other": roman}).status_code == 404
    assert client.post(f"/api/voices/people/{sid}/merge", json={"other": sid}).json()["code"] == "voice_merge_same"


def test_merge_while_scanning_and_accepting_keeps_files_consistent(world):
    lib, db, steven, roman = world["library"], world["db"], world["steven"], world["roman"]
    sid, rid = world["ids"]["steven"], world["ids"]["roman"]
    for n in range(12):
        make(lib, f"c{n}", {"SPEAKER_00": 100, "SPEAKER_01": 100},
             embeddings={"SPEAKER_00": near(roman, 0.9), "SPEAKER_01": near(steven, 0.7)})
    errors = []

    def run(fn):
        def inner():
            try:
                fn()
            except Exception as e:  # pragma: no cover - reported below
                errors.append(repr(e))
        return inner

    def scan_loop():
        for _ in range(4):
            voices.scan_archive(lib, db)

    def accept_loop():
        for n in range(12):
            try:
                voices.accept(lib, f"c{n}", "SPEAKER_01", db)
            except voices.VoiceError:
                pass  # e.g. the match changed meanwhile

    threads = [threading.Thread(target=run(f)) for f in (scan_loop, scan_loop, accept_loop)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    voices.merge_people(lib, db, sid, rid)
    for t in threads:
        t.join()
    assert not errors, errors
    assert [p["name"] for p in db.list()] == ["Steven"]
    voices.scan_archive(lib, db)  # one more pass after the dust has settled
    for n in range(12):
        for info in voices.read_transcript(lib, f"c{n}")["speakers"].values():
            assert (info.get("match") or {}).get("person_id") in (None, sid)  # nobody points to the merged person
        assert all(v == "Steven" for v in lib.names(f"c{n}").values()), (n, lib.names(f"c{n}"), {k: {x: y for x, y in v.items() if x != "vector"} for k, v in voices.read_transcript(lib, f"c{n}")["speakers"].items()})
    assert not list(db.path.parent.glob("*.tmp"))
    json.loads(db.path.read_text(encoding="utf-8"))  # a valid file


# --- search by speaker ----------------------------------------------------------------------


def test_search_filters_by_speaker_and_person(world, client):
    lib = world["library"]
    make(lib, "s1", {"SPEAKER_00": 30, "SPEAKER_01": 30}, names={"SPEAKER_00": "Jana Nováková", "SPEAKER_01": "Petr"},
         texts={"SPEAKER_00": "balkon a bylinky", "SPEAKER_01": "bylinky na balkoně"})
    make(lib, "s2", {"SPEAKER_00": 30}, names={"SPEAKER_00": "Steven"}, texts={"SPEAKER_00": "bylinky všude"})
    index = SearchIndex(lib, world["settings"].data_dir / "search-test.sqlite3")
    try:
        index.sync()
        assert index.search("bylinky")["total"] == 3
        jana = index.search("bylinky", speaker="jana novakova")  # case and diacritics ignored
        assert [(h["transcript_id"], h["speaker"]) for h in jana["results"]] == [("s1", "SPEAKER_00")]
        assert index.search("bylinky", speaker="SPEAKER_01")["total"] == 1  # raw label
        assert index.search("bylinky", speaker="nobody")["total"] == 0
        # no text: everything this speaker said, in archive order
        everything = index.search("", speaker="Petr")
        assert everything["total"] == 1 and everything["results"][0]["text"].strip() == "bylinky na balkoně"
        assert index.search("")["total"] == 0 and index.search("", speaker="   ")["total"] == 0
    finally:
        index.close()
    sid = world["ids"]["steven"]
    r = client.get("/api/search", params={"q": "bylinky", "person": sid}).json()
    assert [h["transcript_id"] for h in r["results"]] == ["s2"]
    assert client.get("/api/search", params={"q": "bylinky", "person": "nope"}).status_code == 404
    assert client.get("/api/search", params={"q": "", "speaker": "petr"}).json()["total"] == 1
    assert client.get("/api/search", params={"q": "bylinky", "speaker": "Petr", "transcript": "s2"}).json()["total"] == 0


# --- the background job and the API ---------------------------------------------------------


def test_scan_job_through_the_queue(world, client):
    populate(world)
    lib = world["library"]
    r = client.post("/api/voices/scan", json={})
    assert r.status_code == 200, r.text
    job = r.json()["job"]
    assert job["task"] == "voice_scan" and job["source"] == "all"
    wait_for(lambda: client.get("/api/voices").json()["scan_job"] is None)
    done = next(j for j in client.get("/api/jobs").json() if j["id"] == job["id"])
    assert done["status"] == "done", done["error"]
    assert lib.names("t1") == {"SPEAKER_00": "Steven"} and lib.names("t3")["SPEAKER_00"] == "Roman"

    r = client.post("/api/voices/scan", json={"person_id": world["ids"]["roman"]}).json()
    assert r["job"]["source"] == f"person:{world['ids']['roman']}"
    assert client.post("/api/voices/scan", json={"person_id": "nope"}).status_code == 404


def test_scan_api_reports_empty_voice_memory(settings, client):
    r = client.post("/api/voices/scan", json={}).json()
    assert r["empty"] is True
    assert client.get("/api/voices").json()["people"] == []
    voices.open_db(settings).enroll("Steven", random_voice(), speech_s=60)
    assert client.post("/api/voices/scan", json={}).json()["empty"] is False


def test_scan_compute_missing_queues_voiceprint_jobs(world, client, monkeypatch):
    populate(world)
    lib = world["library"]
    (lib.root / "t5" / "audio.wav").write_bytes(b"RIFF")  # has audio and a speaker, but no voiceprints
    monkeypatch.setattr(client.app.state, "settings", world["settings"].with_overrides(hf_token="x"))
    added = []
    queue = client.app.state.queue
    monkeypatch.setattr(queue, "listing", lambda: [])
    monkeypatch.setattr(queue, "add", lambda req: added.append(req) or object())
    monkeypatch.setattr(queue, "public", lambda job: {"id": 1, "task": "voice_scan", "source": "all"})
    r = client.post("/api/voices/scan", json={"compute_missing": True}).json()
    assert [(a.task, a.source) for a in added] == [("voice_scan", "all"), ("voiceprints", "t5")]
    assert r["compute_queued"] == 1


def test_appearances_api(world, client):
    populate(world)
    voices.scan_archive(world["library"], world["db"])
    sid = world["ids"]["steven"]
    r = client.get(f"/api/voices/people/{sid}/appearances").json()
    assert r["name"] == "Steven" and r["transcripts"] == 4
    assert [a["transcript_id"] for a in r["appearances"]] == ["t6", "t3", "t2", "t1"]
    assert r["speech_s"] == 200  # confirmed/auto/manual only: t6 + t1
    r = client.get(f"/api/voices/people/{sid}/appearances", params={"suggestions": "false"}).json()
    assert [a["transcript_id"] for a in r["appearances"]] == ["t6", "t1"]
    assert client.get("/api/voices/people/nope/appearances").status_code == 404
    s = client.get(f"/api/voices/people/{sid}/suggestions").json()
    assert [x["transcript_id"] for x in s["suggestions"]] == ["t1", "t3", "t2"]


# --- review fixes -----------------------------------------------------------------------------


def test_interrupted_merge_can_be_finished(world, monkeypatch):
    lib, db = world["library"], world["db"]
    sid, roman = world["ids"]["steven"], world["ids"]["roman"]
    make(lib, "r1", {"SPEAKER_00": 100}, embeddings={"SPEAKER_00": near(world["roman"], 0.9)})
    voices.scan_archive(lib, db)
    assert lib.names("r1") == {"SPEAKER_00": "Roman"}
    real, calls = voices._rewire, []

    def crashing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return 0  # as if the process died before the first pass wrote anything
        raise RuntimeError("crash after the database step")

    monkeypatch.setattr(voices, "_rewire", crashing)
    with pytest.raises(RuntimeError):
        voices.merge_people(lib, db, sid, roman)
    monkeypatch.setattr(voices, "_rewire", real)
    assert [p["name"] for p in db.list()] == ["Steven"]  # the database step is done ...
    assert lib.names("r1") == {"SPEAKER_00": "Roman"}  # ... the transcript still points to the removed person
    out = voices.merge_people(lib, db, sid, roman)  # the same call finishes it
    assert out["resumed"] and out["transcripts"] == 1
    assert lib.names("r1") == {"SPEAKER_00": "Steven"}
    assert voices.read_transcript(lib, "r1")["speakers"]["SPEAKER_00"]["match"]["person_id"] == sid
    with pytest.raises(voices.VoiceError) as e:  # and a finished merge stays a 404
        voices.merge_people(lib, db, sid, roman)
    assert e.value.code == "voice_person_not_found"


def test_delete_everything_cancels_scan_jobs(world, client, monkeypatch):
    queue = client.app.state.queue
    cancelled = []
    monkeypatch.setattr(queue, "listing", lambda: [
        {"id": 1, "task": "voice_scan", "status": "queued", "source": "all"},
        {"id": 2, "task": "voiceprints", "status": "running", "source": "t1"},
        {"id": 3, "task": "transcribe", "status": "running", "source": "x"}])
    monkeypatch.setattr(queue, "get", lambda i: i)
    monkeypatch.setattr(queue, "cancel", lambda job: cancelled.append(job))
    assert client.delete("/api/voices").json()["jobs_cancelled"] == 2
    assert cancelled == [1, 2]


def test_archive_endpoints_respect_voice_memory_off(world, client, monkeypatch):
    sid = world["ids"]["steven"]
    monkeypatch.setenv("MGTRANSCRIBER_VOICE_MEMORY", "false")
    for path in (f"/api/voices/people/{sid}/appearances", f"/api/voices/people/{sid}/suggestions"):
        r = client.get(path)
        assert r.status_code == 409 and r.json()["code"] == "voice_disabled"
    r = client.get("/api/search", params={"q": "x", "person": sid})
    assert r.status_code == 409 and r.json()["code"] == "voice_disabled"
    assert client.post(f"/api/voices/people/{sid}/suggestions/accept", json={}).status_code == 409
    assert client.get("/api/search", params={"q": "x", "speaker": "Steven"}).status_code == 200  # a plain filter


def test_naming_a_speaker_waits_for_a_running_identification(world, client):
    lib = world["library"]
    make(lib, "n1", {"SPEAKER_00": 30})
    done = []

    def put():
        client.put("/api/transcripts/n1/speakers", json={"SPEAKER_00": "Typed now"})
        done.append(1)

    with voices.transcript_lock(lib, "n1"):  # what identify_transcript holds while it works
        t = threading.Thread(target=put)
        t.start()
        time.sleep(0.4)
        assert not done and lib.names("n1") == {}
    t.join(10)
    assert done and lib.names("n1") == {"SPEAKER_00": "Typed now"}
