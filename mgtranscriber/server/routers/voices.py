"""Voice memory: the people whose voices are remembered, and voice actions on a transcript.

All data stays on this computer (voice database in ``data/``, ``voices.json`` per transcript).
See ``mgtranscriber/voices.py`` and ``docs/VOICES.md``.
"""

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ... import voices
from ...pipeline import TranscriptionRequest
from ..deps import LibraryDep, QueueDep, SettingsDep
from ..errors import ApiError
from .setup import client_is_local
from .transcripts import attachment

router = APIRouter(prefix="/api/voices", tags=["voices"])

LABEL = r"^[\w.-]{1,40}$"


class Rename(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class LabelAction(BaseModel):
    label: str = Field(pattern=LABEL)


class Item(LabelAction):
    transcript: str = Field(min_length=1, max_length=200)


class Decide(BaseModel):
    """Bulk confirm/reject for one person. ``items``: chosen (transcript, label) pairs; without them
    ``accept`` takes every unconfirmed match to the person scoring at least ``min_score``."""
    items: list[Item] | None = Field(None, max_length=2000)
    min_score: float | None = Field(None, ge=0, le=1)


class Merge(BaseModel):
    other: str = Field(min_length=1, max_length=40)  # the person merged into this one (and removed)
    name: str | None = Field(None, max_length=60)  # name of the merged person; default: this person's


class Scan(BaseModel):
    person_id: str | None = Field(None, max_length=40)  # only this person; default: everyone
    compute_missing: bool = False  # also queue voiceprint jobs for transcripts that have none


class Enroll(LabelAction):
    name: str = Field("", max_length=60)
    person_id: str | None = Field(None, max_length=40)
    force: bool = False  # add the sample even though it does not resemble the person's voice


def _db(settings) -> voices.VoiceDB:
    return voices.open_db(settings)


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except voices.VoiceError as e:
        raise ApiError(e.status, e.code, e.message)
    except voices.LockTimeout:
        raise ApiError(503, "voice_busy", "The voice data is locked by another process.")


@router.get("")
def list_people(request: Request, settings: SettingsDep, library: LibraryDep, queue: QueueDep):
    """People with their samples (no vectors), where the database lives and the thresholds."""
    db = _db(settings)
    return {
        "enabled": voices.enabled(settings),
        "scan_job": _pending(queue, voices.SCAN_TASK),
        "file": str(db.path) if client_is_local(request) else db.path.name,  # the page shows it to this computer
        "model": db.model,
        "thresholds": {"auto": voices.AUTO_THRESHOLD, "suggest": voices.SUGGEST_THRESHOLD, "margin": voices.MARGIN,
                       "min_identify_s": voices.MIN_IDENTIFY_S, "min_enroll_s": voices.MIN_ENROLL_S},
        "people": _with_counts(library, db),
    }


def _with_counts(library, db) -> list[dict]:
    people = _run(db.list)
    counts = _run(voices.appearance_counts, library, db)
    return [{**p, "appearances": counts.get(p["id"], 0)} for p in people]


def _pending(queue, task: str, source: str | None = None) -> dict | None:
    return next((j for j in queue.listing() if j.get("task") == task and j["status"] in ("queued", "running")
                 and (source is None or j["source"] == source)), None)


@router.get("/export")
def export(settings: SettingsDep):
    """The whole voice database as a JSON download (contains biometric voiceprints)."""
    return JSONResponse(_run(_db(settings).export), headers=attachment("voiceprints.json"))


@router.post("/scan")
def scan(body: Scan, settings: SettingsDep, library: LibraryDep, queue: QueueDep):
    """Recognise voices across the whole archive: a background job (task ``voice_scan``) that identifies
    every transcript with voiceprints again (CPU only) and stores matches as after a transcription.
    ``person_id`` limits it to one person. ``compute_missing`` also queues ``voiceprints`` jobs (GPU)
    for transcripts without voiceprints, after the scan. An identical scan already waiting is returned."""
    if not voices.enabled(settings):
        raise ApiError(409, "voice_disabled", "Voice memory is turned off.")
    if body.person_id and not any(p["id"] == body.person_id for p in _run(_db(settings).list)):
        raise ApiError(404, "voice_person_not_found", "Person not found.")
    source = f"person:{body.person_id}" if body.person_id else "all"
    job = _pending(queue, voices.SCAN_TASK, source)
    if job is None:
        job = queue.public(queue.add(TranscriptionRequest(source, task=voices.SCAN_TASK)))
    queued = skipped = 0
    if body.compute_missing:
        for transcript_id in library.ids():
            try:
                if (voices.read_transcript(library, transcript_id).get("speakers")
                        or not settings.hf_token or library.audio_file(transcript_id) is None
                        or not any(k != "UNKNOWN" for k in library.summary(transcript_id)["speakers"])):
                    continue
            except (OSError, ValueError, KeyError, TypeError, LookupError):
                continue
            if _pending(queue, "voiceprints", transcript_id):
                skipped += 1
                continue
            queue.add(TranscriptionRequest(transcript_id, task="voiceprints"))
            queued += 1
    empty = not _run(_db(settings).list)  # nothing to recognise yet: the UI explains the workflow
    return {"job": job, "compute_queued": queued, "compute_pending": skipped, "empty": empty}


@router.delete("")
def delete_everything(settings: SettingsDep, library: LibraryDep, queue: QueueDep):
    """Delete the voice database and the voiceprints stored with every transcript. Queued or running
    voiceprint and archive-scan jobs are cancelled first, so they cannot write voiceprints again. Voice memory stays
    enabled (setting ``voice_memory``): new transcriptions store voiceprints again."""
    cancelled = 0
    for item in queue.listing():
        if item.get("task") in ("voiceprints", voices.SCAN_TASK) and item["status"] in ("queued", "running"):
            job = queue.get(item["id"])
            if job:
                queue.cancel(job)
                cancelled += 1
    removed = _run(voices.delete_all, library, _db(settings))
    return {"deleted": True, "transcripts": removed, "jobs_cancelled": cancelled}


def _need_person(db, person_id: str, settings=None) -> dict:
    if settings is not None and not voices.enabled(settings):
        raise ApiError(409, "voice_disabled", "Voice memory is turned off.")
    person = next((p for p in _run(db.list) if p["id"] == person_id), None)
    if person is None:
        raise ApiError(404, "voice_person_not_found", "Person not found.")
    return person


@router.get("/people/{person_id}/appearances")
def person_appearances(person_id: str, settings: SettingsDep, library: LibraryDep, suggestions: bool = True):
    """Every transcript where this person speaks (newest first): label, seconds of speech, status
    (``confirmed`` | ``auto`` | ``suggest`` | ``manual``), score, date. ``suggestions=false`` leaves
    out unconfirmed suggestions."""
    db = _db(settings)
    person = _need_person(db, person_id, settings)
    items = _run(voices.appearances, library, db, person_id, include_suggest=suggestions)
    return {"person_id": person_id, "name": person["name"], "transcripts": len({i["transcript_id"] for i in items}),
            "speech_s": round(sum(i["speech_s"] or 0 for i in items if i["status"] != "suggest"), 1),
            "appearances": items}


@router.get("/people/{person_id}/suggestions")
def person_suggestions(person_id: str, settings: SettingsDep, library: LibraryDep, queue: QueueDep,
                       limit: int = Query(500, ge=1, le=5000)):
    """Matches to this person across the archive that nobody has confirmed yet, best score first, each
    with a 5 s excerpt (``start``/``end`` in the transcript's audio) to listen to."""
    db = _db(settings)
    person = _need_person(db, person_id, settings)
    return {"person_id": person_id, "name": person["name"], "thresholds": {"auto": voices.AUTO_THRESHOLD},
            "scan_job": _pending(queue, voices.SCAN_TASK),
            "suggestions": _run(voices.suggestions, library, db, person_id, limit=limit)}


def _decide(action: str, person_id: str, body: Decide, settings, library) -> dict:
    db = _db(settings)
    person = _need_person(db, person_id, settings)
    if body.items is not None:
        chosen = [(i.transcript, i.label) for i in body.items]
    else:
        if action == "reject":
            raise ApiError(400, "voice_items_required", "Choose which suggestions to reject.")
        floor = body.min_score if body.min_score is not None else voices.AUTO_THRESHOLD
        # never over a name somebody typed: only an explicit choice of items may replace it
        chosen = [(r["transcript_id"], r["label"]) for r in _run(voices.suggestions, library, db, person_id)
                  if (r["score"] or 0) >= floor and (not r["name"] or r["name"] == person["name"])]
    done, failed = 0, []
    for transcript_id, label in chosen:
        try:
            if action == "accept":
                enough = db.index().sample_count(person_id) < voices.BULK_SAMPLE_CAP
                voices.accept(library, transcript_id, label, db, expect_person=person_id, add_sample=enough)
            else:
                voices.reject(library, transcript_id, label, db, expect_person=person_id)
            done += 1
        except voices.VoiceError as e:
            failed.append({"transcript": transcript_id, "label": label, "code": e.code})
        except voices.LockTimeout:
            failed.append({"transcript": transcript_id, "label": label, "code": "voice_busy"})
        except (LookupError, ValueError, OSError):  # a transcript deleted meanwhile, or a damaged one
            failed.append({"transcript": transcript_id, "label": label, "code": "voice_not_available"})
    return {"person_id": person_id, "name": person["name"], "accepted" if action == "accept" else "rejected": done,
            "failed": failed}


@router.post("/people/{person_id}/suggestions/accept")
def accept_suggestions(person_id: str, body: Decide, settings: SettingsDep, library: LibraryDep):
    """Confirm matches to this person: the chosen ``items``, or all with a score of at least ``min_score``
    (default: the automatic threshold). Adds the voice as a sample only while the person has fewer
    than ``BULK_SAMPLE_CAP`` samples. A match that changed since the list was shown is skipped."""
    return _decide("accept", person_id, body, settings, library)


@router.post("/people/{person_id}/suggestions/reject")
def reject_suggestions(person_id: str, body: Decide, settings: SettingsDep, library: LibraryDep):
    """The chosen matches are not this person (and are not suggested for those labels again)."""
    return _decide("reject", person_id, body, settings, library)


@router.post("/people/{person_id}/merge")
def merge(person_id: str, body: Merge, settings: SettingsDep, library: LibraryDep):
    """Merge person ``other`` into this one: samples, matches, and names that came from the person
    in transcripts. ``other`` is removed. Not undoable."""
    return _run(voices.merge_people, library, _db(settings), person_id, body.other, body.name)


@router.patch("/people/{person_id}")
def rename(person_id: str, body: Rename, settings: SettingsDep):
    _run(_db(settings).rename, person_id, body.name)
    return {"ok": True}


@router.delete("/people/{person_id}")
def delete_person(person_id: str, settings: SettingsDep):
    _run(_db(settings).delete, person_id)
    return {"ok": True}


@router.delete("/people/{person_id}/samples/{sample_id}")
def delete_sample(person_id: str, sample_id: str, settings: SettingsDep):
    _run(_db(settings).delete, person_id, sample_id)
    return {"ok": True}


# --- a transcript's voices ------------------------------------------------------------------


def _state(library, transcript_id: str, settings, queue) -> dict:
    library.path(transcript_id)  # 404 for unknown transcripts
    state = _run(voices.identify_transcript, library, transcript_id, _db(settings), apply_names=False, persist=False)
    pending = next((j for j in queue.listing() if j.get("task") == "voiceprints" and j["source"] == transcript_id
                    and j["status"] in ("queued", "running")), None)
    has_speakers = any(k != "UNKNOWN" for k in library.summary(transcript_id)["speakers"])
    return {**state, "enabled": voices.enabled(settings), "job": pending, "names": library.names(transcript_id),
            "can_compute": has_speakers and bool(settings.hf_token)}


@router.get("/transcripts/{transcript_id}")
def transcript_voices(transcript_id: str, library: LibraryDep, settings: SettingsDep, queue: QueueDep):
    """Per speaker label: seconds of speech and the current best match (live against the database)."""
    return _state(library, transcript_id, settings, queue)


@router.post("/transcripts/{transcript_id}/enroll")
def enroll(transcript_id: str, body: Enroll, library: LibraryDep, settings: SettingsDep, queue: QueueDep):
    """Remember this label's voice as ``name`` (or as the person ``person_id``); the label gets the name."""
    result = _run(voices.enroll_label, library, transcript_id, body.label, _db(settings), name=body.name,
                  person_id=body.person_id, force=body.force)
    return {**result, "state": _state(library, transcript_id, settings, queue), "names": library.names(transcript_id)}


@router.post("/transcripts/{transcript_id}/accept")
def accept(transcript_id: str, body: LabelAction, library: LibraryDep, settings: SettingsDep, queue: QueueDep):
    """Confirm the suggested person: the label gets the name and the voice becomes another sample."""
    _run(voices.accept, library, transcript_id, body.label, _db(settings))
    return {"state": _state(library, transcript_id, settings, queue), "names": library.names(transcript_id)}


@router.post("/transcripts/{transcript_id}/reject")
def reject(transcript_id: str, body: LabelAction, library: LibraryDep, settings: SettingsDep, queue: QueueDep):
    """The suggested person is wrong for this label; it is not suggested for it again."""
    _run(voices.reject, library, transcript_id, body.label, _db(settings))
    return {"state": _state(library, transcript_id, settings, queue), "names": library.names(transcript_id)}


@router.post("/transcripts/{transcript_id}/identify")
def identify(transcript_id: str, library: LibraryDep, settings: SettingsDep, queue: QueueDep):
    """Identify again against the current database and name labels recognised automatically."""
    _run(voices.identify_transcript, library, transcript_id, _db(settings))
    return {"state": _state(library, transcript_id, settings, queue), "names": library.names(transcript_id)}


@router.post("/transcripts/{transcript_id}/compute")
def compute(transcript_id: str, library: LibraryDep, settings: SettingsDep, queue: QueueDep):
    """Queue a job that computes voiceprints for an older transcript (needs its audio and the GPU)."""
    state = _state(library, transcript_id, settings, queue)
    if state["job"]:
        return state["job"]
    if not state["can_compute"]:
        raise ApiError(409, "voice_cannot_compute", "This transcript has no speakers or HF_TOKEN is missing.")
    if library.audio_file(transcript_id) is None:
        raise ApiError(409, "voice_no_audio", "The transcript has no audio file.")
    job = queue.add(TranscriptionRequest(transcript_id, task="voiceprints"))
    return queue.public(job)
