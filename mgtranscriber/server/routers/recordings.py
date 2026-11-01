"""Recording sessions: chunks are uploaded while the browser records, ``finish`` queues a normal job.

See ``server/recordings.py`` for the storage and ``docs/API.md`` for the contract."""

import uuid

from fastapi import APIRouter, Path, Query, Request
from pydantic import BaseModel, Field

from ...pipeline import TranscriptionRequest
from .. import recordings, uploads
from ..deps import FoldersDep, QueueDep, SettingsDep
from ..errors import ApiError
from .jobs import check_engine_choice, check_separation, resolve_folder, resolve_translate_to, with_translation_warning
from .analysis import LLMDep

router = APIRouter(prefix="/api/recordings", tags=["recordings"])

SessionId = Path(pattern=r"^[0-9a-f]{32}$", description="recording session id")


class NewRecording(BaseModel):
    mime: str = Field(..., max_length=200, description="MediaRecorder's mime type, e.g. audio/webm;codecs=opus")
    source: str = Field("microphone", description="microphone | tab | mix")
    slice_ms: int = Field(5000, ge=500, le=60000, description="timeslice the client records with")
    label: str | None = Field(None, max_length=120)


class FinishRecording(BaseModel):
    chunks: int | None = Field(None, ge=1, le=recordings.MAX_CHUNK_INDEX + 1,
                               description="number of chunks the client sent; omitted: up to the highest received")
    allow_gaps: bool = Field(False, description="join what exists even when chunks in the middle are missing")
    title: str | None = Field(None, max_length=120)
    language: str | None = Field(None, pattern=r"^(auto|[a-z]{2,3})$")
    speakers: int | None = Field(None, ge=1, le=20)
    no_diarize: bool = False
    separate_speakers: bool = False
    engine: str | None = Field(None, max_length=40)
    model: str | None = Field(None, max_length=100)
    folder: str | None = Field(None, max_length=1000)
    translate_to: str | None = Field(None, max_length=12)


@router.get("")
def list_recordings(settings: SettingsDep):
    """Unfinished sessions, newest first (the UI offers to recover them)."""
    return recordings.list_sessions(settings)


@router.post("", status_code=201)
def start_recording(body: NewRecording, settings: SettingsDep):
    return recordings.start(settings, body.mime, body.source, body.slice_ms, body.label)


@router.get("/{session_id}")
def recording_detail(settings: SettingsDep, session_id: str = SessionId):
    return recordings.public(recordings.session_dir(settings, session_id))


@router.put("/{session_id}/chunks/{n}")
async def put_chunk(request: Request, settings: SettingsDep, session_id: str = SessionId,
                    n: int = Path(ge=0, le=recordings.MAX_CHUNK_INDEX),
                    t: int | None = Query(None, ge=0, description="recorded milliseconds at the end of this chunk")):
    """Store chunk ``n`` (raw body). Safe to repeat; the body limit (``max_upload_mb``) applies to each request."""
    folder = recordings.session_dir(settings, session_id)
    tmp = folder / f".chunk-{uuid.uuid4().hex}.tmp"
    limit = recordings.MAX_CHUNK_MB * 1024 * 1024
    received = 0
    try:
        with open(tmp, "wb") as out:
            async for part in request.stream():
                received += len(part)
                if received > limit:
                    raise ApiError(413, "request_too_large", f"A chunk is larger than {recordings.MAX_CHUNK_MB} MB.")
                out.write(part)
        return recordings.store_chunk(settings, folder, n, tmp, t)
    finally:
        tmp.unlink(missing_ok=True)


@router.post("/{session_id}/finish", status_code=201)
def finish_recording(body: FinishRecording, queue: QueueDep, settings: SettingsDep, folders: FoldersDep,
                     llm: LLMDep, session_id: str = SessionId):
    """Join the chunks and queue a transcription job like ``POST /api/jobs/upload`` does."""
    recordings.session_dir(settings, session_id)
    check_engine_choice(settings, body.engine or None)
    check_separation(settings, body.engine or None, body.separate_speakers)
    target = resolve_folder(folders, body.folder or None)
    translate = resolve_translate_to(body.translate_to)
    with recordings.finishing(session_id):
        upload, original = recordings.assemble(settings, session_id, body.chunks, body.allow_gaps, body.title)
        try:
            job = queue.add(TranscriptionRequest(source=str(upload), language=body.language or None,
                                                 speakers=body.speakers, diarize=False if body.no_diarize else None,
                                                 engine=body.engine or None, model=body.model or None, folder=target,
                                                 separate_speakers=body.separate_speakers,
                                                 translate_to=translate))
        except BaseException:
            uploads.discard(settings, str(upload))
            raise
        job.title = original
        recordings.discard(settings, session_id)
    return with_translation_warning(queue.public(job), llm, translate)


@router.delete("/{session_id}")
def discard_recording(settings: SettingsDep, session_id: str = SessionId):
    recordings.discard(settings, session_id)
    return {"removed": session_id}
