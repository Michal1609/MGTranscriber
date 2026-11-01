"""Transcription jobs: create, list, cancel and a live SSE stream of state and log."""

import asyncio
import dataclasses
import json
from pathlib import Path

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ... import netguard, privacy, separated
from ...engines import list_engines
from ...jobs import Job, JobQueue
from ...jobs.models import FINISHED
from ...jobs.queue import RETRYABLE
from ...pathguard import UNC_MESSAGE, unc_blocked
from ...pipeline import TranscriptionRequest
from ...sources import is_url
from .. import uploads
from ..deps import FoldersDep, QueueDep, SettingsDep
from .analysis import LLMDep
from ...folders import FolderError
from ...llm.base import LLMError
from ...llm.translate import LANGUAGES
from ...llm.translations import normalize_lang
from ..errors import ApiError

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class NewJob(BaseModel):
    source: str | None = Field(None, min_length=1, max_length=2000)  # URL or local path
    url: str | None = Field(None, min_length=1, max_length=2000)  # alias of ``source`` used by the original UI
    language: str | None = Field(None, pattern=r"^(auto|[a-z]{2,3})$")
    speakers: int | None = Field(None, ge=1, le=20)
    no_diarize: bool = False
    separate_speakers: bool = Field(False, description="one track per voice, each transcribed alone (slower; local engines)")
    engine: str | None = Field(None, max_length=40)
    model: str | None = Field(None, max_length=100)
    folder: str | None = Field(None, max_length=1000)  # library folder; created when new; "" = unfiled
    translate_to: str | None = Field(None, max_length=12, description="translate the finished transcript into this language")

    @model_validator(mode="after")
    def need_source(self):
        if not (self.source or self.url):
            raise ValueError("source is required")
        return self


def get_job(queue: JobQueue, job_id: int) -> Job:
    job = queue.get(job_id)
    if not job:
        raise ApiError(404, "job_not_found", "Job not found.")
    return job


@router.get("")
def list_jobs(queue: QueueDep, status: str | None = None, limit: int | None = Query(None, ge=1, le=200),
              offset: int = Query(0, ge=0)):
    """Without ``limit``: every job as a list (newest first; ``status`` still filters). With ``limit``: a page
    ``{items, total, limit, offset, counts}``. ``status``: active | done | failed | finished | all, or raw statuses."""
    try:
        if limit is None and offset == 0:
            return queue.page(status)["items"]
        return queue.page(status, limit, offset)
    except ValueError as e:
        raise ApiError(400, "invalid_status", str(e), param="status") from None


@router.get("/summary")
def jobs_summary(queue: QueueDep):
    """The running job, the queue length and the recent failures (the home page's strip and the nav badge)."""
    return queue.summary()


@router.post("/clear")
def clear_jobs(queue: QueueDep, status: str = "finished"):
    """Remove finished jobs from the history (``status``: finished | done | failed). Transcripts are untouched."""
    try:
        wanted = queue.statuses(status)
    except ValueError as e:
        raise ApiError(400, "invalid_status", str(e), param="status") from None
    if wanted is None or not wanted <= FINISHED:
        raise ApiError(400, "invalid_status", "Only finished jobs can be cleared.", param="status")
    return {"removed": queue.clear(status)}


@router.post("", status_code=201)
def create_job(body: NewJob, queue: QueueDep, settings: SettingsDep, folders: FoldersDep, llm: LLMDep):
    source = (body.source or body.url).strip()
    if source.startswith("-") or uploads.upload_of(settings, source):  # uploads have their own endpoint
        raise ApiError(400, "invalid_source", "Invalid source.")
    check_source(settings, source)
    check_engine_choice(settings, body.engine)
    check_separation(settings, body.engine, body.separate_speakers)
    folder = resolve_folder(folders, body.folder)
    translate_to = resolve_translate_to(body.translate_to)
    request = TranscriptionRequest(source=source, language=body.language, speakers=body.speakers,
                                   diarize=False if body.no_diarize else None, engine=body.engine, model=body.model,
                                   folder=folder, separate_speakers=body.separate_speakers,
                                   translate_to=translate_to)
    return with_translation_warning(queue.public(queue.add(request)), llm, translate_to)


def check_source(settings, source: str) -> None:
    """Refuse a source the server must not read (local paths off, private address, offline mode)."""
    if not is_url(source) and not settings.local_paths_allowed:
        raise ApiError(403, "local_paths_disabled", "Transcribing files on the server's disk is disabled.")
    if not is_url(source) and unc_blocked(settings, source):  # before any file system call
        raise ApiError(403, "unc_path_disabled", UNC_MESSAGE)
    if is_url(source) and netguard.guard_needed(settings):
        try:
            netguard.require_public_url(source)
        except netguard.BlockedAddress as e:
            raise ApiError(403, "blocked_address", str(e)) from None
        except OSError:
            raise ApiError(400, "unresolvable_host", "Cannot resolve the host of that URL.") from None
    try:  # offline mode: no URLs (the worker checks again)
        privacy.check_source(settings, source)
    except privacy.PrivacyError as e:
        raise ApiError(e.status, e.code, e.message) from None


def resolve_folder(folders, folder: str | None) -> str | None:
    """Normalised folder of a request, registered in the folder list; None = keep the transcript's own."""
    try:
        return folders.ensure(folder)
    except FolderError as e:
        raise ApiError(e.status, e.code, e.message, param="folder") from None


def resolve_translate_to(value: str | None) -> str | None:
    """The language of "Translate to" as exactly a LANGUAGES key (``pt-BR`` -> ``pt``); empty = no translation.
    Refused at submit time (400 ``invalid_language``), not silently when the transcription ends."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        code = normalize_lang(value)
    except LLMError as e:
        raise ApiError(400, e.code, e.message, param="translate_to") from None
    code = code if code in LANGUAGES else code.split("-")[0]
    if code not in LANGUAGES:
        raise ApiError(400, "invalid_language", f"Translation into '{value}' is not supported.", param="translate_to")
    return code


def with_translation_warning(job: dict, llm, translate_to: str | None) -> dict:
    """The job JSON of a new job; when "Translate to" is set and no translation could be queued right now (no provider,
    offline mode ...) it carries ``translation_warning`` with the reason. The transcription runs regardless."""
    if translate_to:
        try:
            llm.provider().check()
        except LLMError as e:
            return {**job, "translation_warning": e.message}
    return job


def check_engine_choice(settings, engine: str | None) -> None:
    if engine and engine not in {e.name for e in list_engines() if e.available}:
        raise ApiError(400, "unknown_engine", f"Engine '{engine}' is not available.")
    try:  # offline mode: no cloud engines (the worker checks again)
        privacy.check_engine(settings, engine)
    except privacy.PrivacyError as e:
        raise ApiError(e.status, e.code, e.message) from None


def check_separation(settings, engine: str | None, separate: bool) -> None:
    """``separate_speakers`` needs a local engine and the separation models (400 ``separation_unavailable``)."""
    if separate and (reason := separated.unavailable_reason(settings, engine or settings.engine)):
        raise ApiError(400, "separation_unavailable", reason, param="separate_speakers")


@router.post("/upload", status_code=201)
async def upload_job(
    queue: QueueDep,
    settings: SettingsDep,
    folders: FoldersDep,
    llm: LLMDep,
    file: UploadFile = File(..., description="audio or video file"),
    language: str | None = Form(None, pattern=r"^(auto|[a-z]{2,3})$"),
    speakers: int | None = Form(None, ge=1, le=20),
    no_diarize: bool = Form(False),
    separate_speakers: bool = Form(False, description="one track per voice, each transcribed alone (slower)"),
    engine: str | None = Form(None, max_length=40),
    model: str | None = Form(None, max_length=100),
    folder: str | None = Form(None, max_length=1000, description="library folder; created when new"),
    translate_to: str | None = Form(None, max_length=12, description="translate the finished transcript into this language"),
):
    """Transcribe an uploaded file. A normal queued job; the file is kept as the transcript's audio.

    Not a local path and not a URL, so it works with ``allow_local_paths`` off and in offline mode.
    The body limit (``max_upload_mb``, 413) is applied by the middleware while the body arrives."""
    check_engine_choice(settings, engine or None)
    check_separation(settings, engine or None, separate_speakers)
    target = resolve_folder(folders, folder or None)  # an empty form field = no folder chosen
    translate = resolve_translate_to(translate_to)
    uploads.check_media_type(file)
    upload, original = await uploads.save_upload(file, settings)
    try:
        job = queue.add(TranscriptionRequest(source=str(upload), language=language or None, speakers=speakers,
                                             diarize=False if no_diarize else None, engine=engine or None,
                                             model=model or None, folder=target,
                                             separate_speakers=separate_speakers, translate_to=translate))
    except BaseException:
        uploads.discard(settings, str(upload))
        raise
    job.title = original  # until the pipeline names it; the stored path is internal
    return with_translation_warning(queue.public(job), llm, translate)


@router.get("/{job_id}")
def job_detail(job_id: int, queue: QueueDep):
    job = get_job(queue, job_id)
    return {**queue.public(job), "log": job.log_since(0)}


@router.delete("/{job_id}")
def cancel_job(job_id: int, queue: QueueDep, settings: SettingsDep):
    job = get_job(queue, job_id)
    queue.cancel(job)
    if job.status == "cancelled":  # a queued upload never reaches the end-of-job hook
        uploads.discard(settings, job.request.source)
    return queue.public(job)


@router.post("/{job_id}/retry", status_code=201)
def retry_job(job_id: int, queue: QueueDep, settings: SettingsDep):
    """Queue a failed, cancelled or interrupted job again with the same request; the old entry is replaced."""
    job = get_job(queue, job_id)
    request = job.request
    if job.status not in RETRYABLE:
        raise ApiError(409, "job_not_retryable", "Only a failed, cancelled or interrupted job can be retried.")
    if request.task == "transcribe":
        local = not is_url(request.source)
        if job.ephemeral or (local and not Path(request.source).is_file()):  # an upload is deleted when its job ends
            raise ApiError(409, "source_gone", "The source file is no longer available; start it again.")
        check_source(settings, request.source)
        check_engine_choice(settings, request.engine)
        check_separation(settings, request.engine, request.separate_speakers)
    new = queue.add(dataclasses.replace(request))
    queue.remove(job)
    return queue.public(new)


@router.delete("/{job_id}/history")
def remove_job(job_id: int, queue: QueueDep):
    """Remove a finished job from the history; its transcript is not touched."""
    job = get_job(queue, job_id)
    if not queue.remove(job):
        raise ApiError(409, "job_active", "A queued or running job cannot be removed; cancel it first.")
    return {"removed": job_id}


@router.get("/{job_id}/events")
async def job_events(job_id: int, request: Request, queue: QueueDep, since: int = 0):
    """SSE: ``log`` (one log line), ``state`` (job JSON when it changes), ``end`` once the job finished."""
    job = get_job(queue, job_id)

    async def stream():
        sent, last_state = max(0, since), None
        while True:
            lines = job.log_since(sent)
            for line in lines:
                yield f"event: log\ndata: {json.dumps(line, ensure_ascii=False)}\n\n"
            sent += len(lines)
            state = json.dumps(queue.public(job), ensure_ascii=False)
            if state != last_state:
                yield f"event: state\ndata: {state}\n\n"
                last_state = state
            if job.status in FINISHED and sent >= job.log_lines:
                yield "event: end\ndata: {}\n\n"
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
