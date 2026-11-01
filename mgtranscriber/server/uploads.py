"""Uploaded media: shared by ``POST /v1/audio/transcriptions`` and ``POST /api/jobs/upload``.

An upload is copied (streamed, never held in memory) to ``data/uploads/<random>/<name>``. The
body size limit (``max_upload_mb``) is enforced by ``limits.BodyLimitMiddleware`` before an endpoint
runs. A finished transcription keeps the file as ``audio.*`` in its folder (the web player) and
records the original file name as its source (``keep_upload_as_audio``).

A web upload is a normal, persistent job: it waits in the queue and survives a server restart, so
its file stays where it is until the job ends (``finish_job``) and leftovers are swept at start
(``sweep``).
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from ..config import Settings
from ..jobs import Job
from ..library import AUDIO_TYPES, Library
from .errors import ApiError

log = logging.getLogger(__name__)

# what the web UI accepts (the extension is checked, the content is left to ffmpeg)
MEDIA_EXTENSIONS = (
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aif", ".aiff", ".amr", ".weba",
    ".webm", ".mp4", ".m4v", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp", ".ts", ".mts", ".m2ts",
)
# browsers report audio/* and video/*; some leave the type empty or say "octet-stream" for formats they do not know
GENERIC_TYPES = {"", "application/octet-stream", "binary/octet-stream", "application/ogg", "application/mp4",
                 "application/x-matroska", "application/vnd.apple.mpegurl"}


def invalid(message: str, param: str | None = None, code: str = "invalid_request", status: int = 400) -> ApiError:
    return ApiError(status, code, message, param=param)


def upload_name(filename: str | None) -> str:
    name = Path(filename or "").name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" .")
    return name[:120] or "audio"


def check_media_type(file: UploadFile) -> None:
    """415 unless the name has a media extension and the declared type (when there is one) is audio or video."""
    suffix = Path(file.filename or "").suffix.lower()
    kind = (file.content_type or "").split(";")[0].strip().lower()
    if suffix not in MEDIA_EXTENSIONS or not (kind.startswith(("audio/", "video/")) or kind in GENERIC_TYPES):
        raise invalid("Only audio and video files can be transcribed.", "file", "unsupported_media_type", 415)


async def save_upload(file: UploadFile, settings: Settings) -> tuple[Path, str]:
    """Copy the upload to ``data/uploads/<random>/`` and return (path, original file name).

    On any failure the folder is removed again; on success the caller owns it (``discard``)."""
    original = upload_name(file.filename)
    upload_dir = settings.data_dir / "uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True)
    upload = upload_dir / original
    try:
        with open(upload, "wb") as out:
            await asyncio.to_thread(shutil.copyfileobj, file.file, out, 1 << 20)
        if upload.stat().st_size == 0:
            raise invalid("The uploaded file is empty.", "file", "empty_file")
    except BaseException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    return upload, original


def adopt_file(path: Path, name: str, settings: Settings) -> tuple[Path, str]:
    """Move a file the server made itself (an assembled recording) into ``data/uploads/<random>/<name>``, where
    it is handled like an upload (``finish_job`` keeps it as audio and removes the folder, ``sweep`` clears leftovers)."""
    original = upload_name(name)
    upload_dir = settings.data_dir / "uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True)
    upload = upload_dir / original
    try:
        shutil.move(path, upload)
    except BaseException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    return upload, original


def keep_upload_as_audio(library: Library, transcript_id: str, upload: Path, original_name: str,
                         via: str = "openai-api") -> None:
    """Move the uploaded file into the transcript folder (web player) and record its name as the source."""
    folder = library.path(transcript_id)
    suffix = upload.suffix.lower()
    if suffix in AUDIO_TYPES and not any(folder.glob("audio.*")):
        shutil.move(upload, folder / f"audio{suffix}")
    meta = library.metadata(transcript_id)
    meta.source = original_name
    meta.extra["uploaded_via"] = via
    library.save_metadata(transcript_id, meta)


def upload_of(settings: Settings, source: str) -> Path | None:
    """The file of a job whose source is one of our uploads (``data/uploads/<random>/<name>``), else None."""
    path = Path(source)
    if path.parent.parent == settings.data_dir / "uploads":
        return path
    return None


def discard(settings: Settings, source: str) -> None:
    upload = upload_of(settings, source)
    if upload:
        shutil.rmtree(upload.parent, ignore_errors=True)


def finish_job(settings: Settings, library: Library, job: Job) -> None:
    """Queue hook: a web upload's job ended. Done: the file becomes the transcript's audio. Otherwise it is deleted."""
    upload = upload_of(settings, job.request.source)
    if upload is None or job.ephemeral:  # not an upload, or a /v1 request that cleans up after itself
        return
    try:
        if job.status == "done" and job.result:
            if upload.is_file():
                keep_upload_as_audio(library, job.result, upload, upload.name, via="web-upload")
    except Exception:
        log.exception("could not keep the upload of job %s as audio", job.id)
    finally:
        shutil.rmtree(upload.parent, ignore_errors=True)


def sweep(settings: Settings, jobs: list[Job]) -> None:
    """At start: delete upload folders no waiting or running job refers to (crash, cancelled while queued)."""
    root = settings.data_dir / "uploads"
    if not root.is_dir():
        return
    wanted = set()
    for job in jobs:
        upload = upload_of(settings, job.request.source)
        if upload and job.status in ("queued", "running"):
            wanted.add(upload.parent.name)
    for folder in root.iterdir():
        if folder.is_dir() and folder.name not in wanted:
            shutil.rmtree(folder, ignore_errors=True)
