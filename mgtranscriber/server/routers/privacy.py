"""Sensitive-data mode: offline status, chain of custody and audit log, redacted audio, secure delete.

See ``mgtranscriber/privacy.py`` and ``docs/PRIVACY.md``.
"""

import logging
import os
import tempfile
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from ... import editor, privacy
from ..deps import LibraryDep, QueueDep, SettingsDep
from ..errors import ApiError
from .transcripts import attachment

router = APIRouter(tags=["privacy"])
log = logging.getLogger(__name__)

# One redacted-audio run at a time (ffmpeg re-encodes the whole file); further requests get 409.
_audio_slot = threading.Lock()
_STALE_TEMP_S = 3600


class _TempFileResponse(FileResponse):
    """A FileResponse that removes its file when the response ends, also when the client went away."""

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            Path(self.path).unlink(missing_ok=True)


def _temp_dir(settings) -> Path:
    """``data_dir/tmp``: redacted audio is made next to the library, not in the shared %TEMP%. Files
    left behind by a crash (older than an hour) are removed first."""
    folder = settings.data_dir / "tmp"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("redacted-*.mp3"):
        try:
            if time.time() - old.stat().st_mtime > _STALE_TEMP_S:
                old.unlink()
        except OSError:
            log.warning("could not remove %s", old)
    return folder


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except privacy.PrivacyError as e:
        raise ApiError(e.status, e.code, e.message)


@router.get("/api/privacy")
def privacy_status(settings: SettingsDep):
    """Whether offline mode is on (set in settings.json ``offline_mode`` or ``MGTRANSCRIBER_OFFLINE_MODE``)."""
    return privacy.status(settings)


@router.get("/api/transcripts/{transcript_id}/custody")
def custody(transcript_id: str, library: LibraryDep):
    """Stored hashes, a fresh comparison with the files, whether the audit chain is intact, and the log."""
    library.path(transcript_id)
    return {**privacy.verify_custody(library, transcript_id), "audit": privacy.read_audit(library, transcript_id)}


@router.post("/api/transcripts/{transcript_id}/custody")
def record_custody(transcript_id: str, library: LibraryDep):
    """Hash the source media and outputs now and store them (for transcripts made before hashing)."""
    library.path(transcript_id)
    outputs = [editor.SEGMENTS_FILE, *(n for n in ("prepis.txt", "titulky.srt") if (library.path(transcript_id) / n).is_file())]
    record = privacy.update_custody(library, transcript_id, outputs, with_source=True)
    privacy.audit(library, transcript_id, "hashed", source_sha256=(record.get("source") or {}).get("sha256"),
                  sha256=record["outputs"].get(editor.SEGMENTS_FILE))
    return {**privacy.verify_custody(library, transcript_id), "audit": privacy.read_audit(library, transcript_id)}


@router.get("/api/transcripts/{transcript_id}/redacted/audio")
def redacted_audio(transcript_id: str, library: LibraryDep, settings: SettingsDep, beep: bool = True):
    """The audio with every redacted word silenced (and a beep in its place), as MP3. One at a time:
    409 ``redaction_running`` while another is being made."""
    library.path(transcript_id)
    media = privacy.source_file(library, transcript_id)
    if media is None:
        raise ApiError(409, "no_audio", "The transcript has no audio file.")
    ranges = privacy.redacted_ranges(library.segments(transcript_id))
    if not _audio_slot.acquire(blocking=False):
        raise ApiError(409, "redaction_running", "Another redacted audio is being made; try again shortly.")
    tmp: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix="redacted-", suffix=".mp3", dir=_temp_dir(settings))
        os.close(fd)
        tmp = Path(name)
        _run(privacy.redacted_audio, media, ranges, tmp, beep)
        privacy.audit(library, transcript_id, "exported", format="audio", redacted=True, ranges=len(ranges),
                      sha256=privacy.sha256_file(tmp))
    except BaseException:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        raise
    finally:
        _audio_slot.release()
    return _TempFileResponse(tmp, media_type="audio/mpeg", headers=attachment("audio.redacted.mp3"))


@router.delete("/api/transcripts/{transcript_id}")
def delete_transcript(transcript_id: str, request: Request, library: LibraryDep, settings: SettingsDep,
                      queue: QueueDep, confirm: str = ""):
    """Delete a transcript for good: its folder (overwritten first), voice samples taken from it and its
    search rows. ``confirm`` must repeat the transcript id."""
    library.path(transcript_id)
    if confirm != transcript_id:
        raise ApiError(400, "confirm_required", "Repeat the transcript id in 'confirm' to delete it.")
    busy = [j for j in queue.listing() if j["status"] in ("queued", "running")
            and privacy.job_targets(library, transcript_id, j)]
    run = request.app.state.llm.runs.get(transcript_id)
    if busy or (run and run.status == "running"):
        raise ApiError(409, "transcript_busy", "A job or an analysis is working on this transcript.")
    # no edit lock here: its lock file lives in the folder that is being removed
    privacy.purge_tombstones(library)  # leftovers of earlier deletes that failed part-way
    result = _run(privacy.delete_transcript, library, transcript_id, settings)
    request.app.state.llm.runs.pop(transcript_id, None)
    return result
