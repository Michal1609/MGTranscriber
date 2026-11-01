"""Recording sessions: the web UI records from the microphone or a shared tab and uploads the audio in chunks
WHILE it records, so a crashed tab, a closed lid or a sleeping computer loses at most the last chunk.

A session is a folder ``data/recordings/<id>/`` (``id`` = 32 hex chars, validated before it touches a path):

    session.json          {id, created, updated, mime, ext, source, slice_ms, duration_ms, label}
    000000.part ...       the chunks exactly as MediaRecorder produced them (one container stream, cut in pieces)

``finish`` joins the chunks (a plain byte concatenation is the MediaRecorder stream again; ffmpeg then remuxes it
so the file has a duration and can be seeked), hands the file to ``uploads`` as an ordinary upload and removes the
session. From there it is a normal queued job (``uploads.finish_job`` keeps the file as the transcript's audio).

Limits: the whole recording counts against ``max_upload_mb`` like an upload, a single chunk is at most
``MAX_CHUNK_MB``, at most ``MAX_SESSIONS`` unfinished sessions exist and one that was not touched for
``KEEP_DAYS`` is swept (at start and whenever a session starts).
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings
from ..fsutil import replace_file
from .errors import ApiError
from .uploads import adopt_file

log = logging.getLogger(__name__)

ID_RE = re.compile(r"^[0-9a-f]{32}$")
KEEP_DAYS = 7
MAX_SESSIONS = 50
MAX_CHUNK_MB = 64
MAX_CHUNK_INDEX = 99_999  # 5 s chunks: five days of recording
SESSION_FILE = "session.json"
CHUNK_RE = re.compile(r"^(\d{6})\.part$")
FFMPEG_TIMEOUT = 15 * 60

# what MediaRecorder produces in the browsers we know -> file extension of the assembled recording
CONTAINERS = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/mpeg": ".mp3", "audio/wav": ".wav",
              "audio/x-wav": ".wav", "video/webm": ".webm", "video/mp4": ".m4a"}
SOURCES = {"microphone", "tab", "mix"}

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(session_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(session_id, threading.Lock())


def root(settings: Settings) -> Path:
    return settings.data_dir / "recordings"


def session_dir(settings: Settings, session_id: str) -> Path:
    """The folder of a session; 404 for an id that is not one of ours (never builds a path from other input)."""
    if not ID_RE.fullmatch(session_id or ""):
        raise ApiError(404, "recording_not_found", "Recording not found.")
    folder = root(settings) / session_id
    if not (folder / SESSION_FILE).is_file():
        raise ApiError(404, "recording_not_found", "Recording not found.")
    return folder


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    replace_file(tmp, path)


def _read(folder: Path) -> dict:
    try:
        return json.loads((folder / SESSION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _chunks(folder: Path) -> dict[int, Path]:
    found = {}
    for p in folder.iterdir():
        m = CHUNK_RE.match(p.name)
        if m and p.is_file():
            found[int(m.group(1))] = p
    return found


def _size(chunks: dict[int, Path]) -> int:
    total = 0
    for p in chunks.values():
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def public(folder: Path) -> dict:
    meta = _read(folder)
    chunks = _chunks(folder)
    duration = meta.get("duration_ms") or len(chunks) * meta.get("slice_ms", 0)
    return {
        "id": folder.name, "created": meta.get("created"), "updated": meta.get("updated"),
        "mime": meta.get("mime"), "source": meta.get("source"), "label": meta.get("label"),
        "chunks": len(chunks), "bytes": _size(chunks), "duration": round(duration / 1000, 1),
        "contiguous": sorted(chunks) == list(range(len(chunks))),
    }


def sweep(settings: Settings, keep_days: float = KEEP_DAYS) -> int:
    """Remove sessions untouched for ``keep_days`` and folders without a session file. Returns how many."""
    base = root(settings)
    if not base.is_dir():
        return 0
    cutoff = time.time() - keep_days * 86400
    removed = 0
    for folder in base.iterdir():
        if not folder.is_dir():
            continue
        marker = folder / SESSION_FILE
        try:
            newest = max([marker.stat().st_mtime if marker.exists() else 0, *(p.stat().st_mtime for p in folder.iterdir())])
        except (OSError, ValueError):
            newest = 0
        if (not marker.exists() and time.time() - newest > 3600) or (marker.exists() and newest < cutoff):
            shutil.rmtree(folder, ignore_errors=True)
            removed += 1
    return removed


def list_sessions(settings: Settings) -> list[dict]:
    base = root(settings)
    if not base.is_dir():
        return []
    items = [public(f) for f in base.iterdir() if f.is_dir() and ID_RE.fullmatch(f.name) and (f / SESSION_FILE).is_file()]
    return sorted(items, key=lambda i: i["created"] or "", reverse=True)


def start(settings: Settings, mime: str, source: str = "microphone", slice_ms: int = 5000, label: str | None = None) -> dict:
    kind = (mime or "").split(";")[0].strip().lower()
    if kind not in CONTAINERS:
        raise ApiError(415, "unsupported_media_type", "Only audio recordings are accepted.", "mime")
    if source not in SOURCES:
        raise ApiError(400, "invalid_request", "Unknown recording source.", "source")
    sweep(settings)
    base = root(settings)
    base.mkdir(parents=True, exist_ok=True)
    if sum(1 for f in base.iterdir() if f.is_dir()) >= MAX_SESSIONS:
        raise ApiError(409, "too_many_recordings", "Too many unfinished recordings; finish or discard some first.")
    session_id = uuid.uuid4().hex
    folder = base / session_id
    folder.mkdir()
    now = _now()
    _write_json(folder / SESSION_FILE, {
        "id": session_id, "created": now, "updated": now, "mime": mime[:200], "ext": CONTAINERS[kind], "source": source,
        "slice_ms": max(500, min(60_000, int(slice_ms))), "duration_ms": 0, "label": (label or "")[:120] or None,
    })
    return public(folder)


class ChunkTooLarge(Exception):
    pass


def chunk_path(folder: Path, index: int) -> Path:
    return folder / f"{index:06d}.part"


def store_chunk(settings: Settings, folder: Path, index: int, tmp: Path, elapsed_ms: int | None) -> dict:
    """Move an already received chunk (``tmp``, inside the session folder) into place. Idempotent: a retry of the
    same index replaces it. 413 when the whole recording would exceed ``max_upload_mb``."""
    with _lock(folder.name):
        if not (folder / SESSION_FILE).is_file():  # discarded or finished while the chunk was on its way
            tmp.unlink(missing_ok=True)
            raise ApiError(404, "recording_not_found", "Recording not found.")
        size = tmp.stat().st_size
        if size == 0:
            tmp.unlink(missing_ok=True)
            raise ApiError(400, "empty_file", "The chunk is empty.")
        chunks = _chunks(folder)
        total = _size(chunks) - (chunks[index].stat().st_size if index in chunks else 0) + size
        limit = settings.max_upload_mb * 1024 * 1024
        if limit > 0 and total > limit:
            tmp.unlink(missing_ok=True)
            raise ApiError(413, "recording_too_large",
                           f"The recording is larger than the limit of {settings.max_upload_mb} MB (setting max_upload_mb).")
        replace_file(tmp, chunk_path(folder, index))
        meta = _read(folder)
        meta["updated"] = _now()
        if elapsed_ms is not None:
            meta["duration_ms"] = max(int(meta.get("duration_ms") or 0), min(int(elapsed_ms), 10 ** 9))
        _write_json(folder / SESSION_FILE, meta)
        return public(folder)


def discard(settings: Settings, session_id: str) -> None:
    folder = session_dir(settings, session_id)
    with _lock(session_id):
        shutil.rmtree(folder, ignore_errors=True)


def _remux(ffmpeg: str, joined: Path, out: Path) -> bool:
    """Copy the audio into a clean container (duration, seek index). False when ffmpeg could not."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(joined), "-vn", "-c:a", "copy", str(out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("ffmpeg remux of %s failed: %s", joined, e)
        return False
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        log.warning("ffmpeg remux of %s failed: %s", joined, proc.stderr.strip()[-300:])
        return False
    return True


def assemble(settings: Settings, session_id: str, expected: int | None = None, allow_gaps: bool = False,
             title: str | None = None) -> tuple[Path, str]:
    """Join the chunks into one file and move it to ``data/uploads/<random>/<name>``; returns (path, name).

    The session folder stays until ``discard`` (the caller removes it once the job is queued), so a failure here
    (missing chunk, ffmpeg) leaves the recording recoverable. ``expected`` = number of chunks the client sent."""
    folder = session_dir(settings, session_id)
    with _lock(session_id):
        meta = _read(folder)
        chunks = _chunks(folder)
        if not chunks:
            raise ApiError(409, "recording_empty", "The recording has no data.")
        count = expected if expected is not None else max(chunks) + 1
        missing = [i for i in range(count) if i not in chunks]
        if 0 in missing:
            raise ApiError(409, "recording_chunks_missing", "The first part of the recording is missing, so it cannot be played.")
        if missing and not allow_gaps:
            raise ApiError(409, "recording_chunks_missing",
                           f"{len(missing)} part(s) of the recording are missing: {missing[:10]}.")
        extra = [i for i in chunks if i >= count]
        order = [i for i in sorted(chunks) if i < count] if expected is not None else sorted(chunks)
        if extra and expected is not None:
            log.info("recording %s: ignoring %d chunk(s) after the announced %d", session_id, len(extra), count)
        ext = meta.get("ext") or ".webm"
        joined = folder / f"joined{ext}"
        with open(joined, "wb") as out:
            for i in order:
                with open(chunks[i], "rb") as src:
                    shutil.copyfileobj(src, out, 1 << 20)
        final = joined
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            clean = folder / f"clean{ext}"
            if _remux(ffmpeg, joined, clean):
                final = clean
        stamp = datetime.fromisoformat(meta["created"]).astimezone().strftime("%Y-%m-%d %H-%M") if meta.get("created") else "recording"
        name = f"{(title or '').strip() or 'Recording ' + stamp}"
        return adopt_file(final, name + ext, settings)


_finishing: set[str] = set()


@contextlib.contextmanager
def finishing(session_id: str):
    """One finish at a time per session (a double click must not queue two jobs)."""
    with _locks_guard:
        if session_id in _finishing:
            raise ApiError(409, "recording_busy", "This recording is already being finished.")
        _finishing.add(session_id)
    try:
        yield
    finally:
        with _locks_guard:
            _finishing.discard(session_id)
