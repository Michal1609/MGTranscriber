"""Sensitive-data mode: offline mode, chain of custody, audit log, redaction, secure delete.

* **Offline mode** (setting ``offline_mode`` in settings.json or ``MGTRANSCRIBER_OFFLINE_MODE``): no
  cloud engine, no cloud LLM, Ollama only on this computer, no URL sources. Enforced where the work
  happens (``engines.get_engine``, ``llm.make_provider``, ``pipeline.run``) and checked early by the
  API (``check_source`` / ``check_engine``), so a client can never get around it.
* **Chain of custody**: SHA-256 of the source media and of the outputs in ``metadata.json``
  (``extra.custody``), updated after every change of the transcript.
* **Audit log**: ``audit.jsonl`` in the transcript folder, one JSON object per line, append-only.
  Each line carries the SHA-256 of the previous line (``prev``), so removing or changing a line
  breaks the chain (``verify_audit``). Deleted transcripts leave one line in ``data/audit.jsonl``.
* **Redaction**: words marked ``redacted`` in prepis.json; ``redact_segments`` replaces each run of
  them by a label for redacted exports, ``redacted_audio`` silences them and plays a beep (ffmpeg).
* **Secure delete**: ``delete_transcript`` overwrites and removes the folder, the voice samples taken
  from it and its search rows.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from .config import ENV_PREFIX, is_loopback_host
from .fsutil import file_lock
from .library import TOMBSTONE_PREFIX
from .models import Segment, Word

if TYPE_CHECKING:
    from .config import Settings
    from .library import Library

log = logging.getLogger(__name__)

AUDIT_FILE = "audit.jsonl"
GLOBAL_AUDIT_FILE = "audit.jsonl"  # in data_dir: deletions (the folder and its own log are gone)
AUDIT_KEY_FILE = "audit.key"  # in data_dir: HMAC key for the ids of deleted transcripts
REDACTED_LABELS = {"cs": "[REDIGOVÁNO]", "sk": "[REDIGOVANÉ]"}
DEFAULT_REDACTED_LABEL = "[REDACTED]"
REDACT_PAD_S = 0.05  # silence a little around each redacted word


class PrivacyError(Exception):
    """A refused action with a stable ``code`` (the API answers ``status``)."""

    def __init__(self, code: str, message: str, status: int = 403):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- offline mode ---------------------------------------------------------------------------


def _truthy(value) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}


def offline(settings: "Settings") -> bool:
    value = os.environ.get(ENV_PREFIX + "OFFLINE_MODE")
    if value in (None, ""):
        value = settings.extra.get("offline_mode", False)
    return _truthy(value)


def check_source(settings: "Settings", source: str) -> None:
    """URL sources are refused in offline mode (yt-dlp would contact the site)."""
    from .sources import is_url

    if offline(settings) and is_url(source.strip()):
        raise PrivacyError("offline_url", "Offline mode: URL sources are disabled. Use a local file.")


NETWORK_ENGINES = {"youtube"}  # not "cloud" (no audio is uploaded), but they only work with URLs


def needs_network(info) -> bool:
    """Whether an engine (``engines.EngineInfo``) sends audio out or only works online."""
    return bool(info.cloud) or info.name in NETWORK_ENGINES


def check_engine(settings: "Settings", engine: str | None) -> None:
    from .engines import list_engines

    name = engine or settings.engine
    info = next((e for e in list_engines() if e.name == name), None)
    if offline(settings) and info is not None and needs_network(info):
        raise PrivacyError("offline_engine", f"Offline mode: the engine '{name}' needs the internet and is disabled.")


def check_llm(settings: "Settings", provider: str, cloud: bool, base_url: str | None = None) -> None:
    """Cloud LLM providers are refused in offline mode; Ollama must run on this computer."""
    if not offline(settings):
        return
    if cloud:
        raise PrivacyError("offline_llm", f"Offline mode: the cloud LLM provider '{provider}' is disabled.")
    if base_url:
        host = urlsplit(base_url if "//" in base_url else "//" + base_url).hostname or ""
        if not is_loopback_host(host):
            raise PrivacyError("offline_llm", f"Offline mode: the LLM server {host} is not on this computer.")


def status(settings: "Settings") -> dict:
    return {"offline": offline(settings)}


# --- hashes, chain of custody, audit log ----------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(content: "str | bytes") -> str:
    return hashlib.sha256(content if isinstance(content, bytes) else content.encode("utf-8")).hexdigest()


def audit(library: "Library", transcript_id: str, event: str, **details) -> dict:
    """Append one record to the transcript's audit log (never rewrites it). Never raises: a failing
    log is reported in the server log but must not undo the action it records."""
    try:
        return _append(library.path(transcript_id, must_exist=False) / AUDIT_FILE, event, details)
    except Exception:
        log.exception("audit log failed for %s (%s)", transcript_id, event)
        return {}


def audit_global(settings: "Settings", event: str, **details) -> dict:
    try:
        return _append(settings.data_dir / GLOBAL_AUDIT_FILE, event, details)
    except Exception:
        log.exception("global audit log failed (%s)", event)
        return {}


ANCHOR_SUFFIX = ".anchor"  # audit.jsonl.anchor: {"count": n, "last": sha256 of the last record}


def _split(path: Path) -> tuple[list[bytes], bytes]:
    """The complete (newline-terminated) lines of a log and the unterminated tail left by a write
    that was cut short (empty when the log ends cleanly)."""
    if not path.is_file():
        return [], b""
    data = path.read_bytes()
    complete, _, tail = data.rpartition(b"\n")
    return (complete.split(b"\n") if complete or data.startswith(b"\n") else []), tail


def _anchor_path(path: Path) -> Path:
    return path.with_name(path.name + ANCHOR_SUFFIX)


def _write_anchor(path: Path, count: int, last: bytes | None) -> None:
    target = _anchor_path(path)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps({"count": count, "last": hashlib.sha256(last).hexdigest() if last else None}),
                   encoding="ascii")
    os.replace(tmp, target)


def _append(path: Path, event: str, details: dict) -> dict:
    """Append one record. A tail left by an interrupted write is cut off first (it was never a record)
    and noted in the new record as ``recovered_bytes``. Afterwards the anchor is updated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        lines, tail = _split(path)
        last = lines[-1] if lines else None
        record = {"time": now_iso(), "event": event, **{k: v for k, v in details.items() if v is not None},
                  "prev": hashlib.sha256(last).hexdigest() if last else None}
        if tail:
            record["recovered_bytes"] = len(tail)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if tail:
            with open(path, "r+b") as f:
                f.truncate(path.stat().st_size - len(tail))
        with open(path, "ab") as f:  # append only
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        try:
            _write_anchor(path, len(lines) + 1, line.rstrip(b"\n"))
        except OSError:
            log.warning("could not write the audit anchor of %s", path)
    return record


def read_audit(library: "Library", transcript_id: str) -> list[dict]:
    """The records; a line that is not JSON reads as ``{"event": "unreadable"}``, an interrupted last
    write as ``{"event": "incomplete"}``."""
    lines, tail = _split(library.path(transcript_id) / AUDIT_FILE)
    out = []
    for raw in lines:
        try:
            out.append(json.loads(raw))
        except ValueError:
            out.append({"event": "unreadable"})
    if tail:
        out.append({"event": "incomplete"})
    return out


def verify_audit(path: Path) -> bool:
    """True when every line's ``prev`` is the hash of the line before it. An unterminated last line
    (an interrupted write) is not a record and does not count."""
    prev = None
    for raw in _split(path)[0]:
        try:
            record = json.loads(raw)
        except ValueError:
            return False
        if record.get("prev") != (hashlib.sha256(prev).hexdigest() if prev is not None else None):
            return False
        prev = raw
    return True


def audit_status(path: Path) -> dict:
    """``{"chain": bool, "records": n, "partial_tail": bool, "anchor": ...}``. ``anchor`` compares the
    log with the count and last hash noted at the last append: ``ok``, ``none`` (older log, no anchor),
    ``truncated`` (fewer records than noted: the end was cut off) or ``mismatch`` (the record at that
    place differs)."""
    lines, tail = _split(path)
    anchor = "none"
    try:
        noted = json.loads(_anchor_path(path).read_text(encoding="ascii"))
        count, last = int(noted["count"]), noted["last"]
    except (OSError, ValueError, KeyError, TypeError):
        noted = None
    if noted is not None:
        if len(lines) < count:
            anchor = "truncated"
        elif count == 0 or hashlib.sha256(lines[count - 1]).hexdigest() == last:
            anchor = "ok"
        else:
            anchor = "mismatch"
    return {"chain": verify_audit(path), "records": len(lines), "partial_tail": bool(tail), "anchor": anchor}


def source_file(library: "Library", transcript_id: str) -> Path | None:
    """The media the transcript was made from: downloaded audio, else the original local file."""
    audio = library.audio_file(transcript_id)
    if audio is not None:
        return audio
    source = library.metadata(transcript_id).source
    if source and not re.match(r"https?://", source, re.IGNORECASE) and Path(source).is_file():
        return Path(source)
    return None


def custody_record(library: "Library", transcript_id: str, outputs: list[str], with_source: bool = True,
                   source: Path | None = None) -> dict:
    """Current SHA-256 hashes of the source media (optional; ``source`` or the transcript's own) and
    of output files in the folder."""
    folder = library.path(transcript_id)
    record: dict = {"algorithm": "sha256", "updated": now_iso(), "outputs": {}}
    for name in outputs:
        if (folder / name).is_file():
            record["outputs"][name] = sha256_file(folder / name)
    if with_source and (media := source or source_file(library, transcript_id)) is not None:
        record["source"] = {"file": media.name, "size": media.stat().st_size, "sha256": sha256_file(media)}
    return record


def update_custody(library: "Library", transcript_id: str, outputs: list[str], with_source: bool = False) -> dict:
    """Store fresh output hashes in metadata.json (``extra.custody``). The source hash is computed
    once (at transcription) and kept, unless ``with_source``."""
    meta = library.metadata(transcript_id)
    old = meta.extra.get("custody") or {}
    record = custody_record(library, transcript_id, outputs, with_source=with_source)
    if not with_source and old.get("source"):
        record["source"] = old["source"]
    meta.extra["custody"] = record
    library.save_metadata(transcript_id, meta)
    return record


def verify_custody(library: "Library", transcript_id: str) -> dict:
    """Compare the stored hashes with the files now: ``{name: "ok" | "changed" | "missing"}``."""
    stored = library.metadata(transcript_id).extra.get("custody") or {}
    folder = library.path(transcript_id)
    result = {}
    for name, digest in (stored.get("outputs") or {}).items():
        path = folder / name
        result[name] = "missing" if not path.is_file() else "ok" if sha256_file(path) == digest else "changed"
    if stored.get("source"):
        media = source_file(library, transcript_id)
        result["source"] = ("missing" if media is None else
                            "ok" if sha256_file(media) == stored["source"].get("sha256") else "changed")
    state = audit_status(folder / AUDIT_FILE)
    return {"stored": stored, "check": result, "audit_state": state,
            "audit_intact": state["chain"] and state["anchor"] in ("ok", "none")}


# --- redaction ------------------------------------------------------------------------------


def redacted_label(language: str | None) -> str:
    return REDACTED_LABELS.get((language or "").split("-")[0].lower(), DEFAULT_REDACTED_LABEL)


def has_redactions(segments: list[Segment]) -> bool:
    return any(w.redacted for s in segments for w in s.words)


def redact_segments(segments: list[Segment], label: str = DEFAULT_REDACTED_LABEL) -> list[Segment]:
    """Copies of ``segments`` whose redacted words are replaced: each run of redacted words becomes
    one ``label`` word (text and words), so no exporter can leak them."""
    out = []
    for seg in segments:
        if not any(w.redacted for w in seg.words):
            out.append(seg)
            continue
        words: list[Word] = []
        for w in seg.words:
            if not w.redacted:
                words.append(w)
            elif words and words[-1].redacted:
                words[-1].end = w.end if w.end is not None else words[-1].end
            else:
                words.append(Word(label, w.start, w.end, None, w.speaker, True))
        out.append(replace(seg, text=" " + " ".join(w.word for w in words), words=words))
    return out


def redacted_ranges(segments: list[Segment], pad: float = REDACT_PAD_S) -> list[tuple[float, float]]:
    """Time ranges to silence: redacted words with times; a redacted word without times silences
    the gap between its timed neighbours (or its whole segment). Merged and sorted."""
    ranges = []
    for seg in segments:
        words = seg.words
        for i, w in enumerate(words):
            if not w.redacted:
                continue
            if w.start is not None and w.end is not None:
                start, end = w.start, w.end
            else:
                start = next((x.end for x in reversed(words[:i]) if x.end is not None), seg.start)
                end = next((x.start for x in words[i + 1:] if x.start is not None), seg.end)
            ranges.append((max(0.0, start - pad), end + pad))
    ranges.sort()
    merged: list[list[float]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(round(a, 3), round(b, 3)) for a, b in merged]


def redacted_audio(source: Path, ranges: list[tuple[float, float]], target: Path, beep: bool = True) -> Path:
    """Write ``source`` with ``ranges`` silenced (and a 1 kHz beep in their place) to ``target`` (mp3)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PrivacyError("ffmpeg_missing", "ffmpeg is not installed; the redacted audio cannot be made.", 501)
    if not ranges:
        cond = "0"
    else:
        cond = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in ranges)
    filters = f"[0:a]volume=enable='{cond}':volume=0[muted]"
    if beep and ranges:
        filters += (f";sine=frequency=1000:sample_rate=44100[tone];[tone]volume=2,"  # sine is at 1/8 amplitude: about -12 dB
                    f"volume=enable='not({cond})':volume=0[beep];"
                    f"[muted][beep]amix=inputs=2:duration=first:normalize=0[out]")
    else:
        filters += ";[muted]anull[out]"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-filter_complex", filters,
           "-map", "[out]", "-vn", "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-q:a", "4", str(target)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise PrivacyError("ffmpeg_failed", f"ffmpeg failed: {proc.stderr.strip()[-500:]}", 500)
    return target


# --- secure delete --------------------------------------------------------------------------


def _overwrite(path: Path) -> None:
    """Overwrite a file with zeros before it is removed (best effort, see docs/PRIVACY.md)."""
    try:
        size = path.stat().st_size
        with open(path, "r+b", buffering=0) as f:
            chunk = b"\0" * (1 << 20)
            left = size
            while left > 0:
                n = f.write(chunk[:min(left, len(chunk))])
                left -= n
            os.fsync(f.fileno())
    except OSError:
        log.warning("could not overwrite %s", path)


def job_targets(library: "Library", transcript_id: str, job: dict) -> bool:
    """Whether a job (``Job.public()``) works on this transcript: its result, or a new run of the same
    source (by path, URL, source key or title; rather too many than too few)."""
    from .library import metadata_source_key

    source, result = job.get("source") or "", job.get("result")
    if transcript_id in (source, result):
        return True
    try:
        meta = library.metadata(transcript_id)
    except Exception:
        return False
    if source and source in (meta.source, (meta.extra.get("source_info") or {}).get("webpage_url")):
        return True
    if job.get("title") and job["title"] == meta.title:
        return True
    if source and not re.match(r"https?://", source, re.IGNORECASE):
        try:
            key = "file:" + os.path.normcase(str(Path(source).resolve()))
        except OSError:
            return False
        return key == (meta.source_key or metadata_source_key(meta.to_dict()))
    return False


def delete_transcript(library: "Library", transcript_id: str, settings: "Settings") -> dict:
    """Overwrite and remove the transcript folder (prepis.json, versions, audio, analysis,
    voiceprints, audit log...), the voice samples taken from it and its search rows.

    The folder is first renamed to a hidden tombstone, so a failure part-way never leaves a
    transcript with zeroed files in the library; the tombstone is removed again at the next delete
    (``purge_tombstones``). Voice samples, the search index and the global audit record are handled
    whatever happens to the files."""
    from . import voices

    folder = library.path(transcript_id)
    if library.active_runs(transcript_id):  # the run's save would create the folder again, half empty
        raise PrivacyError("transcript_busy", "The transcript is being transcribed again. Cancel the job first.", 409)
    tombstone = folder.with_name(f"{TOMBSTONE_PREFIX}{secrets.token_hex(8)}")
    try:
        _retry(lambda: os.replace(folder, tombstone))  # the transcript is gone from the library from here on
    except OSError as e:
        raise PrivacyError("transcript_busy", f"A file of the transcript is open elsewhere ({e.strerror}).", 409)
    files: list[Path] = []
    removed_samples = 0
    error: Exception | None = None
    try:
        files = [p for p in tombstone.rglob("*") if p.is_file()]
        for path in files:
            _overwrite(path)
        shutil.rmtree(tombstone, onerror=_retry_remove)
    except Exception as e:
        error = e
        log.exception("could not remove %s completely; it is retried at the next delete", tombstone)
    finally:
        try:
            removed_samples = voices.open_db(settings).delete_samples_from(transcript_id)
        except Exception:
            log.exception("could not remove voice samples of %s", transcript_id)
        library.changed(transcript_id)  # the search index removes a transcript that is gone
        audit_global(settings, "deleted", transcript=_id_digest(settings, transcript_id), files=len(files),
                     voice_samples=removed_samples, **({"incomplete": True} if error else {}))
    return {"deleted": True, "files": len(files), "voice_samples": removed_samples,
            **({"incomplete": True} if error else {})}


def purge_tombstones(library: "Library") -> int:
    """Remove what failed deletes left behind (overwritten first)."""
    count = 0
    for path in library.root.glob(TOMBSTONE_PREFIX + "*"):
        try:
            for f in (p for p in path.rglob("*") if p.is_file()):
                _overwrite(f)
            shutil.rmtree(path, onerror=_retry_remove)
            count += 1
        except Exception:
            log.warning("could not remove %s", path)
    return count


def _id_digest(settings: "Settings", transcript_id: str) -> str:
    """HMAC of the transcript id with a random key kept in data_dir, so the deleted record cannot
    be matched to a title by guessing (a plain hash of a readable id could)."""
    key_file = settings.data_dir / AUDIT_KEY_FILE
    with file_lock(key_file):
        try:
            key = bytes.fromhex(key_file.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            key = b""
        if len(key) < 16:
            key = secrets.token_bytes(32)
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.write_text(key.hex(), encoding="ascii")
    return hmac.new(key, transcript_id.encode("utf-8"), hashlib.sha256).hexdigest()


def _retry(action) -> None:
    import time

    for delay in (0.05, 0.2, 0.5, 1.0, None):  # Windows: a reader (player, indexer) may hold a file
        try:
            return action()
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def _retry_remove(func, path, _exc):
    import stat
    import time

    for delay in (0.05, 0.2, 0.5, 1.0):  # Windows: a reader (player, indexer) may hold the file
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
            return
        except OSError:
            time.sleep(delay)
    func(path)
