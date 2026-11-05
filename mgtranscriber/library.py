"""The transcript library: one folder per transcript inside the output directory.

Folder layout (file names kept from the original tool so existing output loads unchanged)::

    <output>/<id>/      id = slug of the title + hash of the source (older folders: the title)
        prepis.json     segments with words and speakers (source of truth)
        metadata.json   title, source, date, duration, language, engine, model, speakers
        speakers.json   {"SPEAKER_00": "Alice"} display names chosen by the user
        prepis.txt      readable transcript (exporter "txt")
        titulky.srt     subtitles (exporter "srt")
        audio.*         downloaded audio, used by the web player

Folders without metadata.json (created before it existed) get metadata derived from
prepis.json and the folder itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .folders import in_folder, safe_path
from .fsutil import replace_file
from .models import UNKNOWN_SPEAKER, Segment, speaker_stats

log = logging.getLogger(__name__)

SEGMENTS_FILE = "prepis.json"
METADATA_FILE = "metadata.json"
SPEAKERS_FILE = "speakers.json"
AUDIO_TYPES = {
    ".webm": "audio/webm", ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".opus": "audio/ogg",
    ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
}
SPEAKER_KEY = re.compile(r"[\w.-]{1,40}")
LEGACY_KEYS = {"NEZNÁMÝ": UNKNOWN_SPEAKER}
# Derived files: what other features compute from prepis.json and keep in the transcript folder.
# When the same source is transcribed again the segments and speaker labels change, so each
# derived file needs one of two strategies (see docs/ARCHITECTURE.md, "Derived files"):
#  1. list it here: Library.commit_run() deletes it when the new run has succeeded (never before: a
#     cancelled or failed run leaves the old transcript untouched). For data that is cheap to
#     recompute or meaningless for new speakers. Keep the list in this module: the pipeline runs in
#     the job worker process, where feature modules may not be imported.
#  2. store Library.segments_stamp(id) in it and compare on read: the feature shows "outdated"
#     instead of losing an expensive result (the LLM analysis does this).
STALE_ON_RETRANSCRIBE = [SPEAKERS_FILE, "voices.json"]


class TranscriptNotFound(LookupError):
    pass


class InvalidTranscriptId(ValueError):
    pass


@dataclass
class Metadata:
    title: str
    source: str | None = None  # URL or local path
    date: str | None = None  # ISO 8601, when the transcript was created
    duration: float | None = None  # seconds
    language: str | None = None
    engine: str | None = None
    model: str | None = None
    speakers: list[str] = field(default_factory=list)
    source_key: str | None = None  # identity of the source (sources.source_key), finds the folder again
    folder: str = ""  # virtual folder path ("Podcasts/Cílek"; "" = unfiled), see folders.py
    extra: dict = field(default_factory=dict)  # features may store their own keys here

    @classmethod
    def from_dict(cls, data: dict) -> "Metadata":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data and k != "extra"}
        extra = {**data.get("extra", {}), **{k: v for k, v in data.items() if k not in cls.__dataclass_fields__}}
        return cls(**known, extra=extra)

    def to_dict(self) -> dict:
        return asdict(self)


def slugify(text: str, max_length: int = 50) -> str:
    """ASCII slug: "Přepis – Díl 1" -> "prepis-dil-1"."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")[:max_length].strip("-") or "transcript"


def transcript_id_for(title: str, source_key: str) -> str:
    """Stable, unique id: readable slug of the title plus a short hash of the source.

    The same source always maps to the same id (re-transcription replaces it); two
    different sources with the same title get different ids.
    """
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:8]
    return f"{slugify(title)}-{digest}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iso_to_epoch(value: str | None) -> float | None:
    try:
        return datetime.fromisoformat(value).timestamp() if value else None
    except ValueError:
        return None


def metadata_source_key(data: dict) -> str | None:
    """Source key stored in metadata.json, or derived from what older versions stored."""
    if data.get("source_key"):
        return data["source_key"]
    info = (data.get("extra") or {}).get("source_info") or {}
    if info.get("extractor_key") and info.get("id"):
        return f"{info['extractor_key']}:{info['id']}"  # same rule as sources.source_key
    source = data.get("source")
    if source and not re.match(r"https?://", source, re.IGNORECASE):
        return "file:" + os.path.normcase(str(Path(source).resolve()))
    return None


STAGE_PREFIX = ".run-"  # staging folder of a transcription in progress (Library.begin_run)
STAGE_OLD = "old"  # inside a stage: what commit_run moved out of the transcript, until everything is saved
MANIFEST = "manifest.json"
STAGE_MAX_AGE_S = 24 * 3600  # a stage nothing touched for this long belongs to a killed run
RECOVER_AFTER_S = 10 * 60  # a commit that was interrupted this long ago is undone by the next run
TOMBSTONE_PREFIX = ".deleting-"  # a transcript being deleted for good (privacy.delete_transcript)


def _read_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data) -> None:
    # a temp file of its own: two writers of one file (editor, pipeline) must not share it;
    # no NaN/Infinity, which is not JSON and breaks the API answers
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        replace_file(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def stage_mtime(stage: Path) -> float:
    """Last change in a stage: the folder itself or anything directly in it or in its ``old`` folder."""
    newest = stage.stat().st_mtime
    for sub in (stage, stage / STAGE_OLD):
        try:
            newest = max(newest, sub.stat().st_mtime)
            for f in sub.iterdir():
                newest = max(newest, f.stat().st_mtime)
        except OSError:
            continue
    return newest


def _intact(path: Path, size: int | None) -> bool:
    """A file in ``old/`` that may be put back: it has the size the manifest recorded; a manifest written by an
    older version has none, so a JSON file must at least parse (a truncated copy does not)."""
    try:
        if isinstance(size, int):
            return path.stat().st_size == size
        if path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                json.load(f)
    except (OSError, ValueError):
        return False
    return True


def restore_old(stage: Path, folder: Path) -> None:
    """Undo ``commit_run`` of a run: remove what it created, put the moved and copied files back."""
    old = stage / STAGE_OLD
    manifest = _read_json(old / MANIFEST, {})
    sizes = manifest.get("sizes") if isinstance(manifest.get("sizes"), dict) else {}
    for name in manifest.get("created", []):
        (folder / name).unlink(missing_ok=True)
    for name in manifest.get("files", []):
        if (old / name).is_file() and _intact(old / name, sizes.get(name)):
            replace_file(old / name, folder / name)
        elif (old / name).is_file():
            log.warning("%s in %s is not complete (a copy cut short); the file in %s is kept", name, old, folder)
    (old / MANIFEST).unlink(missing_ok=True)  # restored: nothing left to recover


def discard_stage(stage: Path, output_dir: Path) -> bool:
    """The worker of a run is gone (cancelled, failed, stopped) and did not clean up: put back what an
    interrupted ``commit_run`` moved, remove the stage, and the transcript folder when nothing else is in it.
    Only a ``.run-*`` folder directly inside ``output_dir``'s subfolders is touched. True when the stage is gone."""
    try:
        stage, root = Path(stage).resolve(), Path(output_dir).resolve()
        if not stage.name.startswith(STAGE_PREFIX) or stage.parent.parent != root:
            return False
        if not stage.is_dir():
            return True
        if (stage / STAGE_OLD / MANIFEST).is_file():
            restore_old(stage, stage.parent)
        shutil.rmtree(stage, ignore_errors=True)
        try:
            stage.parent.rmdir()  # only when empty: a folder with a transcript stays
        except OSError:
            pass
        return not stage.exists()
    except Exception:
        log.exception("could not discard %s", stage)  # the stage stays: the next run of the source or the startup sweep restores from it
        return False


@dataclass
class Run:
    """A (re)transcription in progress, see ``Library.begin_run``."""

    folder: Path  # the transcript's folder
    stage: Path  # where the new run puts its audio until it has succeeded
    existed: bool  # the folder was there before this run
    _touched: float = 0.0

    def touch(self, every: float = 30.0) -> None:
        """Mark the stage as alive (progress of a long run); rate limited."""
        now = time.time()
        if now - self._touched >= every:
            self._touched = now
            try:
                os.utime(self.stage)
            except OSError:
                pass


class Library:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._root_resolved = self.root.resolve()
        self._cache: dict[str, tuple[tuple, dict]] = {}
        self._names_cache: dict[str, tuple[tuple, dict[str, str]]] = {}  # id -> ((speakers.json mtime_ns, size), names)
        self._lock = threading.Lock()
        # called with the transcript id after segments, metadata or speaker names were saved
        # through this object (e.g. the search index); a failing listener does not fail the save
        self.listeners: list[Callable[[str], None]] = []
        self._source_keys: dict[str, str] = {}  # id -> source key of folders made by create()
        self._folder_cache: dict[str, tuple[int, str]] = {}  # id -> (metadata.json mtime_ns, folder)

    def changed(self, transcript_id: str) -> None:
        for listener in list(self.listeners):
            try:
                listener(transcript_id)
            except Exception:
                log.exception("library listener failed for %s", transcript_id)

    # --- paths -------------------------------------------------------------------------

    def path(self, transcript_id: str, must_exist: bool = True) -> Path:
        """Folder of a transcript. Never resolves outside the library root."""
        if not transcript_id or transcript_id in {".", ".."} or any(c in transcript_id for c in '/\\:\0'):
            raise InvalidTranscriptId(transcript_id)
        # the id has no separators, so the child is inside the root unless it is a link to elsewhere (resolve() on
        # every call cost about 0.7 ms each on Windows and dominated the listing; the root is resolved once)
        path = self._root_resolved / transcript_id
        if path.is_symlink() or os.path.isjunction(path):
            path = path.resolve()
            if path.parent != self._root_resolved:
                raise InvalidTranscriptId(transcript_id)
        if must_exist and not (path / SEGMENTS_FILE).is_file():
            raise TranscriptNotFound(transcript_id)
        return path

    def create(self, title: str, source_key: str) -> Path:
        """Folder for a transcript of ``source_key`` (see ``transcript_id_for``), created if missing.

        The same source always gets the same folder, even when the title changed or the folder
        predates source ids (found through metadata.json). Nothing in it is touched: a new run
        goes through ``begin_run`` / ``commit_run`` so that the old transcript survives a failed one.
        """
        existing = self.find_by_source(source_key)
        path = self.root / (existing or transcript_id_for(title, source_key))
        self._source_keys[path.name] = source_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def begin_run(self, title: str, source_key: str) -> "Run":
        """Start a (re)transcription: the folder plus a private staging folder for the new audio.

        Nothing of an existing transcript is changed until ``commit_run``; ``abort_run`` (or a
        process killed mid-run: see ``_sweep_stages``) leaves it as it was.
        """
        existing = self.find_by_source(source_key)
        path = self.root / (existing or transcript_id_for(title, source_key))
        existed = path.is_dir()
        self._source_keys[path.name] = source_key
        path.mkdir(parents=True, exist_ok=True)
        self._sweep_stages(path)
        stage = path / f"{STAGE_PREFIX}{secrets.token_hex(6)}"
        # parents=True: another run's abort_run may have removed the (empty) folder just now
        stage.mkdir(parents=True)
        return Run(folder=path, stage=stage, existed=existed)

    def _sweep_stages(self, folder: Path, max_age: float = STAGE_MAX_AGE_S, protect: "set[Path] | None" = None) -> int:
        """Clean up after runs that were killed (the worker cannot do it itself). A stage is only
        touched when nothing in it changed for a long time (``max_age``); one whose commit was interrupted first
        gets the old files back. ``protect``: stages of runs known to be alive, never touched. Returns how many
        stages were removed."""
        removed = 0
        for stage in folder.glob(STAGE_PREFIX + "*"):
            try:
                if (not stage.is_dir() or stage.is_symlink() or os.path.isjunction(stage)
                        or (protect and stage.resolve() in protect)):
                    continue
                age = time.time() - stage_mtime(stage)
                if (stage / STAGE_OLD / MANIFEST).is_file():
                    if age > RECOVER_AFTER_S:
                        restore_old(stage, folder)
                        shutil.rmtree(stage, ignore_errors=True)
                        removed += 1
                elif age > max_age:
                    shutil.rmtree(stage, ignore_errors=True)
                    removed += 1
            except Exception:
                log.exception("could not clean up %s", stage)
        return removed

    def sweep_stages(self, max_age: float = 3600.0, protect: "set[Path] | None" = None) -> int:
        """Remove the stages that killed runs left in ANY folder of the library (including folders that hold
        nothing else, which ``ids()`` does not list), when nothing in them changed for ``max_age`` seconds.
        A folder that is empty afterwards is removed too (never one that holds anything). ``protect``: resolved
        paths of stages of runs that are alive. Returns the number of stages removed."""
        if not self.root.is_dir():
            return 0
        removed = 0
        for folder in list(self.root.iterdir()):
            try:
                if not folder.is_dir() or folder.name.startswith(TOMBSTONE_PREFIX):
                    continue
                if folder.is_symlink() or os.path.isjunction(folder):
                    continue  # a link leads out of the library: its target is not ours to clean
                if not any(True for _ in folder.glob(STAGE_PREFIX + "*")):
                    continue
                removed += self._sweep_stages(folder, max_age, protect)
                try:
                    folder.rmdir()  # only when nothing else is in it
                except OSError:
                    pass
            except OSError:
                continue
        return removed

    def commit_run(self, run: "Run", snapshot: tuple[str, ...] | list[str] = ()) -> None:
        """The new run succeeded: its audio replaces the old audio, derived files of the old run go.

        Call under the editor's lock, right before the new segments are saved. Nothing is deleted:
        what is replaced moves to ``stage/old`` (the files in ``snapshot`` are copied there, for
        the caller is about to rewrite them). Any error afterwards must end in ``abort_run``, which
        puts it all back; ``finish_run`` drops the old files once everything is saved. The old
        audio stays when the new run staged none (captions only, a file read from where it is).
        """
        old = run.stage / STAGE_OLD
        staged = [f for f in run.stage.glob("audio.*") if f.is_file()]
        move = [f.name for f in run.folder.glob("audio.*") if f.is_file()] if staged else []
        move += [n for n in STALE_ON_RETRANSCRIBE if (run.folder / n).is_file()]
        copy = [n for n in snapshot if (run.folder / n).is_file()]
        created = [n for n in [f.name for f in staged] + list(snapshot) if not (run.folder / n).exists()]
        old.mkdir(exist_ok=True)
        # the manifest first: a crash from here on can be undone (restore_old) by the next run
        # sizes: restore_old skips a file in old/ that is not the size it had (a copy cut short by a kill)
        sizes = {n: (run.folder / n).stat().st_size for n in move + copy}
        _write_json(old / MANIFEST, {"created": created, "files": move + copy, "sizes": sizes})
        for name in move:
            replace_file(run.folder / name, old / name)
        for name in copy:  # to a temp name first: old/<name> is either absent or complete, never half a copy
            part = old / (name + ".part")
            shutil.copy2(run.folder / name, part)
            replace_file(part, old / name)
        for f in staged:
            replace_file(f, run.folder / f.name)

    def finish_run(self, run: "Run") -> None:
        """Everything is saved: the old files and the stage can go."""
        shutil.rmtree(run.stage, ignore_errors=True)

    def abort_run(self, run: "Run") -> None:
        """The run was cancelled or failed: put back what ``commit_run`` moved, drop the stage; a
        folder made for the run goes too. Safe to call twice."""
        try:
            if (run.stage / STAGE_OLD / MANIFEST).is_file():
                restore_old(run.stage, run.folder)
        except Exception:
            # keep the stage: the next run of this source restores from it (_sweep_stages)
            log.exception("could not restore %s; the old files stay in %s", run.folder, run.stage)
            return
        shutil.rmtree(run.stage, ignore_errors=True)
        if not run.existed:
            try:
                run.folder.rmdir()  # only when empty
            except OSError:
                pass

    def active_runs(self, transcript_id: str) -> bool:
        """A (re)transcription of this transcript seems to be in progress (a stage changed lately)."""
        folder = self.path(transcript_id, must_exist=False)
        for stage in folder.glob(STAGE_PREFIX + "*"):
            try:
                if stage.is_dir() and time.time() - stage_mtime(stage) <= STAGE_MAX_AGE_S:
                    return True
            except OSError:
                continue
        return False

    def find_by_source(self, source_key: str) -> str | None:
        """Id of the transcript made from ``source_key``, if any."""
        for transcript_id in self.ids():
            data = _read_json(self.root / transcript_id / METADATA_FILE, None)
            if isinstance(data, dict) and metadata_source_key(data) == source_key:
                return transcript_id
        return None

    def ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return [p.name for p in self.root.iterdir()
                if p.is_dir() and not p.name.startswith(TOMBSTONE_PREFIX) and (p / SEGMENTS_FILE).is_file()]

    # --- content -----------------------------------------------------------------------

    def segments(self, transcript_id: str) -> list[Segment]:
        with open(self.path(transcript_id) / SEGMENTS_FILE, encoding="utf-8") as f:
            return [Segment.from_dict(s) for s in json.load(f)]

    def save_segments(self, transcript_id: str, segments: list[Segment]) -> None:
        _write_json(self.path(transcript_id, must_exist=False) / SEGMENTS_FILE, [s.to_dict() for s in segments])
        self.changed(transcript_id)

    def metadata(self, transcript_id: str) -> Metadata:
        path = self.path(transcript_id)
        data = _read_json(path / METADATA_FILE, None)
        if isinstance(data, dict) and data.get("title"):
            return Metadata.from_dict(data)
        return self._legacy_metadata(path)

    def save_metadata(self, transcript_id: str, metadata: Metadata) -> None:
        if metadata.source_key is None:  # callers need not carry it: keep what create() or the old file knew
            old = _read_json(self.root / transcript_id / METADATA_FILE, None)
            metadata.source_key = self._source_keys.get(transcript_id) or (
                metadata_source_key(old) if isinstance(old, dict) else None)
        _write_json(self.path(transcript_id, must_exist=False) / METADATA_FILE, metadata.to_dict())
        if (self.root / transcript_id / SEGMENTS_FILE).is_file():
            self.changed(transcript_id)

    def folder_of(self, transcript_id: str) -> str:
        """The virtual folder of a transcript ("" = unfiled). Cheap: only metadata.json, cached by its mtime."""
        path = self.root / transcript_id / METADATA_FILE
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            return ""
        cached = self._folder_cache.get(transcript_id)
        if cached and cached[0] == stamp:
            return cached[1]
        data = _read_json(path, None)
        folder = safe_path(data.get("folder")) if isinstance(data, dict) else ""
        self._folder_cache[transcript_id] = (stamp, folder)
        return folder

    def folders_map(self) -> dict[str, str]:
        """Transcript id -> folder, for every transcript."""
        return {i: self.folder_of(i) for i in self.ids()}

    def set_folder(self, transcript_id: str, folder: str) -> None:
        """Put the transcript into ``folder`` (already normalised). Saved like any metadata change, so
        listeners (the search index) run. Callers that race with the editor hold ``editor.edit_lock``
        (``FolderStore`` does)."""
        path = self.path(transcript_id) / METADATA_FILE
        if path.exists():
            # patch the one key in the file as it is: going through metadata() would, for a file that cannot
            # be read, rewrite it from the legacy defaults and lose extra (custody, source_info)
            data = _read_json(path, None)
            if not isinstance(data, dict):
                raise ValueError(f"{path} is not readable; its folder was not changed")
            data["folder"] = folder
            _write_json(path, data)
            self.changed(transcript_id)
        else:  # a transcript of the original tool: metadata.json is made from what prepis.json tells
            meta = self.metadata(transcript_id)
            meta.folder = folder
            self.save_metadata(transcript_id, meta)
        self._folder_cache.pop(transcript_id, None)

    def names(self, transcript_id: str) -> dict[str, str]:
        data = _read_json(self.path(transcript_id) / SPEAKERS_FILE, {})
        if not isinstance(data, dict):
            return {}
        # the original web UI stored the name of segments without a speaker under its Czech label
        return {LEGACY_KEYS.get(k, k): v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}

    def _names_for(self, transcript_id: str, stamp: tuple | None) -> dict[str, str]:
        """``names()`` cached by the stamp (mtime_ns, size) of speakers.json; ``None`` = no such file."""
        if stamp is None:
            return {}
        with self._lock:
            named = self._names_cache.get(transcript_id)
        if not named or named[0] != stamp:
            named = (stamp, self.names(transcript_id))
            with self._lock:
                self._names_cache[transcript_id] = named
        return dict(named[1])

    def names_cached(self, transcript_id: str) -> dict[str, str]:
        """Same as ``names()``, but re-read only when speakers.json changed (for loops over the whole archive)."""
        try:
            st = (self.path(transcript_id) / SPEAKERS_FILE).stat()
        except OSError:
            return {}
        return self._names_for(transcript_id, (st.st_mtime_ns, st.st_size))

    def save_names(self, transcript_id: str, names: dict[str, str]) -> dict[str, str]:
        clean = {k: v.strip()[:60] for k, v in names.items() if SPEAKER_KEY.fullmatch(k) and v.strip()}
        _write_json(self.path(transcript_id) / SPEAKERS_FILE, clean)
        self.changed(transcript_id)
        return clean

    def segments_stamp(self, transcript_id: str) -> str:
        """Hash of prepis.json: store it in a derived file to detect later that the transcript changed."""
        return hashlib.sha1((self.path(transcript_id) / SEGMENTS_FILE).read_bytes()).hexdigest()

    def audio_file(self, transcript_id: str) -> Path | None:
        path = self.path(transcript_id)
        return next((p for p in sorted(path.glob("audio.*")) if p.is_file() and p.suffix.lower() in AUDIO_TYPES), None)

    # --- listing -----------------------------------------------------------------------

    def summary(self, transcript_id: str) -> dict:
        """Everything the library list needs. Cached until prepis.json or metadata.json change.

        One directory scan gives the file stamps and the audio file (no per-file exists/stat/glob), and the
        speaker names are re-read only when speakers.json changed: with thousands of transcripts this runs
        for each of them on every list request."""
        path = self.path(transcript_id, must_exist=False)
        stamps: dict[str, int] = {}
        audio_names: list[str] = []
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    name = entry.name
                    if name in (SEGMENTS_FILE, METADATA_FILE, SPEAKERS_FILE):
                        st = entry.stat()
                        stamps[name] = st.st_mtime_ns if name != SPEAKERS_FILE else (st.st_mtime_ns, st.st_size)
                    elif name.startswith("audio.") and Path(name).suffix.lower() in AUDIO_TYPES and entry.is_file():
                        audio_names.append(name)
        except (FileNotFoundError, NotADirectoryError):
            raise TranscriptNotFound(transcript_id) from None
        if SEGMENTS_FILE not in stamps:
            raise TranscriptNotFound(transcript_id)
        stamp = (stamps[SEGMENTS_FILE], stamps.get(METADATA_FILE, 0))
        with self._lock:
            cached = self._cache.get(transcript_id)
        if not cached or cached[0] != stamp:
            segments = self.segments(transcript_id)
            meta = self.metadata(transcript_id)
            stats = speaker_stats(segments)
            computed = {
                "title": meta.title,
                "source": meta.source,
                "date": iso_to_epoch(meta.date) or (path / SEGMENTS_FILE).stat().st_mtime,
                "duration": meta.duration or max((s.end for s in segments), default=0),
                "language": meta.language,
                "engine": meta.engine,
                "model": meta.model,
                "speakers": stats,
                "words": sum(len(s.text.split()) for s in segments),
                "no_speech": not any(s.text.strip() for s in segments),  # nothing was recognised (music, silence)
                "folder": safe_path(meta.folder),
            }
            cached = (stamp, computed)
            with self._lock:
                self._cache[transcript_id] = cached
        names = self._names_for(transcript_id, stamps.get(SPEAKERS_FILE))
        audio = min(audio_names) if audio_names else None
        return {"id": transcript_id, **cached[1], "names": names, "has_audio": audio is not None, "audio": audio}

    def list(self, folder: str | None = None, recursive: bool = False) -> list[dict]:
        """All transcripts, newest first. ``folder``: only those in it (``""`` = unfiled; with
        ``recursive`` also those below it)."""
        items = []
        for transcript_id in self.ids():
            if folder is not None and not in_folder(self.folder_of(transcript_id), folder, recursive):
                continue
            try:
                items.append(self.summary(transcript_id))
            except (OSError, ValueError, KeyError, TypeError):
                continue  # half-written or damaged transcript
        return sorted(items, key=lambda t: -t["date"])

    def _legacy_metadata(self, path: Path) -> Metadata:
        segments_file = path / SEGMENTS_FILE
        with open(segments_file, encoding="utf-8") as f:
            raw = json.load(f)
        segments = [Segment.from_dict(s) for s in raw]
        mtime = datetime.fromtimestamp(segments_file.stat().st_mtime, timezone.utc)
        return Metadata(
            title=path.name,
            date=mtime.isoformat(timespec="seconds"),
            duration=max((s.end for s in segments), default=0),
            engine="whisperx",
            speakers=sorted({s.speaker for s in segments if s.speaker}),
        )
