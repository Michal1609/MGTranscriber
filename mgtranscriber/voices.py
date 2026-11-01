"""Voice memory: recognise known speakers across transcripts by their voice.

Diarization yields one 256-dim speaker embedding per label (pyannote community-1, WeSpeaker
ResNet34). They are compared by cosine similarity of L2-normalised vectors. Measured on
real recordings: the same person scores 0.85-0.92 across different podcasts (0.94+
within one), different people at most 0.34.

Two files:

* the **voice database** (``data/voiceprints.json`` by default, setting ``voices_file``):
  people and their samples. Biometric data: it never leaves this computer.
* ``voices.json`` in each transcript folder: the embedding of every speaker label plus the
  identification result. ``speakers.json`` stays ``{label: name}`` (read by the library,
  exports, search, MCP) and only ever receives names with status ``auto``; suggestions are
  shown in the UI and become names only when the user accepts them.

Decisions (``identify``): a label is ``auto`` when its best person scores >= AUTO_THRESHOLD
with a lead >= MARGIN over the next person, ``suggest`` from SUGGEST_THRESHOLD, else
``unknown``. Assignment is 1:1 within a transcript: two labels never get the same person
automatically (the weaker one becomes a suggestion), so two unknown voices can never be
merged into one known person. Unknown voices are never enrolled automatically.

Embeddings depend on the diarization model, so every sample records its ``model``; samples
of another model are ignored (``list()`` reports them as outdated).
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .fsutil import LockTimeout as LockTimeout  # re-exported: routers catch voices.LockTimeout
from .fsutil import PathLock as _PathLock, file_lock

import numpy as np

if TYPE_CHECKING:
    from .config import Settings
    from .library import Library
    from .progress import ProgressCallback

log = logging.getLogger(__name__)

MODEL = "pyannote/speaker-diarization-community-1"  # = engines.common.DIARIZATION_MODEL
DIM = 256
DB_VERSION = 1
DB_FILE = "voiceprints.json"
TRANSCRIPT_FILE = "voices.json"

AUTO_THRESHOLD = 0.60
SUGGEST_THRESHOLD = 0.45
MARGIN = 0.10
MIN_IDENTIFY_S = 10.0  # less speech than this: too unreliable to identify
MIN_ENROLL_S = 20.0  # less speech than this: not enough for a sample


class VoiceError(Exception):
    """A problem the API reports with a stable code (see ``server.routers.voices``)."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if v.shape[0] != DIM or not np.isfinite(n) or n == 0:
        raise VoiceError("invalid_vector", f"Expected a non-zero {DIM}-dim vector.")
    return v / n


def encode(vector) -> str:
    return base64.b64encode(normalize(vector).astype("<f4").tobytes()).decode("ascii")


def decode(text: str) -> np.ndarray:
    return normalize(np.frombuffer(base64.b64decode(text), dtype="<f4"))


def _read_json(path: Path, default, corrupt: "VoiceError | None" = None):
    """JSON of ``path``; ``default`` when it is missing. A damaged file raises ``corrupt`` or,
    without one, also counts as missing."""
    for attempt in range(40):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except PermissionError:  # Windows: the file is being replaced by another thread or process
            if attempt == 39:
                raise
            time.sleep(0.05)
        except ValueError:  # JSONDecodeError, UnicodeDecodeError
            if corrupt is not None:
                raise corrupt from None
            log.warning("ignoring damaged %s", path)
            return default


def _write_json(path: Path, data) -> None:
    """Atomic write: a unique temp file next to ``path`` (fsynced), then moved into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(40):  # Windows: a reader in another process may hold the file briefly
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.05)
    finally:
        tmp.unlink(missing_ok=True)


# Read-modify-write of the voice database and of a transcript's voices.json happens in the server
# (several threads) and in job worker processes: fsutil.file_lock (one lock per file, also across
# processes). Order when both are needed: transcript first, then the database.


def db_path(settings: "Settings") -> Path:
    custom = os.environ.get("MGTRANSCRIBER_VOICES_FILE") or settings.extra.get("voices_file")
    if custom:
        path = Path(custom)
        return path if path.is_absolute() else settings.root / path
    return settings.data_dir / DB_FILE


def enabled(settings: "Settings") -> bool:
    value = os.environ.get("MGTRANSCRIBER_VOICE_MEMORY", settings.extra.get("voice_memory", True))
    return value if isinstance(value, bool) else str(value).strip().lower() not in {"0", "false", "no", "off"}


# --- database -------------------------------------------------------------------------------


@dataclass
class Match:
    status: str  # auto | suggest | unknown
    person_id: str | None = None
    name: str | None = None
    score: float | None = None  # best person's score (also for unknown, when the DB is not empty)
    margin: float | None = None  # lead over the next person

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VoiceIndex:
    """An immutable snapshot of the database for matching: names of all people and, for people with
    samples of the current model, their matrix of unit vectors. Built once per database version
    (``VoiceDB.index``), so scanning an archive does not decode the samples again for every label."""

    names: dict[str, str]
    mats: dict[str, np.ndarray]  # person id -> (n, DIM)
    means: dict[str, np.ndarray]  # person id -> unit mean of the samples

    def scores(self, vector, person_id: str | None = None) -> dict[str, float]:
        v = normalize(vector)
        return {pid: float(max(float(self.means[pid] @ v), float((mat @ v).max())))
                for pid, mat in self.mats.items() if person_id is None or pid == person_id}

    def sample_count(self, person_id: str) -> int:
        mat = self.mats.get(person_id)
        return 0 if mat is None else len(mat)


@dataclass
class _Sample:
    id: str
    vector: np.ndarray
    model: str
    speech_s: float
    source: dict
    added: str


class VoiceDB:
    """People and their voice samples in one JSON file. Thread-safe within a process; writes are atomic."""

    def __init__(self, path: Path, model: str = MODEL):
        self.path = Path(path)
        self.model = model
        self._lock = file_lock(self.path)  # shared by all instances for this file, and across processes
        self._index: tuple[tuple, VoiceIndex] | None = None

    # storage

    def _load(self) -> dict:
        corrupt = VoiceError("voice_db_corrupt", f"The voice database {self.path} is damaged.", 500)
        data = _read_json(self.path, None, corrupt)
        if data is None:
            return {"version": DB_VERSION, "people": {}}
        if not isinstance(data, dict) or not isinstance(data.get("people"), dict):
            raise corrupt
        version = data.get("version", 0)
        if not isinstance(version, int) or isinstance(version, bool):
            raise corrupt
        if version > DB_VERSION:
            raise VoiceError("voice_db_newer", "The voice database was written by a newer version.", 409)
        for person in data["people"].values():
            if (not isinstance(person, dict) or not isinstance(person.get("name"), str) or not person["name"].strip()
                    or not isinstance(person.setdefault("samples", []), list)
                    or not all(isinstance(x, dict) and isinstance(x.get("id"), str) for x in person["samples"])):
                raise corrupt
        return data

    def _save(self, data: dict) -> None:
        data["version"] = DB_VERSION
        _write_json(self.path, data)

    def _samples(self, person: dict, compatible_only: bool = True) -> list[_Sample]:
        out = []
        for s in person.get("samples", []):
            if compatible_only and (s.get("model") != self.model or s.get("dim", DIM) != DIM):
                continue
            try:
                vec = decode(s["vector"])
            except (VoiceError, ValueError, KeyError):
                continue
            out.append(_Sample(s["id"], vec, s.get("model"), s.get("speech_s", 0.0), s.get("source", {}), s.get("added")))
        return out

    # queries

    def list(self) -> list[dict]:
        with self._lock:
            data = self._load()
        people = []
        for pid, p in data["people"].items():
            samples = p.get("samples", [])
            ok = {s.id for s in self._samples(p)}
            people.append({
                "id": pid,
                "name": p["name"],
                "created": p.get("created"),
                "samples": [{
                    "id": s["id"], "speech_s": s.get("speech_s"), "source": s.get("source", {}),
                    "added": s.get("added"), "model": s.get("model"), "outdated": s["id"] not in ok,
                } for s in samples],
                "speech_s": round(sum(s.get("speech_s") or 0 for s in samples), 1),
            })
        return sorted(people, key=lambda p: p["name"].casefold())

    def person_id(self, name: str) -> str | None:
        key = name.strip().casefold()
        with self._lock:
            data = self._load()
        return next((pid for pid, p in data["people"].items() if p["name"].casefold() == key), None)

    def _stamp(self) -> tuple:
        try:
            st = self.path.stat()
            return (st.st_mtime_ns, st.st_size, st.st_ino)
        except OSError:
            return ()

    def index(self) -> VoiceIndex:
        """The matching snapshot of the database as it is now (rebuilt only after the file changed)."""
        with self._lock:
            stamp = self._stamp()
            cached = self._index
            if cached is not None and cached[0] == stamp:
                return cached[1]
            data = self._load()
            names, mats, means = {}, {}, {}
            for pid, p in data["people"].items():
                names[pid] = p["name"]
                samples = self._samples(p)
                if not samples:
                    continue
                mat = np.stack([s.vector for s in samples])
                mean = mat.mean(axis=0)
                mean /= np.linalg.norm(mean) or 1.0
                mats[pid], means[pid] = mat, mean
            index = VoiceIndex(names, mats, means)
            self._index = (stamp, index)
            return index

    def scores(self, vector, person_id: str | None = None) -> dict[str, float]:
        """Person id -> score: max(cosine to the mean of the samples, cosine to the nearest sample)."""
        return self.index().scores(vector, person_id)

    def identify(self, speakers: dict[str, tuple], rejected: dict[str, list[str]] | None = None,
                 index: VoiceIndex | None = None) -> dict[str, Match]:
        """``{label: (vector, speech_s)}`` -> ``{label: Match}``. ``rejected``: label -> person ids the
        user said are wrong for that label (never suggested again). ``index``: a snapshot to match
        against (default: the current one)."""
        rejected = rejected or {}
        index = index or self.index()
        names = index.names
        best: dict[str, tuple[str | None, float | None, float | None]] = {}
        for label, (vector, speech_s) in speakers.items():
            scores = {pid: s for pid, s in index.scores(vector).items() if pid not in rejected.get(label, ())}
            if not scores:
                best[label] = (None, None, None)
                continue
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            top_pid, top = ranked[0]
            margin = top - ranked[1][1] if len(ranked) > 1 else top + 1.0
            if speech_s is not None and speech_s < MIN_IDENTIFY_S:
                best[label] = (None, round(top, 4), None)
                continue
            best[label] = (top_pid, round(top, 4), round(margin, 4))

        result: dict[str, Match] = {}
        # auto candidates, strongest first; a person already taken downgrades the rest to suggestions
        taken: set[str] = set()
        order = sorted(best, key=lambda lb: -(best[lb][1] or -9))
        for label in order:
            pid, score, margin = best[label]
            if pid is None:
                result[label] = Match("unknown", score=score)
            elif score >= AUTO_THRESHOLD and margin >= MARGIN and pid not in taken:
                taken.add(pid)
                result[label] = Match("auto", pid, names[pid], score, margin)
            elif score >= SUGGEST_THRESHOLD:
                result[label] = Match("suggest", pid, names[pid], score, margin)
            else:
                result[label] = Match("unknown", score=score, margin=margin)
        return {label: result[label] for label in speakers}

    # changes

    def enroll(self, name: str, vector, *, speech_s: float, source: dict | None = None,
               person_id: str | None = None, force: bool = False) -> dict:
        """Add a sample to the person ``person_id`` or named ``name`` (created when new).

        Refuses samples with too little speech, and samples that do not resemble the
        person's existing ones (``force`` overrides) so that one wrong click cannot poison
        a voice."""
        name = (name or "").strip()[:60]
        if not name and not person_id:
            raise VoiceError("voice_name_required", "A name is required.")
        if speech_s < MIN_ENROLL_S:
            raise VoiceError("voice_too_short", f"At least {MIN_ENROLL_S:.0f} s of speech is needed, this speaker has {speech_s:.0f} s.")
        vec = normalize(vector)
        with self._lock:
            data = self._load()
            people = data["people"]
            pid = person_id or next((k for k, p in people.items() if p["name"].casefold() == name.casefold()), None)
            if person_id and person_id not in people:
                raise VoiceError("voice_person_not_found", "Person not found.", 404)
            if pid and not force:
                samples = self._samples(people[pid])
                if samples:
                    sim = max(float(s.vector @ vec) for s in samples)
                    if sim < SUGGEST_THRESHOLD:
                        raise VoiceError("voice_mismatch",
                                         f"This voice does not resemble the saved voice of {people[pid]['name']} ({sim:.2f}).", 409)
            if not pid:
                pid = uuid.uuid4().hex[:12]
                people[pid] = {"name": name, "created": now_iso(), "samples": []}
            sample = {"id": uuid.uuid4().hex[:12], "vector": encode(vec), "model": self.model, "dim": DIM,
                      "speech_s": round(float(speech_s), 1), "source": source or {}, "added": now_iso()}
            people[pid]["samples"].append(sample)
            self._save(data)
        return {"person_id": pid, "name": people[pid]["name"], "sample_id": sample["id"]}

    def rename(self, person_id: str, name: str) -> None:
        name = name.strip()[:60]
        if not name:
            raise VoiceError("voice_name_required", "A name is required.")
        with self._lock:
            data = self._load()
            if person_id not in data["people"]:
                raise VoiceError("voice_person_not_found", "Person not found.", 404)
            if any(p["name"].casefold() == name.casefold() for k, p in data["people"].items() if k != person_id):
                raise VoiceError("voice_name_taken", "Another person already has this name.", 409)
            data["people"][person_id]["name"] = name
            self._save(data)

    def delete(self, person_id: str, sample_id: str | None = None) -> None:
        """Delete a person with all samples, or one sample (the person goes with its last sample)."""
        with self._lock:
            data = self._load()
            person = data["people"].get(person_id)
            if person is None:
                raise VoiceError("voice_person_not_found", "Person not found.", 404)
            if sample_id is None:
                del data["people"][person_id]
            else:
                before = len(person["samples"])
                person["samples"] = [s for s in person["samples"] if s["id"] != sample_id]
                if len(person["samples"]) == before:
                    raise VoiceError("voice_sample_not_found", "Sample not found.", 404)
                if not person["samples"]:
                    del data["people"][person_id]
            self._save(data)

    def merge(self, keep_id: str, drop_id: str, name: str | None = None) -> dict:
        """Move every sample of ``drop_id`` to ``keep_id`` and delete ``drop_id``, in one atomic write.
        ``name`` renames the merged person (default: ``keep_id``'s name)."""
        name = (name or "").strip()[:60]
        with self._lock:
            data = self._load()
            people = data["people"]
            if keep_id == drop_id:
                raise VoiceError("voice_merge_same", "Choose two different people.")
            if keep_id not in people or drop_id not in people:
                raise VoiceError("voice_person_not_found", "Person not found.", 404)
            if name and any(p["name"].casefold() == name.casefold() for k, p in people.items() if k not in (keep_id, drop_id)):
                raise VoiceError("voice_name_taken", "Another person already has this name.", 409)
            keep, drop = people[keep_id], people.pop(drop_id)
            have = {x["id"] for x in keep["samples"]}
            keep["samples"].extend(x for x in drop["samples"] if x["id"] not in have)
            created = [c for c in (keep.get("created"), drop.get("created")) if c]
            if created:
                keep["created"] = min(created)
            if name:
                keep["name"] = name
            self._save(data)
            return {"person_id": keep_id, "name": keep["name"], "samples": len(keep["samples"]),
                    "dropped_name": drop["name"]}

    def sample_ids(self) -> set[str]:
        with self._lock:
            return {x["id"] for p in self._load()["people"].values() for x in p["samples"]}

    def delete_samples_from(self, transcript_id: str) -> int:
        """Delete every sample taken from ``transcript_id`` (people left without samples go too)."""
        with self._lock:
            data = self._load()
            removed = 0
            for pid in list(data["people"]):
                person = data["people"][pid]
                keep = [x for x in person["samples"] if (x.get("source") or {}).get("transcript") != transcript_id]
                removed += len(person["samples"]) - len(keep)
                person["samples"] = keep
                if not keep:
                    del data["people"][pid]
            if removed:
                self._save(data)
            return removed

    def export(self) -> dict:
        with self._lock:
            return self._load()

    def delete_all(self) -> None:
        with self._lock:
            self.path.unlink(missing_ok=True)
            for tmp in self.path.parent.glob(self.path.name + ".*.tmp"):
                tmp.unlink(missing_ok=True)


def open_db(settings: "Settings") -> VoiceDB:
    return VoiceDB(db_path(settings))


# --- per-transcript voices.json -------------------------------------------------------------


def speech_seconds(segments) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in segments:
        if s.speaker:
            out[s.speaker] = out.get(s.speaker, 0.0) + max(0.0, s.end - s.start)
    return out


def transcript_lock(library: "Library", transcript_id: str) -> _PathLock:
    """Lock for read-modify-write of a transcript's voices.json (and the names derived from it)."""
    return file_lock(library.path(transcript_id, must_exist=False) / TRANSCRIPT_FILE)


_LIGHT: dict[str, tuple[tuple, dict]] = {}  # voices.json path -> (stamp, speakers without vectors)
_LIGHT_LOCK = threading.Lock()


def read_speakers_light(library: "Library", transcript_id: str) -> dict:
    """The ``speakers`` of a transcript's voices.json without the vectors, for READ-ONLY use (do not modify the
    result). Cached until the file changes (size, mtime): the voices page and the speaker tools look at every
    transcript of the archive, and re-opening thousands of files for each request dominated their time."""
    path = library.path(transcript_id) / TRANSCRIPT_FILE
    try:
        st = path.stat()
    except OSError:
        return {}
    key, stamp = str(path), (st.st_mtime_ns, st.st_size)
    with _LIGHT_LOCK:
        hit = _LIGHT.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    speakers = {label: {k: v for k, v in info.items() if k != "vector"}
                for label, info in read_transcript(library, transcript_id).get("speakers", {}).items()}
    with _LIGHT_LOCK:
        _LIGHT[key] = (stamp, speakers)
    return speakers


def read_transcript(library: "Library", transcript_id: str) -> dict:
    """The transcript's voices.json; a missing or damaged file counts as no voiceprints."""
    data = _read_json(library.path(transcript_id) / TRANSCRIPT_FILE, None)
    if not isinstance(data, dict) or not isinstance(data.get("speakers"), dict):
        return {}
    data["speakers"] = {label: info for label, info in data["speakers"].items()
                        if isinstance(info, dict) and isinstance(info.get("vector"), str)}
    for info in data["speakers"].values():
        if not isinstance(info.get("match"), (dict, type(None))):
            info["match"] = None
        if not isinstance(info.get("rejected"), list):
            info["rejected"] = []
    return data


def write_transcript(library: "Library", transcript_id: str, data: dict) -> None:
    _write_json(library.path(transcript_id, must_exist=False) / TRANSCRIPT_FILE, data)


def store_embeddings(library: "Library", transcript_id: str, embeddings: dict[str, list[float]],
                     speech: dict[str, float], model: str = MODEL) -> dict:
    speakers = {}
    for label, vector in embeddings.items():
        try:
            speakers[label] = {"vector": encode(vector), "speech_s": round(speech.get(label, 0.0), 1),
                               "match": None, "rejected": []}
        except VoiceError:
            log.warning("skipping invalid embedding for %s in %s", label, transcript_id)
    data = {"version": DB_VERSION, "model": model, "created": now_iso(), "speakers": speakers}
    with transcript_lock(library, transcript_id):
        write_transcript(library, transcript_id, data)
    return data


def _confirmed(info: dict, people: dict[str, str]) -> dict | None:
    """The match the user confirmed for a label, if that person still exists (name kept current)."""
    match = info.get("match") or {}
    pid = match.get("person_id")
    if match.get("confirmed") and pid in people:
        return {**match, "status": "auto", "name": people[pid]}
    return None


def identify_transcript(library: "Library", transcript_id: str, db: VoiceDB, apply_names: bool = True,
                        persist: bool = True, index: VoiceIndex | None = None) -> dict:
    """(Re)identify the speakers of a transcript against the current database. With ``persist``
    the matches are stored in its voices.json; with ``apply_names`` names with status ``auto``
    are added to speakers.json for labels that have no name yet (names the user typed are never
    replaced). Matches the user confirmed stay as long as the person exists. Returns the
    transcript's voice state (``public_state``). With ``index`` (an archive scan: one snapshot of the
    database for all transcripts) nothing else is read from the database and ``enrolled`` is not checked."""
    # a read-only look (persist=False) needs no lock: files are replaced atomically
    with transcript_lock(library, transcript_id) if persist else contextlib.nullcontext():
        if index is not None:
            index = db.index()  # taken under the lock: a merge finishing meanwhile cannot leave a stale person here
        data = read_transcript(library, transcript_id)
        speakers = data.get("speakers", {})
        if not speakers:
            return public_state(data)
        if data.get("model", MODEL) != db.model:
            for info in speakers.values():
                info["match"] = None
            if persist:
                write_transcript(library, transcript_id, data)
            return public_state(data, None if index else db)
        people = index.names if index else {p["id"]: p["name"] for p in db.list()}
        confirmed = {label: m for label, info in speakers.items() if (m := _confirmed(info, people))}
        inputs = {label: (decode(info["vector"]), info.get("speech_s")) for label, info in speakers.items()
                  if label not in confirmed}
        rejected = {label: info.get("rejected", []) for label, info in speakers.items()}
        matches = db.identify(inputs, rejected, index) if inputs else {}
        taken = {m["person_id"] for m in confirmed.values()}
        for label, match in matches.items():
            if match.status == "auto" and match.person_id in taken:  # 1:1 with the confirmed labels too
                match.status = "suggest"
            speakers[label]["match"] = match.to_dict()
        for label, match in confirmed.items():
            speakers[label]["match"] = match
        if not persist:
            return public_state(data, db)
        data["identified"] = now_iso()
        names = library.names(transcript_id) if apply_names else {}
        auto = {label: m.name for label, m in matches.items()
                if apply_names and m.status == "auto" and not names.get(label)}
        for label, name in auto.items():
            speakers[label]["auto_name"] = name  # reject() may take back only names set this way
        write_transcript(library, transcript_id, data)
        if auto:
            library.save_names(transcript_id, {**names, **auto})
        return public_state(data, None if index else db)


def public_state(data: dict, db: VoiceDB | None = None) -> dict:
    """What the API returns for a transcript: no vectors. With ``db``, ``enrolled`` is reported only
    while its sample still exists."""
    samples = db.sample_ids() if db is not None and data.get("speakers") else None

    def enrolled(info: dict):
        value = info.get("enrolled")
        if not isinstance(value, dict) or (samples is not None and value.get("sample_id") not in samples):
            return None
        return value

    return {
        "available": bool(data.get("speakers")),
        "model": data.get("model"),
        "speakers": {label: {"speech_s": info.get("speech_s"), "match": info.get("match"),
                             "rejected": info.get("rejected", []), "enrolled": enrolled(info)}
                     for label, info in data.get("speakers", {}).items()},
    }


def after_transcription(library: "Library", transcript_id: str, embeddings: dict | None, settings: "Settings") -> None:
    """Pipeline hook: store the speaker embeddings of a fresh transcript and identify them.
    Never raises; voice memory must not fail a transcription."""
    try:
        folder = library.path(transcript_id)
        with transcript_lock(library, transcript_id):
            if not embeddings or not enabled(settings):
                (folder / TRANSCRIPT_FILE).unlink(missing_ok=True)  # a re-transcription must not keep old voices
                return
            store_embeddings(library, transcript_id, embeddings, speech_seconds(library.segments(transcript_id)))
            identify_transcript(library, transcript_id, open_db(settings))
    except Exception:
        log.exception("voice memory failed for %s", transcript_id)


def _current_match(library: "Library", transcript_id: str, label: str, db: VoiceDB,
                   expect_person: str | None = None) -> tuple[dict, dict, dict]:
    identify_transcript(library, transcript_id, db, apply_names=False)  # the match the user saw is the current one
    data = read_transcript(library, transcript_id)
    info = data.get("speakers", {}).get(label)
    match = (info or {}).get("match") or {}
    if not match.get("person_id"):
        raise VoiceError("voice_no_suggestion", "There is no suggestion for this speaker.", 409)
    if expect_person and match["person_id"] != expect_person:  # bulk actions never confirm someone else
        raise VoiceError("voice_changed", "The suggestion has changed in the meantime.", 409)
    return data, info, match


def accept(library: "Library", transcript_id: str, label: str, db: VoiceDB, *, expect_person: str | None = None,
           add_sample: bool = True) -> dict:
    """The user confirms a suggestion: the label gets the name and, with enough speech, its voice
    becomes another sample. With less speech (or ``add_sample`` false) the label is only named and
    marked as confirmed. ``expect_person``: refuse when the current match is another person."""
    with transcript_lock(library, transcript_id):
        data, info, match = _current_match(library, transcript_id, label, db, expect_person)
        if (add_sample and (info.get("speech_s") or 0.0) >= MIN_ENROLL_S
                and not public_state(data, db)["speakers"][label]["enrolled"]):
            return enroll_label(library, transcript_id, label, db, name=match["name"], person_id=match["person_id"],
                                force=True)
        info["match"] = {**match, "status": "auto", "confirmed": True}
        info.pop("auto_name", None)
        write_transcript(library, transcript_id, data)
        library.save_names(transcript_id, {**library.names(transcript_id), label: match["name"]})
        return {"person_id": match["person_id"], "name": match["name"], "sample_id": None,
                "state": public_state(data, db)}


def reject(library: "Library", transcript_id: str, label: str, db: VoiceDB, *, expect_person: str | None = None) -> dict:
    """The user says the suggested (or automatic) person is wrong for this label."""
    with transcript_lock(library, transcript_id):
        data, info, match = _current_match(library, transcript_id, label, db, expect_person)
        info.setdefault("rejected", []).append(match["person_id"])
        info["match"] = {"status": "unknown", "person_id": None, "name": None, "score": None, "margin": None}
        auto_name = info.pop("auto_name", None)
        write_transcript(library, transcript_id, data)
        names = library.names(transcript_id)
        if auto_name and names.get(label) == auto_name:
            names.pop(label)  # a name set automatically, which the user says is wrong; typed names stay
            library.save_names(transcript_id, names)
        return public_state(data, db)


def enroll_label(library: "Library", transcript_id: str, label: str, db: VoiceDB, *, name: str = "",
                 person_id: str | None = None, force: bool = False) -> dict:
    """"Remember this voice as ...": add the label's embedding to a person and name the label.
    A label whose voice is already saved (and the sample still exists) is refused."""
    with transcript_lock(library, transcript_id):
        data = read_transcript(library, transcript_id)
        info = data.get("speakers", {}).get(label)
        if info is None:
            raise VoiceError("voice_not_available", "This transcript has no voiceprint for this speaker.", 409)
        if data.get("model", MODEL) != db.model:
            raise VoiceError("voice_model_mismatch", "The voiceprints of this transcript come from another model.", 409)
        if public_state(data, db)["speakers"][label]["enrolled"]:
            raise VoiceError("voice_already_enrolled", "This speaker's voice is already saved.", 409)
        meta_title = library.metadata(transcript_id).title
        result = db.enroll(name, decode(info["vector"]), speech_s=info.get("speech_s") or 0.0, person_id=person_id,
                           force=force, source={"transcript": transcript_id, "label": label, "title": meta_title})
        info["enrolled"] = {"person_id": result["person_id"], "sample_id": result["sample_id"]}
        info["match"] = {"status": "auto", "person_id": result["person_id"], "name": result["name"], "score": 1.0,
                         "margin": None, "confirmed": True}
        info.pop("auto_name", None)
        write_transcript(library, transcript_id, data)
        library.save_names(transcript_id, {**library.names(transcript_id), label: result["name"]})
        return {**result, "state": public_state(data, db)}


def delete_all(library: "Library", db: VoiceDB) -> int:
    """Delete the voice database and the voiceprints of every transcript folder (also folders
    without prepis.json and leftover temp files). Returns the number of voices.json files removed.
    Speaker names in speakers.json are kept (they are not biometric). Voice memory stays enabled."""
    db.delete_all()
    removed = 0
    if library.root.is_dir():
        for folder in (p for p in library.root.iterdir() if p.is_dir()):
            if not any(folder.glob(TRANSCRIPT_FILE + "*")):
                continue
            with file_lock(folder / TRANSCRIPT_FILE):
                for path in folder.glob(TRANSCRIPT_FILE + "*"):
                    if path.name.endswith(".lock"):
                        continue
                    removed += path.name == TRANSCRIPT_FILE
                    path.unlink(missing_ok=True)
    return removed


# --- the whole archive: scan, appearances, suggestions, merge -------------------------------

SCAN_TASK = "voice_scan"
BULK_SAMPLE_CAP = 10  # accepting many suggestions at once adds samples only while a person has fewer


def scan_archive(library: "Library", db: VoiceDB, *, person_id: str | None = None, progress=None) -> dict:
    """Identify again every transcript that has voiceprints and store the matches (auto names are
    added to empty labels, as after a transcription). ``person_id``: only transcripts where that
    person might speak (a label scoring from SUGGEST_THRESHOLD, or already matched to them), so the
    stored matches of everything else stay untouched. Embeddings are already computed, so this is
    CPU only. Returns counts; when ``progress(done, total, transcript_id, counts)`` raises (the job was
    cancelled) the scan stops and returns what was done so far with ``cancelled`` true."""
    index = db.index()
    if person_id is not None and person_id not in index.names:
        raise VoiceError("voice_person_not_found", "Person not found.", 404)
    ids = sorted(library.ids())
    out = {"transcripts": 0, "scanned": 0, "without_voiceprints": 0, "auto": 0, "suggest": 0, "errors": 0,
           "cancelled": False, "people": len(index.names)}
    for n, transcript_id in enumerate(ids):
        if progress:
            try:
                progress(n, len(ids), transcript_id, out)
            except Exception:  # the job runner cancels by raising from the progress callback
                out["cancelled"] = True
                return out
        try:
            data = read_transcript(library, transcript_id)
            speakers = data.get("speakers", {})
            out["transcripts"] += 1
            if not speakers:
                out["without_voiceprints"] += 1
                continue
            index = db.index()  # picks up changes the user made while the scan runs
            if person_id is not None:
                relevant = any((info.get("match") or {}).get("person_id") == person_id
                               or index.scores(decode(info["vector"]), person_id).get(person_id, 0.0) >= SUGGEST_THRESHOLD
                               for info in speakers.values())
                if not relevant:
                    continue
            state = identify_transcript(library, transcript_id, db, index=index)
            out["scanned"] += 1
            for info in state["speakers"].values():
                status = (info.get("match") or {}).get("status")
                if status in ("auto", "suggest"):
                    out[status] += 1
        except (VoiceError, OSError, ValueError, KeyError, TypeError, LookupError):
            log.exception("voice scan failed for %s", transcript_id)
            out["errors"] += 1
    if progress:
        try:
            progress(len(ids), len(ids), None, out)
        except Exception:
            out["cancelled"] = True
    return out


def _display_status(info: dict, match: dict, name: str | None, person_name: str) -> str:
    """confirmed (the user said so) | auto | suggest. An automatic match that the user's own name for
    the label contradicts is only a suggestion."""
    if match.get("confirmed") or isinstance(info.get("enrolled"), dict):
        return "confirmed"
    if match.get("status") == "auto" and (not name or name == person_name):
        return "auto"
    return "suggest"


def _person_hits(speakers: dict, names: dict, person_id: str, person_name: str,
                 include_suggest: bool) -> dict[str, tuple]:
    """Labels of one transcript that are ``person_id``: ``{label: (status, score, speech_s)}``."""
    wanted = person_name.casefold()
    hits: dict[str, tuple] = {}
    for label, info in speakers.items():
        match = info.get("match") or {}
        enrolled = info.get("enrolled")
        if match.get("person_id") != person_id and not (
                isinstance(enrolled, dict) and enrolled.get("person_id") == person_id):
            continue
        status = _display_status(info, match, names.get(label), person_name)
        if status != "suggest" or include_suggest:
            hits[label] = (status, match.get("score"), info.get("speech_s"))
    for label, name in names.items():
        other = (speakers.get(label, {}).get("match") or {}).get("person_id")
        if label not in hits and name.casefold() == wanted and other in (None, person_id):
            hits[label] = ("manual", None, None)
    return hits


def appearances(library: "Library", db: VoiceDB, person_id: str, *, include_suggest: bool = True) -> list[dict]:
    """Every transcript where ``person_id`` speaks, newest first: ``{transcript_id, title, label,
    speech_s, status, score, date, audio, name}`` (``name``: what the label is called now, if anything). ``status``: confirmed | auto | suggest (only with
    ``include_suggest``) | manual (the user typed this person's name for the label)."""
    index = db.index()
    if person_id not in index.names:
        raise VoiceError("voice_person_not_found", "Person not found.", 404)
    person_name = index.names[person_id]
    rows = []
    for transcript_id in sorted(library.ids()):
        try:
            names = library.names_cached(transcript_id)
            hits = _person_hits(read_speakers_light(library, transcript_id), names, person_id, person_name, include_suggest)
            if not hits:
                continue
            summary = library.summary(transcript_id)
        except (OSError, ValueError, KeyError, TypeError, LookupError):
            continue
        for label, (status, score, speech_s) in hits.items():
            if speech_s is None:
                speech_s = round(summary["speakers"].get(label, 0.0), 1) or None
            rows.append({"transcript_id": transcript_id, "title": summary["title"], "label": label,
                         "speech_s": speech_s, "status": status, "score": score, "date": summary["date"],
                         "audio": summary["audio"], "name": names.get(label)})
    return sorted(rows, key=lambda r: (-(r["date"] or 0), r["transcript_id"], r["label"]))


def appearance_stats(library: "Library", db: VoiceDB) -> dict[str, dict]:
    """Person id -> ``{transcripts, seconds, suggested_transcripts, suggested_seconds}`` over the archive,
    by the same rules as ``appearances``: ``transcripts``/``seconds`` count every sure appearance (confirmed,
    auto or typed name: the rows ``find_speaker`` lists by default), the ``suggested_*`` ones only the
    unconfirmed suggestions (never added to the first two). ``seconds`` is the speech of the person's labels."""
    index = db.index()
    stats = {pid: {"transcripts": 0, "seconds": 0.0, "suggested_transcripts": 0, "suggested_seconds": 0.0}
             for pid in index.names}
    for transcript_id in library.ids():
        try:
            speakers, names = read_speakers_light(library, transcript_id), library.names_cached(transcript_id)
            found = {pid: hits for pid, name in index.names.items()
                     if (hits := _person_hits(speakers, names, pid, name, True))}
            if not found:
                continue
            spoken = library.summary(transcript_id)["speakers"]
        except (OSError, ValueError, KeyError, TypeError, LookupError):
            continue
        for pid, hits in found.items():
            for kind, wanted in (("", lambda st: st != "suggest"), ("suggested_", lambda st: st == "suggest")):
                chosen = [(label, secs) for label, (status, _score, secs) in hits.items() if wanted(status)]
                if not chosen:
                    continue
                stats[pid][kind + "transcripts"] += 1
                stats[pid][kind + "seconds"] += sum(secs if secs is not None else spoken.get(label, 0.0)
                                                    for label, secs in chosen)
    for row in stats.values():
        row["seconds"] = round(row["seconds"], 1)
        row["suggested_seconds"] = round(row["suggested_seconds"], 1)
    return stats


def appearance_counts(library: "Library", db: VoiceDB) -> dict[str, int]:
    """Person id -> number of transcripts where they speak (confirmed, auto or typed name; no suggestions)."""
    return {pid: row["transcripts"] for pid, row in appearance_stats(library, db).items() if row["transcripts"]}


def suggestions(library: "Library", db: VoiceDB, person_id: str, *, limit: int | None = None) -> list[dict]:
    """Matches to ``person_id`` across the archive that the user has not confirmed yet (suggestions and
    automatic names), best score first, each with an excerpt of the label's speech to listen to:
    ``snippet`` and ``start``/``end`` (5 s)."""
    rows = [r for r in appearances(library, db, person_id) if r["status"] in ("suggest", "auto")]
    rows.sort(key=lambda r: (-(r["score"] or 0), r["transcript_id"], r["label"]))
    if limit:
        rows = rows[:limit]
    cache: dict[str, list] = {}
    for row in rows:
        tid = row["transcript_id"]
        if tid not in cache:
            try:
                cache[tid] = library.segments(tid)
            except (OSError, ValueError, LookupError):
                cache[tid] = []
        own = [s for s in cache[tid] if s.speaker == row["label"] and s.text.strip()]
        pick = next((s for s in own if s.end - s.start >= 4.0), max(own, key=lambda s: s.end - s.start, default=None))
        if pick:
            row.update(start=round(pick.start, 2), end=round(pick.start + 5.0, 2), snippet=pick.text.strip()[:240])
        else:
            row.update(start=None, end=None, snippet="")
    return rows


def _rewire(library: "Library", old_ids: set[str], old_names: set[str], keep_id: str, name: str) -> int:
    """In every transcript's voices.json: matches, enrolments and rejections of ``old_ids`` point to
    ``keep_id``; a name that came from the person (equal to one of ``old_names``, on a label that
    belongs to them) becomes ``name``. Names the user typed for other labels stay. Returns the number
    of transcripts changed."""
    changed = 0
    for transcript_id in sorted(library.ids()):
        folder = library.path(transcript_id, must_exist=False)
        try:
            text = (folder / TRANSCRIPT_FILE).read_text(encoding="utf-8", errors="replace")
        except OSError:  # missing, or being replaced right now (then the second pass sees it)
            text = ""
            if (folder / TRANSCRIPT_FILE).exists():
                text = " ".join(old_ids)  # unknown: look under the lock
            else:
                continue
        if not any(x in text for x in (*old_ids, *old_names)):  # a quick look without the lock (a name can remain after its id went)
            continue
        with transcript_lock(library, transcript_id):
            data = read_transcript(library, transcript_id)
            names = library.names(transcript_id)
            dirty = False
            renamed: dict[str, str] = {}
            for label, info in data.get("speakers", {}).items():
                mine = False
                match = info.get("match")
                if isinstance(match, dict) and match.get("person_id") in old_ids:
                    match["person_id"], match["name"] = keep_id, name
                    mine = dirty = True
                enrolled = info.get("enrolled")
                if isinstance(enrolled, dict) and enrolled.get("person_id") in old_ids:
                    enrolled["person_id"] = keep_id
                    mine = dirty = True
                rejected = info.get("rejected") or []
                if any(r in old_ids for r in rejected):
                    info["rejected"] = list(dict.fromkeys(keep_id if r in old_ids else r for r in rejected))
                    dirty = True
                if info.get("auto_name") in old_names:  # set automatically for the old person, whatever the match says now
                    info["auto_name"] = name
                    mine = dirty = True
                if mine and names.get(label) in old_names:
                    renamed[label] = name
            if dirty:
                write_transcript(library, transcript_id, data)
                changed += 1
            if renamed:
                library.save_names(transcript_id, {**names, **renamed})
    return changed


def _referenced_names(library: "Library", person_id: str) -> set[str] | None:
    """Names that voices.json files carry for ``person_id`` (match name, automatic name); None when no
    transcript refers to the person any more."""
    names: set[str] = set()
    found = False
    for transcript_id in library.ids():
        try:
            speakers = read_speakers_light(library, transcript_id)
        except (OSError, ValueError, KeyError, TypeError, LookupError):
            continue
        for info in speakers.values():
            match, enrolled = info.get("match"), info.get("enrolled")
            if (isinstance(match, dict) and match.get("person_id") == person_id) or (
                    isinstance(enrolled, dict) and enrolled.get("person_id") == person_id):
                found = True
                names.update(n for n in ((match or {}).get("name"), info.get("auto_name")) if isinstance(n, str))
            elif person_id in (info.get("rejected") or []):
                found = True
    return names if found else None


def merge_people(library: "Library", db: VoiceDB, keep_id: str, drop_id: str, name: str | None = None) -> dict:
    """Merge person ``drop_id`` into ``keep_id``: samples, matches, enrolments and the names that came
    from the person (automatic or confirmed) in transcripts all go to ``keep_id``, named ``name``
    (default: their own name). Order: transcripts, then the database (one atomic write), then the
    transcripts once more (a scan running meanwhile may have matched the old person again). It is not
    one transaction, but it can be finished: when the database step is done and the interruption came
    later, calling it again (same two ids) rewires what still points to the removed person. Locks as
    everywhere: transcript, then database."""
    index = db.index()
    if keep_id == drop_id:
        raise VoiceError("voice_merge_same", "Choose two different people.")
    if keep_id in index.names and drop_id not in index.names:
        # the database step of an interrupted merge is done: finish the transcripts that still point
        # to the removed person (their names are only known from the transcripts themselves)
        old_names = _referenced_names(library, drop_id)
        touched = _rewire(library, {drop_id}, old_names, keep_id, index.names[keep_id]) if old_names is not None else 0
        if not touched:
            raise VoiceError("voice_person_not_found", "Person not found.", 404)
        return {"person_id": keep_id, "name": index.names[keep_id], "samples": index.sample_count(keep_id),
                "transcripts": touched, "resumed": True}
    if keep_id not in index.names or drop_id not in index.names:
        raise VoiceError("voice_person_not_found", "Person not found.", 404)
    final = (name or "").strip()[:60] or index.names[keep_id]
    if any(n.casefold() == final.casefold() for pid, n in index.names.items() if pid not in (keep_id, drop_id)):
        raise VoiceError("voice_name_taken", "Another person already has this name.", 409)
    old_names = {index.names[drop_id], index.names[keep_id]} - {final}
    touched = _rewire(library, {drop_id, keep_id}, old_names, keep_id, final)
    result = db.merge(keep_id, drop_id, final)
    touched += _rewire(library, {drop_id}, old_names, keep_id, final)
    return {**result, "transcripts": touched}


def scan_task(request, settings: "Settings", progress: "ProgressCallback", library: "Library") -> str:
    """Job ``voice_scan``: ``request.source`` is ``all`` or ``person:<id>``. Needs no GPU."""
    from .engines.base import EngineError
    from .progress import Phase, Progress

    source = request.source or "all"
    person_id = source.split(":", 1)[1] if source.startswith("person:") else None
    if not enabled(settings):
        raise EngineError("Voice memory is turned off.")

    def summary(r: dict) -> str:
        if not r.get("people"):
            return (f"Voice scan: {r['transcripts']} transcripts checked. Voice memory is empty, remember a voice first "
                    "(open a transcript, name a speaker, choose \"Remember voice\"), then scan again.")
        return (f"Voice scan{' (cancelled)' if r['cancelled'] else ''}: {r['scanned']} transcripts checked, "
                f"{r['auto']} automatic, {r['suggest']} suggestions, {r['without_voiceprints']} without voiceprints, "
                f"{r['errors']} errors")

    def report(done: int, total: int, transcript_id: str | None, counts: dict) -> None:
        # a log line now and then: a killed worker (cancel) leaves the last counts in the job log
        message = summary(counts) if done and done % 25 == 0 else None
        progress(Progress(Phase.WRITING, 100 * done / max(total, 1), message=message, data={"title": "Voices"}))

    try:
        result = scan_archive(library, open_db(settings), person_id=person_id, progress=report)
    except VoiceError as e:
        raise EngineError(e.message) from None
    try:
        progress(Progress(Phase.WRITING, 100, message=summary(result)))
    except Exception:  # cancelled while finishing
        pass
    return SCAN_TASK


# --- recomputing voiceprints of older transcripts (GPU) -------------------------------------


def label_spans(segments, min_s: float = 2.0, max_s: float = 10.0) -> dict[str, list[tuple[float, float, float]]]:
    """Per label: (start, end, weight) spans of uninterrupted speech, from word speakers where
    available (a segment can contain a speaker change), cut to ``max_s``."""
    spans: dict[str, list] = {}
    for seg in segments:
        runs: list[list] = []
        timed = [w for w in seg.words if w.start is not None and w.end is not None and w.speaker]
        if timed:
            for w in timed:
                if runs and runs[-1][0] == w.speaker and w.start - runs[-1][2] < 1.0:
                    runs[-1][2] = w.end
                else:
                    runs.append([w.speaker, w.start, w.end])
        elif seg.speaker:
            runs.append([seg.speaker, seg.start, seg.end])
        for speaker, start, end in runs:
            if end - start >= min_s:
                spans.setdefault(speaker, []).append((start, min(end, start + max_s), min(end - start, max_s)))
    return spans


def robust_centroid(vectors: np.ndarray, weights: np.ndarray, trim: float = 0.2) -> np.ndarray:
    """Weighted mean of normalised vectors without the ``trim`` share farthest from it."""
    e = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    c = (e * weights[:, None]).sum(0)
    c /= np.linalg.norm(c)
    if len(e) >= 5:
        keep = np.argsort(e @ c)[int(len(e) * trim):]
        c = (e[keep] * weights[keep, None]).sum(0)
        c /= np.linalg.norm(c)
    return c


def compute_embeddings(audio_path: Path, segments, settings: "Settings", progress: "ProgressCallback") -> dict[str, list[float]]:
    """Speaker embeddings of an existing transcript, from its own speaker labels: the
    diarization model's embedding network on each label's speech spans (no new clustering,
    so labels stay the same). Same embedding space as the diarization centroids."""
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    from .progress import Phase, Progress

    spans = label_spans(segments)
    if not spans:
        return {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    progress(Progress(Phase.LOADING, message="Loading the speaker model..."))
    embedder = DiarizationPipeline(model_name=MODEL, token=settings.hf_token, device=device).model._embedding
    audio = whisperx.load_audio(str(audio_path))
    total = sum(len(v) for v in spans.values())
    done = 0
    out = {}
    for label, items in spans.items():
        vecs, weights = [], []
        for start, end, weight in items:
            piece = audio[int(start * 16000):int(end * 16000)]
            if len(piece) < 16000:
                continue
            vecs.append(embedder(torch.from_numpy(piece[None, None, :].astype(np.float32)))[0])
            weights.append(weight)
            done += 1
            progress(Progress(Phase.DIARIZE, 100 * done / total))
        vecs = [v for v in vecs if np.all(np.isfinite(v))]
        if vecs:
            out[label] = robust_centroid(np.stack(vecs), np.array(weights[:len(vecs)])).tolist()
    del embedder
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def compute_task(request, settings: "Settings", progress: "ProgressCallback", library: "Library") -> str:
    """Job: compute voiceprints for a transcript that has none (made before voice memory, or by
    an engine without embeddings), then identify its speakers. Needs the audio file."""
    from .engines.base import EngineError
    from .progress import Phase, Progress

    transcript_id = request.source
    if not settings.hf_token:
        raise EngineError("HF_TOKEN is missing in .env; the speaker model needs it.")
    audio = library.audio_file(transcript_id)
    if audio is None:
        source = library.metadata(transcript_id).source
        audio = Path(source) if source and Path(source).is_file() else None
    if audio is None:
        raise EngineError("The transcript has no audio file; voiceprints cannot be computed.")
    segments = library.segments(transcript_id)
    embeddings = compute_embeddings(audio, segments, settings, progress)
    if not embeddings:
        raise EngineError("No speaker has enough speech (at least 2 s in one piece) for a voiceprint.")
    progress(Progress(Phase.WRITING, message="Identifying speakers..."))
    with transcript_lock(library, transcript_id):
        store_embeddings(library, transcript_id, embeddings, speech_seconds(segments))
        identify_transcript(library, transcript_id, open_db(settings))
    progress(Progress(Phase.WRITING, 100, data={"result": transcript_id}))
    return transcript_id
