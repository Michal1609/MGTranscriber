from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..pipeline import TranscriptionRequest
from ..progress import Phase

ACTIVE = {"queued", "running"}
# "interrupted": the server died while the job ran, too often to try again
FINISHED = {"done", "error", "cancelled", "interrupted"}
MAX_LOG_LINES = 2000  # lines kept in memory; the full log is in the job's log file


@dataclass(eq=False)
class Job:
    id: int
    request: TranscriptionRequest
    status: str = "queued"  # queued | running | done | error | cancelled | interrupted
    phase: str = Phase.QUEUED
    percent: float | None = None  # progress within the phase
    title: str | None = None
    result: str | None = None  # transcript id in the library
    error: str | None = None
    log: list[str] = field(default_factory=list)
    log_dropped: int = 0  # lines trimmed from the start of ``log``; line numbers stay stable
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    attempts: int = 0  # times a worker was started for this job
    cancel_requested: bool = False
    # its input lives only as long as the request that made it (an upload): never queued again after a
    # restart or stop, because the file is gone and nobody waits for the result
    ephemeral: bool = False
    # the input is a web upload (``<data>/uploads/<random>/<name>``, set by the queue when the job is created):
    # clients see ``uploads/<name>``, never the absolute server path
    upload: bool = False
    # outcome of the form's "Translate to": {language, job} when queued, {language, error} when it could not be
    translation: dict | None = None
    error_code: str | None = None  # stable code of a user-facing error (translated by the UI), e.g. empty_transcript
    no_speech: bool = False  # the transcription found no speech: the transcript is kept, but empty
    log_file: Path | None = None  # append-only full log, set by the queue
    done: threading.Event = field(default_factory=threading.Event, repr=False)  # set once the job finished
    _log_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_log(self, line: str) -> None:
        with self._log_lock:
            self.log.append(line)
            if len(self.log) > MAX_LOG_LINES:
                extra = len(self.log) - MAX_LOG_LINES
                del self.log[:extra]
                self.log_dropped += extra
            if self.log_file:
                try:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(line.replace("\n", " ") + "\n")
                except OSError:
                    pass  # the in-memory log still has the line

    def log_since(self, line_number: int) -> list[str]:
        """Lines from absolute line number ``line_number`` on."""
        with self._log_lock:
            return self.log[max(line_number - self.log_dropped, 0):]

    @property
    def log_lines(self) -> int:
        with self._log_lock:
            return self.log_dropped + len(self.log)

    def load_log(self) -> None:
        """Fill the in-memory tail from ``log_file`` (after a restart)."""
        try:
            lines = self.log_file.read_text(encoding="utf-8").splitlines() if self.log_file else []
        except OSError:
            return
        if lines:
            with self._log_lock:
                self.log = lines[-MAX_LOG_LINES:]
                self.log_dropped = len(lines) - len(self.log)

    @property
    def shown_source(self) -> str:
        """The source as clients and logs show it."""
        source = self.request.source
        return f"uploads/{Path(source).name}" if self.upload else source

    def public(self, position: int | None = None) -> dict:
        """The JSON the API and SSE stream send. ``url``/``no_diarize`` are kept for older clients."""
        r = self.request
        source = self.shown_source
        return {
            "id": self.id,
            "url": source,
            "source": source,
            "upload": self.upload,
            "language": r.language,
            "speakers": r.speakers,
            "no_diarize": r.diarize is False,
            "separate_speakers": r.separate_speakers,
            "engine": r.engine,
            "model": r.model,
            "task": r.task,
            "folder": r.folder,
            "translate_to": r.translate_to,
            "translation": self.translation,
            "no_speech": self.no_speech,
            "status": self.status,
            "phase": self.phase,
            "percent": self.percent,
            "title": self.title,
            "result": self.result,
            "error": self.error,
            "error_code": self.error_code,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "position": position,
            "log_lines": self.log_lines,
        }

    def to_dict(self) -> dict:
        """Stored form; the log lives in its own file."""
        return {
            "id": self.id, "request": self.request.to_dict(), "status": self.status, "phase": self.phase,
            "percent": self.percent, "title": self.title, "result": self.result, "error": self.error,
            "created": self.created, "started": self.started, "finished": self.finished, "attempts": self.attempts,
            **({"ephemeral": True} if self.ephemeral else {}),
            **({"upload": True} if self.upload else {}),
            **({"translation": self.translation} if self.translation else {}),
            **({"no_speech": True} if self.no_speech else {}),
            **({"error_code": self.error_code} if self.error_code else {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        job = cls(
            id=data["id"], request=TranscriptionRequest.from_dict(data["request"]), status=data["status"],
            phase=data.get("phase", data["status"]), percent=data.get("percent"), title=data.get("title"),
            result=data.get("result"), error=data.get("error"), created=data.get("created", time.time()),
            started=data.get("started"), finished=data.get("finished"), attempts=data.get("attempts", 0),
            ephemeral=data.get("ephemeral", False), upload=bool(data.get("upload", False)), translation=data.get("translation"),
            no_speech=bool(data.get("no_speech", False)), error_code=data.get("error_code"),
            # stores written before log files existed carry the log inline
            log=list(data.get("log", [])), log_dropped=data.get("log_dropped", 0),
        )
        if job.status in FINISHED:
            job.done.set()
        return job
