"""How a job is executed. The queue talks to runners only through events.

``SubprocessRunner`` is the production runner (one process per job, frees VRAM).
``InProcessRunner`` runs the pipeline in a thread; it is meant for tests and debugging.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
from typing import Callable, Protocol

from .. import tasks
from ..config import Settings
from ..engines import EngineError
from ..pipeline import TranscriptionRequest
from . import winjob

log = logging.getLogger(__name__)

# event dicts as described in worker.py, plus {"event": "log", "line": "..."}
EventSink = Callable[[dict], None]


class Cancelled(Exception):
    pass


class Execution(Protocol):
    def wait(self) -> int: ...
    def cancel(self) -> None: ...


class Runner(Protocol):
    def start(self, request: TranscriptionRequest, settings: Settings, on_event: EventSink) -> Execution: ...


# --- raw log filter (drops progress-bar noise) ---------------------------------------

PROGRESS_BAR = re.compile(r"\d+%\s*\||^\s*\d+(\.\d+)?%\s*$")
WARNING_START = re.compile(r"\.py:\d+: \w*Warning:")
WARNING_END = re.compile(r"^  \S")  # source line printed after a warning text


class LogFilter:
    """Drops progress bars, Python warnings and torchcodec noise from the raw log."""

    def __init__(self):
        self.in_warning = False
        self.warning_lines = 0

    def keep(self, line: str) -> bool:
        if self.in_warning:
            self.warning_lines += 1
            if WARNING_END.match(line) or self.warning_lines > 60:
                self.in_warning = False
            return False
        if WARNING_START.search(line):
            self.in_warning, self.warning_lines = True, 0
            return False
        return bool(line.strip()) and not PROGRESS_BAR.search(line) and "torchcodec" not in line.lower()


# --- subprocess -----------------------------------------------------------------------------

WORKER_MODULE = __package__ + ".worker"


class SubprocessExecution:
    def __init__(self, request: TranscriptionRequest, settings: Settings, on_event: EventSink):
        payload = {"request": request.to_dict(), "settings": settings.to_dict()}
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        extra = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {"start_new_session": True}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", WORKER_MODULE], cwd=settings.root, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", **extra,  # universal newlines: \r splits tqdm bars
        )
        winjob.assign(self.proc)  # dies with the server, even when the server crashes
        self.on_event = on_event
        self.got_result = False
        self.cancelled = False
        self.last_exception: str | None = None
        self.proc.stdin.write(json.dumps(payload))
        self.proc.stdin.close()
        self.threads = [threading.Thread(target=self._read_events, daemon=True),
                        threading.Thread(target=self._read_log, daemon=True)]
        for t in self.threads:
            t.start()

    def _emit(self, event: dict) -> None:
        # A failing consumer must not stop the readers: an undrained pipe would block the worker.
        try:
            self.on_event(event)
        except Exception:
            log.exception("job event handler failed")

    def _read_events(self) -> None:
        for line in self.proc.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                event = None
            if not isinstance(event, dict):
                event = {"event": "log", "line": line.rstrip("\n")}
            if event.get("event") in {"done", "error"}:
                self.got_result = True
            self._emit(event)

    def _read_log(self) -> None:
        filt = LogFilter()
        for raw in self.proc.stderr:
            line = raw.rstrip("\n")
            if re.match(r"^[\w.]+(Error|Exception): .+", line):
                self.last_exception = line
            if filt.keep(line):
                self._emit({"event": "log", "line": line})

    def wait(self) -> int:
        code = self.proc.wait()
        for t in self.threads:
            t.join()
        if code != 0 and not self.got_result and not self.cancelled:
            self._emit({"event": "error", "message": self.last_exception or f"Worker exited with code {code}."})
        return code

    def cancel(self) -> None:
        self.cancelled = True
        if self.proc.poll() is not None:
            return
        if sys.platform == "win32":  # whole tree: the worker may have children such as ffmpeg
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
        else:
            os.killpg(self.proc.pid, signal.SIGKILL)


class SubprocessRunner:
    def start(self, request: TranscriptionRequest, settings: Settings, on_event: EventSink) -> Execution:
        return SubprocessExecution(request, settings, on_event)


# --- in-process -----------------------------------------------------------------------------


class InProcessExecution:
    def __init__(self, request: TranscriptionRequest, settings: Settings, on_event: EventSink):
        self.cancelled = threading.Event()
        self.code = 0

        def progress(p):
            if self.cancelled.is_set():
                raise Cancelled()
            on_event({"event": "progress", **p.to_dict()})

        def target():
            try:
                result = tasks.execute(request, settings, progress)
                on_event({"event": "done", "result": result})
            except Cancelled:
                self.code = 1
            except (EngineError, FileNotFoundError) as e:
                self.code = 1
                on_event({"event": "error", "message": str(e)})
            except Exception as e:
                self.code = 1
                on_event({"event": "error", "message": f"{type(e).__name__}: {e}"})

        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def wait(self) -> int:
        self.thread.join()
        return self.code

    def cancel(self) -> None:
        self.cancelled.set()


class InProcessRunner:
    def start(self, request: TranscriptionRequest, settings: Settings, on_event: EventSink) -> Execution:
        return InProcessExecution(request, settings, on_event)
