"""Persistent job queue. One job runs at a time (the GPU is the bottleneck), others wait.

State is saved to ``settings.jobs_file`` (small: no logs) and each job's log is appended
to its own file in ``job-logs/`` next to it, so the queue survives a server restart:

- queued jobs stay queued;
- a job interrupted by a clean shutdown (``stop()``) is queued again;
- a job that was running when the server died is queued again, unless that already
  happened ``MAX_ATTEMPTS`` times (it probably kills the server); it is then "interrupted";
- an ephemeral job (``add(..., ephemeral=True)``, e.g. an upload to /v1 whose file is
  deleted with the request) is never resumed: ``stop()`` cancels it, and after a crash
  it is "interrupted".
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..fsutil import replace_file
from ..pipeline import TranscriptionRequest
from ..progress import Phase
from .models import ACTIVE, FINISHED, Job
from .runners import Execution, Runner, SubprocessRunner

log = logging.getLogger(__name__)

KEEP_FINISHED = 100  # finished jobs kept in history
SAVE_INTERVAL = 2.0  # seconds between saves caused by progress events alone
MAX_ATTEMPTS = 2  # worker starts per job before a crash-restart gives up
STOP_TIMEOUT = 15.0  # seconds stop() waits for the running worker to go away
RECENT_FAILURE_S = 24 * 3600  # "failed recently" in summary()
# tabs of the Jobs page; ``status`` filters accept these names and the raw statuses (comma separated)
GROUPS = {"active": ACTIVE, "done": {"done"}, "failed": FINISHED - {"done"}, "finished": FINISHED}
RETRYABLE = FINISHED - {"done"}


class JobQueue:
    def __init__(self, settings: Settings, runner: Runner | None = None, store: Path | None = None, autostart: bool = True):
        self.settings = settings
        self.runner = runner or SubprocessRunner()
        self.store = store or settings.jobs_file
        self.log_dir = self.store.parent / "job-logs"
        self.jobs: dict[int, Job] = {}
        self.pending: list[int] = []
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.next_id = 1
        self.execution: Execution | None = None
        self.running: Job | None = None
        # hooks for features (search index, post-processing); run one at a time on a daemon thread, so they
        # never block the queue and a slow hook (an HTTP call) cannot hold up interpreter exit or Ctrl+C
        self.on_finished: list[Callable[[Job], None]] = []
        self._hooks: queue.SimpleQueue[Job | None] = queue.SimpleQueue()
        self._hook_thread: threading.Thread | None = None
        self._last_save = 0.0
        self._stages: dict[int, Path] = {}  # job id -> staging folder its worker reported (library.begin_run)
        self._stop = False
        self._thread: threading.Thread | None = None
        self._load()
        if autostart:
            self.start()

    # --- public API ----------------------------------------------------------------------

    def start(self) -> None:
        if not self._thread:
            self._thread = threading.Thread(target=self._loop, name="job-queue", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = STOP_TIMEOUT) -> None:
        """Stop the queue; a running job is killed and queued again for the next start."""
        with self.lock:
            self._stop = True
            execution = self.execution
        self.wake.set()
        if execution:
            execution.cancel()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        with self.lock:
            if self.running and self.running.status == "running":  # the worker did not exit in time
                self._requeue_after_stop(self.running)
            for job_id in [i for i in self.pending if self.jobs[i].ephemeral]:  # whoever waits gets an answer
                self.pending.remove(job_id)
                self._end_ephemeral(self.jobs[job_id])
            self._save()
        self._hooks.put(None)  # the hook thread ends after the hooks already queued

    def add(self, request: TranscriptionRequest, ephemeral: bool = False) -> Job:
        """Queue a job. ``ephemeral``: its input disappears with the caller, so it is never resumed."""
        with self.lock:
            job = Job(self.next_id, request, log_file=self._log_path(self.next_id), ephemeral=ephemeral,
                      upload=self._is_upload(request.source))
            self.next_id += 1
            try:  # a lost jobs.json restarts ids at 1: never append to an old job's log
                job.log_file.unlink(missing_ok=True)
            except OSError:
                log.warning("could not delete %s", job.log_file)
            self.jobs[job.id] = job
            self.pending.append(job.id)
            self._save()
        self.wake.set()
        return job

    def get(self, job_id: int) -> Job | None:
        return self.jobs.get(job_id)

    def wait(self, job: Job, timeout: float | None = None) -> bool:
        """Block until ``job`` finished; False on timeout."""
        return job.done.wait(timeout)

    def position(self, job: Job) -> int | None:
        with self.lock:
            return self.pending.index(job.id) + 1 if job.id in self.pending else None

    def public(self, job: Job) -> dict:
        return job.public(self.position(job))

    def listing(self) -> list[dict]:
        with self.lock:
            return [self.public(j) for j in sorted(self.jobs.values(), key=lambda j: -j.id)]

    @staticmethod
    def statuses(status: str | None) -> set[str] | None:
        """Statuses selected by a filter ("active", "failed", "done,error", ...); None = all; ValueError if unknown."""
        names = {n.strip() for n in (status or "").split(",") if n.strip()}
        if not names or "all" in names:
            return None
        known = {s for group in GROUPS.values() for s in group}
        unknown = names - GROUPS.keys() - known
        if unknown:
            raise ValueError(f"unknown status: {', '.join(sorted(unknown))}")
        return set().union(*(GROUPS.get(n, {n}) for n in names))

    def page(self, status: str | None = None, limit: int | None = None, offset: int = 0) -> dict:
        """One page of jobs: ``{"items", "total", "limit", "offset", "counts"}``. Active jobs: running first, then
        in queue order; the rest newest first."""
        wanted = self.statuses(status)
        with self.lock:
            jobs = list(self.jobs.values())
            counts = {"all": len(jobs), **{g: sum(j.status in s for j in jobs) for g, s in GROUPS.items()}}
            if wanted is not None:
                jobs = [j for j in jobs if j.status in wanted]
            if wanted is not None and wanted <= ACTIVE:
                order = {job_id: i for i, job_id in enumerate(self.pending)}
                jobs.sort(key=lambda j: (j.status != "running", order.get(j.id, 0), j.id))
            else:
                jobs.sort(key=lambda j: -j.id)
            chosen = jobs[offset:] if limit is None else jobs[offset:offset + limit]
            return {"items": [self.public(j) for j in chosen], "total": len(jobs), "limit": limit, "offset": offset,
                    "counts": counts}

    def summary(self) -> dict:
        """What the home page shows: the running job, how many wait, how many failed lately."""
        since = time.time() - RECENT_FAILURE_S
        with self.lock:
            running = self.running if self.running and self.running.status == "running" else None
            failed = sum(1 for j in self.jobs.values()
                         if j.status in ("error", "interrupted") and (j.finished or j.created) > since)
            return {"running": self.public(running) if running else None, "queued": len(self.pending),
                    "active": len(self.pending) + (1 if running else 0), "failed_recent": failed}

    def remove(self, job: Job) -> bool:
        """Drop a finished job from the history (its transcript stays). False when it is still active."""
        with self.lock:
            if job.status not in FINISHED or self.jobs.get(job.id) is not job:
                return False
            del self.jobs[job.id]
            self._drop_log(job)
            self._save()
            return True

    def clear(self, status: str | None = None) -> int:
        """Remove every finished job (or those in ``status``; active ones are never removed); returns the count."""
        wanted = self.statuses(status) or FINISHED
        with self.lock:
            gone = [j for j in self.jobs.values() if j.status in wanted and j.status in FINISHED]
            for job in gone:
                del self.jobs[job.id]
                self._drop_log(job)
            if gone:
                self._save()
            return len(gone)

    def _drop_log(self, job: Job) -> None:
        try:
            if job.log_file:
                job.log_file.unlink(missing_ok=True)
        except OSError:
            log.warning("could not delete %s", job.log_file)

    def cancel(self, job: Job) -> None:
        with self.lock:
            if job.status == "queued":
                self.pending.remove(job.id)
                job.status = job.phase = "cancelled"
                job.finished = time.time()
                job.done.set()
                self._save()
                return
            if job.status != "running":
                return
            job.cancel_requested = True
            execution = self.execution
        if execution:
            execution.cancel()

    # --- worker --------------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop:
            try:
                self._step()
            except Exception:  # never let the queue thread die
                log.exception("job queue step failed")
                time.sleep(1)

    def _step(self) -> None:
        self.wake.wait()
        with self.lock:
            if self._stop:
                return
            if not self.pending:
                self.wake.clear()
                return
            job = self.jobs[self.pending.pop(0)]
            job.status, job.phase, job.percent, job.started = "running", Phase.STARTING, None, time.time()
            job.error, job.attempts = None, job.attempts + 1
            self.running = job
            self._save()
        try:
            self._run(job)
        except Exception as e:
            job.status, job.error = "error", f"Server error: {e}"
        self._discard_stage(job)
        with self.lock:
            self.execution = self.running = None
            if self._stop and job.status in ("running", "queued"):  # stopped by stop(): resume on the next start
                if job.status == "running":  # "queued": stop() already did it after its timeout
                    self._requeue_after_stop(job)
                return
            job.finished = time.time()
            self._save()
        job.done.set()
        if self.on_finished:
            self._submit_hooks(job)

    def active_stages(self) -> "set[Path]":
        """Resolved staging folders of the job that runs now (a sweep must not touch them)."""
        with self.lock:
            job = self.running
            stage = self._stages.get(job.id) if job else None
        try:
            return {stage.resolve()} if stage else set()
        except OSError:
            return set()

    def _discard_stage(self, job: Job) -> None:
        """The worker is gone. A job that did not finish (cancelled, failed, stopped: the worker was killed and
        could not clean up) leaves its staging folder; remove it and give back what an interrupted commit moved."""
        with self.lock:
            stage = self._stages.pop(job.id, None)
        if stage is not None and job.status != "done":
            from ..library import discard_stage

            discard_stage(stage, self.settings.output_dir)

    def _requeue_after_stop(self, job: Job) -> None:
        if job.ephemeral:
            self._end_ephemeral(job)
            return
        job.status, job.phase, job.percent, job.started = "queued", Phase.QUEUED, None, None
        job.attempts = max(job.attempts - 1, 0)  # a clean shutdown is not the job's fault
        job.add_log("[server stopped, job will run again after restart]")
        self._save()

    def _submit_hooks(self, job: Job) -> None:
        self._hooks.put(job)
        if not self._hook_thread:
            self._hook_thread = threading.Thread(target=self._hook_loop, name="job-hooks", daemon=True)
            self._hook_thread.start()

    def _hook_loop(self) -> None:
        while (job := self._hooks.get()) is not None:
            for hook in list(self.on_finished):
                self._call_hook(hook, job)

    def _end_ephemeral(self, job: Job) -> None:
        job.status = job.phase = "cancelled"
        job.finished = time.time()
        job.add_log("[server stopped, job cancelled]")
        self._save()
        job.done.set()

    @staticmethod
    def _call_hook(hook: Callable[[Job], None], job: Job) -> None:
        try:
            hook(job)
        except Exception as e:
            log.exception("on_finished hook failed")
            job.add_log(f"[hook failed: {e}]")

    def _run(self, job: Job) -> None:
        job.add_log(f"> {job.shown_source}")
        execution = self.runner.start(job.request, self.settings, lambda event: self._on_event(job, event))
        with self.lock:
            self.execution = execution
            stop_now = job.cancel_requested or self._stop  # cancelled between dequeue and start
        if stop_now:
            execution.cancel()
        code = execution.wait()

        if job.cancel_requested:
            job.status = job.phase = "cancelled"
            job.error = None
            job.add_log("[cancelled]")
        elif job.result and job.error is None:
            job.status = job.phase = "done"
            job.percent = None
            job.title = job.title or job.result
        elif self._stop:
            pass  # stays "running"; _step queues it again
        else:
            job.status = job.phase = "error"
            job.error = job.error or f"Worker exited with code {code}."
            job.add_log(f"[error, exit code {code}]")

    def _on_event(self, job: Job, event: dict) -> None:
        kind = event.get("event")
        if kind == "error" and self._stop:
            return  # the worker was killed by stop()
        with self.lock:
            phase_changed = False
            if kind == "progress":
                if event.get("phase") and event["phase"] != job.phase:
                    job.phase, phase_changed = event["phase"], True
                job.percent = event.get("percent")
                if event.get("message"):
                    job.add_log(event["message"])
                data = event.get("data") or {}
                job.title = data.get("title", job.title)
                if data.get("stage"):
                    self._stages[job.id] = Path(data["stage"])
            elif kind == "log":
                job.add_log(event.get("line", ""))
                return  # log lines go to the log file only
            elif kind == "done":
                job.result = event.get("result")
            elif kind == "error":
                job.error = event.get("message") or "Unknown error."
                job.add_log(f"Error: {job.error}")
            if phase_changed or kind in {"done", "error"} or time.time() - self._last_save > SAVE_INTERVAL:
                self._save()

    # --- persistence ---------------------------------------------------------------------

    def _is_upload(self, source: str) -> bool:
        """True for a file of ours under ``<data>/uploads/<random>/`` (not any folder that happens to be named so)."""
        path = Path(source)
        return path.parent.parent == self.settings.data_dir / "uploads"

    def _log_path(self, job_id: int) -> Path:
        return self.log_dir / f"{job_id}.log"

    def _load(self) -> None:
        try:
            data = json.loads(self.store.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for item in data.get("jobs", []):
            try:
                job = Job.from_dict(item)
            except (KeyError, TypeError):
                continue
            job.log_file = self._log_path(job.id)
            if "upload" not in item:  # store written before the flag existed
                job.upload = self._is_upload(job.request.source)
            if job.log and not job.log_file.exists():  # version 1 store: move the inline log to its file
                self._write_log_file(job)
            elif not job.log:
                job.load_log()
            if job.ephemeral and job.status in ACTIVE:  # its input went away with the server
                job.status = job.phase = "interrupted"
                job.error = "The server stopped before this job finished; send the request again."
                job.finished = job.finished or time.time()
                job.done.set()
                job.add_log("[server restarted, job interrupted]")
            elif job.status == "running":  # the server died while this job ran
                if job.attempts >= MAX_ATTEMPTS:
                    job.status = job.phase = "interrupted"
                    job.error = "The server stopped during this job repeatedly; not trying again."
                    job.finished = job.finished or time.time()
                    job.done.set()
                    job.add_log("[server restarted, job interrupted]")
                else:
                    job.status, job.phase, job.percent, job.started = "queued", Phase.QUEUED, None, None
                    job.add_log("[server restarted, job queued again]")
            self.jobs[job.id] = job
        self.pending = sorted(j.id for j in self.jobs.values() if j.status == "queued")
        self.next_id = max(self.jobs, default=0) + 1
        self.wake.set()

    def _write_log_file(self, job: Job) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            job.log_file.write_text("".join(line + "\n" for line in job.log), encoding="utf-8")
        except OSError:
            log.exception("could not write %s", job.log_file)

    def save(self) -> None:
        """Persist the job list after a hook changed a finished job."""
        self._save()

    def _save(self) -> None:
        """Write the job list. Never raises: a failed save must not stop the queue."""
        with self.lock:
            finished = sorted((j for j in self.jobs.values() if j.status not in ACTIVE), key=lambda j: -j.id)
            for old in finished[KEEP_FINISHED:]:
                del self.jobs[old.id]
                self._drop_log(old)
            data = {"version": 2, "jobs": [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: j.id)]}
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.store.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                replace_file(tmp, self.store)
            except OSError:
                log.exception("could not save %s", self.store)
            self._last_save = time.time()
