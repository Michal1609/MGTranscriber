import json
import subprocess
import sys
import threading
import time
from pathlib import Path
import pytest

from mgtranscriber import fsutil
from mgtranscriber.jobs import InProcessRunner, JobQueue, SubprocessRunner, queue as queue_module
from mgtranscriber.library import Library
from mgtranscriber.pipeline import TranscriptionRequest, run
from tests.conftest import wait_for


def test_pipeline_writes_library_entry(settings, media_file):
    events = []
    transcript_id = run(TranscriptionRequest(str(media_file("talk.wav"))), settings, events.append)
    assert transcript_id.startswith("talk-") and len(transcript_id) == len("talk-") + 8
    folder = settings.output_dir / transcript_id
    assert {"prepis.json", "prepis.txt", "titulky.srt", "metadata.json"} <= {p.name for p in folder.iterdir()}
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["title"] == "talk"
    assert meta["engine"] == "mock" and meta["model"] == "mock-1" and meta["language"] == "en"
    assert meta["speakers"] == ["SPEAKER_00", "SPEAKER_01"] and meta["duration"] == 5.0
    assert any(e.data and e.data.get("title") == "talk" for e in events)
    assert {"loading", "transcribe", "writing"} <= {e.phase for e in events}
    # the same file again maps to the same transcript; another file with the same name does not
    assert run(TranscriptionRequest(str(media_file("talk.wav"))), settings) == transcript_id


def test_same_name_in_two_folders_gives_two_transcripts(settings, tmp_path):
    ids = set()
    for folder in ("a", "b"):
        path = tmp_path / folder / "talk.wav"
        path.parent.mkdir()
        path.write_bytes(b"RIFF")
        ids.add(run(TranscriptionRequest(str(path)), settings))
    assert len(ids) == 2
    assert {Library(settings.output_dir).metadata(i).title for i in ids} == {"talk"}


def test_queue_runs_job_in_process(settings, media_file):
    queue = JobQueue(settings, runner=InProcessRunner())
    job = queue.add(TranscriptionRequest(str(media_file("a.wav"))))
    assert queue.wait(job, timeout=20)
    assert job.status == "done" and job.title == "a"
    assert Library(settings.output_dir).summary(job.result)["engine"] == "mock"
    queue.stop()


def test_queue_reports_engine_error(settings, media_file):
    queue = JobQueue(settings, runner=InProcessRunner())
    job = queue.add(TranscriptionRequest(str(media_file("fail.wav"))))
    assert queue.wait(job, timeout=20)
    assert job.status == "error" and "mock failure" in job.error
    queue.stop()


def test_queue_persists_and_requeues_after_restart(settings, media_file):
    queue = JobQueue(settings, autostart=False)
    first = queue.add(TranscriptionRequest(str(media_file("one.wav")), language="cs", speakers=2))
    second = queue.add(TranscriptionRequest(str(media_file("two.wav"))))
    queue.cancel(second)
    # simulate a crash while the first job was running
    first.status, first.attempts = "running", 1
    queue._save()

    restarted = JobQueue(settings, autostart=False)
    again = restarted.get(first.id)
    assert again.status == "queued" and again.request.speakers == 2 and again.request.language == "cs"
    assert restarted.get(second.id).status == "cancelled" and restarted.get(second.id).done.is_set()
    assert restarted.pending == [first.id]
    assert restarted.add(TranscriptionRequest("x")).id == 3


def test_job_that_keeps_killing_the_server_is_not_retried(settings, media_file):
    queue = JobQueue(settings, autostart=False)
    job = queue.add(TranscriptionRequest(str(media_file("one.wav"))))
    job.status, job.attempts = "running", queue_module.MAX_ATTEMPTS
    queue._save()
    restarted = JobQueue(settings, autostart=False)
    again = restarted.get(job.id)
    assert again.status == "interrupted" and again.error and again.done.is_set()
    assert restarted.pending == []


def test_logs_go_to_files_and_survive_restart(settings, media_file):
    queue = JobQueue(settings, runner=InProcessRunner())
    job = queue.add(TranscriptionRequest(str(media_file("logged.wav"))))
    assert queue.wait(job, timeout=20)
    queue.stop()
    stored = json.loads(settings.jobs_file.read_text(encoding="utf-8"))
    assert "log" not in stored["jobs"][0]
    assert "Mock transcribing..." in job.log_file.read_text(encoding="utf-8")
    restarted = JobQueue(settings, autostart=False)
    assert "Mock transcribing..." in restarted.get(job.id).log_since(0)


def test_old_store_with_inline_log_is_migrated(settings):
    settings.data_dir.mkdir(parents=True)
    settings.jobs_file.write_text(json.dumps({"version": 1, "jobs": [{
        "id": 7, "request": {"source": "x"}, "status": "done", "log": ["one", "two"], "log_dropped": 0}]}))
    queue = JobQueue(settings, autostart=False)
    assert queue.get(7).log_since(0) == ["one", "two"]
    queue._save()
    assert JobQueue(settings, autostart=False).get(7).log_since(0) == ["one", "two"]


def test_failing_save_does_not_stop_the_queue(settings, media_file, monkeypatch):
    def locked(*_):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(queue_module, "replace_file", locked)  # jobs.json stays locked for good
    queue = JobQueue(settings, runner=InProcessRunner())
    first = queue.add(TranscriptionRequest(str(media_file("a.wav"))))  # does not raise
    assert queue.wait(first, timeout=20) and first.status == "done"
    second = queue.add(TranscriptionRequest(str(media_file("b.wav"))))
    assert queue.wait(second, timeout=20) and second.status == "done"
    queue.stop()


def test_save_retries_a_briefly_locked_file(settings, monkeypatch):
    real, calls = fsutil.os.replace, []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(5, "Access is denied")
        real(src, dst)

    monkeypatch.setattr(fsutil.os, "replace", flaky)
    queue = JobQueue(settings, autostart=False)
    queue.add(TranscriptionRequest("x"))
    assert len(calls) == 3 and settings.jobs_file.is_file()


def test_slow_hook_does_not_block_the_queue(settings, media_file):
    release = threading.Event()
    queue = JobQueue(settings, runner=InProcessRunner())
    queue.on_finished.append(lambda job: release.wait(10))
    first = queue.add(TranscriptionRequest(str(media_file("a.wav"))))
    second = queue.add(TranscriptionRequest(str(media_file("b.wav"))))
    assert queue.wait(first, timeout=20) and queue.wait(second, timeout=20)
    release.set()
    queue.stop()



def test_hooks_run_in_order_one_at_a_time(settings, media_file):
    calls, running = [], []

    def hook(name):
        def run(job):
            running.append(name)
            assert len(running) == 1  # never two hooks at once
            calls.append((name, job.id))
            time.sleep(0.02)
            running.remove(name)
        return run

    queue = JobQueue(settings, runner=InProcessRunner())
    queue.on_finished += [hook("a"), hook("b")]
    jobs = [queue.add(TranscriptionRequest(str(media_file(f"{i}.wav")))) for i in range(2)]
    wait_for(lambda: len(calls) == 4)
    assert calls == [("a", jobs[0].id), ("b", jobs[0].id), ("a", jobs[1].id), ("b", jobs[1].id)]
    queue.stop()


HANGING_HOOK_SCRIPT = """
import sys, threading, tempfile
from pathlib import Path
import tests.conftest
from mgtranscriber.config import PROJECT_ROOT, Settings
from mgtranscriber.jobs import InProcessRunner, JobQueue
from mgtranscriber.pipeline import TranscriptionRequest

tmp = Path(tempfile.mkdtemp())
(tmp / "a.wav").write_bytes(b"RIFF fake")
settings = Settings(root=PROJECT_ROOT, output_dir=tmp / "output", data_dir=tmp / "data", engine="mock",
                    model="", language="en")
queue = JobQueue(settings, runner=InProcessRunner())
started = threading.Event()
queue.on_finished.append(lambda job: (started.set(), threading.Event().wait()))  # never returns
job = queue.add(TranscriptionRequest(str(tmp / "a.wav")))
assert queue.wait(job, 30) and started.wait(30)
print("hook running", flush=True)
"""


def test_hanging_hook_does_not_block_interpreter_exit():
    # a slow on_finished hook (an HTTP call) must not hold up shutdown: hook threads are daemons
    done = subprocess.run([sys.executable, "-c", HANGING_HOOK_SCRIPT], capture_output=True, text=True,
                          timeout=60, cwd=str(Path(__file__).resolve().parent.parent))
    assert done.returncode == 0, done.stderr
    assert "hook running" in done.stdout

def test_failing_event_handler_does_not_hang_the_worker(settings, media_file):
    seen = []

    def broken(event):
        seen.append(event)
        raise RuntimeError("handler bug")

    execution = SubprocessRunner().start(TranscriptionRequest(str(media_file("x.wav"))), settings, broken)
    result = []
    waiter = threading.Thread(target=lambda: result.append(execution.wait()), daemon=True)
    waiter.start()
    waiter.join(60)
    assert result == [0]
    assert any(e.get("event") == "done" for e in seen)


def test_subprocess_runner_end_to_end(settings, media_file):
    queue = JobQueue(settings, runner=SubprocessRunner())
    job = queue.add(TranscriptionRequest(str(media_file("sub.wav"))))
    assert queue.wait(job, timeout=60)
    assert job.status == "done", job.log
    assert job.title == "sub"
    assert "Mock transcribing..." in job.log
    queue.stop()


def test_subprocess_cancel(settings, media_file):
    queue = JobQueue(settings, runner=SubprocessRunner())
    job = queue.add(TranscriptionRequest(str(media_file("slow.wav"))))
    wait_for(lambda: job.phase == "transcribe", timeout=60)
    queue.cancel(job)
    assert queue.wait(job, timeout=30)
    assert job.status == "cancelled" and job.error is None
    queue.stop()


def test_stop_kills_the_worker_and_requeues_the_job(settings, media_file):
    queue = JobQueue(settings, runner=SubprocessRunner())
    job = queue.add(TranscriptionRequest(str(media_file("slow.wav"))))
    wait_for(lambda: job.phase == "transcribe", timeout=60)
    proc = queue.execution.proc
    queue.stop()
    assert proc.poll() is not None
    assert job.status == "queued" and job.error is None and job.attempts == 0
    restarted = JobQueue(settings, autostart=False)
    assert restarted.pending == [job.id]


def test_subprocess_reports_worker_error(settings, media_file):
    queue = JobQueue(settings, runner=SubprocessRunner())
    job = queue.add(TranscriptionRequest(str(media_file("fail.wav"))))
    assert queue.wait(job, timeout=60)
    assert job.status == "error" and "mock failure" in job.error
    queue.stop()


PARENT_SCRIPT = """
import sys, time
sys.path.insert(0, sys.argv[1])
import tests.conftest
from mgtranscriber.config import Settings
from mgtranscriber.jobs import SubprocessRunner
from mgtranscriber.pipeline import TranscriptionRequest
settings = Settings.from_dict(__import__("json").loads(sys.argv[2]))
ex = SubprocessRunner().start(TranscriptionRequest(sys.argv[3]), settings, lambda e: None)
print(ex.proc.pid, flush=True)
time.sleep(120)
"""


def process_alive(pid: int) -> bool:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32")
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel32.CloseHandle(handle)
    return code.value == 259  # STILL_ACTIVE


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
def test_worker_dies_when_the_server_process_is_killed(settings, media_file):
    parent = subprocess.Popen(
        [sys.executable, "-c", PARENT_SCRIPT, str(settings.root), json.dumps(settings.to_dict()),
         str(media_file("silent.wav"))], stdout=subprocess.PIPE, text=True)
    worker_pid = int(parent.stdout.readline())
    time.sleep(3)  # let the worker reach the silent part
    assert process_alive(worker_pid)
    parent.kill()  # TerminateProcess: no cleanup code runs, like a crash
    parent.wait()
    wait_for(lambda: not process_alive(worker_pid), timeout=5)


@pytest.mark.parametrize("runner", [InProcessRunner, SubprocessRunner])
def test_registered_task_runs_through_the_queue(settings, legacy_transcript, runner):
    from mgtranscriber import tasks

    tasks._register_from_env()  # conftest set the env var after mgtranscriber.tasks may have been imported
    queue = JobQueue(settings, runner=runner())
    job = queue.add(TranscriptionRequest(legacy_transcript, task="mock-task"))
    assert queue.wait(job, timeout=60)
    assert job.status == "done" and job.result == legacy_transcript, job.log
    assert queue.public(job)["task"] == "mock-task"
    assert Library(settings.output_dir).metadata(legacy_transcript).extra["mock_task"] is True
    queue.stop()


def test_unknown_task_fails_cleanly(settings):
    queue = JobQueue(settings, runner=InProcessRunner())
    job = queue.add(TranscriptionRequest("x", task="nope"))
    assert queue.wait(job, timeout=20)
    assert job.status == "error" and "Unknown task" in job.error
    queue.stop()


def test_new_job_does_not_append_to_an_old_log(settings, media_file):
    """If jobs.json is lost, ids restart at 1 while job-logs/1.log may still exist."""
    log_dir = settings.data_dir / "job-logs"
    log_dir.mkdir(parents=True)
    (log_dir / "1.log").write_text("old job line\n", encoding="utf-8")
    queue = JobQueue(settings, runner=InProcessRunner())
    job = queue.add(TranscriptionRequest(str(media_file("a.wav"))))
    assert job.id == 1 and queue.wait(job, timeout=20)
    assert "old job line" not in job.log_file.read_text(encoding="utf-8")
    queue.stop()


def test_locked_log_file_does_not_break_saving(settings, monkeypatch):
    import pathlib

    queue = JobQueue(settings, autostart=False)
    for i in range(3):
        job = queue.add(TranscriptionRequest(f"x{i}"))
        queue.cancel(job)
    real_unlink = pathlib.Path.unlink

    def locked(self, missing_ok=False):
        if self.suffix == ".log":
            raise PermissionError(5, "Access is denied")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(queue_module, "KEEP_FINISHED", 1)
    monkeypatch.setattr(pathlib.Path, "unlink", locked)
    queue._save()  # prunes two finished jobs; their logs cannot be deleted
    assert len(queue.jobs) == 1
    assert json.loads(settings.jobs_file.read_text(encoding="utf-8"))["jobs"][0]["id"] == 3


class StuckExecution:
    """A worker that ignores cancel() for a while (e.g. stuck in a CUDA call)."""

    def __init__(self):
        self.released = threading.Event()

    def wait(self):
        self.released.wait(30)
        return 1

    def cancel(self):
        pass


class StuckRunner:
    def __init__(self):
        self.execution = StuckExecution()

    def start(self, request, settings, on_event):
        on_event({"event": "progress", "phase": "transcribe", "percent": 10})
        return self.execution


def test_stop_timeout_still_requeues_without_counting_an_attempt(settings):
    runner = StuckRunner()
    queue = JobQueue(settings, runner=runner)
    job = queue.add(TranscriptionRequest("x"))
    wait_for(lambda: job.phase == "transcribe")
    queue.stop(timeout=0.3)
    stored = json.loads(settings.jobs_file.read_text(encoding="utf-8"))["jobs"][0]
    assert stored["status"] == "queued" and stored["attempts"] == 0
    runner.execution.released.set()  # the worker finally exits; nothing is counted twice
    wait_for(lambda: not queue._thread.is_alive())
    assert job.status == "queued" and job.attempts == 0
    assert JobQueue(settings, autostart=False).pending == [job.id]
