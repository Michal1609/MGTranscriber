"""Staging folders (.run-*) of killed runs are removed: right after a job is cancelled/fails, and by a sweep."""

import os
import threading
import time

from mgtranscriber.jobs import JobQueue
from mgtranscriber.library import MANIFEST, STAGE_OLD, STAGE_PREFIX, Library, discard_stage
from mgtranscriber.pipeline import TranscriptionRequest
from tests.conftest import wait_for


def make_stage(output, name="talk-1", transcript=False, age_s=0.0, audio=b"x" * 100):
    folder = output / name
    stage = folder / f"{STAGE_PREFIX}abc123"
    stage.mkdir(parents=True)
    (stage / "audio.webm").write_bytes(audio)
    if transcript:
        (folder / "prepis.json").write_text("[]", encoding="utf-8")
    if age_s:
        old = time.time() - age_s
        for p in (stage / "audio.webm", stage, folder):
            os.utime(p, (old, old))
    return folder, stage


class KilledWorkerRunner:
    """A worker that reports its stage, then is killed by cancel() without any chance to clean up."""

    def __init__(self, output, name="talk-1", transcript=False):
        self.output, self.name, self.transcript = output, name, transcript
        self.stage = None
        self.started = threading.Event()

    def start(self, request, settings, on_event):
        runner = self
        folder, stage = make_stage(self.output, self.name, self.transcript)
        self.stage = stage

        class Execution:
            def __init__(self):
                self.gone = threading.Event()

            def wait(self):
                self.gone.wait(20)
                return 1

            def cancel(self):
                self.gone.set()

        on_event({"event": "progress", "phase": "loading", "data": {"stage": str(stage)}})
        runner.started.set()
        return Execution()


def run_and_cancel(settings, runner):
    queue = JobQueue(settings, runner=runner)
    job = queue.add(TranscriptionRequest("whatever"))
    assert runner.started.wait(10)
    queue.cancel(job)
    assert queue.wait(job, timeout=10)
    wait_for(lambda: job.status == "cancelled")
    queue.stop()
    return job


def test_a_cancelled_job_leaves_no_stage_and_no_empty_folder(settings):
    settings.output_dir.mkdir(parents=True)
    runner = KilledWorkerRunner(settings.output_dir)
    job = run_and_cancel(settings, runner)
    assert job.status == "cancelled"
    assert not runner.stage.exists()
    assert not (settings.output_dir / "talk-1").exists()  # the folder held nothing else


def test_a_cancelled_rerun_keeps_the_existing_transcript(settings):
    settings.output_dir.mkdir(parents=True)
    runner = KilledWorkerRunner(settings.output_dir, transcript=True)
    run_and_cancel(settings, runner)
    assert not runner.stage.exists()
    assert (settings.output_dir / "talk-1" / "prepis.json").is_file()


def test_discard_restores_what_an_interrupted_commit_moved(settings):
    settings.output_dir.mkdir(parents=True)
    folder, stage = make_stage(settings.output_dir, transcript=True)
    (stage / STAGE_OLD).mkdir()
    (stage / STAGE_OLD / "audio.webm").write_bytes(b"old audio")
    (stage / STAGE_OLD / MANIFEST).write_text('{"created": [], "files": ["audio.webm"]}', encoding="utf-8")
    assert discard_stage(stage, settings.output_dir)
    assert not stage.exists()
    assert (folder / "audio.webm").read_bytes() == b"old audio"


def test_discard_never_leaves_the_library(settings, tmp_path):
    settings.output_dir.mkdir(parents=True)
    outside = tmp_path / "elsewhere" / "talk" / f"{STAGE_PREFIX}1"
    outside.mkdir(parents=True)
    assert not discard_stage(outside, settings.output_dir)
    assert outside.exists()
    other = settings.output_dir / "talk"
    other.mkdir()
    assert not discard_stage(other, settings.output_dir)  # not a .run-* folder
    assert other.exists()


def test_startup_sweep_removes_old_stages_and_keeps_live_ones(settings):
    out = settings.output_dir
    out.mkdir(parents=True)
    _, orphan = make_stage(out, "orphan-1", age_s=7200)  # nothing else in its folder
    keep_folder, with_transcript = make_stage(out, "done-2", transcript=True, age_s=7200)
    _, live = make_stage(out, "live-3")  # a run in progress: changed just now
    _, running = make_stage(out, "running-4", age_s=7200)  # old, but the queue says it is running
    removed = Library(out).sweep_stages(protect={running.resolve()})
    assert removed == 2
    assert not orphan.exists() and not orphan.parent.exists()
    assert not with_transcript.exists() and (keep_folder / "prepis.json").is_file()
    assert live.is_dir() and (live / "audio.webm").is_file()
    assert running.is_dir()


def test_the_queue_reports_the_stage_of_its_running_job(settings):
    settings.output_dir.mkdir(parents=True)
    runner = KilledWorkerRunner(settings.output_dir)
    queue = JobQueue(settings, runner=runner)
    job = queue.add(TranscriptionRequest("whatever"))
    assert runner.started.wait(10)
    wait_for(lambda: queue.active_stages())
    assert queue.active_stages() == {runner.stage.resolve()}
    queue.cancel(job)
    assert queue.wait(job, timeout=10)
    queue.stop()


# --- review fixes: half-written snapshot copies, links in the library, heartbeat ---------------------------------

def committed_run(settings, content=b'[{"text": "original transcript"}]'):
    """A transcript with a run whose commit_run has finished (old/ holds the snapshot)."""
    from mgtranscriber.library import Run

    folder, stage = make_stage(settings.output_dir, transcript=True)
    (folder / "prepis.json").write_bytes(content)
    run = Run(folder=folder, stage=stage, existed=True)
    Library(settings.output_dir).commit_run(run, snapshot=["prepis.json"])
    return folder, stage, content


def test_commit_copies_through_a_temp_name_so_old_is_absent_or_complete(settings, monkeypatch):
    import shutil


    settings.output_dir.mkdir(parents=True)
    folder, stage = make_stage(settings.output_dir, transcript=True)
    original = b'[{"text": "original transcript"}]'
    (folder / "prepis.json").write_bytes(original)

    def killed_halfway(src, dst, *a, **k):  # the process dies after writing 10 bytes of the copy
        with open(dst, "wb") as f:
            f.write(original[:10])
        raise SystemExit("killed")

    from mgtranscriber.library import Run
    monkeypatch.setattr(shutil, "copy2", killed_halfway)
    try:
        Library(settings.output_dir).commit_run(Run(folder=folder, stage=stage, existed=True), snapshot=["prepis.json"])
    except SystemExit:
        pass
    assert not (stage / STAGE_OLD / "prepis.json").exists()  # only prepis.json.part is cut short
    assert discard_stage(stage, settings.output_dir)
    assert (folder / "prepis.json").read_bytes() == original  # the intact original survived the restore


def test_restore_does_not_trust_a_truncated_copy_from_an_older_version(settings):
    folder, stage, original = committed_run(settings)
    old = stage / STAGE_OLD
    (old / "prepis.json").write_bytes(original[:10])  # what an older version left behind
    assert discard_stage(stage, settings.output_dir)
    assert (folder / "prepis.json").read_bytes() == original


def test_restore_puts_back_a_complete_copy_and_checks_sizes_without_json(settings):
    import json as _json

    folder, stage, original = committed_run(settings)
    manifest = stage / STAGE_OLD / MANIFEST
    data = _json.loads(manifest.read_text(encoding="utf-8"))
    assert data["sizes"]["prepis.json"] == len(original)
    (folder / "prepis.json").write_bytes(b'[{"text": "rewritten by the new run"}]')
    assert discard_stage(stage, settings.output_dir)
    assert (folder / "prepis.json").read_bytes() == original
    # an older manifest (no sizes) with a complete JSON copy is restored too
    folder3 = settings.output_dir / "legacy-1"
    stage3 = folder3 / f"{STAGE_PREFIX}zzz"
    (stage3 / STAGE_OLD).mkdir(parents=True)
    (folder3 / "prepis.json").write_text('{"new": 1}', encoding="utf-8")
    (stage3 / STAGE_OLD / "prepis.json").write_text('[{"old": 1}]', encoding="utf-8")
    (stage3 / STAGE_OLD / MANIFEST).write_text('{"created": [], "files": ["prepis.json"]}', encoding="utf-8")
    assert discard_stage(stage3, settings.output_dir)
    assert (folder3 / "prepis.json").read_text(encoding="utf-8") == '[{"old": 1}]'


def test_the_sweep_does_not_follow_a_junction_or_symlink(settings, tmp_path):
    import subprocess

    out = settings.output_dir
    out.mkdir(parents=True)
    target = tmp_path / "elsewhere"
    victim = target / f"{STAGE_PREFIX}1"
    victim.mkdir(parents=True)
    (victim / "x.bin").write_bytes(b"x")
    old = time.time() - 7200
    for p in (victim / "x.bin", victim, target):
        os.utime(p, (old, old))
    link = out / "linked"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        if done.returncode != 0:
            import pytest
            pytest.skip("cannot create a symlink or junction here")
    assert Library(out).sweep_stages() == 0
    assert victim.is_dir() and (victim / "x.bin").is_file()  # the target is untouched
    assert link.exists()  # and the link itself was not removed


def test_the_run_keeps_its_stage_fresh_while_it_is_silent(settings, monkeypatch):
    from mgtranscriber import pipeline
    from mgtranscriber.library import Run
    import threading

    monkeypatch.setattr(pipeline, "HEARTBEAT_S", 0.05)
    settings.output_dir.mkdir(parents=True)
    folder, stage = make_stage(settings.output_dir, age_s=7200)
    before = stage.stat().st_mtime
    stop = threading.Event()
    t = threading.Thread(target=pipeline._heartbeat, args=(Run(folder=folder, stage=stage, existed=False), stop), daemon=True)
    t.start()
    wait_for(lambda: stage.stat().st_mtime > before + 3600)
    stop.set()
    t.join(2)
    assert not t.is_alive()
