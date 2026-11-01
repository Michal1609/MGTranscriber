"""Transcribing a source again must not damage the existing transcript until the new run has succeeded."""

import os
import time

import pytest

from mgtranscriber import pipeline
from mgtranscriber.library import STAGE_PREFIX, Library
from mgtranscriber.pipeline import TranscriptionRequest, run
from mgtranscriber.sources import Media
from tests.mock_engine import MockEngine

URL = "https://www.youtube.com/watch?v=abc123"


class Cancelled(BaseException):
    """What a cancelled job looks like to the pipeline: not an Exception."""


@pytest.fixture
def fake_url(monkeypatch):
    """A URL source whose 'download' writes audio.webm with the content set in ``state['audio']``."""
    state = {"audio": b"first audio"}

    def probe(source):
        return Media(source, "url", "A talk", info={"id": "abc123", "extractor_key": "Youtube"})

    def fetch_audio(media, out_dir, progress=None):
        media.audio_path = out_dir / "audio.webm"
        media.audio_path.write_bytes(state["audio"])
        return media.audio_path

    monkeypatch.setattr(pipeline, "probe", probe)
    monkeypatch.setattr(pipeline, "fetch_audio", fetch_audio)
    return state


def stages(folder):
    return list(folder.glob(STAGE_PREFIX + "*"))


def seed(settings, fake_url):
    """A finished transcript with the things the user added later."""
    tid = run(TranscriptionRequest(URL), settings)
    folder = settings.output_dir / tid
    (folder / "speakers.json").write_text('{"SPEAKER_00": "Alice"}', encoding="utf-8")
    (folder / "voices.json").write_text('{"version": 1}', encoding="utf-8")
    (folder / "analysis.json").write_text('{"summary": "kept"}', encoding="utf-8")
    return tid, folder


def snapshot(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}


@pytest.mark.parametrize("error", [RuntimeError("engine failed"), Cancelled()])
def test_failed_or_cancelled_rerun_leaves_the_old_transcript_untouched(settings, fake_url, monkeypatch, error):
    tid, folder = seed(settings, fake_url)
    before = snapshot(folder)
    seen = {}

    def boom(self, media, options, progress):
        seen["old_audio"] = (folder / "audio.webm").read_bytes()  # still the old one while the new run works
        seen["staged"] = media.audio_path.read_bytes()
        raise error

    fake_url["audio"] = b"second audio"
    monkeypatch.setattr(MockEngine, "transcribe", boom)
    with pytest.raises(type(error)):
        run(TranscriptionRequest(URL), settings)
    assert seen == {"old_audio": b"first audio", "staged": b"second audio"}
    assert snapshot(folder) == before  # audio, text, names, voices, analysis, metadata: byte for byte
    assert not stages(folder)
    assert not (folder / "versions").exists()
    assert Library(settings.output_dir).audio_file(tid).read_bytes() == b"first audio"


def test_successful_rerun_replaces_audio_and_drops_stale_files_only(settings, fake_url):
    tid, folder = seed(settings, fake_url)
    fake_url["audio"] = b"second audio"
    assert run(TranscriptionRequest(URL), settings) == tid
    assert (folder / "audio.webm").read_bytes() == b"second audio"
    assert not (folder / "speakers.json").exists() and not (folder / "voices.json").exists()
    assert (folder / "analysis.json").exists()  # marked outdated by its stamp instead
    assert not stages(folder)
    assert any("retranscribe" in p.name for p in (folder / "versions").iterdir())  # the earlier text stays restorable
    custody = Library(settings.output_dir).metadata(tid).extra["custody"]
    assert custody["source"]["file"] == "audio.webm" and custody["source"]["size"] == len(b"second audio")


def test_failed_first_run_leaves_no_folder(settings, fake_url, monkeypatch):
    def boom(self, media, options, progress):
        raise RuntimeError("engine failed")

    monkeypatch.setattr(MockEngine, "transcribe", boom)
    with pytest.raises(RuntimeError):
        run(TranscriptionRequest(URL), settings)
    assert not settings.output_dir.exists() or not [p for p in settings.output_dir.iterdir()]


def test_stale_stage_of_a_killed_run_is_swept(settings, fake_url):
    tid, folder = seed(settings, fake_url)
    old, fresh = folder / (STAGE_PREFIX + "old"), folder / (STAGE_PREFIX + "fresh")
    for d in (old, fresh):
        d.mkdir()
        (d / "audio.webm").write_bytes(b"partial")
    long_ago = time.time() - 3 * 24 * 3600
    for path in (old / "audio.webm", old):
        os.utime(path, (long_ago, long_ago))
    run(TranscriptionRequest(URL), settings)
    assert not old.exists()  # a run killed days ago (the worker cannot clean up after itself)
    assert fresh.exists()  # a recent one may belong to a run still going


def test_live_stage_with_an_old_folder_but_a_fresh_file_is_not_swept(settings, fake_url):
    tid, folder = seed(settings, fake_url)
    live = folder / (STAGE_PREFIX + "live")
    live.mkdir()
    (live / "audio.webm").write_bytes(b"downloading")  # written just now
    long_ago = time.time() - 3 * 24 * 3600
    os.utime(live, (long_ago, long_ago))  # the folder's own mtime says nothing about a file being written
    run(TranscriptionRequest(URL), settings)
    assert live.exists()


def test_progress_touches_the_stage(settings, fake_url, monkeypatch):
    tid, folder = seed(settings, fake_url)
    library = Library(settings.output_dir)
    r = library.begin_run("A talk", "Youtube:abc123")
    long_ago = time.time() - 3 * 24 * 3600
    os.utime(r.stage, (long_ago, long_ago))
    r.touch()
    assert time.time() - r.stage.stat().st_mtime < 60
    library.abort_run(r)


def failing(monkeypatch, where):
    """Make one step of the final write fail the way an antivirus or a player holding a file does."""
    from mgtranscriber import library as lib

    if where == "replace":
        real = lib.replace_file

        def replace(src, dst):
            if "audio" in dst.name and ".run-" in str(src) and "old" not in src.parts:
                raise PermissionError("file in use")  # the new audio cannot be moved in (restoring still works)
            return real(src, dst)

        monkeypatch.setattr(lib, "replace_file", replace)
    elif where == "save_segments":
        def boom(self, *a, **k):
            raise PermissionError("file in use")
        monkeypatch.setattr(lib.Library, "save_segments", boom)
    else:  # metadata, after the new segments and exports are already written
        def boom(self, *a, **k):
            raise PermissionError("file in use")
        monkeypatch.setattr(lib.Library, "save_metadata", boom)


@pytest.mark.parametrize("where", ["replace", "save_segments", "save_metadata"])
def test_failure_while_writing_restores_the_old_state_byte_for_byte(settings, fake_url, monkeypatch, where):
    tid, folder = seed(settings, fake_url)
    before = snapshot(folder)
    fake_url["audio"] = b"second audio"
    failing(monkeypatch, where)
    with pytest.raises(PermissionError):
        run(TranscriptionRequest(URL), settings)
    assert snapshot(folder) == before
    assert not stages(folder)
    assert not (folder / "versions").exists()  # not even the backup the run made on the way


def test_nothing_staged_keeps_the_old_audio(settings, fake_url, monkeypatch, tmp_path):
    """Captions only, or media read from where it is: no new audio, so the old one is not deleted."""
    tid, folder = seed(settings, fake_url)

    def fetch_nothing(media, out_dir, progress=None):
        media.audio_path = tmp_path / "elsewhere.wav"
        media.audio_path.write_bytes(b"RIFF")
        return media.audio_path

    monkeypatch.setattr(pipeline, "fetch_audio", fetch_nothing)
    assert run(TranscriptionRequest(URL), settings) == tid
    assert (folder / "audio.webm").read_bytes() == b"first audio"
    assert not (folder / "speakers.json").exists()  # the names still belong to the old run
    assert not stages(folder)


def test_interrupted_commit_is_undone_by_the_next_run(settings, fake_url):
    """A process killed between commit and save: the next run of the source puts the old files back."""
    tid, folder = seed(settings, fake_url)
    before = snapshot(folder)
    library = Library(settings.output_dir)
    r = library.begin_run("A talk", "Youtube:abc123")
    (r.stage / "audio.webm").write_bytes(b"second audio")
    library.commit_run(r, snapshot=["prepis.json", "metadata.json"])
    assert not (folder / "speakers.json").exists()  # ... and the process dies here
    long_ago = time.time() - 3600
    for path in [*(r.stage / "old").iterdir(), r.stage / "old", *r.stage.iterdir(), r.stage]:
        os.utime(path, (long_ago, long_ago))
    library.begin_run("A talk", "Youtube:abc123")  # the next run sweeps
    assert snapshot(folder) == before


def test_begin_run_survives_a_concurrent_abort_of_the_empty_folder(settings, monkeypatch):
    library = Library(settings.output_dir)
    other = library.begin_run("New talk", "Youtube:new1")
    real = Library._sweep_stages

    def abort_in_between(self, folder):
        real(self, folder)
        library.abort_run(other)  # removes the empty folder after our mkdir, before our stage.mkdir

    monkeypatch.setattr(Library, "_sweep_stages", abort_in_between)
    mine = library.begin_run("New talk", "Youtube:new1")
    assert mine.stage.is_dir()
    library.abort_run(mine)


def test_secure_delete_is_refused_while_a_run_is_in_progress(settings, fake_url):
    from mgtranscriber import privacy

    tid, folder = seed(settings, fake_url)
    library = Library(settings.output_dir)
    r = library.begin_run("A talk", "Youtube:abc123")
    with pytest.raises(privacy.PrivacyError) as err:
        privacy.delete_transcript(library, tid, settings)
    assert err.value.status == 409 and err.value.code == "transcript_busy"
    assert folder.is_dir()
    library.abort_run(r)
    assert privacy.delete_transcript(library, tid, settings)["deleted"]
