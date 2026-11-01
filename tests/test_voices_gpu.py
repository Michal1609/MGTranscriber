"""Voice memory end to end on a real GPU: remember a voice in one part of a podcast, recognise it
in another part, and compute voiceprints for an older transcript. Run with: uv run pytest -m gpu

Two 2-minute clips are cut from the same local audio as tests/make_clip.py (third-party content,
git-ignored in tests/fixtures/).
"""

import shutil
import subprocess

import pytest

from mgtranscriber import voices
from mgtranscriber.config import load_settings
from mgtranscriber.library import Library
from mgtranscriber.pipeline import TranscriptionRequest, run
from mgtranscriber.tasks import execute
from tests.make_clip import ROOT, find_source
from tests.test_e2e_gpu import MODEL

pytestmark = pytest.mark.gpu
CLIPS = {"voices-a.wav": 0, "voices-b.wav": 600}  # start second; 120 s each


@pytest.fixture(scope="module")
def clips():
    source = find_source()
    if source is None or not shutil.which("ffmpeg"):
        pytest.skip("source audio or ffmpeg missing")
    out = {}
    for name, start in CLIPS.items():
        path = ROOT / "tests" / "fixtures" / name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", "120", "-i", str(source),
                            "-ac", "1", "-ar", "16000", str(path)], check=True)
        out[name] = path
    return out


@pytest.fixture
def settings(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    s = load_settings().with_overrides(output_dir=tmp_path / "output", data_dir=tmp_path / "data")
    if not s.hf_token:
        pytest.skip("HF_TOKEN missing (diarization)")
    return s


def transcribe(path, settings):
    return run(TranscriptionRequest(str(path), language="en", model=MODEL, diarize=True, engine="whisperx",
                                    speakers=2), settings)


def test_remember_in_one_part_recognise_in_another(clips, settings):
    library = Library(settings.output_dir)
    db = voices.open_db(settings)

    a = transcribe(clips["voices-a.wav"], settings)
    state = voices.public_state(voices.read_transcript(library, a))
    assert state["available"] and len(state["speakers"]) == 2
    raw = voices.read_transcript(library, a)["speakers"]
    assert all(len(voices.decode(info["vector"])) == voices.DIM for info in raw.values())
    assert library.names(a) == {}  # nobody is known yet

    # remember both speakers of part A
    by_speech = sorted(state["speakers"], key=lambda lb: -state["speakers"][lb]["speech_s"])
    for label, name in zip(by_speech, ["Most", "Less"]):
        voices.enroll_label(library, a, label, db, name=name)
    assert library.names(a) == {by_speech[0]: "Most", by_speech[1]: "Less"}

    # part B (10 minutes later): both are recognised automatically, as different people
    b = transcribe(clips["voices-b.wav"], settings)
    names = library.names(b)
    matches = {lb: info["match"] for lb, info in voices.read_transcript(library, b)["speakers"].items()}
    assert sorted(names.values()) == ["Less", "Most"], matches
    assert all(m["status"] == "auto" and m["score"] >= voices.AUTO_THRESHOLD for m in matches.values())

    # an older transcript without voiceprints: the voiceprints job computes them from its labels
    (library.path(b) / voices.TRANSCRIPT_FILE).unlink()
    library.save_names(b, {})
    library.path(b)  # still a transcript
    shutil.copy(clips["voices-b.wav"], library.path(b) / "audio.wav")  # local sources keep no audio copy
    assert execute(TranscriptionRequest(b, task="voiceprints"), settings, library=library) == b
    assert sorted(library.names(b).values()) == ["Less", "Most"]
