"""Real WhisperX run on a 40 s clip with a small model. Run with: uv run pytest -m gpu"""

import json
import os

import pytest

from mgtranscriber.config import load_settings
from mgtranscriber.library import Library
from mgtranscriber.pipeline import TranscriptionRequest, run
from tests.make_clip import ensure_clip

pytestmark = pytest.mark.gpu
MODEL = os.environ.get("MGTRANSCRIBER_TEST_MODEL", "tiny")


@pytest.fixture(scope="module")
def clip():
    try:
        return ensure_clip()
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_whisperx_end_to_end(clip, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    settings = load_settings().with_overrides(output_dir=tmp_path / "output", data_dir=tmp_path / "data")
    diarize = bool(settings.hf_token)

    phases = []
    transcript_id = run(
        TranscriptionRequest(str(clip), language="en", model=MODEL, diarize=diarize, engine="whisperx"),
        settings, lambda p: phases.append(p.phase),
    )

    library = Library(settings.output_dir)
    segments = library.segments(transcript_id)
    text = " ".join(s.text for s in segments).lower()
    assert "ai safety" in text or "safe ai" in text
    assert all(s.words for s in segments)
    meta = json.loads((library.path(transcript_id) / "metadata.json").read_text(encoding="utf-8"))
    assert meta["engine"] == "whisperx" and meta["model"] == MODEL and meta["language"] == "en"
    assert 39 <= meta["duration"] <= 41
    assert {"transcribe", "align", "writing"} <= set(phases)
    if diarize:
        assert "diarize" in phases
        assert len(meta["speakers"]) >= 2
    assert (library.path(transcript_id) / "prepis.txt").read_text(encoding="utf-8").startswith("[00:00:00]")
