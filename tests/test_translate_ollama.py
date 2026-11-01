"""Translation against a real Ollama (pytest -m ollama): a short English transcript into Czech."""

import re

import pytest

from mgtranscriber.llm.base import LLMError
from mgtranscriber.llm.ollama import OllamaProvider
from mgtranscriber.llm.translate import build_translation
from mgtranscriber.models import Segment
from tests.test_llm_ollama import SCRIPT

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def provider():
    p = OllamaProvider()
    try:
        p.check()
    except LLMError as exc:
        pytest.skip(f"Ollama not usable: {exc.message}")
    return p


def test_real_translation_en_to_cs_keeps_segments_aligned(provider):
    segments = [Segment(i * 10.0, i * 10.0 + 8, " " + text, f"SPEAKER_0{i % 2}") for i, text in enumerate(SCRIPT)]
    out = build_translation(segments, {"SPEAKER_00": "Eva"}, provider, "cs", source="en", title="Garden talk")
    assert len(out["segments"]) == len(segments)
    for seg, tr in zip(segments, out["segments"]):
        assert (tr["start"], tr["end"], tr["speaker"]) == (seg.start, seg.end, seg.speaker)
        assert tr["text"].strip() and tr["text"].strip() != seg.text.strip()
    assert out["untranslated"] == []
    joined = " ".join(s["text"] for s in out["segments"]).lower()
    assert re.search(r"rajčat|paradaj", joined) and re.search(r"kompost", joined)  # tomatoes, compost
