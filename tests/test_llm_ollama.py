"""Integration tests against a real Ollama (pytest -m ollama). Skipped when it is not running."""

import pytest

from mgtranscriber.llm import qa
from mgtranscriber.llm.analysis import analyze
from mgtranscriber.llm.base import LLMError
from mgtranscriber.llm.ollama import OllamaProvider, same_model
from mgtranscriber.models import Segment

pytestmark = pytest.mark.ollama

SCRIPT = [
    "Welcome to the show. Today we talk about growing tomatoes at home.",
    "Tomatoes need at least six hours of direct sun every day, so choose a south facing balcony.",
    "Water them in the morning and never wet the leaves, otherwise fungus spreads quickly.",
    "Next topic: composting. A good compost needs green waste, brown waste and some air.",
    "Turn the pile once a week and keep it as damp as a wrung out sponge.",
    "Finally, I will publish a shopping list for beginners in the show notes tomorrow.",
]


@pytest.fixture(scope="module")
def provider():
    p = OllamaProvider()
    try:
        p.check()
        p.embed(["ping"])
    except LLMError as exc:
        pytest.skip(f"Ollama not usable: {exc.message}")
    return p


@pytest.fixture(scope="module")
def segments():
    return [Segment(i * 40.0, i * 40.0 + 35, " " + text, f"SPEAKER_0{i % 2}") for i, text in enumerate(SCRIPT * 4)]


def test_real_analysis_has_real_times_and_verified_quotes(provider, segments):
    result = analyze(segments, {"SPEAKER_00": "Eva"}, provider, language="en", title="Garden talk")
    starts = {s.start for s in segments}
    assert result["summary"] and result["chapters"]
    assert all(c["start"] in starts for c in result["chapters"])
    assert all(q["time"] in starts for q in result["quotes"])
    assert result["validation"]["max_time_snap_s"] < 60


def test_real_qa_answers_and_refuses(provider, segments):
    yes = qa.ask(segments, {}, "How much sun do tomatoes need?", provider, language="en")
    assert yes["found"] and yes["citations"] and any("sun" in c["text"].lower() for c in yes["citations"])
    no = qa.ask(segments, {}, "What is the recipe for goulash?", provider, language="en")
    assert no["found"] is False and no["citations"] == []


def test_same_model_matching():
    assert same_model("gemma3:4b", "gemma3:4b") and same_model("bge-m3", "bge-m3:latest")
    assert not same_model("gemma3:4b", "gemma3:12b")
