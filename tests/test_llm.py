import re

import pytest

from mgtranscriber.llm import qa
from mgtranscriber.llm.analysis import analyze, even_chapters
from mgtranscriber.llm.base import BadModelOutput, parse_json_object, repair_json, strict_schema
from mgtranscriber.llm.qa import ask
from mgtranscriber.llm.transcript import (Index, build_turns, chunk_turns, fmt_time, parse_time, speaker_labels)
from tests.llm_fakes import STEP, FakeProvider, make_segments

SEGS = make_segments()
LINES = re.compile(r"^\[([\d:]+)\] (.+?): (.*)$", re.M)


def topic_map(user: str) -> dict:
    """A well-behaved but imprecise 'model': one topic per chunk-start line, times a few seconds off."""
    lines = LINES.findall(user)
    first, last = lines[0], lines[len(lines) // 2]
    quote_text = SEGS[int(re.search(r"Sentence number (\d+)", last[2]).group(1))].text.strip()
    return {
        "summary": f"Part about {first[2][:20]}", "key_points": [],
        "topics": [{"start": first[0], "title": f"Topic at {first[0]}", "summary": "s"},
                   {"start": fmt_time(parse_time(last[0]) + 7), "title": f"Topic at {last[0]}", "summary": "s"}]
                  + [{"start": fmt_time(parse_time(lines[k][0]) + 3), "title": f"Topic at {lines[k][0]}", "summary": ""}
                     for k in (len(lines) // 4, 3 * len(lines) // 4)],
        "quotes": [{"time": last[0], "speaker": last[1], "text": quote_text},
                   {"time": last[0], "speaker": last[1], "text": "A sentence nobody ever said in this recording."}],
        "action_items": [{"time": last[0], "owner": "", "text": "Water the plants"}],
    }


def script(**over):
    base = dict(
        map=topic_map,
        reduce=lambda u: {"summary": "Whole summary.", "key_points": ["a", "b"]},
        chapters=lambda u: {"chapters": [
            {"start": m.group(1), "title": f"Chapter {i}", "summary": "x"}
            for i, m in enumerate(re.finditer(r"^(\S+) \| ", u, re.M)) if i % 2 == 0]},
        translate=lambda u: {"items": ["Zalij kytky" if "Water" in x else x for x in re.findall(r"^\d+\. (.*)$", u, re.M)]},
    )
    return {**base, **over}


# --- text helpers -----------------------------------------------------------------------

def test_time_parsing_and_formatting():
    assert parse_time("1:02:03") == 3723 and parse_time("[02:03]") == 123 and parse_time(45) == 45
    assert parse_time("nonsense") is None and parse_time(None) is None
    assert fmt_time(75) == "01:15" and fmt_time(3723) == "1:02:03"


def test_turns_merge_same_speaker_and_labels_use_names():
    from mgtranscriber.models import Segment
    segs = [Segment(0, 1, "Hi", "SPEAKER_00"), Segment(1.2, 2, "there", "SPEAKER_00"), Segment(3, 4, "Yo", "SPEAKER_01")]
    turns = build_turns(segs)
    assert [t.text for t in turns] == ["Hi there", "Yo"]
    assert speaker_labels(["SPEAKER_00", "SPEAKER_01"], {"SPEAKER_00": "Ada [x]:"}) == {
        "SPEAKER_00": "Ada x", "SPEAKER_01": "S1"}


def test_chunking_covers_everything_within_budget():
    turns = build_turns(SEGS)
    labels = speaker_labels(["SPEAKER_00", "SPEAKER_01"], {})
    chunks = chunk_turns(turns, labels, 300)
    assert len(chunks) > 3
    assert [t for c in chunks for t in c] == turns  # nothing lost, nothing duplicated, in order
    from mgtranscriber.llm.transcript import turn_line
    from mgtranscriber.llm.base import estimate_tokens
    assert all(sum(estimate_tokens(turn_line(t, labels)) for t in c) <= 300 * 1.3 for c in chunks)


def test_index_snaps_and_verifies_quotes():
    idx = Index(SEGS)
    assert idx.snap(80) == (75.0, 5.0) and idx.snap(-5)[0] == 0.0 and idx.snap(10_000)[0] == SEGS[-1].start
    exact = SEGS[10].text.strip()
    assert idx.find_quote(exact, around=0) == (10 * STEP, 1.0)  # exact match is found anywhere
    assert idx.find_quote(exact.upper() + "!!", around=10 * STEP)[1] == 1.0  # case/punctuation insensitive
    time, score = idx.find_quote("Sentence number 10 talks about something entirely different", around=10 * STEP)
    assert time == 10 * STEP and 0.5 < score < 0.85
    assert idx.find_quote("A sentence nobody ever said in this recording.", around=0)[1] < 0.5
    assert idx.find_quote("hi", around=0) == (None, 0.0)


def test_json_helpers():
    assert parse_json_object('{"a": [1, 2') == {"a": [1, 2]}  # cut off by the output limit
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(BadModelOutput):
        parse_json_object("not json")
    assert repair_json('{"a": "b') == '{"a": "b"}'
    schema = {"type": "object", "properties": {"x": {"type": "array", "items": {"type": "object", "properties": {}}}}}
    assert strict_schema(schema)["properties"]["x"]["items"]["additionalProperties"] is False


# --- chapters --------------------------------------------------------------------------

def test_even_chapters_drops_close_ones_and_fills_gaps():
    def ch(*starts):
        return [{"start": float(s), "title": f"t{s}", "summary": ""} for s in starts]

    candidates = ch(0, 60, 400, 800, 1300, 1800, 2400, 3000, 3300)
    # the model returned chapters bunched in the first half and two too close together
    result = even_chapters(ch(0, 60, 400, 800), candidates, 3600, target=8)
    starts = [c["start"] for c in result]
    assert starts == sorted(starts) and 60 not in starts  # 60 s after the first one is dropped
    assert max(b - a for a, b in zip(starts, starts[1:] + [3600])) <= 3600 / 8 * 2 + 1  # no huge gap left
    assert starts[0] == 0


# --- analysis ---------------------------------------------------------------------------

def test_analyze_map_reduce_validates_times_quotes_and_translates():
    provider = FakeProvider(**script())
    seen = []
    result = analyze(SEGS, {"SPEAKER_00": "Ada"}, provider, language="cs", title="Test talk",
                     progress=lambda p, m: seen.append((p, m)))
    real_starts = {s.start for s in SEGS}
    assert provider.count("map") > 3 and provider.count("reduce") == 1 and provider.count("chapters") == 1
    assert result["summary"] == "Whole summary." and result["key_points"] == ["a", "b"]
    # every chapter time is the start of a real segment, although the "model" was up to 7 s off
    assert result["chapters"] and all(c["start"] in real_starts for c in result["chapters"])
    assert result["chapters"][0]["start"] == 0
    assert result["validation"]["max_time_snap_s"] <= 7.0
    # the real quotes survive with their real time and speaker id; the invented one is dropped
    assert result["quotes"] and all(q["text"] != "A sentence nobody ever said in this recording." for q in result["quotes"])
    assert result["validation"]["quotes_dropped"] >= provider.count("map")
    q = result["quotes"][0]
    assert q["time"] in real_starts and q["speaker"] in {"SPEAKER_00", "SPEAKER_01"}
    assert {a["text"] for a in result["action_items"]} == {"Zalij kytky"}  # translated, deduplicated
    # prompts: language, speaker names, title, and timestamps instead of invented ones
    kind, system, user = provider.calls[0]
    assert "Czech" in system and "Ada" in system and "Test talk" in system and "[00:00] Ada:" in user
    assert result["language"] == "cs" and result["provider"] == "fake" and result["usage"]["calls"] == len(provider.calls)
    assert seen[0][0] < seen[-1][0] <= 99


def test_analyze_single_chunk_for_big_context_providers():
    provider = FakeProvider(chunk_tokens=100_000, max_context=200_000, **script(
        map=lambda u: {**topic_map(u), "summary": "Whole thing.", "key_points": ["k1"]}))
    result = analyze(SEGS, {}, provider, language="en")
    assert provider.count("map") == 1 and provider.count("reduce") == 0
    assert result["summary"] == "Whole thing." and result["key_points"] == ["k1"]
    assert "This is the whole transcript" in provider.calls[0][2]


def test_analyze_survives_garbage_times_and_empty_output():
    garbage = lambda u: {"summary": "s", "key_points": [], "action_items": [], "quotes": [],
                         "topics": [{"start": "soon", "title": "x", "summary": ""}, {"start": "99:99:99", "title": "y"}]}
    provider = FakeProvider(**script(map=garbage))
    result = analyze(SEGS, {}, provider)
    real_starts = {s.start for s in SEGS}
    assert all(c["start"] in real_starts for c in result["chapters"])


def test_analyze_can_be_cancelled():
    from mgtranscriber.llm.base import Cancelled
    provider = FakeProvider(**script())
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        if calls["n"] > 2:
            raise Cancelled()

    with pytest.raises(Cancelled):
        analyze(SEGS, {}, provider, cancel=cancel)
    assert len(provider.calls) < 3


# --- Q&A -------------------------------------------------------------------------------

def answer_with(citation_index: int | None, found: bool = True):
    def handler(user: str) -> dict:
        cites = []
        if citation_index is not None:
            cites = [{"time": fmt_time(citation_index * STEP), "text": SEGS[citation_index].text.strip()}]
        return {"found": found, "answer": "It is discussed.", "citations": cites}
    return handler


@pytest.fixture
def loose_thresholds(monkeypatch):
    # the fake bag-of-words embedding scores lower than a real model
    monkeypatch.setattr(qa, "MIN_RETRIEVAL_SCORE", 0.2)
    monkeypatch.setattr(qa, "MIN_CITATION_SCORE", 0.2)


def test_ask_retrieval_answers_with_verified_citation(loose_thresholds):
    idx = 25  # astronomy
    provider = FakeProvider(max_context=1500, **script(answer=answer_with(idx)))
    result = ask(SEGS, {}, "What about astronomy?", provider, language="en")
    assert result["retrieval"]["mode"] == "retrieval"
    assert result["found"] and result["citations"][0]["time"] == idx * STEP
    sent = provider.calls[-1][2]
    assert "astronomy" in sent and len(sent) < 0.5 * sum(len(s.text) for s in SEGS) * 3  # not the whole transcript


def test_ask_says_not_in_transcript_when_nothing_is_similar(loose_thresholds):
    provider = FakeProvider(max_context=1500, **script(answer=answer_with(3)))
    result = ask(SEGS, {}, "Share your goulash recipe with paprika", provider, language="cs")
    assert result["found"] is False and result["answer"] == "V přepisu to není." and result["citations"] == []
    assert provider.count("answer") == 0  # no model call, so nothing to hallucinate


def test_ask_requires_a_verified_citation():
    provider = FakeProvider(max_context=100_000, **script(answer=lambda u: {
        "found": True, "answer": "Confident but unsupported.",
        "citations": [{"time": "00:10", "text": "The model made this sentence up entirely."}]}))
    result = ask(SEGS, {}, "Anything?", provider, language="en")
    assert result["retrieval"]["mode"] == "full"
    assert result["found"] is False and result["answer"] == "The transcript does not say."
    # and a found=false from the model stays false even with a real quote
    provider = FakeProvider(max_context=100_000, **script(answer=answer_with(5, found=False)))
    assert ask(SEGS, {}, "Anything?", provider)["found"] is False


def test_ask_needs_embeddings_for_long_transcripts_without_a_big_context():
    from mgtranscriber.llm.base import LLMError
    provider = FakeProvider(max_context=1500, embeddings=False, **script(answer=answer_with(1)))
    with pytest.raises(LLMError) as err:
        ask(SEGS, {}, "Anything?", provider)
    assert err.value.code == "llm_transcript_too_long"


def test_even_chapters_with_nothing_returns_nothing():
    assert even_chapters([], [], 600.0, 5) == []


def test_analyze_skips_list_items_that_are_not_objects_or_have_odd_types():
    def odd(u):
        data = topic_map(u)
        data["topics"] += ["a string", None, 7, {"start": "00:10", "title": 5}, {"start": "00:20", "title": ["x"]}]
        data["quotes"] += ["q", None]
        data["action_items"] += ["do it", {"text": 3, "owner": 4}]
        return data

    result = analyze(SEGS, {}, FakeProvider(**script(map=odd)))
    assert result["chapters"] and result["quotes"] and result["action_items"]
