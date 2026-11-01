"""Transcript editor: word re-alignment, validation, saving through the API, versions, derived files."""

import json

import pytest
from fastapi.testclient import TestClient

from mgtranscriber import editor, pipeline, privacy, voices
from mgtranscriber.editor import EditError, parse_segments, realign_words
from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.llm.store import load_analysis, save_analysis
from mgtranscriber.models import Segment, Word
from mgtranscriber.server import create_app

W = Word


def words_of(seg):
    return [(w.word, w.start, w.end) for w in seg]


# --- word re-alignment ----------------------------------------------------------------------


def test_unchanged_text_keeps_the_words():
    old = [W("Hello", 0.0, 0.5, 0.9), W("world.", 0.6, 1.0, 0.8)]
    assert realign_words(old, "Hello world.", 0.0, 1.0, "A") is old


def test_corrected_word_keeps_its_slot_times():
    old = [W("Helo", 0.0, 0.5, 0.9, "A"), W("world.", 0.6, 1.0, 0.8, "A")]
    out = realign_words(old, "Hello world.", 0.0, 1.0, "A")
    assert words_of(out) == [("Hello", 0.0, 0.5), ("world.", 0.6, 1.0)]
    assert out[0].score is None and out[1].score == 0.8  # the corrected word is no longer the model's


def test_inserted_words_get_times_between_neighbours():
    old = [W("one", 0.0, 1.0), W("four", 3.0, 4.0)]
    out = realign_words(old, "one two three four", 0.0, 4.0, None)
    assert [w.word for w in out] == ["one", "two", "three", "four"]
    assert out[1].start == 1.0 and out[2].end == 3.0
    assert out[1].end == pytest.approx(1.0 + 2.0 * 4 / 10)  # "two" + space = 4 of 10 characters
    assert out[0].start == 0.0 and out[3].end == 4.0


def test_deleted_words_disappear_and_punctuation_changes_still_match():
    old = [W("Well,", 0.0, 0.3), W("um,", 0.4, 0.6), W("yes", 0.7, 1.0)]
    out = realign_words(old, "Well yes!", 0.0, 1.0, None)
    assert words_of(out) == [("Well", 0.0, 0.3), ("yes!", 0.7, 1.0)]


def test_words_without_any_times_are_spread_over_the_segment():
    out = realign_words([], "a bb ccc", 10.0, 13.0, "B")
    assert out[0].start == 10.0 and out[-1].end == 13.0 and all(w.speaker == "B" for w in out)


def test_redaction_survives_edits():
    old = [W("call", 0, 1), W("Jan", 1, 2, redacted=True), W("Novák", 2, 3, redacted=True), W("now", 3, 4)]
    out = realign_words(old, "call Jana Nováka right now", 0, 4, None)
    # the changed stretch had redacted words: all of it is redacted (rather more than less)
    assert [w.word for w in out if w.redacted] == ["Jana", "Nováka", "right"]
    assert not any(w.redacted for w in out if w.word in ("now", "call"))


def test_rewording_a_redacted_word_redacts_all_its_replacements():
    # review finding: positional pairing redacted "Mr" and left "Nowak" readable
    old = [W("call", 0, 1), W("Novák", 1, 2, redacted=True), W("now", 2, 3)]
    out = realign_words(old, "call Mr Nowak now", 0, 3, None)
    assert [(w.word, bool(w.redacted)) for w in out] == [("call", False), ("Mr", True), ("Nowak", True), ("now", False)]


def test_words_added_next_to_a_redacted_word_are_redacted():
    old = [W("call", 0, 1), W("Novák", 1, 2, redacted=True), W("now", 2, 3)]
    after = realign_words(old, "call Novák Jan now", 0, 3, None)
    assert [w.word for w in after if w.redacted] == ["Novák", "Jan"]
    before = realign_words(old, "call Petr Novák now", 0, 3, None)
    assert [w.word for w in before if w.redacted] == ["Petr", "Novák"]
    apart = realign_words(old, "please call Novák now", 0, 3, None)
    assert [w.word for w in apart if w.redacted] == ["Novák"]


def test_long_segments_keep_their_alignment():
    # beyond the LCS table limit the common prefix and suffix still match (an edit in the middle)
    n = 1000
    old = [W(f"w{i}", i, i + 0.5, 0.9, redacted=True if i == 990 else None) for i in range(n)]
    text = " ".join(["w0", "inserted", *(f"w{i}" for i in range(1, n))])
    out = realign_words(old, text, 0, n, None)
    assert out[500].word == "w499" and (out[500].start, out[500].score) == (499, 0.9)
    assert [w.word for w in out if w.redacted] == ["w990"]
    # a middle too large for the table is one changed stretch: redaction kept for all of it
    shuffled = " ".join(f"x{i}" for i in range(n))
    out = realign_words(old, shuffled, 0, n, None)
    assert all(w.redacted for w in out) and out[10].start == 10


def test_parse_segments_rejects_non_finite_and_misordered_values():
    good = {"start": 0, "end": 1, "text": "a", "words": [{"word": "a", "start": 0, "end": 1, "score": 0.5}]}
    for bad in ([{**good, "words": [{"word": "a", "score": float("nan")}]}],
                [{**good, "words": [{"word": "a", "score": float("inf")}]}],
                [{**good, "end": float("inf")}],
                [{**good, "words": [{"word": "a", "speaker": "../x"}]}],
                [{**good, "start": 5, "end": 6}, good],  # out of order
                [{**good, "end": 3}, {**good, "start": 1, "end": 4}]):  # overlapping by 2 s
        with pytest.raises(EditError):
            parse_segments(bad)
    # word times are kept inside their segment
    seg = parse_segments([{"start": 1, "end": 2, "text": "a b", "words": [
        {"word": "a", "start": 0.2, "end": 1.5}, {"word": "b", "start": 1.8, "end": 9}]}])[0]
    assert [(w.start, w.end) for w in seg.words] == [(1, 1.5), (1.8, 2)]


def test_lcs_pairs_matches_the_javascript_version():
    import random
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    rng = random.Random(7)
    vocab = ["a", "b", "c", "Novák", "now", "call", "the", "x,", "X"]
    cases = [([{"word": "call"}, {"word": "Novák", "start": 1, "end": 2, "redacted": True}, {"word": "now"}],
              "call Mr Nowak now")]
    for _ in range(200):
        old = [{"word": rng.choice(vocab), "start": i, "end": i + 0.5, **({"redacted": True} if rng.random() < 0.2 else {})}
               for i in range(rng.randint(0, 12))]
        cases.append((old, " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 12)))))
    module = (Path(__file__).parent.parent / "static" / "js" / "components" / "editor.js").as_uri()
    script = f"""
globalThis.document = {{ documentElement: {{}}, querySelector: () => null, addEventListener() {{}} }};
globalThis.localStorage = {{ getItem: () => null, setItem() {{}} }};
globalThis.window = globalThis;
const {{ realign }} = await import({json.dumps(module)});
let input = ""; for await (const chunk of process.stdin) input += chunk;
const out = JSON.parse(input).map(([old, text]) => realign(old, text, 0, 20, null).map((w) => [w.word, !!w.redacted, w.start ?? null]));
process.stdout.write(JSON.stringify(out));
"""
    proc = subprocess.run([node, "--input-type=module", "-e", script], input=json.dumps(cases), capture_output=True,
                          text=True, encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)
    for (old, text), got in zip(cases, js):
        py = realign_words([W(w["word"], w.get("start"), w.get("end"), None, None, w.get("redacted")) for w in old],
                           text, 0, 20, None)
        assert [[w.word, bool(w.redacted)] for w in py] == [g[:2] for g in got], (old, text)
        assert [w.start for w in py] == pytest.approx([g[2] for g in got], abs=0.0015), (old, text)  # rounding
    assert js[0] == [["call", False, None], ["Mr", True, 1], ["Nowak", True, 2], ["now", False, None]]


def test_parse_segments_validates_and_normalizes():
    segs = parse_segments([
        {"start": 0, "end": 1, "text": "  Hello   there ", "speaker": "SPEAKER_00",
         "words": [{"word": "Hello", "start": 0, "end": 0.4}, {"word": "there", "start": 0.5, "end": 1}]},
        {"start": 1, "end": 2, "text": "   "},  # emptied: removed
        {"start": 2, "end": 3, "text": "x", "speaker": "UNKNOWN"},
    ])
    assert [s.text for s in segs] == [" Hello there", " x"] and segs[1].speaker is None
    for bad in ([{"start": 2, "end": 1, "text": "x"}], [{"start": 0, "end": 1, "text": "x", "speaker": "../a"}],
                [{"start": "0", "end": 1, "text": "x"}], [{"start": 0, "end": 1, "text": 5}], {"a": 1},
                [{"start": float("nan"), "end": 1, "text": "x"}]):
        with pytest.raises(EditError):
            parse_segments(bad)


# --- API ------------------------------------------------------------------------------------


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        yield c


@pytest.fixture
def tid(settings, library, media_file):
    return pipeline.run(pipeline.TranscriptionRequest(str(media_file("talk.wav"))), settings, library=library)


def get(client, tid):
    r = client.get(f"/api/transcripts/{tid}/segments")
    assert r.status_code == 200, r.text
    return r.json()


def put(client, tid, data, segments):
    return client.put(f"/api/transcripts/{tid}/segments", json={"base": data["stamp"], "segments": segments})


def test_get_segments_has_words_one_to_one(client, tid):
    data = get(client, tid)
    for seg in data["segments"]:
        assert [w["word"] for w in seg["words"]] == seg["text"].split()
    assert data["versions"] == []


def test_edit_text_speaker_and_save(client, tid, library):
    data = get(client, tid)
    segs = data["segments"]
    segs[0]["text"] = "Hello there, friend."
    segs[2]["speaker"] = "SPEAKER_02"  # a new speaker
    r = put(client, tid, data, segs)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"]["segments_changed"] == 2 and body["version"]
    saved = library.segments(tid)
    assert saved[0].text == " Hello there, friend." and saved[2].speaker == "SPEAKER_02"
    assert [w.word for w in saved[0].words] == ["Hello", "there,", "friend."]
    assert saved[0].words[0].start == 0.5  # kept
    assert library.metadata(tid).speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    assert "friend" in (library.path(tid) / "prepis.txt").read_text(encoding="utf-8")  # stored export rewritten
    # the old state is a version and can be restored
    [version] = body["versions"]
    r = client.post(f"/api/transcripts/{tid}/versions/{version['id']}/restore")
    assert r.status_code == 200
    assert library.segments(tid)[0].text == " Hello there."
    assert len(client.get(f"/api/transcripts/{tid}/versions").json()) == 2  # the edited state was kept too


def test_split_and_merge(client, tid, library):
    data = get(client, tid)
    first, second, third = data["segments"]
    # split the first segment after its first word (what the editor sends)
    a = {**first, "text": "Hello", "end": first["words"][0]["end"], "words": first["words"][:1]}
    b = {**first, "text": "there.", "start": first["words"][1]["start"], "words": first["words"][1:]}
    # merge the second and third
    merged = {**second, "text": f"{second['text']} {third['text']}", "end": third["end"],
              "words": second["words"] + third["words"]}
    r = put(client, tid, data, [a, b, merged])
    assert r.status_code == 200, r.text
    saved = library.segments(tid)
    assert [s.text.strip() for s in saved] == ["Hello", "there.", "Hi, how are you? Fine."]
    assert (saved[1].start, saved[1].end) == (1.1, 2.0)
    assert saved[2].end == 5.0 and len(saved[2].words) == 5


def test_conflict_and_noop(client, tid, library):
    data = get(client, tid)
    assert put(client, tid, data, data["segments"]).json()["changed"]["segments_changed"] == 0
    segs = data["segments"]
    segs[0]["text"] = "Changed."
    assert put(client, tid, data, segs).status_code == 200
    r = put(client, tid, data, segs)  # the same base again: the transcript changed meanwhile
    assert r.status_code == 409 and r.json()["code"] == "edit_conflict"
    r = client.put(f"/api/transcripts/{tid}/segments", json={"base": None, "segments": []})
    assert r.status_code == 422
    r = client.put(f"/api/transcripts/{tid}/segments", json={"base": None, "segments": [{"text": "x"}]})
    assert r.status_code == 422 and r.json()["code"] == "invalid_segments"
    assert client.post(f"/api/transcripts/{tid}/versions/..%2Fprepis.json/restore").status_code in (404, 405)
    assert client.post(f"/api/transcripts/{tid}/versions/prepis.20200101T000000000Z.json/restore").json()["code"] == "version_not_found"


def test_versions_are_coalesced_while_editing(client, tid, library):
    data = get(client, tid)
    for n in range(3):
        segs = data["segments"]
        segs[0]["text"] = f"Edit {n}"
        r = put(client, tid, data, segs)
        data = {**data, "stamp": r.json()["stamp"]}
    assert len(editor.list_versions(library, tid)) == 1


def test_save_keeps_voices_and_marks_analysis_outdated(client, tid, library, settings):
    voices.store_embeddings(library, tid, {"SPEAKER_00": [1.0] + [0.0] * 255}, {"SPEAKER_00": 30})
    library.save_names(tid, {"SPEAKER_00": "Alice"})
    save_analysis(library, tid, {"summary": "x"})
    assert load_analysis(library, tid)["stale"] is False
    data = get(client, tid)
    data["segments"][1]["text"] = "Hi, how are you doing?"
    assert put(client, tid, data, data["segments"]).status_code == 200
    assert (library.path(tid) / voices.TRANSCRIPT_FILE).is_file()
    assert library.names(tid) == {"SPEAKER_00": "Alice"}
    assert load_analysis(library, tid)["stale"] is True
    # search sees the new text
    hits = client.get("/api/search", params={"q": "doing"}).json()
    assert hits["total"] == 1 and hits["results"][0]["transcript_id"] == tid


def test_edit_is_audited_and_hashed(client, tid, library):
    data = get(client, tid)
    data["segments"][0]["text"] = "Secret name here."
    put(client, tid, data, data["segments"])
    log = privacy.read_audit(library, tid)
    assert [e["event"] for e in log] == ["created", "edited"]
    assert "Secret" not in json.dumps(log)  # the log never repeats the text
    custody = library.metadata(tid).extra["custody"]
    assert custody["outputs"]["prepis.json"] == privacy.sha256_file(library.path(tid) / "prepis.json")
    assert custody["source"]["sha256"]  # kept from the transcription


def test_retranscription_backs_up_edits(settings, library, media_file, tid):
    segs = library.segments(tid)
    segs[0].text = " Edited by hand."
    editor.save(library, tid, segs, base=None)
    pipeline.run(pipeline.TranscriptionRequest(str(media_file("talk.wav"))), settings, library=library)
    reasons = [v["reason"] for v in editor.list_versions(library, tid)]
    assert "retranscribe" in reasons
    restored = editor.version_segments(library, tid, editor.list_versions(library, tid)[0]["id"])
    assert restored[0].text == " Edited by hand."
    assert privacy.read_audit(library, tid)[-1]["event"] == "retranscribed"


def test_word_redacted_flag_roundtrip():
    seg = Segment.from_dict({"start": 0, "end": 1, "text": " a b", "words": [{"word": "a", "redacted": True}, {"word": "b"}]})
    assert seg.words[0].redacted is True and seg.words[1].redacted is None
    assert seg.to_dict()["words"] == [{"word": "a", "redacted": True}, {"word": "b"}]
