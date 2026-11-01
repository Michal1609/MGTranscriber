"""The benchmark code (benchmarks/): data selection, session building, DER, tables. No GPU, no downloads."""

import io
import json
import tarfile
import wave

import numpy as np

from benchmarks import bench_cs, longform
from mgtranscriber.models import Segment, Word


def make_tar(members: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def clip(i: int, speaker: str, sitting: str = "ps2013-001", dur: float = 10.0) -> dict:
    return {"vert": f"{sitting}-{i}.tsv", "speaker": speaker, "text": f"text {i}", "dur": dur,
            "start_token": f"{sitting}-01-002-020.u{i}.p1.s1.w1", "flac": b"x"}


# --- common voice -----------------------------------------------------------------------

def test_cv_select_limits_clips_per_client_and_repeats():
    rows = [{"client_id": f"c{i % 4}", "path": f"{i}.mp3", "sentence": f"věta {i // 2}"} for i in range(40)]
    taken = bench_cs.cv_select(rows, 100, per_speaker=3)
    assert len(taken) == 12
    assert all(sum(r["client_id"] == c for r in taken) == 3 for c in {"c0", "c1", "c2", "c3"})
    assert len({r["sentence"] for r in taken}) == len(taken)
    assert bench_cs.cv_select(rows, 5) == bench_cs.cv_select(rows, 5)  # fixed seed


# --- ParCzech tar ranges ----------------------------------------------------------------

def test_clips_from_blob_starts_in_the_middle_and_drops_cut_member():
    members = []
    for i in range(6):
        meta = {"true_text": f"věta {i}", "speaker": "A.1950", "dur": 1.5, "vert": "v.tsv", "start_token_id": f"ps2013-001-1-1-1.u{i}"}
        members += [(f"{i:010d}.json", json.dumps(meta).encode()), (f"{i:010d}.flac", bytes([i]) * 1500)]
    tar = make_tar(members)
    # start at the 3rd member (a 512-aligned header) minus noise, end in the middle of the last file
    start = 0
    for _name, data in members[:4]:
        start += 512 + (len(data) + 511) // 512 * 512
    blob = b"\x07" * 1024 + tar.rstrip(b"\0")[start:-500]
    clips = longform.clips_from_blob(blob)
    assert [c["text"] for c in clips] == ["věta 2", "věta 3", "věta 4"]
    assert clips[0]["flac"] == bytes([2]) * 1500


def test_token_key_orders_numerically():
    tokens = ["ps2013-004-01-002-020.u10.p1.s1.w1", "ps2013-004-01-002-020.u9.p1.s1.w1"]
    assert sorted(tokens, key=longform.token_key)[0].endswith("u9.p1.s1.w1")


# --- sessions ---------------------------------------------------------------------------

def test_make_sessions_groups_by_sitting_and_needs_two_speakers():
    clips = [clip(i, "A" if i < 20 else "B") for i in range(40)]  # 400 s, two speakers: kept
    clips += [clip(i, "C", sitting="ps2013-002") for i in range(40)]  # one speaker: dropped
    clips += [clip(i, "D" if i % 2 else "E", sitting="ps2013-003") for i in range(5)]  # too short: dropped
    sessions = longform.make_sessions(clips)
    assert len(sessions) == 1
    assert {c["speaker"] for c in sessions[0]} == {"A", "B"}
    assert [c["start_token"] for c in sessions[0]] == sorted(
        (c["start_token"] for c in sessions[0]), key=longform.token_key)


def test_make_sessions_stops_at_the_target_length():
    clips = [clip(i, "A" if i % 7 < 3 else "B", dur=20.0) for i in range(100)]
    (session,) = longform.make_sessions(clips)
    assert longform.SESSION_TARGET_S <= sum(c["dur"] + longform.CLIP_GAP_S for c in session) <= longform.SESSION_MAX_S


def test_choose_prefers_more_speakers_until_target():
    few = [clip(i, "A" if i % 2 else "B", sitting="ps2013-001") for i in range(40)]
    many = [clip(i, "ABCD"[i % 4], sitting="ps2013-002") for i in range(40)]
    chosen = longform.choose([few, many], target_s=100)
    assert chosen == [many]


def test_write_session_spans_match_audio(tmp_path):
    session = [clip(0, "A", dur=1.0), clip(1, "B", dur=2.0)]
    lengths = iter([16000, 32000])
    described = longform.write_session(session, 3, tmp_path, lambda _: np.full(next(lengths), 0.1, dtype=np.float32))
    assert described["id"] == "session03"
    assert [(c["start"], c["end"]) for c in described["clips"]] == [(0.0, 1.0), (1.4, 3.4)]
    with wave.open(str(tmp_path / "session03.wav")) as f:
        assert f.getnframes() == 16000 + 6400 + 32000 + 6400
    assert described["speakers"] == ["A", "B"]


# --- diarization ------------------------------------------------------------------------

def words_segment(speaker: str | None, spans: list[tuple[float, float]], word_speaker: str | None = None) -> Segment:
    words = [Word("w", s, e, None, word_speaker) for s, e in spans]
    return Segment(spans[0][0], spans[-1][1], "w " * len(spans), speaker, words)


CLIPS = [{"speaker": "A", "start": 0.0, "end": 10.0}, {"speaker": "B", "start": 10.4, "end": 20.0}]


def test_perfect_hypothesis_has_zero_der_even_with_other_labels():
    segments = [words_segment("SPEAKER_01", [(0.1, 4.0), (4.2, 9.9)]), words_segment("SPEAKER_00", [(10.5, 15.0), (15.2, 19.9)])]
    result = longform.diarization_error(longform.reference_annotation(CLIPS), longform.hypothesis_annotation(segments))
    assert result["der"] == 0.0
    assert (result["speakers_ref"], result["speakers_hyp"]) == (2, 2)


def test_merged_speakers_are_confusion_and_counted():
    segments = [words_segment("SPEAKER_00", [(0.1, 9.9)]), words_segment("SPEAKER_00", [(10.5, 19.9)])]
    result = longform.diarization_error(longform.reference_annotation(CLIPS), longform.hypothesis_annotation(segments))
    assert result["speakers_hyp"] == 1
    assert 40 < result["confusion"] < 55  # half of the speech belongs to the wrong (merged) speaker
    assert result["miss"] < 5


def test_word_speaker_beats_segment_speaker():
    segment = words_segment("SPEAKER_00", [(0.0, 5.0)], word_speaker="SPEAKER_07")
    assert longform.hypothesis_annotation([segment]).labels() == ["SPEAKER_07"]


def test_pooled_der_weights_by_speech():
    parts = [{"miss": 0, "false_alarm": 0, "confusion": 50, "ref_speech_s": 100},
             {"miss": 0, "false_alarm": 0, "confusion": 0, "ref_speech_s": 300}]
    assert longform.pooled_der(parts) == 12.5


# --- report -----------------------------------------------------------------------------

def test_assign_words_uses_given_gap():
    seg = words_segment(None, [(0.1, 0.5), (1.9, 2.3)])
    texts = bench_cs.assign_words([seg], [(0.0, 1.0), (1.4, 2.5)], gap=0.4)
    assert texts == ["w", "w"]


def test_run_name_and_report_for_sessions():
    assert bench_cs.run_name("fleurs", "app", "parakeet", 300).endswith("-app-parakeet-n300")
    assert bench_cs.run_name("cv", "model", "canary", 300).endswith("-cv-canary-n300")
    assert bench_cs.run_name("parczech", "app", "whisperx:large-v3", 0).endswith("-parczech-app-whisperx-large-v3")
    items = [{"id": "a", "ref": "ahoj světe", "hyp": "ahoj světe"}, {"id": "b", "ref": "dobrý den", "hyp": "dobrý dne"}]
    result = {"run": "app-x", "rtf": 0.1, "seconds_total": 120.0, "der": 8.5, "wer_whole": 25.0,
              "speakers_ref": 10, "speakers_hyp": 9, **bench_cs.summarize(items),
              "sessions": [{"ref_speech_s": 100.0, "miss": 2.0, "false_alarm": 1.0, "confusion": 5.5}]}
    text = bench_cs.report([result])
    assert "DER %" in text and "9 / 10" in text and "8.50" in text


# --- English: LibriSpeech, AMI, normalizer -------------------------------------------------

def test_librispeech_select_spreads_over_speakers_and_is_stable():
    rows = [{"id": f"{s}-{c}-{u:04d}", "text": "x", "path": None} for s in range(5) for c in range(2) for u in range(10)]
    taken = bench_cs.librispeech_select(rows, 100, per_speaker=4)
    assert len(taken) == 20
    assert all(sum(r["id"].startswith(f"{s}-") for r in taken) == 4 for s in range(5))
    assert [r["id"] for r in taken] == sorted(r["id"] for r in taken)
    assert taken == bench_cs.librispeech_select(rows, 100, per_speaker=4)


def test_english_normalizer_unifies_numbers_and_fillers(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_cs, "HERE", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "english_spelling.json").write_text('{"colour": "color"}', encoding="utf-8")
    monkeypatch.setattr(bench_cs, "_english", None)
    try:
        bench_cs.set_language("en")
        assert bench_cs.normalize("Uh, the COLOUR of twenty twenty-three!") == bench_cs.normalize("the color of 2023")
        assert bench_cs.word_counts("I can't go", "i cannot go")[1] >= 2
    finally:
        bench_cs.set_language("cs")
    assert bench_cs.normalize("Ahoj, světe!") == "ahoj světe"


AMI_XML = b"""<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
 <w nite:id="a0" starttime="0.5" endtime="0.9">Okay</w>
 <w nite:id="a1" starttime="0.9" endtime="0.9" punc="true">.</w>
 <vocalsound nite:id="a2" starttime="1.0" endtime="1.1" type="cough"/>
 <w nite:id="a3" starttime="1.2" endtime="1.4" trunc="true">tha</w>
 <w nite:id="a4" starttime="61.0" endtime="61.5">yeah</w>
 <w nite:id="a5" endtime="62.0">notime</w>
</nite:root>"""


def test_ami_words_skip_punctuation_truncated_and_untimed():
    words = longform.ami_words(AMI_XML, "A")
    assert [(w["word"], w["start"]) for w in words] == [("Okay", 0.5), ("yeah", 61.0)]


def test_ami_session_windows_and_turns():
    words = longform.ami_words(AMI_XML, "A") + [{"word": "mhm", "start": 0.7, "end": 1.0, "speaker": "B"}]
    session = longform.ami_session("ES2004a", words, seconds=100.0)
    assert [c["text"] for c in session["clips"]] == ["Okay mhm", "yeah"]
    assert [(c["start"], c["end"]) for c in session["clips"]] == [(0.0, 60.0), (60.0, 100.0)]
    assert session["gap"] == 0.0 and session["speakers"] == ["A", "B"]
    ref = longform.reference_annotation(session["turns"])
    assert sorted(ref.labels()) == ["A", "B"]  # overlapping turns of two speakers are kept


def test_diarization_error_reports_no_overlap_variant():
    turns = [{"start": 0.0, "end": 10.0, "speaker": "A"}, {"start": 5.0, "end": 15.0, "speaker": "B"}]
    segments = [words_segment("X", [(0.0, 10.0)]), words_segment("Y", [(10.0, 15.0)])]
    result = longform.diarization_error(longform.reference_annotation(turns), longform.hypothesis_annotation(segments))
    assert result["der_no_overlap"] == 0.0  # the app gives one speaker at a time: the overlap is what it misses
    assert 15 < result["der"] < 35


def test_peak_normalize_levels_quiet_audio_and_keeps_silence():
    quiet = np.array([0.001, -0.002, 0.0005], dtype=np.float32)
    assert abs(float(np.abs(bench_cs.peak_normalize(quiet)).max()) - 0.7) < 1e-6
    silence = np.zeros(10, dtype=np.float32)
    assert bench_cs.peak_normalize(silence) is silence
