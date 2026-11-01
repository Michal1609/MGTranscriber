"""The "separate voices" mode: merge and filter logic, pipeline, API validation, UI strings. No GPU: the
separation and the engine are fakes."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mgtranscriber import separated, separation
from mgtranscriber.engines import Engine, EngineError, register_engine
from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.library import Library
from mgtranscriber.models import Segment, TranscriptionResult, Word
from mgtranscriber.pipeline import TranscriptionRequest, run
from mgtranscriber.server import create_app
from tests.conftest import PROJECT_ROOT, wait_for

# what each fake track "says": word, start, end; the second voice has a bleed word of the first one
TRACK_WORDS = {
    "A.wav": [("Hello", 0.0, 0.4), ("there.", 0.5, 1.0), ("Yes.", 4.0, 4.5)],
    "B.wav": [("Hello", 0.1, 0.4), ("Me", 1.2, 1.5), ("too.", 1.6, 2.0), ("Stop.", 4.0, 4.4)],
}
TURNS = [(0.0, 1.0, "A"), (4.0, 4.6, "A"), (1.1, 2.1, "B")]  # B's "Hello" (0.1-0.4) and "Stop." are bleed


LANGUAGES_SEEN = []  # options.language of every track the fake engine got, in order


class TrackEngine(Engine):
    name = "trackfake"

    def transcribe(self, media, options, progress):
        assert options.diarize is False and options.speakers is None  # a track is one speaker already
        LANGUAGES_SEEN.append(options.language)
        progress(Progress0(0))
        words = [Word(w, s, e) for w, s, e in TRACK_WORDS[media.audio_path.name]]
        progress(Progress0(100))
        return TranscriptionResult([Segment(words[0].start, words[-1].end, " " + " ".join(w.word for w in words),
                                            None, words)], language="en", duration=5.0, model="fake-1")


def Progress0(percent):
    from mgtranscriber.progress import Phase, Progress

    return Progress(Phase.TRANSCRIBE, percent, "fake")


register_engine("trackfake", "tests.test_separated:TrackEngine")


@pytest.fixture
def fake_separation(monkeypatch):
    calls = []

    def separate(audio_path, workdir, settings, progress=None, num_speakers=None):
        calls.append({"audio": audio_path, "workdir": Path(workdir), "num_speakers": num_speakers})
        tracks = []
        for name in ("B", "C", "A"):  # C is a phantom track (no turns); A speaks first and most
            path = Path(workdir) / f"{name}.wav"
            path.write_bytes(b"RIFF fake")
            tracks.append(separation.SpeakerTrack(name, path))
        return separation.SeparationResult(tracks, list(TURNS))

    monkeypatch.setattr(separation, "separate", separate)
    monkeypatch.setattr(separation, "is_available", lambda settings: (True, None))
    LANGUAGES_SEEN.clear()
    monkeypatch.setattr(separated, "LOCAL_ENGINES", (*separated.LOCAL_ENGINES, "trackfake", "mock"))
    return calls


def words(*items):
    return [Word(w, s, e) for w, s, e in items]


# --- merge and filter -----------------------------------------------------------------------------------

def test_filter_drops_words_outside_the_speakers_turns_with_tolerance():
    turns = [(1.0, 2.0, "B"), (0.0, 5.0, "A")]
    kept = separated.filter_words(words(("a", 0.0, 0.4), ("b", 1.0, 1.4), ("c", 2.2, 2.4), ("d", 2.6, 3.0)),
                                  "B", turns, tolerance=0.3)
    assert [w.word for w in kept] == ["b", "c"]  # c: midpoint 2.3 is within 0.3 of the turn end
    assert {w.speaker for w in kept} == {"B"}


def test_speaker_without_turns_says_nothing():
    assert separated.filter_words(words(("a", 0.0, 0.4)), "C", [(0.0, 1.0, "A")]) == []
    # no turns at all: nothing to judge by, nothing is dropped
    kept = separated.filter_words(words(("a", 0.0, 0.4)), "C", [])
    assert [w.word for w in kept] == ["a"] and kept[0].speaker == "C"


def test_merge_orders_overlapping_segments_by_start():
    tracks = {"A": words(("Long", 0.0, 0.5), ("sentence", 0.6, 1.5), ("here.", 1.6, 3.0)),
              "B": words(("Hm", 0.8, 1.0), ("yes.", 1.2, 1.5))}
    turns = [(0.0, 3.0, "A"), (0.7, 1.6, "B")]
    segments = separated.merge(tracks, turns)
    assert [(s.speaker, s.text.strip()) for s in segments] == [("A", "Long sentence here."), ("B", "Hm. Yes.")] or \
        [s.speaker for s in segments] == ["A", "B"]
    assert segments[0].start <= segments[1].start < segments[0].end  # they overlap in time
    assert all(w.speaker == s.speaker for s in segments for w in s.words)


def test_solo_segments_drop_overlapped_words():
    segs = [Segment(0, 3, " a b c", "A", [Word("a", 0.0, 1.0, speaker="A"), Word("b", 1.0, 2.0, speaker="A"),
                                          Word("c", 2.0, 3.0, speaker="A")]),
            Segment(1.2, 1.8, " x", "B", [Word("x", 1.2, 1.8, speaker="B")]),
            Segment(10, 11, " y", "B", [Word("y", 10.0, 11.0, speaker="B")])]
    solo = separated.solo_segments(segs)
    assert [[w.word for w in s.words] for s in solo] == [["a"], ["c"], ["y"]]  # cut where b was left out


def test_long_pause_ends_a_segment():
    segments = separated.build_segments(words(("a", 0.0, 0.5), ("b", 5.0, 5.5)))
    assert [s.text.strip() for s in segments] == ["a", "b"]


def test_words_without_times_follow_their_neighbour():
    kept = separated.filter_words([Word("a", 1.0, 1.4), Word("b"), Word("c", 1.8, 2.0)], "A", [(0.0, 3.0, "A")])
    assert [(w.word, w.start) for w in kept] == [("a", 1.0), ("b", 1.4), ("c", 1.8)]


# --- the editor and exporters -----------------------------------------------------------------------------

def test_editor_accepts_overlap_of_different_speakers_only():
    from mgtranscriber import editor

    ok = editor.parse_segments([
        {"start": 0, "end": 20, "speaker": "A", "text": "long talk"},
        {"start": 5, "end": 6, "speaker": "B", "text": "yes"},
        {"start": 7, "end": 8, "speaker": "B", "text": "right"},
        {"start": 20.2, "end": 21, "speaker": "A", "text": "end"}])
    assert [s.speaker for s in ok] == ["A", "B", "B", "A"]
    with pytest.raises(editor.EditError):  # the same speaker does not talk over themselves
        editor.parse_segments([{"start": 0, "end": 20, "speaker": "A", "text": "x"},
                               {"start": 5, "end": 6, "speaker": "A", "text": "y"}])
    with pytest.raises(editor.EditError):  # still ordered by start
        editor.parse_segments([{"start": 5, "end": 6, "speaker": "B", "text": "y"},
                               {"start": 0, "end": 20, "speaker": "A", "text": "x"}])


def test_exporters_keep_overlapping_cues_in_order():
    from mgtranscriber import exporters

    segs = [Segment(0, 20, " long", "A", []), Segment(5, 6, " yes", "B", [])]
    for exporter in exporters.list_exporters():
        content = exporters.render(exporter, segs, {})
        text = content.decode("utf-8", "ignore") if isinstance(content, bytes) else content
        if exporter.name in ("srt", "vtt", "txt"):
            assert text.index("long") < text.index("yes")


# --- the pipeline ---------------------------------------------------------------------------------------

def test_pipeline_merges_tracks_and_records_separation(settings, media_file, fake_separation, monkeypatch):
    stored = {}
    seen = {}

    def compute(audio, segments, settings, progress):
        seen["audio"], seen["segments"] = audio, segments
        return {"A": [0.5, 0.25], "B": [0.5, 0.25]}

    monkeypatch.setattr("mgtranscriber.voices.compute_embeddings", compute)
    monkeypatch.setattr(separated, "MIN_LANGUAGE_S", 1.0)  # the fake data has little speech
    monkeypatch.setattr("mgtranscriber.pipeline.voices.after_transcription",
                        lambda library, tid, embeddings, settings: stored.update(embeddings=embeddings))
    source = media_file("talk.wav")
    transcript_id = run(TranscriptionRequest(str(source), engine="trackfake", speakers=2, language="auto",
                                             separate_speakers=True),
                        settings.with_overrides(engine="trackfake", hf_token="hf_x"))
    library = Library(settings.output_dir)
    segments = library.segments(transcript_id)
    assert [(s.speaker, s.text.strip()) for s in segments] == [
        ("A", "Hello there."), ("B", "Me too."), ("A", "Yes.")]  # B's "Hello" and "Stop." were bleed
    assert [s.start for s in segments] == sorted(s.start for s in segments)
    meta = library.metadata(transcript_id)
    assert meta.extra["separation"]["used"] is True and meta.extra["separation"]["speakers"] == ["A", "B"]
    assert meta.speakers == ["A", "B"] and meta.model == "fake-1"
    assert fake_separation[0]["num_speakers"] == 2
    # the language is detected on the first track only and fixed for the rest; the phantom track is not run
    assert LANGUAGES_SEEN == [None, "en"]
    assert meta.language == "en"
    assert stored["embeddings"] == {"A": [0.5, 0.25], "B": [0.5, 0.25]}
    assert seen["audio"] == source  # the original audio, not a track
    # the audio stays the original and no track is left in the transcript's folder
    folder = library.path(transcript_id)
    assert fake_separation[0]["audio"] == source
    assert not (folder / "A.wav").exists()
    assert not any(p.name.startswith(".separation") for p in folder.iterdir())


def test_pipeline_without_the_option_is_unchanged(settings, media_file, fake_separation):
    transcript_id = run(TranscriptionRequest(str(media_file("plain.wav"))), settings)
    assert not fake_separation
    assert "separation" not in Library(settings.output_dir).metadata(transcript_id).extra


def test_pipeline_refuses_cloud_engines_and_missing_models(settings, media_file, monkeypatch):
    with pytest.raises(EngineError, match="local engines"):
        separated.check(settings, "elevenlabs")
    with pytest.raises(EngineError, match="local engines"):
        separated.check(settings, "youtube")
    monkeypatch.setattr(separated, "LOCAL_ENGINES", ("mock",))
    monkeypatch.setattr(separation, "is_available", lambda s: (False, "models are missing"))
    with pytest.raises(EngineError, match="models are missing"):
        run(TranscriptionRequest(str(media_file("x.wav")), separate_speakers=True), settings)


# --- API ------------------------------------------------------------------------------------------------

@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        yield c


def test_api_rejects_cloud_engine(client, media_file):
    r = client.post("/api/jobs", json={"source": str(media_file("c.wav")), "engine": "elevenlabs",
                                       "separate_speakers": True})
    assert r.status_code == 400 and r.json()["code"] == "separation_unavailable"
    r = client.post("/api/jobs", json={"source": str(media_file("c.wav")), "engine": "youtube",
                                       "separate_speakers": True})
    assert r.status_code == 400 and r.json()["code"] == "separation_unavailable"


def test_api_rejects_missing_separation_models(client, media_file, monkeypatch):
    monkeypatch.setattr(separation, "is_available", lambda s: (False, "no model"))
    r = client.post("/api/jobs", json={"source": str(media_file("m.wav")), "engine": "whisperx",
                                       "separate_speakers": True})
    assert r.status_code == 400 and r.json()["code"] == "separation_unavailable"


def test_api_job_carries_the_option_through_to_the_transcript(client, media_file, fake_separation):
    r = client.post("/api/jobs", json={"source": str(media_file("both.wav")), "engine": "trackfake",
                                       "separate_speakers": True})
    assert r.status_code == 201 and r.json()["separate_speakers"] is True
    job_id = r.json()["id"]
    wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] in ("done", "error"))
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done", job["error"]
    detail = client.get(f"/api/transcripts/{job['result']}").json()
    assert detail["metadata"]["extra"]["separation"]["used"] is True
    # an ordinary job reports the option as off
    plain = client.post("/api/jobs", json={"source": str(media_file("plain2.wav"))}).json()
    assert plain["separate_speakers"] is False


def test_upload_accepts_the_option(client, fake_separation):
    r = client.post("/api/jobs/upload", files={"file": ("u.wav", b"RIFF fake", "audio/wav")},
                    data={"engine": "elevenlabs", "separate_speakers": "true"})
    assert r.status_code == 400 and r.json()["code"] == "separation_unavailable"


# --- UI strings -----------------------------------------------------------------------------------------

def test_i18n_keys_exist_in_both_languages():
    root = PROJECT_ROOT / "static" / "js" / "i18n"
    for lang in ("en", "cs"):
        text = (root / f"{lang}.js").read_text(encoding="utf-8")
        for key in ("separate", "separate_hint", "separate_blocked", "separated"):
            assert re.search(rf"^\s+{key}: \"", text, re.M), f"{lang}: {key}"
    home = (PROJECT_ROOT / "static" / "js" / "pages" / "home.js").read_text(encoding="utf-8")
    for key in ("home.separate", "home.separate_hint", "home.separate_blocked"):
        assert key in home
    assert "separate_speakers" in home


def test_api_editor_allows_relabelled_interjection_only_when_separated(client, media_file, fake_separation):
    def make(**extra):
        engine = "trackfake" if extra else "mock"  # the fake engine insists on separated tracks
        r = client.post("/api/jobs", json={"source": str(media_file(f"e{len(extra)}.wav")),
                                           "engine": engine, **extra})
        job_id = r.json()["id"]
        wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] in ("done", "error"))
        return client.get(f"/api/jobs/{job_id}").json()["result"]

    overlapping = [{"start": 0, "end": 20, "speaker": "A", "text": "long talk"},
                   {"start": 5, "end": 6, "speaker": "A", "text": "yes"}]
    separated_id = make(separate_speakers=True)
    assert client.put(f"/api/transcripts/{separated_id}/segments", json={"segments": overlapping}).status_code == 200
    plain_id = make()
    assert client.put(f"/api/transcripts/{plain_id}/segments", json={"segments": overlapping}).status_code == 422


def test_editor_allow_self_overlap_flag():
    from mgtranscriber import editor

    items = [{"start": 0, "end": 20, "speaker": "A", "text": "x"}, {"start": 5, "end": 6, "speaker": "A", "text": "y"}]
    assert len(editor.parse_segments(items, allow_self_overlap=True)) == 2
    with pytest.raises(editor.EditError):
        editor.parse_segments(items)


# --- language, solo words, scale -------------------------------------------------------------------------

def test_language_is_not_forced_when_the_first_track_is_silent_at_the_start(settings, media_file, fake_separation):
    # in the fake data the first track speaks 1.6 s of its first 30 s: below the threshold
    transcript_id = run(TranscriptionRequest(str(media_file("quiet.wav")), engine="trackfake", language="auto",
                                             separate_speakers=True),
                        settings.with_overrides(engine="trackfake"))
    assert LANGUAGES_SEEN == [None, None]
    assert Library(settings.output_dir).metadata(transcript_id).language == "en"  # most speech decides


def test_language_is_fixed_when_the_first_track_speaks_enough(settings, media_file, fake_separation, monkeypatch):
    monkeypatch.setattr(separated, "MIN_LANGUAGE_S", 1.0)
    run(TranscriptionRequest(str(media_file("loud.wav")), engine="trackfake", language="auto",
                             separate_speakers=True), settings.with_overrides(engine="trackfake"))
    assert LANGUAGES_SEEN == [None, "en"]


def test_solo_segments_are_cut_where_a_word_was_left_out():
    from mgtranscriber import voices

    segs = [Segment(0, 1.4, " a b c", "A", [Word("a", 0.0, 0.4, speaker="A"), Word("b", 0.5, 0.9, speaker="A"),
                                            Word("c", 1.0, 1.4, speaker="A")]),
            Segment(0.5, 0.9, " x", "B", [Word("x", 0.5, 0.9, speaker="B")])]
    solo = separated.solo_segments(segs)
    assert [[w.word for w in s.words] for s in solo] == [["a"], ["c"]]  # x is overlapped too
    # the gaps are < 1 s, so without the cut label_spans would join a and c into one span
    spans = voices.label_spans(solo, min_s=0.1)["A"]
    assert [(round(s, 2), round(e, 2)) for s, e, _ in spans] == [(0.0, 0.4), (1.0, 1.4)]


def test_solo_segments_are_cut_at_dropped_bleed_words():
    tracks = {"A": words(("a", 0.0, 0.9), ("bleed", 0.95, 1.0), ("c", 1.05, 1.9))}
    turns = [(0.0, 0.9, "A"), (1.05, 1.9, "A"), (0.9, 1.05, "B")]
    removed = separated.removed_midpoints(tracks, turns, tolerance=0.0)
    assert removed == {"A": [0.975]}
    segs = separated.merge(tracks, turns, tolerance=0.0)
    solo = separated.solo_segments(segs, removed)
    assert [[w.word for w in s.words] for s in solo] == [["a"], ["c"]]


def test_solo_segments_scale_linearly():
    import time

    n = 10000
    a = [Word("w", i * 1.0, i * 1.0 + 0.4, speaker="A") for i in range(n)]
    b = [Word("v", i * 1.0 + 0.5, i * 1.0 + 0.9, speaker="B") for i in range(n)]
    segs = [Segment(i, i + 0.4, " w", "A", [a[i]]) for i in range(n)] + \
           [Segment(i + 0.5, i + 0.9, " v", "B", [b[i]]) for i in range(n)]
    started = time.perf_counter()
    solo = separated.solo_segments(segs)
    assert len(solo) == 2 * n
    assert time.perf_counter() - started < 3.0
