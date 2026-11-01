"""Engines without GPU or network: registry metadata, word timing helpers, caption parsing,
cloud responses (fixtures shaped like the providers' documented examples) and the Gemini batching."""

import json

import numpy as np
import pytest

from mgtranscriber import jsruntime
from mgtranscriber.engines import EngineError, TranscribeOptions, cloud, common, get_engine, list_engines
from mgtranscriber.engines.gemini import GeminiEngine, batches, parse_reply, wav_bytes
from mgtranscriber.engines.onnx_asr_local import tokens_to_words
from mgtranscriber.engines.youtube_captions import parse_json3, pick_track, split_turns
from mgtranscriber.models import Word
from mgtranscriber.sources import Media


def test_registry_lists_engines_with_ui_details(settings, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    engines = {e.name: e for e in list_engines()}
    assert {"whisperx", "parakeet", "canary", "youtube", "elevenlabs", "deepgram", "gemini"} <= set(engines)
    assert engines["elevenlabs"].cloud and not engines["parakeet"].cloud
    assert engines["canary"].needs_language and "cs" in engines["canary"].languages
    public = engines["elevenlabs"].public(settings)
    assert public["missing"] == {"code": "missing_key", "detail": "ELEVENLABS_API_KEY"}
    assert "target" not in public and "packages" not in public
    monkeypatch.setenv("ELEVENLABS_API_KEY", "x")
    assert engines["elevenlabs"].missing(settings) is None


def test_env_value_reads_dotenv(tmp_path, settings, monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    (tmp_path / ".env").write_text("DEEPGRAM_API_KEY=from-file\n", encoding="utf-8")
    assert common.env_value(settings.with_overrides(root=tmp_path), "DEEPGRAM_API_KEY") == "from-file"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "from-env")
    assert common.env_value(settings.with_overrides(root=tmp_path), "DEEPGRAM_API_KEY") == "from-env"


def test_cloud_engines_need_their_key(settings, monkeypatch):
    for name, key in (("elevenlabs", "ELEVENLABS_API_KEY"), ("deepgram", "DEEPGRAM_API_KEY"), ("gemini", "GEMINI_API_KEY")):
        monkeypatch.delenv(key, raising=False)
        with pytest.raises(EngineError, match=key):
            get_engine(name, settings.with_overrides(root=settings.output_dir)).check(TranscribeOptions(diarize=False))


def test_canary_requires_a_language(settings):
    engine = get_engine("canary", settings)
    with pytest.raises(EngineError, match="language"):
        engine.check(TranscribeOptions(language=None, diarize=False))
    with pytest.raises(EngineError, match="ja"):
        engine.check(TranscribeOptions(language="ja", diarize=False))
    engine.check(TranscribeOptions(language="cs", diarize=False))


def test_parakeet_tokens_become_words():
    tokens = [" You", "'", "ve", " been", " work", "ing", " ", " here."]
    times = [0.0, 0.16, 0.24, 0.32, 0.4, 0.48, 0.5, 0.56]
    words = tokens_to_words(tokens, times, offset=10.0, end=11.2)
    assert [w.word for w in words] == ["You've", "been", "working", "here."]
    assert [w.start for w in words] == [10.0, 10.32, 10.4, 10.56]
    assert [w.end for w in words] == [10.32, 10.4, 10.56, 11.2]


def test_split_sentences_on_punctuation_speaker_and_length():
    words = [Word("Hi.", 0, 1, speaker="A"), Word("How", 1, 2, speaker="A"), Word("are", 2, 3, speaker="B"),
             Word("you?", 3, 4, speaker="B")] + [Word("x", 4 + i, 5 + i) for i in range(25)]
    segments = common.split_sentences(words, max_seconds=10)
    assert [s.text for s in segments[:3]] == [" Hi.", " How", " are you?"]
    assert [s.speaker for s in segments[:3]] == ["A", "A", "B"]
    assert all(s.end - s.start <= 10 for s in segments)
    assert sum(len(s.words) for s in segments) == len(words)


def test_spread_words_fills_the_segment():
    out = common.spread_words({"start": 1.0, "end": 3.0, "text": " ab cd"})
    assert out["words"][0]["start"] == 1.0 and out["words"][-1]["end"] == pytest.approx(3.0)


# --- YouTube captions -------------------------------------------------------------------

JSON3_AUTO = {"events": [
    {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "ahoj"}, {"utf8": " jak", "tOffsetMs": 500},
                                                  {"utf8": " se", "tOffsetMs": 900}]},
    {"tStartMs": 1500, "dDurationMs": 10, "aAppend": 1, "segs": [{"utf8": "\n"}]},
    {"tStartMs": 2000, "dDurationMs": 1500, "segs": [{"utf8": "máš"}, {"utf8": " dneska", "tOffsetMs": 400}]},
]}
JSON3_MANUAL = {"events": [{"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "Dobrý den.\nVítejte"}]}]}


def test_parse_json3_auto_captions_have_word_times():
    words = parse_json3(JSON3_AUTO)
    assert [w.word for w in words] == ["ahoj", "jak", "se", "máš", "dneska"]
    assert [w.start for w in words] == [0.0, 0.5, 0.9, 2.0, 2.4]
    assert all(w.end is not None and w.end >= w.start for w in words)


def test_parse_json3_manual_captions_spread_over_the_line():
    words = parse_json3(JSON3_MANUAL)
    assert [w.word for w in words] == ["Dobrý", "den.", "Vítejte"]
    assert words[0].start == 1.0 and words[-1].end == pytest.approx(3.0)


def test_split_turns_breaks_at_speaker_change_marks():
    words = [Word(w, i, i + 1) for i, w in enumerate(["ahoj", "jak", ">>", "dobře", "díky"])]
    segments = split_turns(words, 20)
    assert [s.text for s in segments] == [" ahoj jak", " dobře díky"]


def test_pick_track_prefers_manual_then_original_auto_never_translations():
    info = {"language": "en", "subtitles": {"en-US": [{"ext": "json3"}]},
            "automatic_captions": {"en-orig": [{"ext": "json3"}], "cs": [{"ext": "json3"}]}}
    assert pick_track(info, "en")[:2] == ("manual", "en-US")
    assert pick_track(info, "cs")[:2] == ("manual", "en-US")  # no Czech captions: the video's own language
    info["subtitles"] = {}
    assert pick_track(info, "cs")[:2] == ("auto", "en-orig")
    assert pick_track({"automatic_captions": {"cs": [{}]}, "language": "en"}, "cs") is None


def test_youtube_engine_uses_captions_without_audio(settings, monkeypatch):
    engine = get_engine("youtube", settings)
    assert engine.needs_audio is False
    monkeypatch.setattr(engine, "_captions", lambda media, language, progress: ("auto", "cs", parse_json3(JSON3_AUTO)))
    media = Media("https://www.youtube.com/watch?v=x", "url", "t", info={"duration": 4})
    result = engine.transcribe(media, TranscribeOptions(language="cs", diarize=False), lambda p: None)
    assert result.language == "cs" and result.model == "youtube-auto"
    assert " ".join(s.text.strip() for s in result.segments) == "ahoj jak se máš dneska"


def test_youtube_engine_falls_back_to_audio(settings, monkeypatch, media_file):
    engine = get_engine("youtube", settings.with_overrides(extra={"youtube_fallback_engine": "mock"}))
    monkeypatch.setattr(engine, "_captions", lambda *a: None)
    media = Media("x", "file", "t", audio_path=media_file())
    result = engine.transcribe(media, TranscribeOptions(language="en", diarize=False), lambda p: None)
    assert result.model == "mock:mock-1" and result.segments


# --- cloud ------------------------------------------------------------------------------

ELEVENLABS_REPLY = {"language_code": "ces", "language_probability": 0.98, "text": "Dobrý den. Ahoj!", "words": [
    {"text": "Dobrý", "type": "word", "start": 0.1, "end": 0.4, "speaker_id": "speaker_1", "logprob": -0.1},
    {"text": " ", "type": "spacing", "start": 0.4, "end": 0.45, "speaker_id": "speaker_1"},
    {"text": "den.", "type": "word", "start": 0.45, "end": 0.8, "speaker_id": "speaker_1"},
    {"text": "(laughter)", "type": "audio_event", "start": 0.8, "end": 1.0, "speaker_id": "speaker_1"},
    {"text": "Ahoj!", "type": "word", "start": 1.2, "end": 1.6, "speaker_id": "speaker_0"},
]}
DEEPGRAM_REPLY = {"metadata": {"duration": 2.5}, "results": {"channels": [{"detected_language": "cs", "alternatives": [{
    "transcript": "dobrý den ahoj", "words": [
        {"word": "dobrý", "start": 0.1, "end": 0.4, "confidence": 0.9, "speaker": 0, "punctuated_word": "Dobrý"},
        {"word": "den", "start": 0.45, "end": 0.8, "confidence": 0.9, "speaker": 0, "punctuated_word": "den."},
        {"word": "ahoj", "start": 1.2, "end": 1.6, "confidence": 0.8, "speaker": 1, "punctuated_word": "Ahoj!"}]}]}]}}


def test_parse_elevenlabs():
    words = cloud.parse_elevenlabs(ELEVENLABS_REPLY)
    assert [(w.word, w.speaker) for w in words] == [("Dobrý", "SPEAKER_00"), ("den.", "SPEAKER_00"), ("Ahoj!", "SPEAKER_01")]


def test_parse_deepgram():
    words, language, duration = cloud.parse_deepgram(DEEPGRAM_REPLY)
    assert [w.word for w in words] == ["Dobrý", "den.", "Ahoj!"]
    assert [w.speaker for w in words] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]
    assert (language, duration) == ("cs", 2.5)


class FakeResponse:
    def __init__(self, status, data):
        self.status_code, self._data, self.text = status, data, json.dumps(data)

    def json(self):
        return self._data


@pytest.mark.parametrize("name,key,reply", [("elevenlabs", "ELEVENLABS_API_KEY", ELEVENLABS_REPLY),
                                            ("deepgram", "DEEPGRAM_API_KEY", DEEPGRAM_REPLY)])
def test_cloud_engine_end_to_end_with_mocked_http(name, key, reply, settings, monkeypatch, media_file):
    import httpx

    calls = []
    monkeypatch.setenv(key, "secret")

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(200, reply)

    monkeypatch.setattr(httpx, "post", post)
    engine = get_engine(name, settings)
    engine.check(TranscribeOptions())
    result = engine.transcribe(Media("x", "file", "t", audio_path=media_file("a.wav")),
                               TranscribeOptions(language="cs", speakers=2), lambda p: None)
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.segments[0].text == " Dobrý den."
    url, kwargs = calls[0]
    if name == "elevenlabs":
        assert kwargs["headers"]["xi-api-key"] == "secret"
        assert kwargs["data"]["diarize"] == "true" and kwargs["data"]["num_speakers"] == "2"
    else:
        assert url.endswith("/v1/listen") and kwargs["headers"]["Authorization"] == "Token secret"
        assert kwargs["params"]["language"] == "cs" and kwargs["params"]["diarize"] == "true"


def test_cloud_engine_reports_a_rejected_key(settings, monkeypatch, media_file):
    import httpx

    monkeypatch.setenv("DEEPGRAM_API_KEY", "bad")
    monkeypatch.setattr(httpx, "post", lambda url, **kw: FakeResponse(401, {"err_msg": "Invalid credentials"}))
    with pytest.raises(EngineError, match="rejected the API key"):
        get_engine("deepgram", settings).transcribe(Media("x", "file", "t", audio_path=media_file("a.wav")),
                                                    TranscribeOptions(), lambda p: None)


# --- Gemini -----------------------------------------------------------------------------

def test_batches_respect_the_limit():
    pieces = [(i * 30.0, i * 30.0 + 25.0) for i in range(30)]
    groups = batches(pieces, limit=100)
    assert [len(g) for g in groups] == [4] * 7 + [2]
    assert [p for g in groups for p in g] == pieces


def test_wav_bytes_is_valid_wav():
    data = wav_bytes(np.zeros(16000, dtype=np.float32))
    assert data[:4] == b"RIFF" and len(data) == 44 + 32000


def test_parse_reply_checks_the_count():
    assert parse_reply('{"language": "cs", "parts": ["a", " b "]}', 2) == ("cs", ["a", "b"])
    with pytest.raises(ValueError):
        parse_reply('{"language": "cs", "parts": ["a"]}', 2)


class FakeGeminiClient:
    """Answers like Gemini with the JSON schema; the first multi-part request returns one part too few."""

    def __init__(self):
        self.requests = []
        self.models = self

    def generate_content(self, model, contents, config):
        n = sum(1 for c in contents if isinstance(c, str) and c.startswith("Part "))
        self.requests.append(n)
        parts = [f"věta {len(self.requests)}.{i}" for i in range(n)]
        if len(self.requests) == 1 and n > 1:
            parts = parts[:-1]
        return type("R", (), {"text": json.dumps({"language": "cs", "parts": parts})})()


def test_gemini_hybrid_with_fake_client(settings, monkeypatch):
    pytest.importorskip("google.genai")
    pieces = [(0.0, 20.0), (21.0, 40.0), (41.0, 60.0)]
    monkeypatch.setattr("mgtranscriber.engines.gemini.speech_pieces", lambda audio: pieces)
    monkeypatch.setattr(common, "load_audio", lambda path: np.zeros(16000 * 61, dtype=np.float32))
    aligned = []
    monkeypatch.setattr(common, "align", lambda segs, audio, lang, progress: aligned.append(lang) or segs)
    client = FakeGeminiClient()
    engine = GeminiEngine(settings, client=client)
    engine.check(TranscribeOptions(diarize=False))
    result = engine.transcribe(Media("x", "file", "t", audio_path=settings.root / "x.wav"),
                               TranscribeOptions(language=None, diarize=False), lambda p: None)
    assert client.requests == [3, 1, 2]  # wrong count -> retried as halves
    assert [(s.start, s.end) for s in result.segments] == pieces
    assert result.language == "cs" and aligned == ["cs"]


# --- JS runtime for yt-dlp --------------------------------------------------------------

def test_js_runtime_required_only_for_youtube(monkeypatch):
    monkeypatch.setattr(jsruntime, "detect", lambda: None)
    with pytest.raises(jsruntime.JsRuntimeMissing, match="Deno"):
        jsruntime.require("https://www.youtube.com/watch?v=x")
    jsruntime.require("https://vimeo.com/123")
    assert jsruntime.is_youtube("https://youtu.be/x") and not jsruntime.is_youtube("https://notyoutube.com/")


def test_js_runtime_versions(monkeypatch):
    jsruntime.detect.cache_clear()
    monkeypatch.setattr(jsruntime.shutil, "which", lambda name: name)
    versions = {"deno": (2, 2), "node": (22, 1)}
    monkeypatch.setattr(jsruntime, "_version", lambda exe: versions[exe])
    assert jsruntime.detect() == ("node", "22.1")
    jsruntime.detect.cache_clear()
    versions["deno"] = (2, 3)
    assert jsruntime.detect() == ("deno", "2.3")
    jsruntime.detect.cache_clear()


# --- review fixes -----------------------------------------------------------------------

def test_parakeet_space_token_starts_a_new_word():
    # Parakeet's vocabulary has a bare "▁" token; digits come after it without the word-start mark
    words = tokens_to_words([" rok", " ", "2", "0", " je"], [0.0, 0.3, 0.4, 0.5, 0.8], 10.0, 11.0)
    assert [w.word for w in words] == ["rok", "20", "je"]
    assert words[1].start == 10.4


def test_word_ends_do_not_swallow_pauses():
    words = tokens_to_words([" a", " po", " pauze"], [0.0, 0.2, 6.0], 0.0, 30.0)
    assert [w.end for w in words] == [0.2, 1.7, 7.5]  # capped at MAX_WORD_S, also the last word
    words = parse_json3({"events": [{"tStartMs": 0, "segs": [{"utf8": "ahoj"}, {"utf8": " tam", "tOffsetMs": 5000}]}]})
    assert words[0].end == 1.5


def test_cut_duplicate_is_dropped():
    before = [Word("pořád", 19.5, 19.9)]
    assert common.drop_cut_duplicate(before, [Word("řád", 19.7, 20.0), Word("dál", 20.1, 20.4)])[0].word == "řád"
    assert [w.word for w in common.drop_cut_duplicate(before, [Word("pořád", 19.6, 20.0), Word("dál", 20.1, 20.4)])] == ["dál"]
    after = common.drop_cut_duplicate([Word("po", 19.8, 20.0)], [Word("pořád", 19.85, 20.2)])
    assert after == [] or after[0].word != "pořád"
    assert common.drop_cut_duplicate_text(" jdeme dál,", "Dál a dál") == "a dál"
    assert common.drop_cut_duplicate_text(" jdeme dál", "potom") == "potom"


def test_split_by_speaker_after_diarization():
    w = lambda word, start, end, speaker=None: {"word": word, "start": start, "end": end, "speaker": speaker}  # noqa: E731
    seg = {"start": 0.0, "end": 4.0, "text": " Ahoj, jak se máš dnes? Dobře", "speaker": "SPEAKER_00", "words": [
        w("Ahoj,", 0.0, 0.5, "SPEAKER_00"), w("jak", 0.6, 0.9, "SPEAKER_00"), w("se", 1.0, 1.2, "SPEAKER_00"),
        w("máš", 1.3, 1.6, "SPEAKER_01"),  # flicker at the boundary: stays with SPEAKER_00
        w("dnes?", 1.7, 2.0, "SPEAKER_00"), w("Dobře", 2.2, 3.0, "SPEAKER_01"), w("díky", 3.1, 3.5)]}
    out = common.split_by_speaker([seg, {"start": 5.0, "end": 6.0, "text": " x", "words": [{"word": "x", "start": 5.0}]}])
    assert [(s["speaker"], s["text"], s["start"], s["end"]) for s in out[:2]] == [
        ("SPEAKER_00", " Ahoj, jak se máš dnes?", 0.0, 2.0), ("SPEAKER_01", " Dobře díky", 2.2, 3.5)]
    assert out[2]["text"] == " x"
    short = {**seg, "words": seg["words"][:5]}  # only the flicker: the segment stays whole
    assert common.split_by_speaker([short]) == [{**short, "speaker": "SPEAKER_00"}]


def test_language_warning_catches_slovak_in_czech():
    slovak = ["Bolo to veľmi dôležité, ako sa ukázalo.", "Ktorý z nich je tiež ešte tu?",
              "Schengenský prostor nicméně v tomto ohľadu funguje podobne ako jedna zem."]  # Parakeet on FLEURS cs
    czech = ["Bylo to velmi důležité, jak se ukázalo.", "Který z nich je tu ještě?", "Jsou tu i jiní, nejen oni."]
    assert "3 of 6 pieces do not look like 'cs' (Slovak)" in common.language_warning(czech + slovak, "cs")
    assert common.language_warning(czech, "cs") is None
    assert common.language_warning(czech * 20 + slovak[:1], "cs") is None  # one stray piece is not a warning
    assert "(Czech)" in common.language_warning(czech, "sk")
    # Parakeet on FLEURS cs, short VAD pieces
    drift = ["Витворила білоруська українська фронти по те.", "Oliver Saks Na swim članku bezrętowarzeć naznaczę.",
             "Слово цивилизация pochází z latinského civліz."]
    assert "(Cyrillic, Polish)" in common.language_warning(czech * 5 + drift, "cs")
    assert common.language_warning(["anything"], None) is None


def test_pick_track_without_language_skips_translations():
    info = {"subtitles": {"en": [{"ext": "json3"}]}, "automatic_captions": {"cs-orig": [{"ext": "json3"}], "en": [{}]}}
    assert pick_track(info, None)[:2] == ("auto", "cs-orig")  # the English manual track is a translation
    info["subtitles"]["cs"] = [{"ext": "json3"}]
    assert pick_track(info, None)[:2] == ("manual", "cs")


def test_youtube_engine_local_file_and_refused_info(settings, monkeypatch, media_file):
    import yt_dlp

    engine = get_engine("youtube", settings.with_overrides(extra={"youtube_fallback_engine": "mock"}))
    monkeypatch.setattr(engine, "_captions", lambda *a: pytest.fail("a local file has no captions"))
    result = engine.transcribe(Media(str(media_file()), "file", "t", audio_path=media_file()),
                               TranscribeOptions(language="en", diarize=False), lambda p: None)
    assert result.model == "mock:mock-1"

    class RefusingYDL:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, *a, **k): raise yt_dlp.utils.DownloadError("Sign in to confirm you're not a bot")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", RefusingYDL)
    engine = get_engine("youtube", settings)
    assert engine._captions(Media("https://www.youtube.com/watch?v=x", "url", "t"), None, lambda p: None) is None


def test_youtube_fallback_cannot_be_youtube(settings):
    engine = get_engine("youtube", settings.with_overrides(extra={"youtube_fallback_engine": "youtube"}))
    with pytest.raises(EngineError, match="cannot be 'youtube'"):
        engine.check(TranscribeOptions(diarize=False))


def test_cloud_details(settings, monkeypatch, media_file):
    import httpx

    assert cloud.iso639_1("ces") == "cs" and cloud.iso639_1("cs") == "cs" and cloud.iso639_1(None) is None
    calls = []

    def post(url, **kwargs):
        content = kwargs.get("content")
        calls.append((url, kwargs, b"".join(content) if content is not None else None))
        return FakeResponse(200, ELEVENLABS_REPLY if "elevenlabs" in url else DEEPGRAM_REPLY)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
    monkeypatch.delenv("DEEPGRAM_API_URL", raising=False)
    path = media_file("a.wav")
    result = get_engine("elevenlabs", settings).transcribe(Media("x", "file", "t", audio_path=path),
                                                           TranscribeOptions(), lambda p: None)
    assert result.language == "cs"  # "ces" in the reply
    get_engine("deepgram", settings).transcribe(Media("x", "file", "t", audio_path=path), TranscribeOptions(), lambda p: None)
    url, kwargs, body = calls[-1]
    assert url.startswith("https://api.eu.deepgram.com/")  # EU by default
    assert body == path.read_bytes() and kwargs["headers"]["Content-Length"] == str(len(body))
    assert kwargs["timeout"].connect == cloud.CONNECT_TIMEOUT_S and kwargs["timeout"].read == cloud.TIMEOUT_S


def test_parakeet_engine_joins_vad_pieces(settings, monkeypatch, media_file):
    """The wiring of OnnxAsrEngine.transcribe with a fake onnx-asr model: a hard VAD cut repeats a word."""
    import types

    import onnx_asr

    from mgtranscriber.engines import onnx_asr_local

    piece = lambda start, end, tokens, times: types.SimpleNamespace(  # noqa: E731
        start=start, end=end, text="".join(tokens), tokens=tokens, timestamps=times)
    pieces = [piece(0.0, 20.03, [" Bolo", " to", " veľmi"], [0.0, 0.5, 19.8]),
              piece(19.97, 25.0, [" veľmi", " dobré", "."], [0.0, 0.4, 0.8]),  # overlaps the previous piece
              piece(30.0, 33.0, [" Ako", " sa", " máš", " ", "2", "0", "?"], [0.0, 0.5, 1.0, 1.5, 1.6, 1.7, 2.0])]

    class FakeModel:
        def with_vad(self, *a, **k): return self
        def with_timestamps(self): return self
        def recognize(self, audio, **kwargs): return iter(pieces)

    monkeypatch.setattr(onnx_asr_local, "load", lambda *a: (FakeModel(), True))
    monkeypatch.setattr(onnx_asr, "load_vad", lambda *a, **k: object())
    monkeypatch.setattr(common, "load_audio", lambda path: np.zeros(16000 * 34, dtype=np.float32))
    messages = []
    result = get_engine("parakeet", settings).transcribe(
        Media("x", "file", "t", audio_path=media_file()), TranscribeOptions(language="cs", diarize=False),
        lambda p: messages.append(p.message or ""))
    words = [w.word for s in result.segments for w in s.words]
    assert words == ["Bolo", "to", "veľmi", "dobré.", "Ako", "sa", "máš", "20?"]
    assert any("(Slovak)" in m for m in messages)


# --- YouTube's intermittent 403 on the audio download -----------------------------------

YT_URL = "https://www.youtube.com/watch?v=x"


@pytest.fixture
def fake_ytdlp(monkeypatch, tmp_path):
    """yt_dlp.YoutubeDL whose downloads fail with a 403 for the first ``fails`` calls (None = always)."""
    import yt_dlp

    from mgtranscriber import sources

    state = {"calls": 0, "fails": 0, "options": [], "sleeps": []}

    class FakeYoutubeDL:
        def __init__(self, options):
            state["options"].append(options)
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=True):
            state["calls"] += 1
            outdir = tmp_path
            (outdir / "audio.webm.part").write_bytes(b"partial")
            if state["fails"] is None or state["calls"] <= state["fails"]:
                raise yt_dlp.utils.DownloadError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
            (outdir / "audio.webm.part").unlink()
            (outdir / "audio.webm").write_bytes(b"audio")
            return {"ext": "webm"}

        def prepare_filename(self, info):
            return str(tmp_path / "audio.webm")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(sources.time, "sleep", lambda s: state["sleeps"].append(s))
    monkeypatch.setattr(jsruntime, "detect", lambda: ("node", "24.0"))
    return state


def test_fetch_audio_retries_after_a_403(fake_ytdlp, tmp_path):
    from mgtranscriber.sources import fetch_audio

    fake_ytdlp["fails"] = 1
    messages = []
    media = Media(YT_URL, "url", "t")
    path = fetch_audio(media, tmp_path, lambda p: messages.append(p.message))
    assert path == tmp_path / "audio.webm" and path.read_bytes() == b"audio"
    assert fake_ytdlp["calls"] == 2 and fake_ytdlp["sleeps"] == [2]
    assert any("retrying 2/3" in (m or "") for m in messages)
    assert not list(tmp_path.glob("*.part"))
    assert all("extractor_args" not in o for o in fake_ytdlp["options"])


def test_fetch_audio_last_attempt_uses_clients_without_po_token(fake_ytdlp, tmp_path):
    from mgtranscriber.sources import FALLBACK_CLIENTS, fetch_audio

    fake_ytdlp["fails"] = 3
    fetch_audio(Media(YT_URL, "url", "t"), tmp_path)
    assert fake_ytdlp["calls"] == 4
    assert fake_ytdlp["options"][-1]["extractor_args"] == {"youtube": {"player_client": list(FALLBACK_CLIENTS)}}
    assert "web" not in FALLBACK_CLIENTS and "mweb" not in FALLBACK_CLIENTS


def test_fetch_audio_gives_a_clear_error_when_blocked(fake_ytdlp, tmp_path):
    from mgtranscriber.sources import DownloadBlocked, fetch_audio

    fake_ytdlp["fails"] = None
    with pytest.raises(DownloadBlocked, match="winget install DenoLand.Deno") as caught:
        fetch_audio(Media(YT_URL, "url", "t"), tmp_path)
    assert isinstance(caught.value, EngineError)
    assert "zablokoval" in str(caught.value) and "blocked" in str(caught.value)
    assert fake_ytdlp["calls"] == 4 and not list(tmp_path.glob("*.part"))


def test_fetch_audio_does_not_retry_other_errors(monkeypatch, tmp_path):
    import yt_dlp

    from mgtranscriber.sources import fetch_audio

    calls = []

    class Private:
        def __init__(self, options): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False

        def extract_info(self, url, download=True):
            calls.append(url)
            raise yt_dlp.utils.DownloadError("ERROR: Private video")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", Private)
    monkeypatch.setattr(jsruntime, "detect", lambda: ("node", "24.0"))
    with pytest.raises(yt_dlp.utils.DownloadError, match="Private"):
        fetch_audio(Media(YT_URL, "url", "t"), tmp_path)
    assert len(calls) == 1


def test_youtube_engine_keeps_captions_when_the_audio_is_blocked(settings, monkeypatch, tmp_path):
    from mgtranscriber.engines import youtube_captions
    from mgtranscriber.sources import DownloadBlocked

    engine = get_engine("youtube", settings)
    monkeypatch.setattr(engine, "check", lambda options: None)
    monkeypatch.setattr(engine, "_captions", lambda media, language, progress: ("auto", "cs", parse_json3(JSON3_AUTO)))

    def blocked(media, out_dir, progress=None):
        raise DownloadBlocked("blocked")

    monkeypatch.setattr(youtube_captions, "fetch_audio", blocked)
    messages = []
    media = Media(YT_URL, "url", "t", info={"duration": 4}, workdir=tmp_path)
    result = engine.transcribe(media, TranscribeOptions(language="cs", diarize=True), lambda p: messages.append(p.message))
    assert result.model == "youtube-auto" and result.segments and not any(s.speaker for s in result.segments)
    assert any("without speakers" in (m or "") for m in messages)
