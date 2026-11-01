"""Existing YouTube captions as a fast engine: no GPU, no audio download (unless speakers are wanted).

Order: manual captions in the requested language, YouTube's automatic captions of the original
audio (``<lang>-orig``), then the same for the video's own language. Auto-translated captions are
never used, and without a requested language neither are manual captions in another language than
the video's (they are usually translations). Captions are read as ``json3``: automatic captions carry a time per word, manual ones
per line (word times are spread over the line).

With diarization on, the audio is downloaded and pyannote assigns speakers by word times.
When the video has no usable captions (none, or YouTube refused them), the audio is transcribed
by the fallback engine (``youtube_fallback_engine`` in settings.json, default ``whisperx``).
So is a local file, and a video YouTube refuses to describe (bot check, geo block): the audio
download then either works or reports the real reason.
"""

from __future__ import annotations

import json

from ..models import Segment, TranscriptionResult, Word
from ..progress import Phase, Progress, ProgressCallback
from ..sources import YTDLP_BASE, DownloadBlocked, Media, fetch_audio, is_url
from . import common
from .base import Engine, EngineError, TranscribeOptions

AUTO_SEGMENT_S = 8.0  # automatic captions have little punctuation, so cut segments by time


def pick_track(info: dict, language: str | None) -> tuple[str, str, list[dict]] | None:
    """``(kind, language key, formats)`` of the best caption track, or None."""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    # the video's language: from the metadata, or from YouTube's ASR track of the original audio
    original = info.get("language") or next((k.removesuffix("-orig") for k in auto if k.endswith("-orig")), None)
    for lang in dict.fromkeys(filter(None, [language, original])):
        for key in [lang, *sorted(k for k in manual if k.startswith(lang + "-"))]:
            if manual.get(key):
                return "manual", key, manual[key]
        if auto.get(lang + "-orig"):
            return "auto", lang + "-orig", auto[lang + "-orig"]
        if lang == original and auto.get(lang):
            return "auto", lang, auto[lang]
    if not language and not original and manual:  # nothing tells the video's language: take what there is
        key = next(iter(manual))
        return "manual", key, manual[key]
    return None


def parse_json3(data: dict) -> list[Word]:
    """Words with times from YouTube's json3 caption format."""
    words: list[Word] = []
    for event in data.get("events") or []:
        segs = event.get("segs")
        if not segs:
            continue
        start = event.get("tStartMs", 0) / 1000
        end = start + event.get("dDurationMs", 0) / 1000
        timed = len(segs) > 1 or "tOffsetMs" in segs[0]
        if timed:
            for seg in segs:
                for token in seg.get("utf8", "").split():
                    words.append(Word(token, round(start + seg.get("tOffsetMs", 0) / 1000, 3)))
        else:
            text = " ".join(seg.get("utf8", "") for seg in segs).replace("\n", " ")
            if text.strip():
                spread = common.spread_words({"start": start, "end": end, "text": text})
                words += [Word(w["word"], w["start"], w["end"]) for w in spread["words"]]
    return common.set_word_ends(words)


def split_turns(words: list[Word], max_seconds: float) -> list[Segment]:
    """Segments that never span a ``>>`` (YouTube's mark for a change of speaker); the marks are dropped."""
    segments: list[Segment] = []
    turn: list[Word] = []
    for word in [*words, None]:
        if word is None or word.word == ">>":
            segments += common.split_sentences(turn, max_seconds)
            turn = []
        else:
            turn.append(word)
    return segments


class YouTubeCaptionsEngine(Engine):
    name = "youtube"
    needs_audio = False
    supports_diarization = True

    def check(self, options: TranscribeOptions) -> None:
        if self._fallback_name() == self.name:
            raise EngineError("youtube_fallback_engine cannot be 'youtube' (it would call itself); use e.g. whisperx")
        if options.diarize:
            common.check_diarization_access(self.settings)

    def _captions(self, media: Media, language: str | None, progress: ProgressCallback):
        import yt_dlp

        progress(Progress(Phase.DOWNLOAD, None, "Looking for existing captions..."))
        with yt_dlp.YoutubeDL(YTDLP_BASE) as ydl:
            try:
                info = ydl.extract_info(media.source, download=False)
            except Exception as e:  # bot check, geo block, private video: the audio download reports it properly
                progress(Progress(Phase.DOWNLOAD, None, f"Captions are not available ({type(e).__name__}: {e})"))
                return None
            track = pick_track(info, language)
            if track is None:
                return None
            kind, key, formats = track
            fmt = next((f for f in formats if f.get("ext") == "json3"), None)
            if fmt is None:
                return None
            try:
                raw = ydl.urlopen(fmt["url"]).read()
                words = parse_json3(json.loads(raw)) if raw else []
            except Exception as e:  # blocked (PO token), network, bad JSON: fall back to the audio
                progress(Progress(Phase.DOWNLOAD, None, f"Captions could not be read ({type(e).__name__}: {e})"))
                return None
        return (kind, key.removesuffix("-orig").split("-")[0], words) if words else None

    def transcribe(self, media: Media, options: TranscribeOptions, progress: ProgressCallback) -> TranscriptionResult:
        if not is_url(media.source):
            progress(Progress(Phase.DOWNLOAD, None, "A local file has no YouTube captions"))
            return self._fallback(media, options, progress)
        found = self._captions(media, options.language, progress)
        if found is None:
            return self._fallback(media, options, progress)
        kind, language, words = found
        if options.language and language != options.language:
            progress(Progress(Phase.DOWNLOAD, None, f"No captions in '{options.language}', using the video's '{language}'"))
        progress(Progress(Phase.TRANSCRIBE, 100, f"Using {kind} captions ({language}), {len(words)} words"))
        segments = split_turns(words, AUTO_SEGMENT_S if kind == "auto" else 20.0)
        voices: dict = {}
        if options.diarize and segments:
            try:
                fetch_audio(media, media.workdir, progress)
            except DownloadBlocked:  # the captions are good enough on their own
                progress(Progress(Phase.DOWNLOAD, None,
                                  "Warning: YouTube blocked the audio download, so the captions are used without speakers"))
            else:
                audio = common.load_audio(media.audio_path)
                diarized = common.diarize([s.to_dict() for s in segments], audio, self.settings, options.speakers,
                                          progress, voices)
                segments = [Segment.from_dict(s) for s in common.split_by_speaker(diarized)]
        return TranscriptionResult(segments, language=language, duration=media.info.get("duration"),
                                   model=f"youtube-{kind}", speaker_embeddings=voices or None)

    def _fallback_name(self) -> str:
        return self.settings.extra.get("youtube_fallback_engine") or "whisperx"

    def _fallback(self, media: Media, options: TranscribeOptions, progress: ProgressCallback) -> TranscriptionResult:
        from . import get_engine

        name = self._fallback_name()
        progress(Progress(Phase.DOWNLOAD, None, f"No usable captions, transcribing the audio with {name}"))
        engine = get_engine(name, self.settings)
        fallback = TranscribeOptions(language=options.language or media.info.get("language"), model=None,
                                     speakers=options.speakers, diarize=options.diarize and engine.supports_diarization,
                                     batch_size=options.batch_size, extra=options.extra)
        engine.check(fallback)
        fetch_audio(media, media.workdir, progress)
        result = engine.transcribe(media, fallback, progress)
        result.model = f"{name}:{result.model}"
        return result
