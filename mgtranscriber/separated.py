"""The "separate voices" mode: one audio track per speaker, each transcribed on its own, then merged.

With overlapping speech the engines write down only the dominant voice. Here ``separation`` splits the
recording into one track per speaker; every track is transcribed with the chosen local engine (without
diarization: the track *is* the speaker), each word gets the speaker of its track, words that lie outside
that speaker's activity are dropped (bleed of the other voice into the track) and all words are merged by
time into segments. Segments of different speakers may overlap in time; they are ordered by start.
"""

from __future__ import annotations

import bisect
import dataclasses
import logging
import tempfile
from pathlib import Path
from typing import Iterable

from .config import Settings
from .engines import EngineError, TranscribeOptions
from .engines.base import Engine
from .engines import common
from .models import Segment, TranscriptionResult, Word
from .progress import Phase, Progress, ProgressCallback, no_progress
from .sources import Media

log = logging.getLogger(__name__)

#: engines that run on this computer and transcribe an audio file themselves
LOCAL_ENGINES = ("whisperx", "parakeet", "canary")
TOLERANCE_S = 0.3  # a word's midpoint may lie this far outside its speaker's turns
MIN_LANGUAGE_S = 5.0  # speech in the first 30 s a track needs for its language to be used for all tracks
GAP_S = 2.0  # a pause of the same speaker longer than this ends a segment
METADATA_KEY = "separation"  # in the transcript metadata's ``extra``


def unavailable_reason(settings: Settings, engine_name: str) -> str | None:
    """Why separation cannot be used with ``engine_name`` here, or None when it can."""
    if engine_name not in LOCAL_ENGINES:
        return (f"Voice separation works with the local engines ({', '.join(LOCAL_ENGINES)}), "
                f"not with '{engine_name}'.")
    try:
        from . import separation
    except ImportError as e:
        return f"Voice separation is not available: {e}"
    ok, reason = separation.is_available(settings)
    return None if ok else (reason or "Voice separation is not available.")


def check(settings: Settings, engine_name: str) -> None:
    """Fail fast (before any download or model) when the request cannot be separated."""
    if reason := unavailable_reason(settings, engine_name):
        raise EngineError(reason)


def _intervals(turns: Iterable[tuple[float, float, str]], speaker: str, tolerance: float) -> tuple[list, list]:
    """The speaker's turns widened by ``tolerance`` and merged: ``(starts, ends)`` of disjoint, sorted intervals."""
    spans = sorted((max(0.0, s - tolerance), e + tolerance) for s, e, who in turns if who == speaker)
    starts: list[float] = []
    ends: list[float] = []
    for s, e in spans:
        if ends and s <= ends[-1]:
            ends[-1] = max(ends[-1], e)
        else:
            starts.append(s)
            ends.append(e)
    return starts, ends


def _inside(t: float, starts: list[float], ends: list[float]) -> bool:
    i = bisect.bisect_right(starts, t) - 1
    return i >= 0 and t <= ends[i]


def _timed(words: list[Word]) -> list[Word]:
    """Words with a start and an end; one the engine could not time follows its neighbour."""
    out: list[Word] = []
    for word in words:
        if word.start is None:
            if not out:
                continue  # nothing to hang it on
            word = dataclasses.replace(word, start=out[-1].end if out[-1].end is not None else out[-1].start)
        if word.end is None or word.end < word.start:
            word = dataclasses.replace(word, end=word.start)
        out.append(word)
    return out


def filter_words(words: list[Word], speaker: str, turns: list[tuple[float, float, str]],
                 tolerance: float = TOLERANCE_S) -> list[Word]:
    """``words`` (of the track of ``speaker``) labelled with the speaker, without those whose midpoint lies
    outside the speaker's turns. A speaker without any turn says nothing (a phantom track, or a model
    hallucinating on silence); only when there are no turns at all is nothing filtered."""
    starts, ends = _intervals(turns, speaker, tolerance)
    judge = bool(turns)
    out: list[Word] = []
    for word in _timed(words):
        if judge and not _inside((word.start + word.end) / 2, starts, ends):
            continue
        out.append(dataclasses.replace(word, speaker=speaker))
    return out


def build_segments(words: list[Word], gap: float = GAP_S) -> list[Segment]:
    """Segments of one speaker's words: sentences, cut at long pauses and at 20 s as for every engine."""
    segments: list[Segment] = []
    run: list[Word] = []

    def flush() -> None:
        if run:
            segments.extend(common.split_sentences(list(run)))
            run.clear()

    for word in sorted(words, key=lambda w: (w.start, w.end)):
        if run and word.start - run[-1].end > gap:
            flush()
        run.append(word)
    flush()
    return segments


def merge(tracks: dict[str, list[Word]], turns: list[tuple[float, float, str]],
          tolerance: float = TOLERANCE_S) -> list[Segment]:
    """Words per speaker -> all segments ordered by start (segments of different speakers may overlap)."""
    segments: list[Segment] = []
    for speaker, words in tracks.items():
        segments += build_segments(filter_words(words, speaker, turns, tolerance))
    segments.sort(key=lambda s: (s.start, s.end, s.speaker or ""))
    return segments


def _first_seconds(turns: list[tuple[float, float, str]], speaker: str, window: float = 30.0) -> float:
    return sum(max(0.0, min(e, window) - s) for s, e, who in turns if who == speaker and s < window)


def _order(tracks: list, turns: list[tuple[float, float, str]]) -> list:
    """Tracks worth transcribing, the one with the most speech in its first 30 s first: an engine detects the
    language from the start of a track, and that must not be a silent one. Tracks with no turn are dropped."""
    if turns:
        tracks = [t for t in tracks if any(who == t.speaker for _, _, who in turns)]
    return sorted(tracks, key=lambda t: (-_first_seconds(turns, t.speaker),
                                         -sum(e - s for s, e, who in turns if who == t.speaker)))


def removed_midpoints(tracks: dict[str, list[Word]], turns: list[tuple[float, float, str]],
                      tolerance: float = TOLERANCE_S) -> dict[str, list[float]]:
    """Per speaker the sorted midpoints of the words ``merge`` dropped (bleed of other voices)."""
    out: dict[str, list[float]] = {}
    for speaker, words in tracks.items():
        starts, ends = _intervals(turns, speaker, tolerance)
        mids = [(w.start + w.end) / 2 for w in _timed(words)
                if turns and not _inside((w.start + w.end) / 2, starts, ends)]
        if mids:
            out[speaker] = sorted(mids)
    return out


def solo_segments(segments: list[Segment], removed: dict[str, list[float]] | None = None) -> list[Segment]:
    """Segments reduced to the words nobody else speaks over: overlapped audio mixes voices, so it is not
    used for a voiceprint. A segment is cut where a word was left out (an overlapped one, or one in
    ``removed``: speaker -> midpoints of dropped words), so no span bridges the gap. Pieces without a word
    are dropped. Linear in the words (a sweep over each speaker's merged intervals)."""
    removed = removed or {}
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for seg in segments:
        for w in seg.words:
            if w.start is not None and w.end is not None:
                by_speaker.setdefault(w.speaker or seg.speaker or "", []).append((w.start, w.end))
    merged: dict[str, tuple[list[float], list[float]]] = {}
    for who, spans in by_speaker.items():
        starts: list[float] = []
        ends: list[float] = []
        for s, e in sorted(spans):
            if ends and s < ends[-1]:
                ends[-1] = max(ends[-1], e)
            else:
                starts.append(s)
                ends.append(e)
        merged[who] = (starts, ends)

    def clashes(word: Word, who: str) -> bool:
        for other, (starts, ends) in merged.items():
            if other == who:
                continue
            i = bisect.bisect_left(starts, word.end) - 1  # last interval starting before the word ends
            if i >= 0 and ends[i] > word.start:
                return True
        return False

    def cut_between(who: str, a: float, b: float) -> bool:
        mids = removed.get(who)
        if not mids:
            return False
        i = bisect.bisect_left(mids, a)
        return i < len(mids) and mids[i] <= b

    out: list[Segment] = []
    for seg in segments:
        piece: list[Word] = []

        def flush() -> None:
            if piece:
                text = " " + " ".join(w.word.strip() for w in piece)
                out.append(Segment(piece[0].start, piece[-1].end, text, seg.speaker, list(piece)))
                piece.clear()

        for w in seg.words:
            who = w.speaker or seg.speaker or ""
            if w.start is None or w.end is None or clashes(w, who):
                flush()
                continue
            if piece and cut_between(who, piece[-1].end, w.start):
                flush()
            piece.append(w)
        flush()
    return out


def _embeddings(media: Media, segments: list[Segment], settings: Settings,
                progress: ProgressCallback, removed: dict[str, list[float]] | None = None) -> dict | None:
    """Voiceprints per speaker as for a normal transcript (the diarization model's embedder on the original
    audio, from the speaker turns). None when that model is not available: voice memory is only skipped."""
    if not settings.hf_token or not media.audio_path:
        return None
    try:
        from . import voices

        found = voices.compute_embeddings(media.audio_path, solo_segments(segments, removed), settings,
                                          progress)
    except Exception as e:  # voice memory is a bonus; never fail the transcription for it
        log.warning("Voiceprints of the separated voices skipped: %s", e)
        return None
    return found or None


def _scaled(progress: ProgressCallback, index: int, count: int, speaker: str) -> ProgressCallback:
    """Report one track's engine progress as a slice of the whole, with the track named in the message."""
    def report(event: Progress) -> None:
        percent = None if event.percent is None else (index + event.percent / 100) / count * 100
        message = f"[{speaker}, track {index + 1}/{count}] {event.message}" if event.message else None
        progress(Progress(event.phase, percent, message, event.data))
    return report


def transcribe(engine: Engine, media: Media, options: TranscribeOptions, settings: Settings,
               progress: ProgressCallback = no_progress, num_speakers: int | None = None
               ) -> tuple[TranscriptionResult, dict]:
    """Separate ``media.audio_path`` and transcribe every track. Returns the merged result (the audio of
    the transcript stays the original) and the record for the transcript metadata."""
    from . import separation

    check(settings, engine.name)
    base = media.workdir or media.audio_path.parent
    # the tracks are scratch files: they live next to the audio only until the merged transcript exists
    with tempfile.TemporaryDirectory(prefix=".separation-", dir=base, ignore_cleanup_errors=True) as scratch:
        progress(Progress(Phase.LOADING, message="Separating voices..."))
        separated = separation.separate(media.audio_path, Path(scratch), settings, progress, num_speakers)
        tracks = _order(separated.tracks, separated.turns)
        if not tracks:
            raise EngineError("Voice separation found no speaker in the recording.")

        # the language is detected once, on the track that speaks first and most, and fixed for the others;
        # when nobody speaks enough at the start (music, silence) each track detects its own
        track_options = dataclasses.replace(options, diarize=False, speakers=None)
        words: dict[str, list[Word]] = {}
        language = options.language
        fixed = False
        detected: list[tuple[float, str]] = []  # (the track's total speech, its detected language)
        duration = 0.0
        model = None
        count = len(tracks)
        for i, track in enumerate(tracks):
            progress(Progress(Phase.TRANSCRIBE, 100 * i / count,
                              f"Transcribing voice {i + 1}/{count} ({track.speaker})..."))
            result = engine.transcribe(dataclasses.replace(media, audio_path=track.path), track_options,
                                       _scaled(progress, i, count, track.speaker))
            words.setdefault(track.speaker, []).extend(w for seg in result.segments for w in seg.words)
            if result.language and not fixed:
                total = sum(e - s for s, e, who in separated.turns if who == track.speaker)
                detected.append((total, result.language))
                if _first_seconds(separated.turns, track.speaker) >= MIN_LANGUAGE_S:
                    fixed = True
                    language = result.language
                    track_options = dataclasses.replace(track_options, language=language)
            duration = max(duration, result.duration or 0.0)
            model = model or result.model

    if not fixed and detected:
        language = max(detected)[1]  # the track with the most speech decides what is written in the metadata
    segments = merge(words, separated.turns)
    removed = removed_midpoints(words, separated.turns)
    merged = TranscriptionResult(segments, language=language, duration=duration or None, model=model,
                                 speaker_embeddings=_embeddings(media, segments, settings, progress, removed))
    info = {"used": True, "tracks": len(tracks), "speakers": sorted(t.speaker for t in tracks),
            "tolerance_s": TOLERANCE_S}
    return merged, info
