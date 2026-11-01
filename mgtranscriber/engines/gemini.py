"""Gemini hybrid: text from Google Gemini (cloud), timing and speakers computed locally.

Gemini writes good text but its own timestamps drift on long audio. So the audio is first cut
locally into speech pieces of at most ~25 s (Silero VAD). Pieces are sent in batches of a few
minutes, each piece as its own audio part, and Gemini returns one text per piece. The piece
boundaries give the segment times; wav2vec2 alignment then times the words and pyannote assigns
the speakers, exactly as for WhisperX.

Opt-in only: the audio pieces are uploaded to Google. Key: ``GEMINI_API_KEY`` (environment or .env).
"""

from __future__ import annotations

import io
import json
import wave
from typing import TYPE_CHECKING

from ..models import Segment, TranscriptionResult
from ..progress import Phase, Progress, ProgressCallback
from ..sources import Media
from . import common
from .base import Engine, EngineError, TranscribeOptions
from .cloud import iso639_1

if TYPE_CHECKING:
    import numpy as np

KEY_ENV = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-2.5-flash"
PIECE_S = 25.0  # target length of one audio part
BATCH_S = 300.0  # audio per request; 16 kHz WAV keeps a batch well under the 20 MB inline limit
SCHEMA = {
    "type": "object",
    "properties": {"language": {"type": "string"}, "parts": {"type": "array", "items": {"type": "string"}}},
    "required": ["language", "parts"],
}
PROMPT = """You are a careful transcriptionist. Below are {n} consecutive audio parts of one recording, labelled Part 1 to Part {n}.
Transcribe each part verbatim{language}, with normal punctuation and capitalization.
Do not translate, summarize, add speaker names, timestamps or descriptions of sounds.
A word cut at the boundary of a part belongs to the part where most of it is heard. A part without speech gets "".
Return JSON: "language" = ISO 639-1 code of the spoken language, "parts" = exactly {n} strings in order."""


def speech_pieces(audio: "np.ndarray", sample_rate: int = common.SAMPLE_RATE) -> list[tuple[float, float]]:
    """Speech regions (seconds), merged into pieces of at most ``PIECE_S`` seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    regions = get_speech_timestamps(audio, VadOptions(max_speech_duration_s=PIECE_S, min_silence_duration_ms=500,
                                                      speech_pad_ms=200), sampling_rate=sample_rate)
    pieces: list[list[float]] = []
    for r in regions:
        start, end = r["start"] / sample_rate, r["end"] / sample_rate
        if pieces and end - pieces[-1][0] <= PIECE_S and start - pieces[-1][1] < 2.0:
            pieces[-1][1] = end
        else:
            pieces.append([start, end])
    return [(round(s, 3), round(e, 3)) for s, e in pieces]


def batches(pieces: list[tuple[float, float]], limit: float = BATCH_S) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    total = limit
    for piece in pieces:
        length = piece[1] - piece[0]
        if total + length > limit:
            out.append([])
            total = 0.0
        out[-1].append(piece)
        total += length
    return out


def wav_bytes(samples: "np.ndarray", sample_rate: int = common.SAMPLE_RATE) -> bytes:
    import numpy as np

    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buffer.getvalue()


def parse_reply(text: str, expected: int) -> tuple[str | None, list[str]]:
    data = json.loads(text)
    parts = [str(p).strip() for p in data.get("parts") or []]
    if len(parts) != expected:
        raise ValueError(f"expected {expected} parts, got {len(parts)}")
    return (data.get("language") or None), parts


class GeminiEngine(Engine):
    name = "gemini"
    supports_diarization = True
    default_model = DEFAULT_MODEL

    def __init__(self, settings, client=None):
        super().__init__(settings)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=common.env_value(self.settings, KEY_ENV))
        return self._client

    def check(self, options: TranscribeOptions) -> None:
        if self._client is None:
            if not common.env_value(self.settings, KEY_ENV):
                raise EngineError(f"{KEY_ENV} is not set. Add it to .env to use Gemini (audio is uploaded to Google).")
            try:
                import google.genai  # noqa: F401
            except ImportError:
                raise EngineError("The google-genai package is missing. Install it with: uv sync")
        if options.diarize:
            common.check_diarization_access(self.settings)

    def _ask(self, model: str, clips: list[bytes], language: str | None) -> tuple[str | None, list[str]]:
        from google.genai import types

        hint = f" in the language with ISO code '{language}'" if language else ""
        contents: list = [PROMPT.format(n=len(clips), language=hint)]
        for i, clip in enumerate(clips, 1):
            contents += [f"Part {i}:", types.Part.from_bytes(data=clip, mime_type="audio/wav")]
        config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, temperature=0)
        try:
            response = self.client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:  # SDK errors: surface the message, it says what is wrong (key, quota, model)
            raise EngineError(f"Gemini error: {getattr(e, 'message', None) or e}")
        return parse_reply(response.text or "", len(clips))

    def _transcribe_batch(self, model: str, audio, batch: list[tuple[float, float]], language: str | None,
                          depth: int = 0) -> tuple[str | None, list[str]]:
        """One request per batch; if Gemini returns the wrong number of parts, retry with halves."""
        sr = common.SAMPLE_RATE
        clips = [wav_bytes(audio[int(s * sr):int(e * sr)]) for s, e in batch]
        try:
            return self._ask(model, clips, language)
        except (ValueError, json.JSONDecodeError):
            if len(batch) == 1 or depth >= 4:
                raise EngineError("Gemini did not return the expected transcript format.")
            half = len(batch) // 2
            lang1, first = self._transcribe_batch(model, audio, batch[:half], language, depth + 1)
            lang2, second = self._transcribe_batch(model, audio, batch[half:], language, depth + 1)
            return lang1 or lang2, first + second

    def transcribe(self, media: Media, options: TranscribeOptions, progress: ProgressCallback) -> TranscriptionResult:
        model = options.model or self.default_model
        audio, original = common.load_audio_pair(media.audio_path, self.settings.level_audio)
        duration = len(audio) / common.SAMPLE_RATE
        progress(Progress(Phase.TRANSCRIBE, 0, f"Finding speech, then sending it to Gemini ({model})..."))
        pieces = speech_pieces(audio)
        groups = batches(pieces)
        language = options.language
        segments: list[dict] = []
        for i, batch in enumerate(groups):
            detected, texts = self._transcribe_batch(model, audio, batch, language)
            language = language or detected
            segments += [{"start": s, "end": e, "text": " " + t} for (s, e), t in zip(batch, texts) if t]
            progress(Progress(Phase.TRANSCRIBE, 100 * (i + 1) / len(groups)))

        language = iso639_1(language) or "en"
        voices: dict = {}
        if segments:
            segments = common.align(segments, audio, language, progress)
            if options.diarize:
                segments = common.split_by_speaker(
                    common.diarize(segments, original, self.settings, options.speakers, progress, voices))
        return TranscriptionResult([Segment.from_dict(s) for s in segments], language=language, duration=duration,
                                   model=model, speaker_embeddings=voices or None)
