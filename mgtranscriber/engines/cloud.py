"""Cloud speech-to-text services. Opt-in only: the audio file is uploaded to the provider.

API keys are read from the environment (or ``.env``), never from settings files:
``ELEVENLABS_API_KEY``, ``DEEPGRAM_API_KEY``. Both services return word timestamps and speakers,
which are turned into the same segments the local engines produce.

Deepgram is called on its EU endpoint by default (``api.eu.deepgram.com``, generally available
since December 2025, same API keys, Nova models): processing stays in the EU, which matters for
recordings of people under the GDPR. ``DEEPGRAM_API_URL=https://api.deepgram.com`` switches to
the global (US) endpoint.
"""

from __future__ import annotations

import mimetypes
from typing import ClassVar

from ..models import TranscriptionResult, Word
from ..progress import Phase, Progress, ProgressCallback
from ..sources import Media
from . import common
from .base import Engine, EngineError, TranscribeOptions

TIMEOUT_S = 3600.0  # a long recording can take minutes to upload and process
CONNECT_TIMEOUT_S = 15.0  # but an unreachable service should fail fast
CHUNK = 1 << 20
DEEPGRAM_URL = "https://api.eu.deepgram.com"

# ElevenLabs answers ISO 639-3 codes ("ces"); the rest of the app (alignment models, exports) uses ISO 639-1.
ISO639_3_TO_1 = {
    "afr": "af", "amh": "am", "ara": "ar", "asm": "as", "aze": "az", "bak": "ba", "bel": "be", "ben": "bn",
    "bod": "bo", "bos": "bs", "bre": "br", "bul": "bg", "cat": "ca", "ces": "cs", "cmn": "zh", "cym": "cy",
    "dan": "da", "deu": "de", "ell": "el", "eng": "en", "est": "et", "eus": "eu", "fao": "fo", "fas": "fa",
    "fil": "tl", "fin": "fi", "fra": "fr", "gle": "ga", "glg": "gl", "guj": "gu", "hat": "ht", "hau": "ha",
    "haw": "haw", "heb": "he", "hin": "hi", "hrv": "hr", "hun": "hu", "hye": "hy", "ind": "id", "isl": "is",
    "ita": "it", "jav": "jw", "jpn": "ja", "kan": "kn", "kat": "ka", "kaz": "kk", "khm": "km", "kor": "ko",
    "lao": "lo", "lat": "la", "lav": "lv", "lin": "ln", "lit": "lt", "ltz": "lb", "mal": "ml", "mar": "mr",
    "mkd": "mk", "mlg": "mg", "mlt": "mt", "mon": "mn", "mri": "mi", "msa": "ms", "mya": "my", "nep": "ne",
    "nld": "nl", "nno": "nn", "nor": "no", "nob": "no", "oci": "oc", "pan": "pa", "pol": "pl", "por": "pt",
    "pus": "ps", "ron": "ro", "rus": "ru", "san": "sa", "sin": "si", "slk": "sk", "slv": "sl", "sna": "sn",
    "snd": "sd", "som": "so", "spa": "es", "sqi": "sq", "srp": "sr", "sun": "su", "swa": "sw", "swe": "sv",
    "tam": "ta", "tat": "tt", "tel": "te", "tgk": "tg", "tgl": "tl", "tha": "th", "tuk": "tk", "tur": "tr",
    "ukr": "uk", "urd": "ur", "uzb": "uz", "vie": "vi", "yid": "yi", "yor": "yo", "yue": "yue", "zho": "zh",
}


def iso639_1(code: str | None) -> str | None:
    """``ces`` -> ``cs``; two-letter codes and unknown ones pass through (lower-cased)."""
    if not code:
        return None
    code = code.strip().lower().replace("_", "-").split("-")[0]
    return ISO639_3_TO_1.get(code, code)


def read_chunks(path):
    """Stream a file to the request instead of reading a long recording into memory."""
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            yield chunk


def normalize_speakers(words: list[Word]) -> list[Word]:
    """Rename provider labels (``speaker_1``, ``0``...) to ``SPEAKER_00``... in order of appearance."""
    names: dict[str, str] = {}
    for word in words:
        if word.speaker is not None:
            word.speaker = names.setdefault(str(word.speaker), f"SPEAKER_{len(names):02d}")
    return words


def parse_elevenlabs(data: dict) -> list[Word]:
    words = [Word(w["text"].strip(), w.get("start"), w.get("end"), speaker=w.get("speaker_id"))
             for w in data.get("words") or [] if w.get("type", "word") == "word" and w.get("text", "").strip()]
    return normalize_speakers(words)


def parse_deepgram(data: dict) -> tuple[list[Word], str | None, float | None]:
    channel = (data.get("results", {}).get("channels") or [{}])[0]
    alternative = (channel.get("alternatives") or [{}])[0]
    words = [Word(w.get("punctuated_word") or w["word"], w.get("start"), w.get("end"), w.get("confidence"),
                  None if w.get("speaker") is None else str(w["speaker"]))
             for w in alternative.get("words") or []]
    return normalize_speakers(words), channel.get("detected_language"), data.get("metadata", {}).get("duration")


class CloudEngine(Engine):
    supports_diarization = True
    key_env: ClassVar[str]
    service: ClassVar[str]

    @property
    def api_key(self) -> str | None:
        return common.env_value(self.settings, self.key_env)

    def check(self, options: TranscribeOptions) -> None:
        if not self.api_key:
            raise EngineError(f"{self.key_env} is not set. Add it to .env to use {self.service} (audio is uploaded there).")

    def _post(self, url: str, **kwargs) -> dict:
        import httpx

        try:
            response = httpx.post(url, timeout=httpx.Timeout(TIMEOUT_S, connect=CONNECT_TIMEOUT_S), **kwargs)
        except httpx.HTTPError as e:
            raise EngineError(f"{self.service} is unreachable: {e}")
        if response.status_code in (401, 403):
            raise EngineError(f"{self.service} rejected the API key ({self.key_env}): {response.text[:300]}")
        if response.status_code >= 400:
            raise EngineError(f"{self.service} error {response.status_code}: {response.text[:500]}")
        return response.json()

    def _result(self, words: list[Word], language: str | None, duration: float | None, model: str) -> TranscriptionResult:
        segments = common.split_sentences(words)
        return TranscriptionResult(segments, language=language, duration=duration or max((s.end for s in segments), default=0),
                                   model=model)


class ElevenLabsEngine(CloudEngine):
    name = "elevenlabs"
    default_model = "scribe_v2"
    key_env = "ELEVENLABS_API_KEY"
    service = "ElevenLabs"
    url = "https://api.elevenlabs.io/v1/speech-to-text"

    def transcribe(self, media: Media, options: TranscribeOptions, progress: ProgressCallback) -> TranscriptionResult:
        model = options.model or self.default_model
        form = {"model_id": model, "timestamps_granularity": "word", "tag_audio_events": "false",
                "diarize": "true" if options.diarize else "false"}
        if options.language:
            form["language_code"] = options.language
        if options.diarize and options.speakers:
            form["num_speakers"] = str(options.speakers)
        progress(Progress(Phase.TRANSCRIBE, None, f"Uploading to ElevenLabs ({model})..."))
        path = media.audio_path
        with open(path, "rb") as f:
            data = self._post(self.url, headers={"xi-api-key": self.api_key}, data=form,
                              files={"file": (path.name, f, mimetypes.guess_type(path.name)[0] or "application/octet-stream")})
        progress(Progress(Phase.TRANSCRIBE, 100, "ElevenLabs finished"))
        return self._result(parse_elevenlabs(data), iso639_1(data.get("language_code")) or options.language, None, model)


class DeepgramEngine(CloudEngine):
    name = "deepgram"
    default_model = "nova-3"
    key_env = "DEEPGRAM_API_KEY"
    service = "Deepgram"

    def transcribe(self, media: Media, options: TranscribeOptions, progress: ProgressCallback) -> TranscriptionResult:
        model = options.model or self.default_model
        base = common.env_value(self.settings, "DEEPGRAM_API_URL") or DEEPGRAM_URL
        params = {"model": model, "punctuate": "true", "smart_format": "true"}
        if options.language:
            params["language"] = options.language
        else:
            params["detect_language"] = "true"
        if options.diarize:
            params["diarize"] = "true"
        progress(Progress(Phase.TRANSCRIBE, None, f"Uploading to Deepgram ({model})..."))
        path = media.audio_path
        data = self._post(base.rstrip("/") + "/v1/listen", params=params, content=read_chunks(path), headers={
            "Authorization": f"Token {self.api_key}", "Content-Length": str(path.stat().st_size),
            "Content-Type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"})
        progress(Progress(Phase.TRANSCRIBE, 100, "Deepgram finished"))
        words, detected, duration = parse_deepgram(data)
        return self._result(words, iso639_1(detected) or options.language, duration, model)
