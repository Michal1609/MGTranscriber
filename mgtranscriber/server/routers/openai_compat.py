"""OpenAI-compatible speech-to-text API: ``POST /v1/audio/transcriptions`` and ``GET /v1/models``.

Any OpenAI client works with ``base_url="http://<host>:<port>/v1"``. Uploads become
normal jobs in the queue (so they never compete for the GPU) and the request waits
for the result; closing the connection cancels the job. The transcript also lands in
the library like any other job (the upload is kept as its audio).

Extensions to the OpenAI request: ``diarize`` (bool) and ``num_speakers`` (int).
Extensions to the response: ``speaker`` on segments and words, ``transcript_id``.

Models: ``whisper-1`` and the other OpenAI model names mean the server default; a
local model name (``large-v3``, ``small``, ...) picks that Whisper model; an engine
name picks that engine with its default model, ``<engine>/<model>`` both.
"""

from __future__ import annotations

import asyncio
import math
import re
import shutil

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response

from ... import privacy
from ...engines import list_engines
from ...exporters import get_exporter
from ...jobs import Job, JobQueue
from ...models import Segment
from ...pipeline import TranscriptionRequest
from ..deps import LibraryDep, QueueDep, SettingsDep
from ..errors import ApiError
from ..uploads import invalid, keep_upload_as_audio, save_upload

router = APIRouter(prefix="/v1", tags=["openai"])

OPENAI_MODELS = ("whisper-1", "gpt-transcribe", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-transcribe-diarize")
WHISPER_MODELS = ("tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
                  "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo", "distil-large-v3")
FORMATS = ("json", "text", "srt", "vtt", "verbose_json")
WAIT_STEP = 1.0  # seconds between disconnect checks while the job runs
# verbose_json reports the language by name, like Whisper
LANGUAGE_NAMES = {
    "en": "english", "cs": "czech", "sk": "slovak", "de": "german", "fr": "french", "es": "spanish",
    "it": "italian", "pt": "portuguese", "pl": "polish", "ru": "russian", "uk": "ukrainian", "nl": "dutch",
    "sv": "swedish", "da": "danish", "no": "norwegian", "fi": "finnish", "hu": "hungarian", "ro": "romanian",
    "tr": "turkish", "el": "greek", "ja": "japanese", "zh": "chinese", "ko": "korean", "ar": "arabic",
    "he": "hebrew", "hi": "hindi",
}


def resolve_model(model: str, default_engine: str) -> tuple[str | None, str | None]:
    """OpenAI ``model`` -> (engine, model); None = server default."""
    engines = {e.name for e in list_engines() if e.available}
    if model in OPENAI_MODELS:
        return None, None
    if "/" in model:
        engine, _, name = model.partition("/")
        if engine in engines and name:
            return engine, name
    elif model in engines:
        return model, None
    elif model in WHISPER_MODELS and "whisperx" in engines:
        return "whisperx", model
    raise invalid(f"The model '{model}' does not exist. See GET /v1/models.", "model", "model_not_found", 404)


def model_ids(default_engine: str) -> list[str]:
    ids = list(OPENAI_MODELS[:1])
    engines = [e.name for e in list_engines() if e.available]
    if "whisperx" in engines:
        ids += WHISPER_MODELS
    ids += [e for e in engines if e not in ids]
    return ids


def model_object(model_id: str) -> dict:
    return {"id": model_id, "object": "model", "created": 0, "owned_by": "mgtranscriber"}


@router.get("/models")
def models(settings: SettingsDep):
    return {"object": "list", "data": [model_object(m) for m in model_ids(settings.engine)]}


@router.get("/models/{model_id:path}")
def model(model_id: str, settings: SettingsDep):
    if model_id not in model_ids(settings.engine):
        resolve_model(model_id, settings.engine)  # raises model_not_found for unknown ids
    return model_object(model_id)


async def wait_for_job(request: Request, queue: JobQueue, job: Job) -> bool:
    """Wait until the job finished; False (and the job is cancelled) when the client went away."""
    while not await asyncio.to_thread(job.done.wait, WAIT_STEP):
        if await request.is_disconnected():
            queue.cancel(job)
            return False
    return True


def filled_words(seg: Segment) -> list[dict]:
    """Words with start/end always set (the aligner leaves numbers and symbols without times)."""
    out, last = [], seg.start
    for i, w in enumerate(seg.words):
        start = w.start if w.start is not None else last
        end = w.end
        if end is None:
            nxt = next((x.start for x in seg.words[i + 1:] if x.start is not None), seg.end)
            end = max(start, nxt)
        last = end
        out.append({"word": w.word.strip(), "start": start, "end": end, "speaker": w.speaker or seg.speaker})
    return out


def verbose_json(segments: list[Segment], language: str | None, duration: float, granularities: set[str],
                 transcript_id: str) -> dict:
    body = {
        "task": "transcribe",
        "language": LANGUAGE_NAMES.get(language or "", language or "unknown"),
        "duration": duration,
        "text": " ".join(s.text.strip() for s in segments),
        "transcript_id": transcript_id,
    }
    if "segment" in granularities or not granularities:
        body["segments"] = [
            {"id": i, "seek": 0, "start": s.start, "end": s.end, "text": s.text, "tokens": [], "temperature": 0.0,
             "avg_logprob": 0.0, "compression_ratio": 0.0, "no_speech_prob": 0.0, "speaker": s.speaker}
            for i, s in enumerate(segments)
        ]
    if "word" in granularities:
        body["words"] = [w for s in segments for w in filled_words(s)]
    body["usage"] = {"type": "duration", "seconds": math.ceil(duration)}
    return body


@router.post("/audio/transcriptions")
async def transcriptions(
    request: Request,
    queue: QueueDep,
    library: LibraryDep,
    settings: SettingsDep,
    file: UploadFile = File(..., description="audio or video file"),
    model: str = Form("whisper-1"),
    language: str | None = Form(None, description="ISO-639-1 code; empty = detect"),
    prompt: str | None = Form(None, description="accepted, ignored"),
    response_format: str = Form("json"),
    temperature: float | None = Form(None, description="accepted, ignored"),
    timestamp_granularities: list[str] = Form([], alias="timestamp_granularities[]"),
    timestamp_granularities_plain: list[str] = Form([], alias="timestamp_granularities"),
    stream: bool = Form(False),
    diarize: bool = Form(False, description="extension: label speakers"),
    num_speakers: int | None = Form(None, ge=1, le=20, description="extension: exact number of speakers"),
):
    if response_format not in FORMATS:
        raise invalid(f"response_format must be one of {', '.join(FORMATS)}.", "response_format")
    if stream:
        raise invalid("Streaming is not supported by this server.", "stream")
    granularities = set(timestamp_granularities) | set(timestamp_granularities_plain)
    if granularities - {"word", "segment"}:
        raise invalid("timestamp_granularities may contain 'word' and 'segment'.", "timestamp_granularities")
    if granularities and response_format != "verbose_json":
        raise invalid("timestamp_granularities requires response_format=verbose_json.", "timestamp_granularities")
    language = (language or "").strip().lower() or "auto"
    if not re.fullmatch(r"auto|[a-z]{2,3}", language):
        raise invalid("language must be an ISO-639-1 code.", "language")
    engine, model_name = resolve_model(model, settings.engine)
    try:
        privacy.check_engine(settings, engine)  # offline mode: no cloud engines
    except privacy.PrivacyError as e:
        raise ApiError(e.status, e.code, e.message, "model") from None

    upload, original = await save_upload(file, settings)
    try:
        # ephemeral: the upload is deleted with this request, so the job must never be resumed after a restart
        job = queue.add(TranscriptionRequest(source=str(upload), language=language, speakers=num_speakers,
                                             diarize=diarize or bool(num_speakers), engine=engine, model=model_name),
                        ephemeral=True)
        if not await wait_for_job(request, queue, job):
            return Response(status_code=499)
        if job.status != "done" or not job.result:
            raise ApiError(500, "transcription_failed", job.error or f"Transcription {job.status}.", kind="server_error")
        keep_upload_as_audio(library, job.result, upload, original)
    finally:
        shutil.rmtree(upload.parent, ignore_errors=True)

    segments = library.segments(job.result)
    meta = library.metadata(job.result)
    duration = meta.duration or max((s.end for s in segments), default=0.0)
    text = " ".join(s.text.strip() for s in segments)
    if response_format == "json":
        return {"text": text, "usage": {"type": "duration", "seconds": math.ceil(duration)}}
    if response_format == "verbose_json":
        return verbose_json(segments, meta.language, duration, granularities, job.result)
    if response_format == "text":
        return PlainTextResponse(text + "\n")
    exporter = get_exporter(response_format)
    return Response(exporter.render(segments, {}), media_type=exporter.media_type)
