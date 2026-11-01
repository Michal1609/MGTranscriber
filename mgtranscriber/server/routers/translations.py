"""Translations of a transcript: list, read, start (a queued ``translate`` job) and delete."""

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ...llm import LLMError
from ...llm.translate import LANGUAGES
from ...llm.translations import (delete_translation, enqueue_translation, list_translations, load_translation,
                                 normalize_lang)
from ..deps import LibraryDep, QueueDep
from ..errors import ApiError

router = APIRouter(prefix="/api/transcripts", tags=["translations"])
TASK = "translate"


class TranslateRequest(BaseModel):
    language: str = Field(..., min_length=2, max_length=12)  # target language code, e.g. "cs"
    provider: str | None = Field(None, max_length=40)  # default: the LLM settings
    model: str | None = Field(None, max_length=120)


def _api_error(exc: LLMError) -> ApiError:
    return ApiError(exc.status, exc.code, exc.message)


def _lang(code: str) -> str:
    try:
        return normalize_lang(code)
    except LLMError as exc:
        raise _api_error(exc) from None


def _jobs(queue, transcript_id: str, active_only: bool = True) -> list[dict]:
    return [j for j in queue.listing() if j.get("task") == TASK and j["source"] == transcript_id
            and (not active_only or j["status"] in ("queued", "running"))]


@router.get("/{transcript_id}/translations")
def list_stored(transcript_id: str, library: LibraryDep, queue: QueueDep):
    """Stored translations (no text), the language names the UI may offer and the translate jobs in the queue."""
    library.path(transcript_id)
    return {"source_lang": library.metadata(transcript_id).language, "translations": list_translations(library, transcript_id),
            "jobs": [{"id": j["id"], "language": j["language"], "status": j["status"], "percent": j["percent"],
                      "position": j["position"], "provider": j["engine"], "model": j["model"]}
                     for j in _jobs(queue, transcript_id)],
            "languages": LANGUAGES}


@router.get("/{transcript_id}/translations/{lang}")
def get_translation(transcript_id: str, lang: str, library: LibraryDep):
    """One translation: ``segments`` (start, end, speaker, text), ``stale`` when the original changed since."""
    library.path(transcript_id)
    data = load_translation(library, transcript_id, _lang(lang))
    if data is None:
        raise ApiError(404, "translation_not_found", "There is no translation in this language.")
    return data


@router.post("/{transcript_id}/translations", status_code=202)
def start_translation(transcript_id: str, body: TranslateRequest, request: Request, library: LibraryDep,
                      queue: QueueDep):
    """Queue a translation job (it shares the job queue with transcriptions, so it never competes with one
    for the GPU). Problems such as 'Ollama is not running' are returned at once. A translation of the same
    language that is already waiting or running is returned instead of a new job."""
    library.path(transcript_id)
    try:
        return enqueue_translation(queue, request.app.state.llm, library, transcript_id, body.language,
                                   body.provider, body.model)
    except LLMError as exc:
        raise _api_error(exc) from None


@router.delete("/{transcript_id}/translations/{lang}")
def delete_stored(transcript_id: str, lang: str, library: LibraryDep, queue: QueueDep):
    """Delete the stored translation and cancel a waiting or running job for the same language."""
    library.path(transcript_id)
    lang = _lang(lang)
    cancelled = 0
    for item in _jobs(queue, transcript_id):
        if item["language"] == lang and (job := queue.get(item["id"])):
            queue.cancel(job)
            cancelled += 1
    return {"deleted": delete_translation(library, transcript_id, lang), "jobs_cancelled": cancelled}
