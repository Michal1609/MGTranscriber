"""The "Translate to" option of a new transcription: queue the translation when the transcription is done.

A JobQueue hook. It uses ``enqueue_translation``, the same path as the detail page's Translate button and the watched
sources. Watch jobs never set ``request.translate_to`` (a source has its own), and translate jobs do not either, so
nothing is queued twice and nothing loops. A failure never fails the transcription: the reason is kept on the job (``job.translation``, shown by the Jobs page)
and written to its log.
"""

from __future__ import annotations

import logging

from .base import LLMError
from .translations import enqueue_translation

log = logging.getLogger(__name__)


def on_job_finished(queue, llm, library, job) -> None:
    request = job.request
    lang = getattr(request, "translate_to", None)
    if not lang or job.status != "done" or not job.result or request.task != "transcribe":
        return
    if getattr(job, "no_speech", False):  # an empty transcript: nothing to translate, and that is not an error
        job.translation = {"language": lang, "skipped": "no_speech"}
        job.add_log(f"[translation to {lang} not queued: no speech was found]")
        queue.save()
        return
    try:
        queued = enqueue_translation(queue, llm, library, job.result, lang)
        job.translation = {"language": lang, "job": queued.get("id")}
        job.add_log(f"[translation to {lang} queued]")
    except LLMError as exc:
        if exc.code == "translation_same_language":  # the transcript is already in that language: nothing to do
            return
        _failed(queue, job, lang, exc.message)
    except Exception as exc:  # noqa: BLE001 - must not turn a finished transcription into a failed one
        log.warning("translation for job %s failed to queue: %s", job.id, exc)
        _failed(queue, job, lang, str(exc))
    else:
        queue.save()


def _failed(queue, job, lang: str, reason: str) -> None:
    job.translation = {"language": lang, "error": reason}
    job.add_log(f"[translation to {lang} not queued: {reason}]")
    queue.save()
