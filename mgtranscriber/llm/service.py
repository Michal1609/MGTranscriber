"""Runs analyses in background threads of the server process and answers questions.

The LLM runs elsewhere (Ollama's own process, or a cloud API), so these threads do not hold the
GPU themselves and never touch the transcription job queue. One analysis per transcript at a
time; at most ``MAX_PARALLEL`` run at once.
"""

from __future__ import annotations

import logging
import threading
import time

from ..config import Settings
from ..library import Library
from . import LLMConfig, load_config, make_provider
from .analysis import analyze
from .base import AnalysisBusy, Cancelled, LLMError, LLMProvider
from .qa import Retriever, ask
from .store import load_analysis, save_analysis, segments_stamp
from .transcript import build_turns, speaker_labels

log = logging.getLogger(__name__)
MAX_PARALLEL = 2


class Run:
    """State of one analysis (kept in memory until the next one starts for that transcript)."""

    def __init__(self, provider: str, model: str, language: str):
        self.status = "running"  # running | done | error | cancelled
        self.percent = 0.0
        self.message = "queued"
        self.error: dict | None = None
        self.provider, self.model, self.language = provider, model, language
        self.started = time.time()
        self.finished: float | None = None
        self.cancel = threading.Event()

    def public(self) -> dict:
        return {"status": self.status, "percent": round(self.percent, 1), "message": self.message, "error": self.error,
                "provider": self.provider, "model": self.model, "language": self.language, "started": self.started,
                "finished": self.finished}


MAX_RETRIEVERS = 8  # transcripts whose question-answering index is kept in memory


class AnalysisManager:
    def __init__(self, settings: Settings, library: Library, env: dict[str, str] | None = None):
        self.settings = settings
        self.library = library
        self.env = env
        self.runs: dict[str, Run] = {}
        self.retrievers: dict[tuple, Retriever] = {}  # most recently used last; at most MAX_RETRIEVERS
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(MAX_PARALLEL)
        self.closed = False

    @property
    def config(self) -> LLMConfig:
        return load_config(self.settings, self.env)

    def provider(self, provider: str | None = None, model: str | None = None) -> LLMProvider:
        return make_provider(self.config, self.settings, provider, model, env=self.env)

    def shutdown(self) -> None:
        for run in list(self.runs.values()):
            run.cancel.set()
        self.closed = True  # worker threads are daemons: a model call that is still running cannot hold up exit

    def _output_language(self, requested: str | None, transcript_id: str) -> str:
        """Language of the generated text; "auto" (or nothing set) = the language of the transcript."""
        language = requested or self.config.language
        if language == "auto":
            return self.library.metadata(transcript_id).language or "en"
        return language

    # --- analysis ----------------------------------------------------------------------

    def start(self, transcript_id: str, provider: str | None = None, model: str | None = None,
              language: str | None = None) -> Run:
        """Validate quickly (so the user sees 'Ollama is not running' at once), then run in the background."""
        self.library.path(transcript_id)  # raises TranscriptNotFound
        if self.closed:
            raise LLMError("The server is shutting down.", "shutting_down")
        cfg = self.config
        main = self.provider(provider, model)
        main.check()
        final = None
        if cfg.final_model and cfg.final_model != main.model and (provider or cfg.provider) == cfg.provider:
            final = main.with_model(cfg.final_model)
        language = self._output_language(language, transcript_id)
        with self.lock:
            current = self.runs.get(transcript_id)
            if current and current.status == "running":
                raise AnalysisBusy("An analysis of this transcript is already running.")
            run = Run(main.name, main.model, language)
            self.runs[transcript_id] = run
        threading.Thread(target=self._run_when_free, args=(transcript_id, run, main, final), daemon=True,
                         name=f"analysis-{transcript_id[:30]}").start()
        return run

    def _run_when_free(self, transcript_id: str, run: Run, provider: LLMProvider, final: LLMProvider | None) -> None:
        while not self.slots.acquire(timeout=0.5):
            if run.cancel.is_set() or self.closed:
                run.status, run.message, run.finished = "cancelled", "cancelled", time.time()
                return
        try:
            self._run(transcript_id, run, provider, final)
        finally:
            self.slots.release()

    def _run(self, transcript_id: str, run: Run, provider: LLMProvider, final: LLMProvider | None) -> None:
        def progress(percent: float, message: str) -> None:
            run.percent, run.message = percent, message

        def check_cancel() -> None:
            if run.cancel.is_set():
                raise Cancelled()

        try:
            stamp = segments_stamp(self.library, transcript_id)
            segments = self.library.segments(transcript_id)
            names = self.library.names(transcript_id)
            title = self.library.metadata(transcript_id).title
            result = analyze(segments, names, provider, language=run.language, final_provider=final, title=title,
                             progress=progress, cancel=check_cancel)
            check_cancel()
            save_analysis(self.library, transcript_id, result, stamp)
            run.status, run.percent, run.message = "done", 100.0, "done"
        except Cancelled:
            run.status, run.message = "cancelled", "cancelled"
        except LLMError as exc:
            run.status, run.message, run.error = "error", "error", {"code": exc.code, "message": exc.message}
        except Exception as exc:  # noqa: BLE001 - a background thread must never die silently
            log.exception("analysis of %s failed", transcript_id)
            run.status, run.message, run.error = "error", "error", {"code": "analysis_failed", "message": str(exc)}
        finally:
            run.finished = time.time()

    def status(self, transcript_id: str) -> dict:
        self.library.path(transcript_id)
        run = self.runs.get(transcript_id)
        return {"run": run.public() if run else None, "analysis": load_analysis(self.library, transcript_id)}

    def cancel(self, transcript_id: str) -> bool:
        run = self.runs.get(transcript_id)
        if run and run.status == "running":
            run.cancel.set()
            return True
        return False

    def on_job_finished(self, job) -> None:
        """JobQueue hook: analyse right after a transcription when the setting ``llm_auto`` is on."""
        if not self.config.auto or job.status != "done" or not job.result:
            return
        if getattr(job.request, "task", "transcribe") != "transcribe":  # e.g. voiceprints jobs reuse the transcript id
            return
        try:
            self.start(job.result)
            job.add_log("[analysis started]")
        except LLMError as exc:
            job.add_log(f"[analysis not started: {exc.message}]")

    # --- Q&A ---------------------------------------------------------------------------

    def ask(self, transcript_id: str, question: str, provider: str | None = None, model: str | None = None,
            language: str | None = None) -> dict:
        llm = self.provider(provider, model)
        llm.check()
        segments = self.library.segments(transcript_id)
        names = self.library.names(transcript_id)
        meta = self.library.metadata(transcript_id)
        language = self._output_language(language, transcript_id)
        stamp = (transcript_id, (self.library.path(transcript_id) / "prepis.json").stat().st_mtime,
                 llm.name, llm.model, tuple(sorted(names.items())))
        retriever = self.retrievers.get(stamp)
        if retriever is not None:  # a hit becomes the most recent
            with self.lock:
                if stamp in self.retrievers:
                    self.retrievers[stamp] = self.retrievers.pop(stamp)
        if retriever is None:
            turns = build_turns(segments)
            labels = speaker_labels(sorted({t.speaker for t in turns}), names)
            retriever = Retriever(turns, labels, llm)
            with self.lock:
                self.retrievers = {k: v for k, v in self.retrievers.items() if k[0] != transcript_id}
                self.retrievers[stamp] = retriever
                while len(self.retrievers) > MAX_RETRIEVERS:  # each holds a transcript's text and embeddings
                    del self.retrievers[next(iter(self.retrievers))]
        result = ask(segments, names, question, llm, language=language, retriever=retriever, title=meta.title)
        result["provider"], result["model"] = llm.name, llm.model
        return result
