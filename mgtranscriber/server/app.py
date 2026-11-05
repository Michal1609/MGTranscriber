"""FastAPI application factory.

Shared services live on ``app.state`` (settings, library, queue) and routers reach them
through ``deps``. Each feature is a router in ``routers/``, listed in ``routers.ROUTERS``.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import APP_NAME, __version__
from ..config import Settings, load_settings
from ..folders import FolderStore
from ..jobs import JobQueue, Runner
from .. import settings_store
from ..library import InvalidTranscriptId, Library, TranscriptNotFound
from ..llm import auto_translate
from ..llm.service import AnalysisManager
from ..search import SearchIndex
from .errors import ApiError, error_body, is_openai_path
from ..watch import WatchManager
from . import recordings, uploads
from .limits import BodyLimitMiddleware
from .mcp_http import mount_mcp
from .routers import ROUTERS
from .security import AuthThrottle, SecurityMiddleware, Sessions

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


class RevalidatingStaticFiles(StaticFiles):
    """The frontend files, always revalidated: without Cache-Control the browser caches ES modules
    heuristically (from Last-Modified) and keeps running old code after an update. ``no-cache`` still
    allows the cheap 304 through ETag/Last-Modified."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(settings: Settings | None = None, queue: JobQueue | None = None, runner: Runner | None = None,
               static_dir: Path = STATIC_DIR) -> FastAPI:
    settings = settings or load_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threading.Thread(target=app.state.search.sync, name="search-sync", daemon=True).start()  # builds a missing index
        if settings_store.effective(settings, "watch_enabled"):
            app.state.watch.start()
        async with app.state.mcp.session_manager.run():
            yield
        app.state.watch.stop()
        app.state.queue.stop()
        app.state.llm.shutdown()

    app = FastAPI(title=APP_NAME, version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.state.settings = settings
    app.state.sessions = Sessions(settings.api_key, settings.data_dir / "sessions.json")
    app.state.auth_throttle = AuthThrottle()
    app.state.library = Library(settings.output_dir)
    app.state.folders = FolderStore(app.state.library, settings.data_dir / "folders.json")
    app.state.queue = queue or JobQueue(settings, runner=runner)
    uploads.sweep(settings, list(app.state.queue.jobs.values()))  # files of web uploads that no job waits for
    recordings.sweep(settings)  # unfinished recording sessions nobody came back to (7 days)
    threading.Thread(  # stages (.run-*) of killed runs older than an hour; the one of a running job is protected
        target=lambda: app.state.library.sweep_stages(protect=app.state.queue.active_stages()),
        name="stage-sweep", daemon=True).start()
    # first: the file becomes the transcript's audio before anything else looks at the transcript
    app.state.queue.on_finished.append(lambda job: uploads.finish_job(settings, app.state.library, job))
    app.state.search = SearchIndex(app.state.library, settings.search_file)
    app.state.library.listeners.append(app.state.search.reindex)  # renames, edits
    app.state.queue.on_finished.append(lambda job: job.result and app.state.search.reindex(job.result))
    app.state.llm = AnalysisManager(settings, app.state.library)
    app.state.watch = WatchManager(settings, app.state.queue, app.state.llm, library=app.state.library)
    app.state.queue.on_finished.append(app.state.watch.on_job_finished)
    app.state.queue.on_finished.append(app.state.llm.on_job_finished)  # setting llm_auto: analyse after transcribing
    app.state.queue.on_finished.append(  # "Translate to" of the new-transcription form
        lambda job: auto_translate.on_job_finished(app.state.queue, app.state.llm, app.state.library, job))

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        body = error_body(request.url.path, exc.status, exc.code, exc.message, exc.param, exc.kind)
        return JSONResponse(body, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        if not is_openai_path(request.url.path):  # FastAPI's 422 shape, without echoing the request values (``input``)
            errors = [{k: v for k, v in e.items() if k != "input"} for e in exc.errors()]
            return JSONResponse({"detail": jsonable_encoder(errors)}, status_code=422)
        first = exc.errors()[0] if exc.errors() else {}
        param = str(first.get("loc", ["", ""])[-1]) or None
        body = error_body(request.url.path, 400, "invalid_request", f"{param}: {first.get('msg', 'invalid')}", param)
        return JSONResponse(body, status_code=400)

    @app.exception_handler(TranscriptNotFound)
    async def not_found(_: Request, exc: TranscriptNotFound):
        return JSONResponse({"detail": "Transcript not found.", "code": "transcript_not_found"}, status_code=404)

    @app.exception_handler(InvalidTranscriptId)
    async def invalid_id(_: Request, exc: InvalidTranscriptId):
        return JSONResponse({"detail": "Invalid transcript id.", "code": "invalid_transcript_id"}, status_code=400)

    for router in ROUTERS:
        app.include_router(router)
    mount_mcp(app)
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_mb * 1024 * 1024)
    app.add_middleware(SecurityMiddleware, settings=settings, sessions=app.state.sessions,
                       throttle=app.state.auth_throttle)  # outermost: auth first

    # last: everything not matched by an API route is the single-page frontend
    app.mount("/", RevalidatingStaticFiles(directory=static_dir, html=True), name="static")
    return app
