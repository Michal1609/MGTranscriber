"""LLM analysis of a transcript: start/poll/cancel, questions, providers, Markdown export."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field

from ... import privacy
from ...llm import LLMError, provider_status
from ...llm.markdown import render as render_markdown
from ...llm.service import AnalysisManager
from ...llm.store import delete_analysis, load_analysis
from ..deps import LibraryDep, SettingsDep
from ..errors import ApiError
from .transcripts import export_segments

router = APIRouter(tags=["analysis"])


def get_llm(request: Request) -> AnalysisManager:
    return request.app.state.llm


LLMDep = Annotated[AnalysisManager, Depends(get_llm)]


class AnalysisRequest(BaseModel):
    provider: str | None = Field(None, max_length=40)
    model: str | None = Field(None, max_length=120)
    language: str | None = Field(None, max_length=12)


class AskRequest(AnalysisRequest):
    question: str = Field(..., min_length=2, max_length=1000)


def api_error(exc: LLMError) -> ApiError:
    return ApiError(exc.status, exc.code, exc.message)


@router.get("/api/llm")
def llm_info(llm: LLMDep, settings: SettingsDep):
    """Configured defaults and what each provider offers. Never contains API keys."""
    cfg = llm.config
    return {"config": cfg.public(), "providers": provider_status(cfg, settings, llm.env)}


@router.get("/api/transcripts/{transcript_id}/analysis")
def get_analysis(transcript_id: str, llm: LLMDep):
    return llm.status(transcript_id)


@router.post("/api/transcripts/{transcript_id}/analysis", status_code=202)
def start_analysis(transcript_id: str, body: AnalysisRequest, llm: LLMDep):
    try:
        run = llm.start(transcript_id, body.provider, body.model, body.language)
    except LLMError as exc:
        raise api_error(exc)
    return {"run": run.public()}


@router.post("/api/transcripts/{transcript_id}/analysis/cancel")
def cancel_analysis(transcript_id: str, llm: LLMDep, library: LibraryDep):
    """Cancels a running analysis. Never touches the stored one."""
    library.path(transcript_id)
    return {"cancelled": llm.cancel(transcript_id)}


@router.delete("/api/transcripts/{transcript_id}/analysis")
def delete_stored_analysis(transcript_id: str, llm: LLMDep, library: LibraryDep):
    """Deletes the stored analysis (not while one is being made: cancel it first)."""
    library.path(transcript_id)
    run = llm.runs.get(transcript_id)
    if run and run.status == "running":
        raise ApiError(409, "analysis_running", "An analysis is running; cancel it first.")
    return {"deleted": delete_analysis(library, transcript_id)}


@router.post("/api/transcripts/{transcript_id}/ask")
def ask_question(transcript_id: str, body: AskRequest, llm: LLMDep):
    try:
        return llm.ask(transcript_id, body.question.strip(), body.provider, body.model, body.language)
    except LLMError as exc:
        raise api_error(exc)


# Registered before the generic export route, so it wins for ``md``: same URL, but with the analysis.
@router.get("/api/transcripts/{transcript_id}/export/md")
def export_markdown(transcript_id: str, library: LibraryDep, names: bool = True, redact: bool = False,
                    title: bool = False, lang: str | None = None):
    """``redact=true``: redacted words replaced, and without the analysis (it may quote them); the
    title (it may name what was redacted) only with ``title=true``. ``lang``: the transcript part is the
    stored translation (the analysis stays as it is: it has its own language)."""
    mapping = library.names(transcript_id) if names else {}
    shown = None if redact and not title else library.metadata(transcript_id).title
    text = render_markdown(shown, export_segments(library, transcript_id, redact, lang),
                           mapping, None if redact else load_analysis(library, transcript_id))
    privacy.audit(library, transcript_id, "exported", format="md", redacted=redact or None, names=names,
                  sha256=privacy.sha256_bytes(text), **({"lang": lang} if lang else {}))
    name = "transcript.redacted.md" if redact else f"transcript.{lang.lower()}.md" if lang else "transcript.md"
    return Response(text, media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})
