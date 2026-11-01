"""Markdown export. The registry entry renders the transcript only; when the transcript has an
analysis, ``GET /api/transcripts/{id}/export/md`` (``server/routers/analysis.py``) prepends the
summary, chapters, quotes and action items."""

from ..llm.markdown import render_transcript
from . import Exporter, register

register(Exporter("md", "MD", "md", "text/markdown; charset=utf-8", render_transcript))
