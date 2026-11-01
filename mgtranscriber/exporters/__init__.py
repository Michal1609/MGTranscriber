"""Export format registry.

An exporter renders segments (with optional speaker display names) to text or bytes.
Exporters with ``stored_as`` are also written into the transcript folder when a
transcription finishes. To add a format, create a module here that calls ``register()``
and import it at the bottom of this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..models import Segment

Renderer = Callable[[list[Segment], dict[str, str]], "str | bytes"]


@dataclass(frozen=True)
class ExportContext:
    """What a document format may show besides the segments (exporters with ``takes_context``)."""

    title: str = ""
    source: str | None = None
    date: str | None = None  # ISO 8601
    duration: float | None = None
    language: str | None = None
    redacted: bool = False  # the segments were redacted for this export
    options: dict = field(default_factory=dict)  # format options from the query string, e.g. layout=table


@dataclass(frozen=True)
class Exporter:
    name: str  # URL/CLI identifier, e.g. "srt"
    label: str  # button text
    extension: str  # without dot
    media_type: str
    render: Renderer
    stored_as: str | None = None  # file name in the transcript folder, if written after transcription
    applies_names: bool = True  # whether speaker display names change the output
    takes_context: bool = False  # True: render(segments, names, context: ExportContext)

    @property
    def download_name(self) -> str:
        return self.stored_as or f"transcript.{self.extension}"

    def public(self) -> dict:
        return {"name": self.name, "label": self.label, "extension": self.extension,
                "download_name": self.download_name, "applies_names": self.applies_names}


_REGISTRY: dict[str, Exporter] = {}


def register(exporter: Exporter) -> Exporter:
    _REGISTRY[exporter.name] = exporter
    return exporter


def get_exporter(name: str) -> Exporter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown export format '{name}'. Available: {', '.join(_REGISTRY)}") from None


def list_exporters() -> list[Exporter]:
    return list(_REGISTRY.values())


def render(exporter: Exporter, segments: list[Segment], names: dict[str, str],
           context: ExportContext | None = None) -> "str | bytes":
    """Render with the context for exporters that take one."""
    if exporter.takes_context:
        return exporter.render(segments, names, context or ExportContext())
    return exporter.render(segments, names)


def speaker_label(speaker: str, names: dict[str, str]) -> str:
    return names.get(speaker) or speaker


# built-in formats register themselves; import order = button order in the UI
from . import txt, srt, json_export, md, vtt, docx_export  # noqa: E402,F401,I001
