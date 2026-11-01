"""Structured progress reporting used by the pipeline, engines, CLI and job queue."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable


class Phase:
    """Well-known phases. Engines may report others; the UI shows unknown ones as text."""

    QUEUED = "queued"
    STARTING = "starting"
    LOADING = "loading"
    DOWNLOAD = "download"
    TRANSCRIBE = "transcribe"
    ALIGN = "align"
    DIARIZE = "diarize"
    WRITING = "writing"


@dataclass
class Progress:
    phase: str
    percent: float | None = None  # 0-100 within the phase, None = indeterminate
    message: str | None = None  # human-readable detail (English), also used as a log line
    data: dict | None = None  # structured extras, e.g. {"title": ...} once the source is known

    def to_dict(self) -> dict:
        return asdict(self)


ProgressCallback = Callable[[Progress], None]


def no_progress(_: Progress) -> None:
    pass
