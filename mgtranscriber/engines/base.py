"""The engine interface. An engine turns media into normalized segments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from ..models import TranscriptionResult
from ..progress import ProgressCallback

if TYPE_CHECKING:
    from ..config import Settings
    from ..sources import Media


class EngineError(Exception):
    """A problem the user can fix (missing token, no access, unsupported option)."""


@dataclass
class TranscribeOptions:
    language: str | None = None  # None = auto-detect
    model: str | None = None  # engine-specific model name; None = engine default
    speakers: int | None = None  # exact number of speakers, if known
    diarize: bool = True
    batch_size: int = 8
    extra: dict = field(default_factory=dict)  # engine-specific options


class Engine(ABC):
    name: ClassVar[str]
    needs_audio: ClassVar[bool] = True  # False = works from the URL alone (e.g. existing captions)
    supports_diarization: ClassVar[bool] = False
    default_model: ClassVar[str | None] = None

    def __init__(self, settings: "Settings"):
        self.settings = settings

    def check(self, options: TranscribeOptions) -> None:
        """Fail fast before downloading or loading anything. Raise EngineError."""

    @abstractmethod
    def transcribe(self, media: "Media", options: TranscribeOptions, progress: ProgressCallback) -> TranscriptionResult:
        """Transcribe ``media`` and report progress through ``progress``."""
