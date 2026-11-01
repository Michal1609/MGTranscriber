"""Normalized transcript data shared by engines, the library, exporters and the API.

The JSON shape is the one WhisperX writes to ``prepis.json`` (a list of segments with
optional ``speaker`` and ``words``), so existing transcripts load unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

UNKNOWN_SPEAKER = "UNKNOWN"


@dataclass
class Word:
    word: str
    start: float | None = None
    end: float | None = None
    score: float | None = None
    speaker: str | None = None
    redacted: bool | None = None  # marked in the editor; redacted exports replace it (see privacy.py)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v is not False}


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        words = [
            Word(w.get("word", ""), w.get("start"), w.get("end"), w.get("score"), w.get("speaker"),
                 True if w.get("redacted") else None)
            for w in data.get("words") or []
        ]
        return cls(float(data["start"]), float(data["end"]), data.get("text", ""), data.get("speaker"), words)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"start": self.start, "end": self.end, "text": self.text}
        if self.speaker is not None:
            out["speaker"] = self.speaker
        out["words"] = [w.to_dict() for w in self.words]
        return out


@dataclass
class TranscriptionResult:
    """What an engine returns."""

    segments: list[Segment]
    language: str | None = None
    duration: float | None = None  # seconds of audio, if the engine knows it
    model: str | None = None
    speaker_embeddings: dict[str, list[float]] | None = None  # raw pyannote voice embeddings per speaker label


def speaker_of(segment: Segment) -> str:
    return segment.speaker or UNKNOWN_SPEAKER


def speaker_stats(segments: list[Segment]) -> dict[str, float]:
    """Seconds of speech per speaker, sorted by speaker id."""
    stats: dict[str, float] = {}
    for seg in segments:
        spk = speaker_of(seg)
        stats[spk] = stats.get(spk, 0.0) + seg.end - seg.start
    return dict(sorted(stats.items()))


def speaker_turns(segments: list[Segment]) -> list[dict]:
    """Merge consecutive segments of the same speaker into blocks."""
    turns: list[dict] = []
    for seg in segments:
        spk = speaker_of(seg)
        text = seg.text.strip()
        if turns and turns[-1]["speaker"] == spk:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = seg.end
        else:
            turns.append({"speaker": spk, "start": seg.start, "end": seg.end, "text": text})
    return turns
