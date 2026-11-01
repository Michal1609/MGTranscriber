"""Transcript text for LLMs: turns, chunking, and checking model output against the real segments."""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..models import Segment, speaker_of
from .base import estimate_tokens

MAX_TURN_CHARS = 700  # a long monologue is split so a chunk boundary can fall inside it
MAX_TURN_GAP = 1.5  # seconds of silence that still count as the same turn
QUOTE_MIN_SCORE = 0.85


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def parse_time(value) -> float | None:
    """'1:02:03' / '02:03' / '[02:03]' / 123 -> seconds (a bare number is seconds); None if unreadable."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    parts = re.findall(r"\d+(?:[.,]\d+)?", str(value or ""))
    if not parts:
        return None
    nums = [float(p.replace(",", ".")) for p in parts[-3:]]
    return sum(n * 60 ** i for i, n in enumerate(reversed(nums)))


@dataclass
class Turn:
    start: float
    end: float
    speaker: str  # speaker id (SPEAKER_00 / UNKNOWN)
    text: str


def clean_label(name: str) -> str:
    return re.sub(r"[\[\]:\s]+", " ", name).strip()[:40] or "?"


def speaker_labels(speakers: list[str], names: dict[str, str]) -> dict[str, str]:
    """Unique short label per speaker: the display name when set, else S0, S1 ..."""
    labels: dict[str, str] = {}
    used: set[str] = set()
    for spk in sorted(speakers):
        m = re.search(r"(\d+)$", spk)
        label = clean_label(names[spk]) if names.get(spk) else (f"S{int(m.group(1))}" if m else spk)
        while label in used:
            label += "'"
        used.add(label)
        labels[spk] = label
    return labels


def build_turns(segments: list[Segment]) -> list[Turn]:
    turns: list[Turn] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        spk = speaker_of(seg)
        last = turns[-1] if turns else None
        if last and last.speaker == spk and seg.start - last.end <= MAX_TURN_GAP and len(last.text) < MAX_TURN_CHARS:
            last.text += " " + text
            last.end = seg.end
        else:
            turns.append(Turn(seg.start, seg.end, spk, text))
    return turns


def turn_line(turn: Turn, labels: dict[str, str]) -> str:
    return f"[{fmt_time(turn.start)}] {labels.get(turn.speaker, turn.speaker)}: {turn.text}"


def render_turns(turns: list[Turn], labels: dict[str, str]) -> str:
    return "\n".join(turn_line(t, labels) for t in turns)


def chunk_turns(turns: list[Turn], labels: dict[str, str], max_tokens: int) -> list[list[Turn]]:
    """Split at turn boundaries into chunks of about ``max_tokens``; prefer a cut at a pause of 2 s or more."""
    chunks: list[list[Turn]] = []
    cur: list[Turn] = []
    used = 0
    for turn in turns:
        cost = estimate_tokens(turn_line(turn, labels))
        if cur and used + cost > max_tokens:
            cut, acc = len(cur), 0
            for k in range(len(cur) - 1, 0, -1):  # look back over the last 20 % for a pause
                acc += estimate_tokens(turn_line(cur[k], labels))
                if acc > max_tokens * 0.2:
                    break
                if cur[k].start - cur[k - 1].end >= 2.0:
                    cut = k
                    break
            chunks.append(cur[:cut])
            cur = cur[cut:]
            used = sum(estimate_tokens(turn_line(t, labels)) for t in cur)
        cur.append(turn)
        used += cost
    if cur:
        chunks.append(cur)
    return chunks


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


class Index:
    """Looks times and quotes up in the real segments, so nothing the model says is trusted blindly."""

    def __init__(self, segments: list[Segment]):
        self.segments = [s for s in segments if s.text.strip()]
        self.starts = [s.start for s in self.segments]
        self.norm = [_norm(s.text) for s in self.segments]
        self._full = ""
        self._offsets: list[int] = []
        for n in self.norm:
            self._offsets.append(len(self._full))
            self._full += n + " "

    def snap(self, seconds: float) -> tuple[float, float]:
        """Start of the nearest segment and how far (s) the requested time was from it."""
        if not self.starts:
            return seconds, 0.0
        k = bisect.bisect_left(self.starts, seconds)
        near = [self.starts[j] for j in (k - 1, k) if 0 <= j < len(self.starts)]
        best = min(near, key=lambda x: abs(x - seconds))
        return best, abs(best - seconds)

    def segment_at(self, start: float) -> Segment | None:
        k = bisect.bisect_left(self.starts, start)
        return self.segments[k] if k < len(self.segments) and self.starts[k] == start else None

    def find_quote(self, text: str, around: float | None = None, window: float = 240.0) -> tuple[float | None, float]:
        """(start of the segment where the quote begins, similarity 0..1). Exact matches are searched
        in the whole transcript; fuzzy ones near ``around``."""
        q = _norm(text)
        if len(q) < 8:
            return None, 0.0
        hits = [m.start() for m in re.finditer(re.escape(q), self._full)]
        if hits:
            # several exact hits: take the one closest to the time the model gave
            def seg_index(off: int) -> int:
                return max(0, bisect.bisect_right(self._offsets, off) - 1)
            idx = min((seg_index(h) for h in hits),
                      key=lambda i: abs(self.starts[i] - around) if around is not None else i)
            return self.starts[idx], 1.0
        if len(q) > 400:
            return None, 0.0
        if around is None:
            cand = range(len(self.segments))
        else:
            cand = [i for i, s in enumerate(self.starts) if abs(s - around) <= window]
        best: tuple[float | None, float] = (None, 0.0)
        for i in cand:
            for n in (1, 2, 3):
                if i + n > len(self.segments):
                    break
                joined = " ".join(self.norm[i:i + n])
                m = SequenceMatcher(None, q, joined, autojunk=False).find_longest_match(0, len(q), 0, len(joined))
                score = m.size / len(q)
                if score > best[1]:
                    # the quote starts in the segment that contains the beginning of the match
                    seg, pos = i, m.b
                    for length in [len(x) + 1 for x in self.norm[i:i + n]][:-1]:
                        if pos < length:
                            break
                        pos -= length
                        seg += 1
                    best = (self.starts[seg], score)
        return best


def windows(turns: list[Turn], words: int = 220, overlap_turns: int = 2) -> list[list[Turn]]:
    """Overlapping windows of consecutive turns (~``words`` words) for retrieval."""
    out: list[list[Turn]] = []
    cur: list[Turn] = []
    count = 0
    for turn in turns:
        cur.append(turn)
        count += len(turn.text.split())
        if count >= words:
            out.append(cur)
            cur = cur[-overlap_turns:]
            count = sum(len(t.text.split()) for t in cur)
    if cur and (not out or len(cur) > overlap_turns):
        out.append(cur)
    return out
