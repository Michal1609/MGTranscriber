"""Question answering over a transcript with verified citations.

Short transcripts (or providers with a big context) are sent whole. Otherwise the transcript is
split into overlapping windows, embedded once, and the windows closest to the question are sent.
An answer counts only if the model cites passages that really occur in the transcript, and (when
retrieval is used) the question is similar enough to the transcript at all; otherwise the result
is ``found = False`` ("the transcript does not say").
"""

from __future__ import annotations

import math
import threading

from ..models import Segment
from .analysis import _obj, _arr, STR, language_name
from .base import LLMError, LLMProvider, estimate_tokens
from .transcript import (Index, QUOTE_MIN_SCORE, Turn, build_turns, parse_time, render_turns,
                         speaker_labels, windows)

# citations come before the answer: a model that has to pick the supporting sentences first writes a
# better grounded answer (and small models otherwise tend to skip the citations)
ANSWER_SCHEMA = _obj(found={"type": "boolean"}, citations=_arr(_obj(time=STR, text=STR)), answer=STR)
NOT_FOUND = {"cs": "V přepisu to není.", "en": "The transcript does not say."}
#: cosine similarity (best window vs. question) below which no model call is made. bge-m3 gives
#: ~0.35-0.47 for off-topic questions and 0.47-0.67 for relevant ones on an 88 min interview (docs/LLM.md);
#: the model and the citation checks below catch what slips through.
MIN_RETRIEVAL_SCORE = 0.44
#: the cited passage itself must also be related to the question
MIN_CITATION_SCORE = 0.35
FULL_CONTEXT_SHARE = 0.6  # send the whole transcript when it fits in this share of the model's context
TOP_WINDOWS = 8


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)


def not_found_text(language: str) -> str:
    return NOT_FOUND.get(language, NOT_FOUND["en"])


class Retriever:
    """Embeds the transcript windows once; reuse the object for several questions."""

    def __init__(self, turns: list[Turn], labels: dict[str, str], provider: LLMProvider):
        self.provider = provider
        self._turns, self._labels = turns, labels  # windows and texts are made on first use (a whole-transcript answer never needs them)
        self._windows: list | None = None
        self._texts: list[str] | None = None
        self._vectors: list[list[float]] | None = None
        self._lock = threading.RLock()  # vectors() holds it while reading .texts, whose _prepare() takes it too

    def _prepare(self) -> None:
        with self._lock:
            if self._texts is None:
                self._windows = windows(self._turns)
                self._texts = [render_turns(w, self._labels) for w in self._windows]
                self._turns = self._labels = None

    @property
    def windows(self) -> list:
        self._prepare()
        return self._windows

    @property
    def texts(self) -> list[str]:
        self._prepare()
        return self._texts

    def vectors(self) -> list[list[float]] | None:
        with self._lock:  # two simultaneous questions must not embed the whole transcript twice
            if self._vectors is None:
                self._vectors = self.provider.embed(self.texts)
            return self._vectors

    def search(self, question: str, k: int = TOP_WINDOWS) -> list[tuple[float, int]] | None:
        vectors = self.vectors()
        if vectors is None:
            return None
        q = self.provider.embed([question])[0]
        scored = sorted(((cosine(q, v), i) for i, v in enumerate(vectors)), reverse=True)
        return scored[:k]


def ask(segments: list[Segment], names: dict[str, str], question: str, provider: LLMProvider, *, language: str = "en",
        retriever: Retriever | None = None, title: str | None = None) -> dict:
    turns = build_turns(segments)
    index = Index(segments)
    labels = speaker_labels(sorted({t.speaker for t in turns}), names)
    full_text = render_turns(turns, labels)
    whole = estimate_tokens(full_text) <= provider.max_context * FULL_CONTEXT_SHARE
    info: dict = {"mode": "full" if whole else "retrieval"}
    question_vec = None

    if whole:
        context = full_text
    else:
        retriever = retriever or Retriever(turns, labels, provider)
        hits = retriever.search(question)
        if hits is None:
            raise LLMError(f"The transcript is too long for {provider.name}/{provider.model} and the provider has no "
                           "embeddings to pick the relevant parts.", "llm_transcript_too_long")
        info["best_score"] = round(hits[0][0], 3)
        if hits[0][0] < MIN_RETRIEVAL_SCORE:
            return {"found": False, "answer": not_found_text(language), "citations": [], "retrieval": info}
        keep = sorted(i for _, i in hits)
        context = "\n\n---\n\n".join(retriever.texts[i] for i in keep)
        info["windows"] = len(keep)
        question_vec = provider.embed([question])[0]

    system = (
        "You answer questions about a transcript, strictly from the provided text. Each line is "
        f"'[mm:ss] SPEAKER: text'. Answer in {language_name(language)}. Be concise (1-5 sentences). "
        "Set found=true only if the text directly answers the question; otherwise found=false, no citations and an "
        "empty answer. Never use outside knowledge. First fill citations: 1-3 items (required when found=true), "
        "time = the timestamp of the line, text = an EXACT copy of one sentence (max 40 words) from that line, in the "
        "transcript's original language. Then write the answer using only those citations.")
    user = (f"<transcript>\n{context}\n</transcript>\n\nQuestion: {question}"
            + (f"\n(The recording is titled '{title}'.)" if title else ""))
    data = provider.generate_json(system, user, ANSWER_SCHEMA, max_tokens=1200)

    citations = []
    for c in [c for c in (data.get("citations") or []) if isinstance(c, dict)][:6]:
        text = (c.get("text") or "").strip() if isinstance(c.get("text"), str) else ""
        found, score = index.find_quote(text, parse_time(c.get("time")), window=900)
        if found is None or score < QUOTE_MIN_SCORE:
            continue
        if question_vec is not None:  # the passage must be related to the question, not just present
            if cosine(question_vec, provider.embed([text])[0]) < MIN_CITATION_SCORE:
                continue
        seg = index.segment_at(found)
        citations.append({"time": found, "speaker": (seg.speaker if seg and seg.speaker else "UNKNOWN"), "text": text})
    citations = citations[:4]
    answer = (data.get("answer") or "").strip()
    ok = bool(data.get("found")) and bool(answer) and bool(citations)
    info["citations_proposed"] = len(data.get("citations") or [])
    if not ok:
        return {"found": False, "answer": not_found_text(language), "citations": [], "retrieval": info}
    return {"found": True, "answer": answer, "citations": citations, "retrieval": info}
