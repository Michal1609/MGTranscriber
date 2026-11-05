"""Translate a whole transcript, segment by segment, keeping the segment structure 1:1.

Segments are sent in small batches together with a few neighbours as context (already translated
lines before, original lines after) and the speaker names. The model must return exactly one item
per segment id; a batch whose ids or count do not match is retried once and then split in halves, down
to single segments. Text is never moved between segments: an item is only ever stored under its own
id. A single segment that cannot be translated keeps its original text and is listed in
``untranslated``. Word timestamps are dropped (a translation has no word alignment).
"""

from __future__ import annotations

import time
from typing import Callable

from ..models import Segment, speaker_of
from .analysis import _arr, _obj, language_name
from .base import BadModelOutput, LLMProvider, estimate_tokens
from .transcript import speaker_labels

Progress = Callable[[float, str], None]

TRANSLATE_SCHEMA = _obj(items=_arr(_obj(id={"type": "integer"}, text={"type": "string"})))
CONTEXT_BEFORE = 3
CONTEXT_AFTER = 2
MAX_RETRIES = 1  # a failing batch is tried once more, then split; a single segment is tried 1 + MAX_RETRIES times

LANGUAGES = {
    "cs": "Czech", "en": "English", "sk": "Slovak", "de": "German", "pl": "Polish", "es": "Spanish",
    "fr": "French", "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "uk": "Ukrainian", "ru": "Russian",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish", "hu": "Hungarian", "ro": "Romanian",
    "bg": "Bulgarian", "hr": "Croatian", "sl": "Slovenian", "sr": "Serbian", "el": "Greek", "tr": "Turkish",
    "ar": "Arabic", "he": "Hebrew", "hi": "Hindi", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "vi": "Vietnamese", "id": "Indonesian", "th": "Thai", "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian",
}


def lang_name(code: str | None) -> str:
    if not code:
        return "the original language"
    return LANGUAGES.get(code.split("-")[0].lower()) or language_name(code)


def system_prompt(source: str | None, target: str, speakers: dict[str, str], title: str | None) -> str:
    who = (" Speakers: " + ", ".join(speakers.values()) + ".") if speakers else ""
    about = f" The recording is titled '{title}'." if title else ""
    return (
        f"You are a professional subtitle translator from {lang_name(source)} into {lang_name(target)}. "
        "You translate spoken language, line by line, keeping the meaning, tone and register. "
        "Keep names of people, brands and products as they are. Every line is translated on its own: "
        "never merge lines, never split a line, never move words from one line to another, even when a sentence "
        f"continues in the next line. Do not add notes or explanations.{who}{about} Output only JSON matching the schema.")


def batch_prompt(batch: list[int], segments: list[Segment], done: dict[int, str], labels: dict[str, str],
                 target: str) -> str:
    def line(i: int, text: str) -> str:
        return f"{i} [{labels.get(speaker_of(segments[i]), '?')}] {text}"

    first, last = batch[0], batch[-1]
    before = [i for i in range(max(0, first - CONTEXT_BEFORE), first) if segments[i].text.strip()]
    after = [i for i in range(last + 1, min(len(segments), last + 1 + CONTEXT_AFTER)) if segments[i].text.strip()]
    parts = []
    if before:
        parts.append("Context, already translated (do not output these):\n" + "\n".join(
            line(i, done.get(i, segments[i].text.strip())) for i in before))
    parts.append(f"Translate these {len(batch)} lines into {lang_name(target)}:\n" + "\n".join(
        line(i, segments[i].text.strip()) for i in batch))
    if after:
        parts.append("Context, what follows (do not output these):\n" + "\n".join(
            line(i, segments[i].text.strip()) for i in after))
    ids = ", ".join(str(i) for i in batch)
    parts.append('Return JSON {"items": [{"id": <line number>, "text": <translation>}, ...]} with exactly '
                 f"{len(batch)} items, one per translated line, using the same ids ({ids}). "
                 "The text of an item is the translation of that line only, without the [speaker] label.")
    return "\n\n".join(parts)


def _valid(reply: dict, batch: list[int]) -> dict[int, str] | None:
    """{id: text} when the reply has exactly the ids asked for, each once, with non-empty text."""
    items = reply.get("items")
    if not isinstance(items, list) or len(items) != len(batch):
        return None
    out: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            return None
        ident, text = item.get("id"), item.get("text")
        if isinstance(ident, str) and ident.strip().isdigit():
            ident = int(ident)
        if isinstance(ident, bool) or not isinstance(ident, int) or not isinstance(text, str) or not text.strip():
            return None
        if ident in out:
            return None
        out[ident] = text.strip()
    return out if set(out) == set(batch) else None


def plan_batches(indices: list[int], segments: list[Segment], max_tokens: int, max_items: int) -> list[list[int]]:
    batches: list[list[int]] = []
    cur: list[int] = []
    used = 0
    for i in indices:
        cost = estimate_tokens(segments[i].text) + 8
        if cur and (used + cost > max_tokens or len(cur) >= max_items):
            batches.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += cost
    if cur:
        batches.append(cur)
    return batches


def translate_segments(segments: list[Segment], names: dict[str, str], provider: LLMProvider, target: str, *,
                       source: str | None = None, title: str | None = None, progress: Progress | None = None,
                       cancel: Callable[[], None] | None = None) -> dict:
    """Returns ``{"texts": [str per segment, same order], "untranslated": [indices], "calls": n}``.
    Segments without text stay empty; ``texts[i]`` always belongs to ``segments[i]``."""
    indices = [i for i, s in enumerate(segments) if s.text.strip()]
    texts = [""] * len(segments)
    done: dict[int, str] = {}
    untranslated: list[int] = []
    if not indices:
        return {"texts": texts, "untranslated": [], "calls": 0}
    labels = speaker_labels(sorted({speaker_of(segments[i]) for i in indices}), names)
    system = system_prompt(source, target, labels, title)
    local = provider.local
    batches = plan_batches(indices, segments, 450 if local else 1500, 8 if local else 30)
    total_chars = sum(len(segments[i].text) for i in indices) or 1
    state = {"chars": 0, "calls": 0}

    def report(message: str) -> None:
        if progress:
            progress(min(99.0, 1 + 98 * state["chars"] / total_chars), message)

    def call(batch: list[int]) -> dict[int, str] | None:
        if cancel:
            cancel()
        state["calls"] += 1
        budget = sum(estimate_tokens(segments[i].text) + 12 for i in batch) * 3 + 200
        try:
            reply = provider.generate_json(system, batch_prompt(batch, segments, done, labels, target),
                                           TRANSLATE_SCHEMA, max_tokens=min(budget, max(provider.max_output, 1024) * 2))
        except BadModelOutput:
            return None
        return _valid(reply, batch)

    def run(batch: list[int], attempts: int) -> None:
        for _ in range(attempts):
            got = call(batch)
            if got is not None:
                for i, text in got.items():
                    texts[i] = done[i] = text
                state["chars"] += sum(len(segments[i].text) for i in batch)
                return
        if len(batch) == 1:  # nothing left to split: keep the original, never guess
            i = batch[0]
            texts[i] = segments[i].text.strip()
            untranslated.append(i)
            state["chars"] += len(segments[i].text)
            return
        half = len(batch) // 2
        run(batch[:half], 1 + MAX_RETRIES)
        run(batch[half:], 1 + MAX_RETRIES)

    for n, batch in enumerate(batches, 1):
        report(f"translate {n}/{len(batches)}")
        run(batch, 1 + MAX_RETRIES)
    report("translate done")
    return {"texts": texts, "untranslated": sorted(untranslated), "calls": state["calls"]}


def build_translation(segments: list[Segment], names: dict[str, str], provider: LLMProvider, target: str, *,
                      source: str | None = None, title: str | None = None, progress: Progress | None = None,
                      cancel: Callable[[], None] | None = None) -> dict:
    """The content of ``translations/<lang>.json`` (without stamp): same start/end/speaker, translated text."""
    started = time.time()
    result = translate_segments(segments, names, provider, target, source=source, title=title, progress=progress,
                                cancel=cancel)
    out_segments = []
    for seg, text in zip(segments, result["texts"]):
        item = {"start": seg.start, "end": seg.end, "text": text}
        if seg.speaker is not None:
            item["speaker"] = seg.speaker
        out_segments.append(item)
    return {"source_lang": source, "provider": provider.name, "model": provider.model,
            "untranslated": result["untranslated"], "usage": dict(provider.usage),
            "seconds": round(time.time() - started, 1), "segments": out_segments}


# --- the queued job ("translate", see tasks.py) ---------------------------------------------

def translate_task(request, settings, progress, library) -> str:
    """Job: ``request.source`` = transcript id, ``request.language`` = target language, ``request.engine`` =
    LLM provider and ``request.model`` = LLM model (both default to the LLM settings)."""
    from .. import privacy
    from ..progress import Phase
    from ..progress import Progress as Event
    from . import load_config, make_provider
    from .base import EmptyTranscript
    from .store import segments_stamp
    from .translations import normalize_lang, save_translation

    transcript_id = request.source
    target = normalize_lang(request.language)
    meta = library.metadata(transcript_id)
    source = (meta.language or "").lower() or None
    cfg = load_config(settings)
    name = (request.engine or cfg.provider).lower()
    model = request.model or (cfg.final_model if name == cfg.provider else None)
    provider = make_provider(cfg, settings, name, model)
    progress(Event(Phase.LOADING, message=f"Translate to {target} with {provider.name}/{provider.model}",
                   data={"title": meta.title}))
    provider.check()
    stamp = segments_stamp(library, transcript_id)
    segments = library.segments(transcript_id)
    names = library.names(transcript_id)
    if not any(s.text.strip() for s in segments):
        raise EmptyTranscript()

    def report(percent: float, message: str) -> None:
        progress(Event("translate", percent, message))

    data = build_translation(segments, names, provider, target, source=source, title=meta.title, progress=report)
    if data["untranslated"]:
        progress(Event("translate", None, f"{len(data['untranslated'])} segment(s) could not be translated "
                                          "and keep the original text"))
    save_translation(library, transcript_id, target, data, stamp)
    privacy.audit(library, transcript_id, "translated", lang=target, provider=provider.name, model=provider.model,
                  segments=len(data["segments"]), untranslated=len(data["untranslated"]) or None)
    progress(Event(Phase.WRITING, 100, data={"result": transcript_id}))
    return transcript_id
