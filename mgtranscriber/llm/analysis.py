"""Analysis of a whole transcript: summary, key points, chapters, quotes and action items.

Strategy (see docs/LLM.md): the transcript is cut into chunks that fit the model's context
(a single chunk for cloud models with a large window), each chunk is analysed (map), then the
partial summaries are combined (reduce) and the topic markers are merged into chapters. Every
time and every quote the model returns is checked against the real segments.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Callable

from ..models import Segment
from .base import LLMProvider
from .transcript import (Index, QUOTE_MIN_SCORE, _norm, build_turns, chunk_turns, fmt_time, parse_time,
                         render_turns, speaker_labels)

ANALYSIS_VERSION = 1
LANGUAGE_NAMES = {"cs": "Czech", "en": "English", "sk": "Slovak", "de": "German", "pl": "Polish",
                  "es": "Spanish", "fr": "French"}
MAX_QUOTES = 12
Progress = Callable[[float, str], None]


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def _obj(**props) -> dict:
    return {"type": "object", "properties": props, "required": list(props)}


def _arr(item: dict) -> dict:
    return {"type": "array", "items": item}


STR = {"type": "string"}

TOPIC = _obj(start=STR, title=STR, summary=STR)
MAP_SCHEMA = _obj(
    summary=STR, key_points=_arr(STR), topics=_arr(TOPIC),
    quotes=_arr(_obj(time=STR, speaker=STR, text=STR)),
    action_items=_arr(_obj(time=STR, owner=STR, text=STR)),
)
REDUCE_SCHEMA = _obj(summary=STR, key_points=_arr(STR))
CHAPTERS_SCHEMA = _obj(chapters=_arr(TOPIC))
TRANSLATE_SCHEMA = _obj(items=_arr(STR))


def system_prompt(language: str, labels: dict[str, str], names: dict[str, str], title: str | None) -> str:
    lang = language_name(language)
    named = [f"{labels[s]}" for s in labels if names.get(s)]
    speakers = f" The speakers are: {', '.join(labels.values())}." + (
        " Use these names when you mention people." if named else " The labels are anonymous.")
    about = f" The recording is titled '{title}'." if title else ""
    return (
        "You analyse transcripts of spoken audio (podcast, interview, lecture, meeting). Each line is "
        f"'[mm:ss] SPEAKER: text' (or [h:mm:ss]).{speakers}{about} "
        f"Write every free-text field in {lang}, even when the transcript is in another language. "
        "Copy quotes VERBATIM in the original language of the transcript. "
        "Timestamps must be copied from the transcript lines, never invented. "
        "Describe only what is in the transcript. Output only JSON matching the schema.")


def map_prompt(text: str, index: int, count: int, minutes: float, language: str) -> str:
    lang = language_name(language)
    topics = max(2, min(12, math.ceil(minutes / 5)))
    quotes = max(1, min(6, math.ceil(minutes / 12)))
    sentences = "3-5" if minutes <= 20 else "6-10"
    where = ("This is the whole transcript." if count == 1
             else f"This is part {index + 1} of {count} of a longer transcript.")
    key_points = ("key_points: 5-8 most important takeaways, one sentence each.\n" if count == 1
                  else "key_points: empty array.\n")
    return (
        f"{where}\n\n<transcript>\n{text}\n</transcript>\n\nReturn JSON with:\n"
        f"- summary: {sentences} sentences about what is discussed.\n"
        f"- {key_points}"
        f"- topics: about {topics} topic shifts in order; start = timestamp of the line where the topic begins "
        "(copy it from a line), title <= 8 words, summary 1 sentence.\n"
        f"- quotes: up to {quotes} of the most striking self-contained sentences (verbatim, max 40 words) with the "
        "timestamp of their line and the speaker label.\n"
        "- action_items: concrete tasks, commitments, decisions or recommendations that a participant states they "
        "will do or that the listener should do; empty array if there are none (never invent any); owner if clear, "
        "else empty string; time = timestamp of the line.\n"
        f"IMPORTANT: write summary, key_points, topic titles/summaries and action item texts in {lang}; "
        "quotes stay verbatim in the transcript language.")


def reduce_prompt(parts: list[dict], language: str) -> str:
    lang = language_name(language)
    joined = "\n\n".join(f"Part {i + 1} [{p['range']}]: {p['summary']}" for i, p in enumerate(parts))
    return (f"Below are summaries of consecutive parts of one long recording.\n\n{joined}\n\n"
            f"Write in {lang}: summary = a coherent summary of the whole recording (6-10 sentences, no repetition, "
            "no 'part 1/part 2' wording), key_points = the 5-8 most important takeaways (one sentence each).")


def chapters_prompt(topics: list[dict], duration: float, target: int, language: str) -> str:
    lang = language_name(language)
    joined = "\n".join(f"{t['start_fmt']} | {t['title']} | {t.get('summary', '')}" for t in topics)
    return (f"Candidate topic markers of a {fmt_time(duration)} long recording (start | title | summary):\n{joined}\n\n"
            f"Merge them into exactly {target} chapters of roughly equal importance, in chronological order, covering "
            f"the whole recording. For each chapter: start = copy EXACTLY one of the start timestamps above, "
            f"title <= 8 words, summary 1-2 sentences, all in {lang}. The first chapter starts at the first timestamp.")


TRANSLATE_SYSTEM = "You translate short texts. Output only JSON matching the schema."
TRANSLATE_BATCH = 8  # small batches: models drop or copy items in long lists


def translate_prompt(items: list[str], language: str) -> str:
    lang = language_name(language)
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(items))
    return (f"Translate every text below into {lang}. Texts that are already {lang} stay as they are; every text in "
            f"another language MUST be translated. Return exactly {len(items)} items in the same order.\n\n{numbered}")


# --- helpers -------------------------------------------------------------------------------

def target_chapter_count(duration: float) -> int:
    return max(3, min(24, round(duration / 420)))


def _dicts(value) -> list[dict]:
    """The dict items of a model-produced list (models sometimes put strings or nulls in lists)."""
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def _str(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def even_chapters(chapters: list[dict], candidates: list[dict], duration: float, target: int) -> list[dict]:
    """Make chapters reasonably even: drop ones too close together, fill long gaps from the candidates.

    ``chapters``/``candidates`` are dicts with ``start`` (a real segment start), ``title``, ``summary``.
    """
    if not chapters:
        chapters = list(candidates[:1])
    if not chapters:
        return []
    span = max(duration, 1.0)
    min_gap = max(90.0, span / (target * 2.0))
    kept: list[dict] = []
    for ch in sorted(chapters, key=lambda c: c["start"]):
        if not kept or ch["start"] - kept[-1]["start"] >= min_gap:
            kept.append(ch)
    if not kept:
        return []
    first = min((c["start"] for c in candidates), default=kept[0]["start"])
    if kept[0]["start"] - first > span * 0.05:  # the opening topic is missing
        opening = next(c for c in sorted(candidates, key=lambda c: c["start"]))
        kept.insert(0, opening)
    big = span / target * 2.0
    while True:
        edges = [c["start"] for c in kept] + [span]
        gaps = sorted(((edges[i + 1] - edges[i], i) for i in range(len(kept))), reverse=True)
        added = False
        for size, i in gaps:
            if size <= big:
                break
            lo, hi, mid = edges[i], edges[i + 1], (edges[i] + edges[i + 1]) / 2
            inside = [c for c in candidates if lo + min_gap <= c["start"] <= hi - min_gap
                      and all(abs(c["start"] - k["start"]) >= min_gap for k in kept)]
            if inside:
                kept.append(min(inside, key=lambda c: abs(c["start"] - mid)))
                kept.sort(key=lambda c: c["start"])
                added = True
                break
        if not added:
            return kept


def _spread(items: list[dict], limit: int) -> list[dict]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def analyze(segments: list[Segment], names: dict[str, str], provider: LLMProvider, *, language: str = "en",
            final_provider: LLMProvider | None = None, title: str | None = None,
            progress: Progress | None = None, cancel: Callable[[], None] | None = None) -> dict:
    """Run the full analysis. ``final_provider`` (optional) does the reduce/chapter/translation steps
    with another, usually stronger model."""
    final = final_provider or provider
    started = time.time()

    def step(percent: float, message: str) -> None:
        if cancel:
            cancel()
        if progress:
            progress(percent, message)

    turns = build_turns(segments)
    if not turns:
        raise ValueError("The transcript is empty.")
    index = Index(segments)
    speakers = sorted({t.speaker for t in turns})
    labels = speaker_labels(speakers, names)
    duration = max(s.end for s in segments)
    system = system_prompt(language, labels, names, title)
    chunks = chunk_turns(turns, labels, provider.chunk_tokens)
    total_steps = len(chunks) + 3
    step(1, "prepare")

    parts: list[dict] = []
    for i, chunk in enumerate(chunks):
        step(2 + 85 * i / total_steps, f"map {i + 1}/{len(chunks)}")
        minutes = (chunk[-1].end - chunk[0].start) / 60
        data = provider.generate_json(system, map_prompt(render_turns(chunk, labels), i, len(chunks), minutes, language),
                                      MAP_SCHEMA, max_tokens=provider.max_output if len(chunks) == 1 else 2048)
        data["range"] = f"{fmt_time(chunk[0].start)}-{fmt_time(chunk[-1].end)}"
        parts.append(data)

    # --- summary and key points ---
    step(2 + 85 * len(chunks) / total_steps, "reduce")
    if len(parts) == 1:
        summary, key_points = parts[0].get("summary", ""), list(parts[0].get("key_points") or [])
    else:
        red = final.generate_json(system, reduce_prompt(parts, language), REDUCE_SCHEMA, max_tokens=1500)
        summary, key_points = red.get("summary", ""), list(red.get("key_points") or [])

    # --- chapters from topic markers, snapped to real segment starts ---
    step(2 + 85 * (len(chunks) + 1) / total_steps, "chapters")
    candidates: list[dict] = []
    deviations: list[float] = []
    for part in parts:
        for topic in _dicts(part.get("topics")):
            seconds = parse_time(topic.get("start"))
            if seconds is None or not _str(topic.get("title")):
                continue
            start, dev = index.snap(seconds)
            deviations.append(dev)
            candidates.append({"start": start, "start_fmt": fmt_time(start), "title": _str(topic["title"]),
                               "summary": _str(topic.get("summary"))})
    candidates.sort(key=lambda c: c["start"])
    seen: set[float] = set()
    candidates = [c for c in candidates if not (c["start"] in seen or seen.add(c["start"]))]
    target = target_chapter_count(duration)
    chapters = candidates
    if len(candidates) > target:
        merged = final.generate_json(system, chapters_prompt(candidates, duration, target, language), CHAPTERS_SCHEMA,
                                     max_tokens=3000)
        chapters = []
        by_start = {c["start"]: c for c in candidates}
        for ch in _dicts(merged.get("chapters")):
            seconds = parse_time(ch.get("start"))
            if seconds is None or not _str(ch.get("title")):
                continue
            start, dev = index.snap(seconds)
            deviations.append(dev)
            chapters.append({"start": start, "title": _str(ch["title"]),
                             "summary": _str(ch.get("summary")) or _str(by_start.get(start, {}).get("summary"))})
    chapters = even_chapters(chapters, candidates, duration, target)
    chapters = [{"start": c["start"], "title": c["title"], "summary": c.get("summary", "")} for c in chapters]

    # --- quotes: verbatim and located ---
    quotes, dropped = [], 0
    for part in parts:
        for q in _dicts(part.get("quotes")):
            text = _str(q.get("text"))
            found, score = index.find_quote(text, parse_time(q.get("time")))
            if found is None or score < QUOTE_MIN_SCORE:
                dropped += 1
                continue
            seg = index.segment_at(found)
            quotes.append({"time": found, "speaker": (seg.speaker if seg and seg.speaker else "UNKNOWN"),
                           "text": text})
    quotes.sort(key=lambda q: q["time"])
    uniq_quotes, seen_q = [], set()
    for q in quotes:
        key = _norm(q["text"])
        if q["time"] not in seen_q and key not in seen_q:
            seen_q.update({q["time"], key})
            uniq_quotes.append(q)
    quotes = _spread(uniq_quotes, MAX_QUOTES)

    # --- action items ---
    actions, seen_a = [], set()
    for part in parts:
        for a in _dicts(part.get("action_items")):
            text = _str(a.get("text"))
            if not text or _norm(text) in seen_a:
                continue
            seen_a.add(_norm(text))
            seconds = parse_time(a.get("time"))
            actions.append({"time": index.snap(seconds)[0] if seconds is not None else None,
                            "owner": _str(a.get("owner")), "text": text})

    # --- one pass to put everything in the target language (models copy the speaker's language) ---
    texts = [(a, "text") for a in actions] + [(c, k) for c in chapters for k in ("title", "summary") if c.get(k)]
    if texts:
        step(2 + 85 * (len(chunks) + 2) / total_steps, "translate")
        try:
            for i in range(0, len(texts), TRANSLATE_BATCH):
                batch = texts[i:i + TRANSLATE_BATCH]
                tr = final.generate_json(TRANSLATE_SYSTEM, translate_prompt([d[k] for d, k in batch], language),
                                         TRANSLATE_SCHEMA, max_tokens=2000)
                items = tr.get("items") or []
                if len(items) == len(batch) and all(isinstance(x, str) and x.strip() for x in items):
                    for (d, k), x in zip(batch, items):
                        d[k] = x.strip()
        except Exception:  # noqa: BLE001 - translation is a nicety; keep the originals
            pass

    step(99, "done")
    usage = dict(provider.usage)
    if final is not provider:
        usage = {k: usage[k] + final.usage[k] for k in usage}
    return {
        "version": ANALYSIS_VERSION,
        "language": language,
        "provider": provider.name,
        "model": provider.model,
        "final_model": final.model if final is not provider else None,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration": duration,
        "summary": summary.strip(),
        "key_points": [str(k).strip() for k in key_points if str(k).strip()],
        "chapters": chapters,
        "quotes": quotes,
        "action_items": actions,
        "validation": {"chunks": len(chunks), "max_time_snap_s": round(max(deviations, default=0.0), 2),
                       "quotes_dropped": dropped, "quotes_kept": len(quotes)},
        "usage": usage,
        "seconds": round(time.time() - started, 1),
    }
