"""Stored translations of a transcript: ``translations/<lang>.json`` in the transcript folder.

A translation keeps the segment structure of the original (start, end and speaker unchanged, text
translated). It carries the ``segments_stamp`` of ``prepis.json`` it was made from, so a changed
original (edit, new transcription) marks it ``stale`` instead of silently losing the work.
Word-level timestamps do not exist in a translation: segments are stored without ``words``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..fsutil import replace_file
from ..library import Library, now_iso
from ..models import Segment
from .base import LLMError, ensure_has_text
from .store import segments_stamp

TRANSLATIONS_DIR = "translations"
TRANSLATION_VERSION = 1
LANG_RE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})?$")


def normalize_lang(code: str | None) -> str:
    """``cs``, ``pt-br`` ...; raises LLMError (400) for anything that is not a language code, so the
    value is always safe as a file name."""
    value = (code or "").strip().lower().replace("_", "-")
    if not LANG_RE.fullmatch(value):
        error = LLMError(f"'{code}' is not a language code (for example cs, en, de).", "invalid_language")
        error.status = 400
        raise error
    return value


def enqueue_translation(queue, llm, library: Library, transcript_id: str, language: str,
                        provider: str | None = None, model: str | None = None) -> dict:
    """Queue a ``translate`` job (the path of the API, the detail page and watched sources). A job of the same
    language that is already waiting or running is returned instead of a new one. Raises LLMError: 409
    ``translation_same_language``, ``invalid_language`` or whatever the provider check reports."""
    from ..pipeline import TranscriptionRequest

    task = "translate"
    lang = normalize_lang(language)
    ensure_has_text(library, transcript_id)
    source = (library.metadata(transcript_id).language or "").lower()
    if source and source.split("-")[0] == lang.split("-")[0]:
        error = LLMError("The transcript is already in this language.", "translation_same_language")
        error.status = 409
        raise error
    existing = next((j for j in queue.listing() if j.get("task") == task and j["source"] == transcript_id
                     and j["status"] in ("queued", "running") and j["language"] == lang), None)
    if existing:
        return existing
    llm.provider(provider, model).check()
    job = queue.add(TranscriptionRequest(transcript_id, language=lang, engine=provider or None,
                                         model=model or None, task=task))
    return queue.public(job)


def translations_dir(library: Library, transcript_id: str) -> Path:
    return library.path(transcript_id) / TRANSLATIONS_DIR


def translation_path(library: Library, transcript_id: str, lang: str) -> Path:
    return translations_dir(library, transcript_id) / f"{normalize_lang(lang)}.json"


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("segments"), list) else None


def _with_state(data: dict, current_stamp: str | None, library: Library, transcript_id: str) -> dict:
    stamp = data.get("segments_stamp")
    data["stale"] = bool(stamp) and stamp != current_stamp
    return data


def load_translation(library: Library, transcript_id: str, lang: str) -> dict | None:
    """The stored translation (with ``stale``), or None."""
    data = _read(translation_path(library, transcript_id, lang))
    if data is None:
        return None
    return _with_state(data, segments_stamp(library, transcript_id), library, transcript_id)


def translation_segments(data: dict) -> list[Segment]:
    return [Segment(float(s["start"]), float(s["end"]), str(s.get("text", "")), s.get("speaker"))
            for s in data["segments"]]


def list_translations(library: Library, transcript_id: str) -> list[dict]:
    """Metadata of every stored translation (no segments), sorted by language."""
    folder = translations_dir(library, transcript_id)
    if not folder.is_dir():
        return []
    stamp = segments_stamp(library, transcript_id)
    out = []
    for path in sorted(folder.glob("*.json")):
        data = _read(path)
        if data is None or not LANG_RE.fullmatch(path.stem):
            continue
        data = _with_state(data, stamp, library, transcript_id)
        out.append({"lang": path.stem, "source_lang": data.get("source_lang"), "provider": data.get("provider"),
                    "model": data.get("model"), "created": data.get("created"), "stale": data["stale"],
                    "segments": len(data["segments"]), "untranslated": len(data.get("untranslated") or [])})
    return out


def save_translation(library: Library, transcript_id: str, lang: str, data: dict, stamp: str | None = None) -> None:
    """Write atomically, then tell the library (the search index follows)."""
    path = translation_path(library, transcript_id, lang)
    path.parent.mkdir(exist_ok=True)
    data = {**data, "version": TRANSLATION_VERSION, "target_lang": normalize_lang(lang),
            "segments_stamp": stamp or segments_stamp(library, transcript_id)}
    data.setdefault("created", now_iso())
    data.pop("stale", None)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    replace_file(tmp, path)
    library.changed(transcript_id)


def delete_translation(library: Library, transcript_id: str, lang: str) -> bool:
    path = translation_path(library, transcript_id, lang)
    if not path.is_file():
        return False
    path.unlink()
    try:
        path.parent.rmdir()  # the folder goes with its last translation
    except OSError:
        pass
    library.changed(transcript_id)
    return True


def translations_stamp(folder: Path) -> str:
    """mtimes of the translation files (for the search index's change detection)."""
    d = folder / TRANSLATIONS_DIR
    try:
        return "+".join(f"{p.stem}:{p.stat().st_mtime_ns}" for p in sorted(d.glob("*.json")))
    except OSError:
        return ""
