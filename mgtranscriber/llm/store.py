"""analysis.json in the transcript folder."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from ..library import Library

ANALYSIS_FILE = "analysis.json"


def analysis_path(library: Library, transcript_id: str) -> Path:
    return library.path(transcript_id) / ANALYSIS_FILE


def segments_stamp(library: Library, transcript_id: str) -> str | None:
    """Hash of prepis.json: the transcript text an analysis was made from."""
    try:
        return hashlib.sha1((library.path(transcript_id) / "prepis.json").read_bytes()).hexdigest()
    except OSError:
        return None


def load_analysis(library: Library, transcript_id: str) -> dict | None:
    """The stored analysis; ``stale`` is true when the transcript changed (e.g. transcribed again) since."""
    try:
        with open(analysis_path(library, transcript_id), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    stamp = data.get("segments_stamp")
    data["stale"] = bool(stamp) and stamp != segments_stamp(library, transcript_id)
    return data


def save_analysis(library: Library, transcript_id: str, data: dict, stamp: str | None = None) -> None:
    path = analysis_path(library, transcript_id)
    data = {**data, "segments_stamp": stamp or segments_stamp(library, transcript_id)}
    data.pop("stale", None)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for delay in (0, 0.05, 0.2, 0.5):  # antivirus/indexers briefly lock files on Windows
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(tmp, path)


def delete_analysis(library: Library, transcript_id: str) -> bool:
    path = analysis_path(library, transcript_id)
    if path.is_file():
        path.unlink()
        return True
    return False
