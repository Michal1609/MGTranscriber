"""Transcript editor: segments with words for editing, saving with version history, restoring.

See ``mgtranscriber/editor.py`` and ``docs/EDITOR.md``.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ... import editor
from ...fsutil import LockTimeout
from ..deps import LibraryDep, SettingsDep
from ..errors import ApiError

router = APIRouter(prefix="/api/transcripts", tags=["editor"])


class SaveSegments(BaseModel):
    base: str | None = Field(None, max_length=64)  # segments stamp the edit started from; null = overwrite
    segments: list[dict]


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except editor.EditError as e:
        raise ApiError(e.status, e.code, e.message)
    except LockTimeout:
        raise ApiError(503, "edit_busy", "The transcript is being saved elsewhere; try again.")


def _editable(library, transcript_id: str) -> dict:
    segments = [editor.normalize(s) for s in library.segments(transcript_id)]
    return {
        "stamp": library.segments_stamp(transcript_id),
        "segments": [s.to_dict() for s in segments],
        "names": library.names(transcript_id),
        "language": library.metadata(transcript_id).language,
        "versions": editor.list_versions(library, transcript_id),
    }


@router.get("/{transcript_id}/segments")
def get_segments(transcript_id: str, library: LibraryDep):
    """Segments with words for the editor (words 1:1 to the text tokens) and the stamp to save against."""
    return _editable(library, transcript_id)


@router.put("/{transcript_id}/segments")
def save_segments(transcript_id: str, body: SaveSegments, library: LibraryDep, settings: SettingsDep):
    """Save edited segments. ``409 edit_conflict`` when the transcript changed since ``base``."""
    library.path(transcript_id)
    separated = bool(((library.metadata(transcript_id).extra or {}).get("separation") or {}).get("used"))
    segments = _run(editor.parse_segments, body.segments, allow_self_overlap=separated)
    if not segments:
        raise ApiError(422, "invalid_segments", "A transcript needs at least one segment.")
    result = _run(editor.save, library, transcript_id, segments, base=body.base, settings=settings)
    return {**result, "versions": editor.list_versions(library, transcript_id)}


@router.get("/{transcript_id}/versions")
def versions(transcript_id: str, library: LibraryDep):
    library.path(transcript_id)
    return editor.list_versions(library, transcript_id)


@router.post("/{transcript_id}/versions/{version}/restore")
def restore(transcript_id: str, version: str, library: LibraryDep, settings: SettingsDep):
    """Make an earlier version current; the current state is kept as a version first."""
    library.path(transcript_id)
    _run(editor.restore, library, transcript_id, version, settings)
    return _editable(library, transcript_id)
