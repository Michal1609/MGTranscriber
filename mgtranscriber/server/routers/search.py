"""Full-text search across all transcripts."""

from fastapi import APIRouter, Query

from ... import voices
from ...folders import FolderError, in_folder, normalize_path
from ..deps import LibraryDep, SearchDep, SettingsDep
from ..errors import ApiError

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("")
def search(index: SearchDep, settings: SettingsDep, library: LibraryDep, q: str = Query("", max_length=500),
           limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0),
           transcript: str | None = Query(None, max_length=200),
           speaker: str | None = Query(None, max_length=100), person: str | None = Query(None, max_length=40),
           folder: str | None = Query(None, max_length=1000), recursive: bool = True):
    """Matching segments, best first: transcript id and title, segment index, start/end time,
    speaker (raw id and display name), the segment text and a snippet with highlight ranges.

    ``speaker`` keeps only segments of a speaker with that display name or raw label; ``person``
    (a voice-memory person id) does the same by that person's name. With either, ``q`` may be empty.

    ``folder`` limits the search to a library folder (empty = unfiled); ``recursive`` (default) includes
    its subfolders. Resolved to transcript ids here, so moving transcripts never reindexes them."""
    ids = None
    if folder is not None:
        try:
            ids = [i for i, f in library.folders_map().items() if in_folder(f, normalize_path(folder), recursive)]
        except FolderError as e:
            raise ApiError(e.status, e.code, e.message, param="folder") from None
    if person:
        if not voices.enabled(settings):
            raise ApiError(409, "voice_disabled", "Voice memory is turned off.")
        found = next((p for p in voices.open_db(settings).list() if p["id"] == person), None)
        if found is None:
            raise ApiError(404, "voice_person_not_found", "Person not found.")
        speaker = found["name"]
    return index.search(q, limit=limit, transcript_id=transcript, offset=offset, speaker=speaker, transcript_ids=ids)
