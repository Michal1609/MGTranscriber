"""The transcript library: list, detail, speaker names, exports and audio."""

import re
from dataclasses import replace
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import FileResponse

from ... import privacy
from ...folders import FolderError, normalize_path
from ...exporters import ExportContext, get_exporter, list_exporters, render
from ...library import AUDIO_TYPES, SEGMENTS_FILE
from ...models import speaker_of
from ..deps import LibraryDep
from ..errors import ApiError
from .setup import client_is_local

router = APIRouter(prefix="/api/transcripts", tags=["transcripts"])


def attachment(name: str) -> dict:
    return {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"}


@router.get("")
def list_transcripts(library: LibraryDep, response: Response, folder: str | None = Query(None, max_length=1000),
                     recursive: bool = True, limit: int | None = Query(None, ge=1, le=10000),
                     offset: int = Query(0, ge=0)):
    """All transcripts, newest first. ``folder``: only those in that library folder (empty = unfiled);
    ``recursive`` (default true, like search and MCP) includes its subfolders; ``recursive=false`` only the
    folder itself. Unfiled (empty) has no subfolders. ``limit``/``offset`` return one page of that list (still a
    plain array; the full count is in the ``X-Total-Count`` header); without them the whole list is returned."""
    try:
        items = library.list() if folder is None else library.list(normalize_path(folder), recursive)
    except FolderError as e:
        raise ApiError(e.status, e.code, e.message, param="folder") from None
    response.headers["X-Total-Count"] = str(len(items))
    if limit is None and not offset:
        return items
    return items[offset:] if limit is None else items[offset:offset + limit]


def _http_url(value) -> str | None:
    """Only http(s) links are offered to the page (never javascript:, data:, file: ...)."""
    if isinstance(value, str) and value.strip()[:8].lower().startswith(("http://", "https://")):
        parts = urlsplit(value.strip())
        if parts.scheme in ("http", "https") and parts.netloc:
            return value.strip()
    return None


def _text(value, limit: int = 300) -> str | None:
    return value.strip()[:limit] or None if isinstance(value, str) else None


def build_origin(meta, watch: dict | None, full_paths: bool) -> dict | None:
    """Where a transcript came from, for the detail page: ``{url, file, channel, channel_url, watch}`` with only
    the known keys; None when nothing is known (recordings, uploads). A local source is shown as its file name,
    the full path only to this computer (``full_paths``)."""
    info = meta.extra.get("source_info") if isinstance(meta.extra.get("source_info"), dict) else {}
    origin: dict = {}
    url = _http_url(info.get("webpage_url")) or _http_url(meta.source)
    if url:
        origin["url"] = url
    elif meta.source and not meta.extra.get("uploaded_via"):  # an upload's source is a temporary server path
        name = _text(meta.source.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1])
        if name and not re.match(r"^[A-Za-z][A-Za-z0-9+.-]+:", meta.source):
            origin["file"] = name
            if full_paths:
                origin["path"] = meta.source
    channel = _text(info.get("channel")) or _text(info.get("uploader"))
    if channel:
        origin["channel"] = channel
        channel_url = _http_url(info.get("channel_url")) or _http_url(info.get("uploader_url"))
        if channel_url:
            origin["channel_url"] = channel_url
    if isinstance(watch, dict) and watch.get("id"):
        origin["watch"] = {"id": str(watch["id"]), "name": _text(watch.get("name")) or "", "kind": str(watch.get("kind") or "")}
    return origin or None


@router.get("/{transcript_id}")
def get_transcript(transcript_id: str, library: LibraryDep, request: Request):
    summary = library.summary(transcript_id)
    meta = library.metadata(transcript_id)
    watch = meta.extra.get("watch")
    if not watch:
        manager = getattr(request.app.state, "watch", None)
        watch = manager.origin_of(transcript_id) if manager else None
    return {
        **summary,
        "origin": build_origin(meta, watch, client_is_local(request)),
        "metadata": meta.to_dict(),
        "segments": [
            {"start": s.start, "end": s.end, "speaker": speaker_of(s), "text": s.text.strip()}
            for s in library.segments(transcript_id)
        ],
    }


@router.put("/{transcript_id}/speakers")
def save_speakers(transcript_id: str, names: dict[str, str], library: LibraryDep):
    from ... import voices

    library.path(transcript_id)  # 404 for unknown transcripts
    with voices.transcript_lock(library, transcript_id):  # identification must not overwrite what is typed now
        return library.save_names(transcript_id, names)


def export_context(library, transcript_id: str, redacted: bool = False, options: dict | None = None,
                   show_title: bool = False) -> ExportContext:
    """A redacted export leaves out the source (path or URL) and, unless ``show_title``, the title:
    both may name what was redacted."""
    meta = library.metadata(transcript_id)
    hide = redacted and not show_title
    return ExportContext(title=None if hide else meta.title, source=None if redacted else meta.source, date=meta.date, duration=meta.duration,
                         language=meta.language, redacted=redacted, options=options or {})


def export_segments(library, transcript_id: str, redact: bool, lang: str | None = None):
    """The segments to export; with ``redact`` every run of redacted words becomes one label. ``lang``: the
    stored translation (404 when there is none). A translation carries no word marks, so it cannot be
    redacted: a redacted export in a translated language is refused rather than leaking the words."""
    if lang:
        from ...llm import LLMError
        from ...llm.translations import load_translation, normalize_lang, translation_segments
        try:
            lang = normalize_lang(lang)
        except LLMError as exc:
            raise ApiError(exc.status, exc.code, exc.message, param="lang") from None
        if redact:
            raise ApiError(409, "translation_not_redactable",
                           "A translation cannot be exported redacted; export the original instead.", param="lang")
        data = load_translation(library, transcript_id, lang)
        if data is None:
            raise ApiError(404, "translation_not_found", "There is no translation in this language.", param="lang")
        return translation_segments(data)
    segments = library.segments(transcript_id)
    if redact:
        segments = privacy.redact_segments(segments, privacy.redacted_label(library.metadata(transcript_id).language))
    return segments


@router.get("/{transcript_id}/export/{fmt}")
def export(transcript_id: str, fmt: str, library: LibraryDep, names: bool = True,
           redact: bool = False, layout: str | None = None, title: bool = False, lang: str | None = None):
    """``redact=true``: the redacted version (redacted words replaced; no source, and no title unless
    ``title=true``). ``layout=table``: DOCX as a table. ``lang=cs``: the stored translation (SRT/VTT then
    work as subtitles in that language)."""
    try:
        exporter = get_exporter(fmt)
    except KeyError:
        raise ApiError(404, "unknown_format", f"Unknown export format '{fmt}'.")
    mapping = library.names(transcript_id) if names else {}
    options = {"layout": layout} if layout in ("table",) else {}
    segments = export_segments(library, transcript_id, redact, lang)
    context = export_context(library, transcript_id, redact, options, show_title=title)
    if lang:
        context = replace(context, language=lang.lower())
    content = render(exporter, segments, mapping, context)
    privacy.audit(library, transcript_id, "exported", format=fmt, redacted=redact or None, names=names,
                  sha256=privacy.sha256_bytes(content), **({"lang": lang} if lang else {}))
    name = exporter.download_name
    if lang:
        stem, _, ext = name.rpartition(".")
        name = f"{stem}.{lang.lower()}.{ext}"
    if redact:
        stem, _, ext = name.rpartition(".")
        name = f"{stem}.redacted.{ext}"
    return Response(content, media_type=exporter.media_type, headers=attachment(name))


@router.get("/{transcript_id}/files/{file}")
def get_file(transcript_id: str, file: str, library: LibraryDep, names: bool = False):
    """Audio for the player (supports Range) and the original file names (prepis.txt, ...) for old links."""
    audio = library.audio_file(transcript_id)
    if audio and file == audio.name:
        return FileResponse(audio, media_type=AUDIO_TYPES.get(audio.suffix.lower()))
    if file == SEGMENTS_FILE:
        return FileResponse(library.path(transcript_id) / SEGMENTS_FILE, media_type="application/json",
                            headers=attachment(SEGMENTS_FILE))
    by_file = {e.download_name: e for e in list_exporters()}
    if file not in by_file:
        raise ApiError(404, "file_not_found", "File not found.")
    return export(transcript_id, by_file[file].name, library, names=names)
