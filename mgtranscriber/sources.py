"""Where media comes from: URLs (anything yt-dlp supports) and local files."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import jsruntime
from .engines.base import EngineError
from .progress import Phase, Progress, ProgressCallback, no_progress

URL_RE = re.compile(r"https?://", re.IGNORECASE)
# YouTube requires solving a JS challenge; yt-dlp uses whichever runtime is available.
YTDLP_BASE = {"quiet": True, "noprogress": True, "no_warnings": True, "js_runtimes": {"deno": {}, "node": {}}}

ATTEMPTS = 3  # extractions per download; the pauses between them are in RETRY_PAUSES (seconds)
RETRY_PAUSES = (2, 5)
# Player clients that need no GVS PO token (see GVS_PO_TOKEN_POLICY in yt_dlp/extractor/youtube/_base.py);
# the default list also has `web`, whose format URLs require one.
FALLBACK_CLIENTS = ("tv_downgraded", "web_embedded", "tv", "visionos")
BLOCKED_MESSAGE = (
    "YouTube blocked the download (HTTP 403). Try again later; installing Deno "
    "(`winget install DenoLand.Deno`) and restarting the app can help.\n"
    "YouTube zablokoval stahování (HTTP 403). Zkuste to později; pomoci může i instalace Deno "
    "(`winget install DenoLand.Deno`) a restart aplikace."
)


class DownloadBlocked(EngineError):
    """YouTube refused the audio download again and again; a readable message, not a bug."""


@dataclass
class Media:
    source: str  # URL or local path as given by the user
    kind: str  # "url" | "file"
    title: str
    audio_path: Path | None = None  # set once audio is available locally
    info: dict = field(default_factory=dict)  # extra metadata (uploader, upload_date, webpage_url, duration, ...)
    workdir: Path | None = None  # the transcript folder, for engines that fetch audio themselves (fetch_audio)


def is_url(source: str) -> bool:
    return URL_RE.match(source) is not None


def source_key(media: Media) -> str:
    """Identity of a source, independent of how it was written: the video (extractor + id) or the absolute file path."""
    if media.kind == "file" and media.audio_path:
        return "file:" + os.path.normcase(str(media.audio_path.resolve()))
    if media.info.get("extractor_key") and media.info.get("id"):
        return f"{media.info['extractor_key']}:{media.info['id']}"
    return media.info.get("webpage_url") or media.source


def probe(source: str) -> Media:
    """Resolve title and metadata without downloading."""
    if not is_url(source):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {source}")
        return Media(source, "file", path.stem, audio_path=path.resolve())

    import yt_dlp

    with yt_dlp.YoutubeDL(YTDLP_BASE) as ydl:
        info = ydl.extract_info(source, download=False)
    keep = ("id", "uploader", "channel", "upload_date", "webpage_url", "duration", "extractor_key", "language")
    return Media(source, "url", info.get("title") or "video", info={k: info.get(k) for k in keep if info.get(k) is not None})


def _is_transient(error: Exception) -> bool:
    """YouTube's intermittent refusal: a format URL that answers 403 (usually one that needs a PO token), or no
    format left after such a refusal. A new extraction hands out other URLs, so the download is worth repeating."""
    text = str(error)
    return "HTTP Error 403" in text or "Requested format is not available" in text


def _remove_partials(out_dir: Path) -> None:
    for pattern in ("audio.*.part", "audio.*.ytdl", "audio.*.part-Frag*"):
        for leftover in out_dir.glob(pattern):
            leftover.unlink(missing_ok=True)


def fetch_audio(media: Media, out_dir: Path, progress: ProgressCallback = no_progress) -> Path:
    """Make sure ``media.audio_path`` exists; downloads the audio track of URLs into ``out_dir/audio.*``.

    YouTube answers some format URLs with HTTP 403 now and then (the same video downloads on the next try).
    yt-dlp's own retries reuse the same URLs, so the whole extraction is repeated; as a last resort with player
    clients that need no PO token. Raises ``DownloadBlocked`` when YouTube keeps refusing.
    """
    if media.audio_path is not None:
        return media.audio_path
    jsruntime.require(media.source)

    import yt_dlp

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progress(Progress(Phase.DOWNLOAD, 100 * d.get("downloaded_bytes", 0) / total))

    opts = {**YTDLP_BASE, "format": "bestaudio/best", "outtmpl": str(out_dir / "audio.%(ext)s"), "progress_hooks": [hook]}
    progress(Progress(Phase.DOWNLOAD, 0, f"Downloading audio: {media.title}"))

    def attempt(options: dict) -> Path:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(media.source, download=True)
            return Path(ydl.prepare_filename(info))

    error: Exception | None = None
    for number, pause in enumerate((0, *RETRY_PAUSES), start=1):
        if pause:
            progress(Progress(Phase.DOWNLOAD, None, f"YouTube refused the download (403), retrying {number}/{ATTEMPTS}..."))
            time.sleep(pause)
        try:
            media.audio_path = attempt(opts)
            return media.audio_path
        except yt_dlp.utils.DownloadError as e:
            if not _is_transient(e):
                raise
            error = e
            _remove_partials(out_dir)
    if jsruntime.is_youtube(media.source):
        progress(Progress(Phase.DOWNLOAD, None,
                          f"YouTube keeps refusing, last attempt with other player clients ({', '.join(FALLBACK_CLIENTS)})..."))
        time.sleep(RETRY_PAUSES[-1])
        try:
            media.audio_path = attempt({**opts, "extractor_args": {"youtube": {"player_client": list(FALLBACK_CLIENTS)}}})
            return media.audio_path
        except yt_dlp.utils.DownloadError as e:
            if not _is_transient(e):
                raise
            error = e
            _remove_partials(out_dir)
    raise DownloadBlocked(BLOCKED_MESSAGE) from error
