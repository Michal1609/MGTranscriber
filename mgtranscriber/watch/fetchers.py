"""Listing what a source currently contains: folder scan, RSS/Atom parsing, YouTube channels/playlists.

Nothing here downloads media or touches the job queue; it only returns ``Item`` objects.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx

from .. import netguard
from .models import WatchError

MEDIA_EXTENSIONS = {".mp3", ".m4a", ".m4b", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aiff", ".aif",
                    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".3gp"}
TEMP_SUFFIXES = (".part", ".tmp", ".temp", ".crdownload", ".download", ".partial", ".!ut", ".filepart", ".swp")
MAX_FOLDER_FILES = 5000
MAX_FOLDER_DIRS = 2000
MAX_FEED_BYTES = 32 * 1024 * 1024
MAX_REDIRECTS = 5
USER_AGENT = "mgtranscriber-watch/1.0 (+local podcast/video transcriber)"
YOUTUBE_FEED = "https://www.youtube.com/feeds/videos.xml"
RSS_ITEM_LIMIT = 15  # what YouTube's feed holds; more needs yt-dlp


@dataclass
class Item:
    key: str  # deduplication key
    source: str  # what to hand to the job queue (path or URL)
    title: str = ""
    published: float | None = None
    alt_keys: tuple[str, ...] = ()
    stable: bool = True  # folders: false while the file may still be written
    extra: dict = field(default_factory=dict)

    @property
    def keys(self) -> tuple[str, ...]:
        return (self.key, *self.alt_keys)

    def public(self) -> dict:
        return {"key": self.key, "source": self.source, "title": self.title, "published": self.published,
                "stable": self.stable}


# --- folder -----------------------------------------------------------------------------------------

def file_key(path: Path, size: int, cache: dict | None = None, mtime: float = 0.0) -> str:
    """Content-based key: size plus a hash of the first and last 64 KB (cheap, survives renames/moves)."""
    ck = (str(path), size, mtime)
    if cache is not None and ck in cache:
        return cache[ck]
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(65536))
        if size > 131072:
            f.seek(size - 65536)
            h.update(f.read(65536))
    key = "file:" + h.hexdigest()
    if cache is not None:
        if len(cache) > 20000:
            cache.clear()
        cache[ck] = key
    return key


def is_media_file(name: str) -> bool:
    low = name.lower()
    return (Path(low).suffix in MEDIA_EXTENSIONS and not low.startswith((".", "~$"))
            and not low.endswith(TEMP_SUFFIXES))


def _is_reparse_point(entry: os.DirEntry) -> bool:
    try:
        attrs = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return True
    return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def scan_folder(folder: str, recursive: bool, observed: dict, stable_s: float, now: float,
                hash_cache: dict | None = None) -> tuple[list[Item], dict, bool]:
    """Media files in ``folder``. A file is ``stable`` when its size and mtime did not change since the previous
    scan (``observed``: path -> (size, mtime)) and it was last modified at least ``stable_s`` seconds ago.
    Returns (items, new observations, truncated)."""
    root = Path(folder)
    if not root.is_dir():
        raise WatchError("watch_folder_missing", f"Folder not found: {folder}")
    items, seen_now, count, truncated, dirs = [], {}, 0, False, 0
    stack = [root]
    while stack and not truncated:
        current = stack.pop()
        dirs += 1
        if dirs > MAX_FOLDER_DIRS:
            truncated = True
            break
        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            if current == root:
                raise WatchError("watch_folder_unreadable", f"Cannot read folder: {exc}") from exc
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if _is_reparse_point(entry):  # NTFS junctions and mount points can loop back
                        continue
                    if recursive and not entry.name.startswith("."):
                        stack.append(Path(entry.path))
                    continue
                if not entry.is_file() or not is_media_file(entry.name):
                    continue
                st = entry.stat()
            except OSError:
                continue
            count += 1
            if count > MAX_FOLDER_FILES:
                truncated = True
                break
            path = entry.path
            sig = (st.st_size, st.st_mtime)
            seen_now[path] = sig
            stable = st.st_size > 0 and observed.get(path) == sig and now - st.st_mtime >= stable_s
            try:
                key = file_key(Path(path), st.st_size, hash_cache, st.st_mtime)
            except OSError:
                continue  # locked or vanished: not ready
            items.append(Item(key=key, source=str(Path(path).resolve()), title=Path(path).stem,
                              published=st.st_mtime, stable=stable))
    items.sort(key=lambda i: i.published or 0, reverse=True)
    return items, seen_now, truncated


# --- HTTP / feeds -----------------------------------------------------------------------------------

def require_public(url: str) -> None:
    """``netguard.require_public_url`` with watch errors: used when the server is reachable from the network, for
    feeds, previews, webhooks and every redirect hop (see ``netguard`` for what is and is not covered)."""
    try:
        netguard.require_public_url(url)
    except netguard.BlockedAddress as exc:
        raise WatchError("watch_blocked_address", str(exc)) from exc
    except OSError as exc:
        raise WatchError("watch_network", f"Cannot resolve {urlparse(url).hostname or url}.") from exc


def http_get(url: str, cache: dict | None = None, timeout: float = 20.0, client: httpx.Client | None = None,
             guard=None) -> bytes | None:
    """GET with a size cap and ETag/Last-Modified revalidation. Returns None when the server says 304.
    Redirects are followed by hand so ``guard(url)`` (raises WatchError) can vet every hop."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"}
    if cache:
        if cache.get("etag"):
            headers["If-None-Match"] = cache["etag"]
        if cache.get("modified"):
            headers["If-Modified-Since"] = cache["modified"]
    own = client or httpx.Client(timeout=timeout)
    try:
        for _ in range(MAX_REDIRECTS + 1):
            if guard:
                guard(url)
            with own.stream("GET", url, headers=headers, follow_redirects=False) as resp:
                if resp.is_redirect and resp.headers.get("location"):
                    url = str(resp.url.join(resp.headers["location"]))
                    if urlparse(url).scheme not in ("http", "https"):
                        raise WatchError("watch_http_error", "The server redirected to a non-http address.")
                    continue
                if resp.status_code == 304:
                    return None
                if resp.status_code == 429:
                    raise WatchError("watch_rate_limited", "The server asks to slow down (HTTP 429).")
                if resp.status_code >= 400:
                    raise WatchError("watch_http_error", f"HTTP {resp.status_code} from {urlparse(url).netloc}")
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > MAX_FEED_BYTES:
                        raise WatchError("watch_feed_too_large", f"The feed is larger than {MAX_FEED_BYTES // 2**20} MB.")
                if cache is not None:
                    cache["etag"] = resp.headers.get("etag")
                    cache["modified"] = resp.headers.get("last-modified")
                return bytes(body)
        raise WatchError("watch_http_error", "Too many redirects.")
    except httpx.HTTPError as exc:
        raise WatchError("watch_network", f"Cannot reach {urlparse(url).netloc or url}: {exc}") from exc
    finally:
        if client is None:
            own.close()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(node, name: str) -> str:
    for c in node:
        if _local(c.tag) == name and c.text and c.text.strip():
            return c.text.strip()
    return ""


def _date(value: str) -> float | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
    except ValueError:
        return None


def _is_media(url: str, mime: str = "") -> bool:
    # web URLs only: an enclosure like C:/x.wav or a UNC path would make the server read a local or SMB file
    if urlparse(url).scheme.lower() not in ("http", "https"):
        return False
    mime = (mime or "").lower()
    if mime.startswith(("audio/", "video/")):
        return True
    return Path(urlparse(url).path).suffix.lower() in MEDIA_EXTENSIONS


def parse_feed(data: bytes) -> list[Item]:
    """Items of an RSS 2.0 or Atom feed that carry audio/video (enclosure) or are YouTube entries."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise WatchError("watch_bad_feed", f"Not a valid RSS/Atom feed: {exc}") from exc
    if _local(root.tag) not in ("rss", "feed", "RDF"):
        raise WatchError("watch_bad_feed", "Not an RSS/Atom feed.")
    items = []
    for e in (e for e in root.iter() if _local(e.tag) in ("item", "entry")):
        title = _text(e, "title")
        guid = _text(e, "guid") or _text(e, "id")
        published = _date(_text(e, "pubDate") or _text(e, "published") or _text(e, "updated") or _text(e, "date"))
        video_id = _text(e, "videoId")
        if video_id:  # YouTube channel/playlist feed
            items.append(Item(key=f"yt:{video_id}", source=f"https://www.youtube.com/watch?v={video_id}",
                              title=title, published=published))
            continue
        media = ""
        for c in e:
            name = _local(c.tag)
            if name == "enclosure" and _is_media(c.get("url", ""), c.get("type", "")):
                media = c.get("url", "")
            elif name == "link" and c.get("rel") == "enclosure" and _is_media(c.get("href", ""), c.get("type", "")):
                media = c.get("href", "")
            elif name == "content" and c.get("url") and (c.get("medium") in ("audio", "video") or _is_media(c["url"], c.get("type", ""))):
                media = c["url"]
            if media:
                break
        if not media:
            continue
        keys = [f"guid:{guid}"] if guid else []
        keys.append(f"url:{media}")
        items.append(Item(key=keys[0], alt_keys=tuple(keys[1:]), source=media,
                          title=title or Path(urlparse(media).path).stem, published=published))
    return items


# --- YouTube ----------------------------------------------------------------------------------------

def youtube_feed_url(target: str, cache: dict | None = None) -> str | None:
    """The channel/playlist RSS feed for a YouTube URL, when it can be derived without yt-dlp."""
    if cache and cache.get("feed"):
        return cache["feed"]
    parsed = urlparse(target)
    query = parse_qs(parsed.query)
    if query.get("list"):
        return f"{YOUTUBE_FEED}?playlist_id={query['list'][0]}"
    m = re.match(r"^/channel/(UC[\w-]{20,})", parsed.path)
    if m:
        return f"{YOUTUBE_FEED}?channel_id={m.group(1)}"
    return None


def _channel_tab(target: str) -> str:
    parsed = urlparse(target)
    if re.match(r"^/(@[^/]+|channel/[^/]+|c/[^/]+|user/[^/]+)/?$", parsed.path):
        return target.rstrip("/") + "/videos"
    return target


def youtube_flat(target: str, limit: int) -> tuple[list[Item], str | None]:
    """List a channel or playlist with yt-dlp flat extraction (no media download). Returns (items, channel feed URL)."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise WatchError("watch_ytdlp_missing", "yt-dlp is not installed.") from exc
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "extract_flat": "in_playlist", "skip_download": True,
            "playlistend": limit, "lazy_playlist": True, "socket_timeout": 30}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(_channel_tab(target), download=False)
    except yt_dlp.utils.DownloadError as exc:
        text = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
        code = "watch_rate_limited" if "429" in text or "Too Many Requests" in text else "watch_youtube_error"
        raise WatchError(code, text.splitlines()[-1][:300] if text else "yt-dlp failed") from exc
    items = []
    for entry in (info or {}).get("entries") or []:
        if not entry or not entry.get("id"):
            continue
        if entry.get("live_status") in ("is_live", "is_upcoming"):
            continue
        vid = entry["id"]
        items.append(Item(key=f"yt:{vid}", source=f"https://www.youtube.com/watch?v={vid}",
                          title=entry.get("title") or vid, published=entry.get("timestamp"),
                          extra={"duration": entry.get("duration")}))
    feed = None
    channel_id = (info or {}).get("channel_id") or (info or {}).get("playlist_channel_id")
    if channel_id and str(channel_id).startswith("UC") and "list=" not in target:
        feed = f"{YOUTUBE_FEED}?channel_id={channel_id}"
    return items[:limit], feed


def list_youtube(target: str, cache: dict, limit: int, client: httpx.Client | None = None, guard=None) -> list[Item]:
    """RSS feed first (light, no JS runtime); yt-dlp flat extraction to resolve @handles or list more than 15 items."""
    feed = youtube_feed_url(target, cache)
    if feed and limit <= RSS_ITEM_LIMIT:
        try:
            body = http_get(feed, cache, client=client, guard=guard)
            return [] if body is None else parse_feed(body)
        except WatchError:
            pass  # feeds 404 now and then: fall back to yt-dlp
    items, resolved = youtube_flat(target, max(limit, 1))
    if resolved:
        cache["feed"] = resolved
        cache.pop("etag", None), cache.pop("modified", None)
    return items
