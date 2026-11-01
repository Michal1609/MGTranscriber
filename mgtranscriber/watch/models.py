"""Watched sources: configuration + state, persisted as one JSON file in ``data/``."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields

KINDS = ("folder", "rss", "youtube")
MODES = ("new", "last_n")
MAX_SEEN = 20000  # processed-item keys remembered per source (least recently seen dropped, see Source.prune)
MAX_HISTORY = 20
MAX_ITEMS = 30


class WatchError(Exception):
    """Raised for invalid input or a failed check; ``code`` is stable so the UI can translate it."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


@dataclass
class Source:
    # --- configuration (what the user edits) ---
    id: str
    kind: str  # folder | rss | youtube
    target: str  # folder path, feed URL, channel/playlist URL
    name: str = ""
    enabled: bool = True
    language: str | None = None  # None = default from settings, "auto" = detect
    speakers: int | None = None
    engine: str | None = None
    model: str | None = None
    interval_s: int = 3600
    mode: str = "new"  # new = only what appears from now on, last_n = also the newest ``last_n`` existing items
    last_n: int = 5
    recursive: bool = False  # folders only
    analyze: bool = False  # LLM analysis after the transcription
    translate_to: str | None = None  # translate every finished transcript into this language (None = no)
    webhook_url: str | None = None
    max_per_check: int = 5
    folder: str = ""  # library folder for what this source transcribes ("" = unfiled)
    # --- state ---
    created: float = field(default_factory=time.time)
    baselined: bool = False
    next_check: float = 0.0
    last_check: float | None = None
    last_ok: float | None = None
    last_error: str | None = None
    last_error_code: str | None = None
    fails: int = 0
    seen: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)  # newest first: {time, ok, found, queued, error, manual, dry_run}
    items: list[dict] = field(default_factory=list)  # newest first: queued items and their job outcome
    pending: dict[str, str] = field(default_factory=dict)  # job id -> item key, until the job finished
    cache: dict = field(default_factory=dict)  # ETag / resolved feed URL

    def remember(self, *keys: str) -> None:
        """Mark keys as processed; the dict keeps recency order (a key seen again moves to the end)."""
        now = time.time()
        for key in keys:
            self.seen.pop(key, None)
            self.seen[key] = now
        self.prune()

    def touch(self, *keys: str) -> None:
        """Refresh keys that are still listed by the source, so they are the last to be forgotten."""
        for key in keys:
            if key in self.seen:
                self.seen[key] = self.seen.pop(key)

    def prune(self, protect=()) -> None:
        """Forget the least recently seen keys beyond ``MAX_SEEN``, but never one the source still lists
        (``protect``): a long feed would otherwise requeue its old episodes."""
        excess = len(self.seen) - MAX_SEEN
        if excess <= 0:
            return
        protect = set(protect)
        for key in list(self.seen):
            if excess <= 0:
                break
            if key not in protect:
                del self.seen[key]
                excess -= 1

    def is_seen(self, keys) -> bool:
        return any(k in self.seen for k in keys)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "Source":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def public(self) -> dict:
        d = self.to_dict()
        d.pop("seen"), d.pop("cache"), d.pop("pending")
        d["seen_count"] = len(self.seen)
        d["pending_jobs"] = len(self.pending)
        return d


# fields a client may set; everything else is state
CONFIG_FIELDS = ("kind", "target", "name", "enabled", "language", "speakers", "engine", "model", "interval_s",
                 "mode", "last_n", "recursive", "analyze", "translate_to", "webhook_url", "max_per_check", "folder")
