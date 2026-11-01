"""Automatic watching of folders, podcast feeds and YouTube channels/playlists (see docs/WATCH.md)."""

from .manager import WatchManager
from .models import Source, WatchError

__all__ = ["WatchManager", "Source", "WatchError"]
