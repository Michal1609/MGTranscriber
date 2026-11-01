"""JavaScript runtime that yt-dlp needs for YouTube (since 2025.11 it solves YouTube's JS challenges with it).

yt-dlp 2026.06+ accepts Deno >= 2.3 (preferred) or Node >= 22. Without one, YouTube serves few or
no formats, so downloads fail with confusing errors; we fail early with instructions instead.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache
from urllib.parse import urlparse

MINIMUM = {"deno": (2, 3), "node": (22, 0)}
HELP = ("YouTube needs a JavaScript runtime for yt-dlp: install Deno 2.3 or newer "
        "(Windows: `winget install DenoLand.Deno`, other systems: https://deno.com) and restart the app. "
        "Node.js 22+ also works.")
YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")


class JsRuntimeMissing(RuntimeError):
    pass


def is_youtube(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in YOUTUBE_HOSTS)


def _version(executable: str) -> tuple[int, ...] | None:
    try:
        out = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)\.(\d+)", out)
    return (int(match[1]), int(match[2])) if match else None


@lru_cache(maxsize=1)
def detect() -> tuple[str, str] | None:
    """The first usable runtime as ``(name, "major.minor")``, or None."""
    for name, minimum in MINIMUM.items():
        executable = shutil.which(name)
        if executable and (version := _version(executable)) and version >= minimum:
            return name, ".".join(map(str, version))
    return None


def require(url: str) -> None:
    """Raise ``JsRuntimeMissing`` with instructions when ``url`` is YouTube and no runtime is usable."""
    if is_youtube(url) and detect() is None:
        raise JsRuntimeMissing(HELP)
