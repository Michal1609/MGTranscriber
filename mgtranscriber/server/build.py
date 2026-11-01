"""Which server code is running: a fingerprint of the package's source files taken when the process started.

The web UI is served from disk, so after an update (git pull, unpacking a new version) a server that was not
restarted still runs the old Python code behind a new page. ``GET /api/info`` reports ``build`` (the fingerprint of
the running code) and ``restart_needed`` (the files on disk no longer match it); an old server that predates this
reports neither, which the UI also treats as "restart". See ``static/js/components/stale.js``.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
_RECHECK_SECONDS = 5.0


def source_stamp() -> str:
    """Short hash of the relative path, size and modification time of every ``.py`` file of the package."""
    digest = hashlib.sha1()
    for path in sorted(PACKAGE.rglob("*.py")):
        try:
            st = path.stat()
        except OSError:
            continue
        digest.update(f"{path.relative_to(PACKAGE).as_posix()}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:12]


STARTED = source_stamp()
_checked: tuple[float, bool] = (0.0, False)


def restart_needed() -> bool:
    """True when the package's files changed since this process started (re-read at most every 5 s)."""
    global _checked
    now = time.monotonic()
    if now - _checked[0] >= _RECHECK_SECONDS:
        _checked = (now, source_stamp() != STARTED)
    return _checked[1]
