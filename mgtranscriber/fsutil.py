"""File helpers shared by the library and the job store."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

# Windows: antivirus, the search indexer or a concurrent reader can hold the target for a moment
REPLACE_RETRIES = (0.05, 0.1, 0.2, 0.5, 1.0)


def replace_file(tmp: Path, target: Path) -> None:
    """``os.replace`` that retries briefly while the target is locked."""
    for delay in (*REPLACE_RETRIES, None):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


# --- file locks -----------------------------------------------------------------------------
# One lock per file for read-modify-write that happens in the server (several threads) and in job
# worker processes: a reentrant thread lock shared by the whole process plus an OS lock on
# "<file>.lock" for other processes.

LOCK_TIMEOUT_S = 60.0


class LockTimeout(TimeoutError):
    pass


class PathLock:
    def __init__(self, path: Path):
        self.lock_path = path.with_name(path.name + ".lock")
        self._rlock = threading.RLock()
        self._depth = 0
        self._fh = None

    def __enter__(self):
        self._rlock.acquire()
        if self._depth == 0:
            try:
                self._fh = _os_lock(self.lock_path)
            except BaseException:
                self._rlock.release()
                raise
        self._depth += 1
        return self

    def __exit__(self, *exc):
        self._depth -= 1
        if self._depth == 0:
            fh, self._fh = self._fh, None
            _os_unlock(fh)
        self._rlock.release()


def _os_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+b")
    deadline = time.monotonic() + LOCK_TIMEOUT_S
    while True:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except OSError:
            if time.monotonic() > deadline:
                fh.close()
                raise LockTimeout(f"{path} is locked by another process.") from None
            time.sleep(0.02)


def _os_unlock(fh) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


_locks: dict[str, PathLock] = {}
_locks_guard = threading.Lock()


def file_lock(path: Path) -> PathLock:
    """The process-wide lock for ``path`` (reentrant; use as ``with file_lock(p): ...``)."""
    key = os.path.normcase(str(Path(path).resolve()))
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = PathLock(Path(path))
        return lock
