"""Virtual folders ("projects") for transcripts.

A transcript's folder is a slash-separated path stored in its ``metadata.json`` (``Metadata.folder``,
e.g. ``"Podcasts/Cílek"``; ``""`` = unfiled). Nothing moves on disk, so transcript ids, links, search,
voices and MCP keep working. One folder per transcript.

``data/folders.json`` only holds the folders the user created, so that *empty* folders can exist.
The folder tree shown to the user is that list plus every folder (and ancestor) some transcript is in,
so a transcript is never in a folder the tree does not show, even when the list was lost or the
transcript came from the CLI.

Changes of a transcript's folder go through ``Library.set_folder`` (listeners and the search index
run), under the editor's per-transcript lock so a concurrent text edit cannot overwrite them.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .fsutil import file_lock, replace_file

if TYPE_CHECKING:
    from .library import Library

log = logging.getLogger(__name__)

MAX_DEPTH = 5  # levels: "A/B/C/D/E"
MAX_NAME = 60  # characters in one folder name
MAX_PATH = 255  # characters in a whole path
MAX_FOLDERS = 2000  # folders the user may create
MAX_ASSIGN = 5000  # transcripts in one bulk move
MOVE_TARGETS = ("parent", "unfiled")  # where the transcripts of a deleted folder go


class FolderError(ValueError):
    """Invalid folder input or state; ``code`` is stable so the UI can translate it."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def normalize_path(value: str | None) -> str:
    """Canonical spelling of a folder path: ``""`` (unfiled) or ``"A/B"``.

    Backslashes count as slashes, names are trimmed, empty names (``//``) collapse; control
    characters, ``.``/``..`` names, too long or too deep paths are refused."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise FolderError("folder_invalid", "A folder path must be text.")
    text = unicodedata.normalize("NFC", value.replace("\\", "/"))
    if any(unicodedata.category(c) in ("Cc", "Cf", "Cs", "Co", "Cn") for c in text):
        raise FolderError("folder_invalid", "A folder name cannot contain control characters.")
    parts = [p.strip() for p in text.split("/")]
    parts = [p for p in parts if p]
    if any(p in (".", "..") for p in parts):
        raise FolderError("folder_invalid", "'.' and '..' are not valid folder names.")
    if len(parts) > MAX_DEPTH:
        raise FolderError("folder_too_deep", f"Folders can be nested at most {MAX_DEPTH} levels deep.")
    if any(len(p) > MAX_NAME for p in parts):
        raise FolderError("folder_name_too_long", f"A folder name can have at most {MAX_NAME} characters.")
    path = "/".join(parts)
    if len(path) > MAX_PATH:
        raise FolderError("folder_name_too_long", f"A folder path can have at most {MAX_PATH} characters.")
    return path


def safe_path(value) -> str:
    """``normalize_path`` for stored data: an unusable value is unfiled instead of an error."""
    try:
        return normalize_path(value)
    except FolderError:
        return ""


def parent_of(path: str) -> str:
    return path.rpartition("/")[0]


def ancestors(path: str) -> list[str]:
    """``"A/B/C"`` -> ``["A", "A/B"]`` (not the path itself)."""
    parts = path.split("/") if path else []
    return ["/".join(parts[:i]) for i in range(1, len(parts))]


def in_folder(item_folder: str, folder: str, recursive: bool = False) -> bool:
    """Is ``item_folder`` ``folder``, or (recursive) below it? ``folder == ""`` is unfiled and only that:
    there is no "everything" folder (callers that want all simply do not filter)."""
    if item_folder == folder:
        return True
    return recursive and folder != "" and item_folder.startswith(folder + "/")


def sort_key(path: str) -> tuple:
    return tuple(p.casefold() for p in path.split("/"))


class FolderStore:
    """The user's folder list (``data/folders.json``) and the operations on folders and their transcripts."""

    def __init__(self, library: "Library", path: Path):
        self.library = library
        self.path = Path(path)
        self._lock = threading.RLock()  # the in-memory list; only ever held for short, non-blocking work
        self._mutex = self.path.with_name(self.path.name + ".lock")  # across processes: see _mutate
        self._explicit: set[str] = set()
        self._stamp: int | None = None  # mtime_ns of the file we last read or wrote
        self._load()

    # --- persistence ------------------------------------------------------------------------

    def _file_stamp(self) -> int | None:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return None

    def _load(self) -> None:
        """Read the file when it changed (another process, e.g. the MCP stdio server, may write it)."""
        stamp = self._file_stamp()
        if stamp is not None and stamp == self._stamp:
            return
        folders: set[str] = set()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for raw in data.get("folders", []) if isinstance(data, dict) else []:
                path = safe_path(raw)
                if path:
                    folders.update(ancestors(path))
                    folders.add(path)
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            log.exception("folders.json is unreadable; starting with the folders of the transcripts only")
            try:
                os.replace(self.path, self.path.with_suffix(".broken"))
            except OSError:
                pass
        self._explicit = folders
        self._stamp = self._file_stamp()

    def _mutate(self):
        """Read-modify-write of folders.json: serialised across threads *and* processes (the MCP stdio server
        and the web server share the file). Hold it only for the list itself, never while transcripts are
        written: that takes a lock, an fsync and a reindex per transcript."""
        return file_lock(self._mutex)

    def _save(self) -> None:
        payload = {"version": 1, "folders": sorted(self._explicit, key=sort_key)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            replace_file(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        self._stamp = self._file_stamp()

    # --- reading ----------------------------------------------------------------------------

    def _known(self, mapping: dict[str, str]) -> set[str]:
        known = set(self._explicit)
        for folder in set(mapping.values()):
            if folder:
                known.update(ancestors(folder))
                known.add(folder)
        return known

    def exists(self, path: str) -> bool:
        with self._lock:
            self._load()
            return normalize_path(path) in self._known(self.library.folders_map())

    def canonical(self, path: str, known: set[str] | None = None) -> str:
        """``path`` normalised, each name spelled like an existing sibling that differs only by case."""
        path = normalize_path(path)
        if known is None:
            with self._lock:
                self._load()
                known = self._known(self.library.folders_map())
        lookup: dict[tuple[str, str], str] = {}
        for k in sorted(known, key=sort_key):
            parent, _, name = k.rpartition("/")
            lookup.setdefault((parent, name.casefold()), name)
        out: list[str] = []
        for part in path.split("/") if path else []:
            out.append(lookup.get(("/".join(out), part.casefold()), part))
        return "/".join(out)

    def tree(self) -> dict:
        """Every folder with its own transcript ``count`` and the ``total`` of its subtree."""
        with self._lock:
            self._load()
            mapping = self.library.folders_map()
            known = self._known(mapping)
        own = Counter(mapping.values())
        total: Counter = Counter()
        for folder, n in own.items():
            if folder:
                for p in [*ancestors(folder), folder]:
                    total[p] += n
        folders = [{"path": p, "name": p.rpartition("/")[2], "parent": parent_of(p), "depth": p.count("/") + 1,
                    "count": own.get(p, 0), "total": total.get(p, 0)}
                   for p in sorted(known, key=sort_key)]
        return {"folders": folders, "unfiled": own.get("", 0), "total": len(mapping),
                "limits": {"max_depth": MAX_DEPTH, "max_name": MAX_NAME, "max_folders": MAX_FOLDERS}}

    def ids_in(self, folder: str, recursive: bool = False) -> list[str]:
        folder = normalize_path(folder)
        return [i for i, f in self.library.folders_map().items() if in_folder(f, folder, recursive)]

    # --- folders ----------------------------------------------------------------------------

    def create(self, path: str) -> tuple[str, bool]:
        """Make a folder (and its parents). Returns (canonical path, created); an existing one is fine."""
        path = normalize_path(path)
        if not path:
            raise FolderError("folder_invalid", "A folder name is required.")
        with self._mutate(), self._lock:
            self._load()
            known = self._known(self.library.folders_map())
            path = self.canonical(path, known)
            if path in known:
                if path not in self._explicit:  # only implied by transcripts: remember it (and parents)
                    self._explicit.update([*ancestors(path), path])
                    self._save()
                return path, False
            new = [p for p in [*ancestors(path), path] if p not in known]
            if len(self._explicit) + len(new) > MAX_FOLDERS:
                raise FolderError("folder_limit", f"At most {MAX_FOLDERS} folders can be created.")
            self._explicit.update([*ancestors(path), path])
            self._save()
            return path, True

    def ensure(self, path: str | None) -> str | None:
        """For job requests: None stays None (keep the transcript's folder), ``""`` is unfiled, else created."""
        if path is None:
            return None
        path = normalize_path(path)
        return self.create(path)[0] if path else ""

    def move(self, path: str, to: str) -> dict:
        """Rename or move a folder with everything below it (transcripts included).

        The list changes first (under the lock), the transcripts follow without it: a transcript that
        could not be written stays visible under its old path and is reported in ``failed``."""
        path, to = normalize_path(path), normalize_path(to)
        if not path or not to:
            raise FolderError("folder_invalid", "A folder name is required.")
        with self._mutate(), self._lock:
            self._load()
            mapping = self.library.folders_map()
            known = self._known(mapping)
            if path not in known:
                raise FolderError("folder_not_found", "Folder not found.", 404)
            if to == path:
                return {"path": path, "moved": 0, "failed": []}
            if to.casefold().startswith(path.casefold() + "/"):
                raise FolderError("folder_cycle", "A folder cannot be moved into itself.")
            # the folder being moved does not count as a sibling: "podcasts" -> "Podcasts" is a rename
            outside = {k for k in known if not in_folder(k, path, recursive=True)}
            to = self.canonical(to, outside)
            if to == path:
                return {"path": path, "moved": 0, "failed": []}
            if to in outside:
                raise FolderError("folder_exists", "A folder with that name already exists there.", 409)
            subtree = [k for k in known if in_folder(k, path, recursive=True)]
            renamed = {k: to + k[len(path):] for k in subtree}
            for new in renamed.values():  # depth and length of the deepest descendants
                normalize_path(new)
            if len(self._explicit - set(subtree)) + len(renamed) > MAX_FOLDERS:
                raise FolderError("folder_limit", f"At most {MAX_FOLDERS} folders can be created.")
            affected = {i: renamed[f] for i, f in mapping.items() if in_folder(f, path, recursive=True)}
            self._explicit = {renamed.get(k, k) for k in self._explicit}
            self._explicit.update(ancestors(to))
            self._explicit.add(to)
            self._save()
        moved, failed = self._set_many(affected, mapping)
        return {"path": to, "moved": moved, "failed": failed}

    def delete(self, path: str, move_to: str = "parent") -> dict:
        """Remove a folder and its subfolders; the transcripts in them go to the parent folder or to
        unfiled. Transcripts are never deleted."""
        if move_to not in MOVE_TARGETS:
            raise FolderError("folder_invalid", "move_to must be 'parent' or 'unfiled'.")
        path = normalize_path(path)
        if not path:
            raise FolderError("folder_invalid", "A folder name is required.")
        with self._mutate(), self._lock:
            self._load()
            mapping = self.library.folders_map()
            known = self._known(mapping)
            if path not in known:
                raise FolderError("folder_not_found", "Folder not found.", 404)
            target = parent_of(path) if move_to == "parent" else ""
            affected = {i: target for i, f in mapping.items() if in_folder(f, path, recursive=True)}
            removed = {k for k in known if in_folder(k, path, recursive=True)}
            self._explicit -= removed
            if target:  # the folder the transcripts go to stays, even when only the deleted one implied it
                self._explicit.update([*ancestors(target), target])
            self._save()
        moved, failed = self._set_many(affected, mapping)
        return {"deleted": sorted(removed, key=sort_key), "moved": moved, "failed": failed, "target": target}

    # --- transcripts ------------------------------------------------------------------------

    def assign(self, ids: Iterable[str], folder: str) -> dict:
        """Put transcripts into ``folder`` (created when new; ``""`` = unfiled). Unknown ids are reported."""
        folder = normalize_path(folder)
        ids = list(dict.fromkeys(ids))
        if len(ids) > MAX_ASSIGN:
            raise FolderError("too_many_transcripts", f"At most {MAX_ASSIGN} transcripts at once.")
        if folder:
            folder = self.create(folder)[0]
        mapping = self.library.folders_map()
        present = {i: folder for i in ids if i in mapping}
        missing = [i for i in ids if i not in mapping]
        moved, failed = self._set_many(present, mapping)
        return {"folder": folder, "moved": moved, "missing": missing, "failed": failed}

    def _set_many(self, targets: dict[str, str], mapping: dict[str, str]) -> tuple[int, list[str]]:
        """Write the new folder of each transcript (id -> folder). Takes no store lock: each transcript is
        locked on its own, so listing and creating folders stay responsive during a big move."""
        from .editor import edit_lock  # lazily: editor imports the library, which imports this module
        from .library import InvalidTranscriptId, TranscriptNotFound

        moved, failed = 0, []
        for transcript_id, target in targets.items():
            if mapping.get(transcript_id) == target:
                continue
            try:
                self.library.path(transcript_id)  # gone meanwhile (secure delete): never re-create its folder
            except (TranscriptNotFound, InvalidTranscriptId):
                continue
            try:
                with edit_lock(self.library, transcript_id):
                    if self.library.folder_of(transcript_id) != target:
                        self.library.set_folder(transcript_id, target)
                moved += 1
            except Exception:  # noqa: BLE001 - one damaged transcript must not stop the others
                log.exception("could not move %s to folder %r", transcript_id, target)
                failed.append(transcript_id)
        return moved, failed
