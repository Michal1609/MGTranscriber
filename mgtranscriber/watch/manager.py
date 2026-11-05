"""The watch scheduler: persisted sources, periodic checks, and queueing of new items.

It only ever *adds jobs to the job queue*; the queue decides when a transcription runs. Checks run one at a
time on a single background thread (kind to YouTube), failing sources back off exponentially.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ..config import Settings
from ..engines import list_engines
from ..folders import FolderError, normalize_path
from ..fsutil import replace_file
from ..pathguard import UNC_MESSAGE, unc_blocked
from ..jobs.models import ACTIVE, FINISHED
from ..llm.base import LLMError
from ..llm.translate import LANGUAGES
from ..llm.translations import enqueue_translation, normalize_lang
from ..pipeline import TranscriptionRequest
from .. import privacy
from ..sources import is_url
from . import fetchers
from .fetchers import Item
from .models import CONFIG_FIELDS, KINDS, MAX_HISTORY, MAX_ITEMS, MODES, Source, WatchError

log = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = {"folder": 60, "rss": 3600, "youtube": 3600}
MIN_INTERVAL_S = {"folder": 10, "rss": 300, "youtube": 1800}  # kind to the servers; settings can override
MAX_INTERVAL_S = 7 * 24 * 3600
MAX_BACKOFF_S = 6 * 3600
MANUAL_COOLDOWN_S = 30  # between two manual checks of one source
TICK_S = 1.0
QUIET_SAVE_S = 600.0  # longest a quiet check's bookkeeping (last_check, history) stays unwritten


class WatchManager:
    def __init__(self, settings: Settings, queue, llm=None, clock=time.time, client: httpx.Client | None = None,
                 library=None):
        self.settings = settings
        self.queue = queue
        self.llm = llm
        self.library = library or getattr(llm, "library", None)  # where the origin of a finished job is stored
        self.clock = clock
        self.client = client
        self.path = settings.data_dir / "watch.json"
        self.sources: dict[str, Source] = {}
        self._lock = threading.RLock()  # state
        self._check_lock = threading.Lock()  # one check at a time
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._observed: dict[str, dict] = {}  # folder scans: source id -> {path: (size, mtime)}
        self._hash_cache: dict = {}
        self._dirty_since: float | None = None  # in-memory changes not yet written (see _save_lazily)
        self._load()

    # --- settings -----------------------------------------------------------------------------

    def _extra(self, key: str, default):
        return self.settings.extra.get(key, default)

    @property
    def stable_s(self) -> float:
        return float(self._extra("watch_stable_s", 10))

    @property
    def max_queued(self) -> int:
        return int(self._extra("watch_max_queued", 20))

    def min_interval(self, kind: str) -> int:
        return int(self._extra(f"watch_min_{kind}_s", MIN_INTERVAL_S[kind]))

    @property
    def guard(self):
        """URL check for outgoing requests: none on loopback (a local Ollama or NAS is legitimate), otherwise
        only public addresses (SSRF protection)."""
        return None if self.settings.is_loopback else fetchers.require_public

    def limits(self) -> dict:
        return {"min_interval_s": {k: self.min_interval(k) for k in KINDS}, "default_interval_s": DEFAULT_INTERVAL_S,
                "max_per_check": 20, "max_last_n": 50, "local_paths_allowed": self.settings.local_paths_allowed,
                "translate_languages": LANGUAGES}

    # --- persistence --------------------------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            log.exception("watch.json is unreadable; starting without sources")
            try:
                os.replace(self.path, self.path.with_suffix(".broken"))
            except OSError:
                pass
            return
        for raw in data.get("sources", []):
            try:
                src = Source.from_dict(raw)
                self.sources[src.id] = src
            except TypeError:
                log.warning("skipping a malformed watch source")

    def _save_lazily(self) -> None:
        """Mark the state as changed in memory only; ``tick`` and ``stop`` write it after ``QUIET_SAVE_S``."""
        with self._lock:
            if self._dirty_since is None:
                self._dirty_since = time.monotonic()

    def _flush_if_due(self, force: bool = False) -> None:
        with self._lock:
            since = self._dirty_since
            if since is not None and (force or time.monotonic() - since >= QUIET_SAVE_S):
                self._save()

    def _save(self) -> None:
        with self._lock:
            self._dirty_since = None
            payload = {"version": 1, "sources": [s.to_dict() for s in self.sources.values()]}
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
                replace_file(tmp, self.path)
            except OSError:
                log.exception("could not save watch.json")

    # --- lifecycle ----------------------------------------------------------------------------

    def start(self) -> None:
        self.reconcile_jobs()
        self._stop.clear()  # also when the thread is still finishing a check after stop(): it must carry on
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._flush_if_due(force=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the scheduler must survive anything
                log.exception("watch tick failed")
            self._wake.wait(TICK_S)
            self._wake.clear()

    def reconcile_jobs(self) -> None:
        """After a restart: settle items whose job finished (or vanished) while the server was down."""
        with self._lock:
            for src in self.sources.values():
                for job_id in list(src.pending):
                    job = self.queue.get(int(job_id))
                    if job is None:
                        self._settle(src, job_id, "unknown", None, None)
                    elif job.status in FINISHED:
                        self.on_job_finished(job)

    # --- CRUD ---------------------------------------------------------------------------------

    def _validate(self, data: dict, current: Source | None = None) -> dict:
        values = {k: data[k] for k in CONFIG_FIELDS if k in data}
        kind = values.get("kind", current.kind if current else None)
        if kind not in KINDS:
            raise WatchError("watch_invalid", f"kind must be one of {', '.join(KINDS)}.")
        if current and kind != current.kind:
            raise WatchError("watch_invalid", "The kind of a source cannot be changed.")
        target = str(values.get("target", current.target if current else "") or "").strip()
        if not target or target.startswith("-"):
            raise WatchError("watch_invalid_target", "A folder or URL is required.")
        values["target"] = target
        if kind == "folder":
            if not self.settings.local_paths_allowed:
                raise WatchError("local_paths_disabled", "Transcribing files on the server's disk is disabled.", 403)
            if unc_blocked(self.settings, target):  # before any file system call
                raise WatchError("unc_path_disabled", UNC_MESSAGE, 403)
            folder = Path(target).expanduser()
            if not folder.is_absolute():
                folder = self.settings.root / folder
            folder = folder.resolve()
            if not folder.is_dir():
                raise WatchError("watch_folder_missing", f"Folder not found: {folder}")
            for own in (self.settings.output_dir, self.settings.data_dir):
                own = own.resolve()
                if folder == own or own in folder.parents or folder in own.parents:
                    raise WatchError("watch_folder_forbidden", "The library and data folders cannot be watched.")
            values["target"] = str(folder)
        else:
            parsed = urlparse(target)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise WatchError("watch_invalid_target", "An http(s) URL is required.")
            if self.guard:
                self.guard(target)
            if kind == "youtube" and not re.search(r"(^|\.)(youtube\.com|youtu\.be)$", parsed.hostname or ""):
                raise WatchError("watch_invalid_target", "A YouTube channel or playlist URL is required.")
        mode = values.get("mode", current.mode if current else "new")
        if mode not in MODES:
            raise WatchError("watch_invalid", "mode must be 'new' or 'last_n'.")
        interval = int(values.get("interval_s") or (current.interval_s if current else DEFAULT_INTERVAL_S[kind]))
        values["interval_s"] = max(self.min_interval(kind), min(interval, MAX_INTERVAL_S))
        if "last_n" in values:
            values["last_n"] = max(1, min(int(values["last_n"]), 50))
        if "max_per_check" in values:
            values["max_per_check"] = max(1, min(int(values["max_per_check"]), 20))
        if "speakers" in values and values["speakers"] is not None:
            values["speakers"] = max(1, min(int(values["speakers"]), 20))
        engine = values.get("engine")
        if engine and engine not in {e.name for e in list_engines() if e.available}:
            raise WatchError("unknown_engine", f"Engine '{engine}' is not available.")
        for key in ("language", "engine", "model", "webhook_url", "name", "translate_to"):
            if key in values:
                values[key] = (str(values[key]).strip() or None) if values[key] is not None else None
        if values.get("translate_to"):
            try:
                code = normalize_lang(values["translate_to"])
            except LLMError as exc:
                raise WatchError("watch_invalid", exc.message) from None
            code = code if code in LANGUAGES else code.split("-")[0]  # stored as exactly a LANGUAGES key
            if code not in LANGUAGES:
                raise WatchError("watch_invalid", f"Translation into '{values['translate_to']}' is not supported.")
            values["translate_to"] = code
        if values.get("webhook_url"):
            hook = urlparse(values["webhook_url"])
            if hook.scheme not in ("http", "https") or not hook.netloc:
                raise WatchError("watch_invalid", "The webhook must be an http(s) URL.")
            if self.guard:
                self.guard(values["webhook_url"])
        if values.get("name") is None and "name" in values:
            values["name"] = ""
        if "folder" in values:
            try:
                values["folder"] = normalize_path(values["folder"])
            except FolderError as exc:
                raise WatchError(exc.code, exc.message) from None
        return values

    def list(self) -> list[dict]:
        with self._lock:
            return [s.public() for s in sorted(self.sources.values(), key=lambda s: s.created)]

    def get(self, source_id: str) -> Source:
        with self._lock:
            src = self.sources.get(source_id)
        if not src:
            raise WatchError("watch_not_found", "Watch source not found.", 404)
        return src

    def create(self, data: dict) -> Source:
        values = self._validate(data)
        values.setdefault("name", "")
        src = Source(id=secrets.token_hex(4), created=self.clock(), **values)
        src.name = src.name or self._default_name(src)
        src.next_check = self.clock()
        with self._lock:
            self.sources[src.id] = src
            self._save()
        self._wake.set()
        return src

    def update(self, source_id: str, data: dict) -> Source:
        with self._lock:
            src = self.get(source_id)
            if data and set(data) == {"enabled"}:  # pausing/resuming needs no re-validation (e.g. local paths now off)
                src.enabled = bool(data["enabled"])
                self._save()
                self._wake.set()
                return src
            values = self._validate(data, src)
            target_changed = values.get("target") not in (None, src.target)
            for key, value in values.items():
                setattr(src, key, value)
            if target_changed:  # a different place: start over (the old items are not that source's)
                src.baselined, src.seen, src.cache, src.created = False, {}, {}, self.clock()
                self._observed.pop(src.id, None)
            if target_changed:
                src.next_check = self.clock()
            elif "interval_s" in values and src.last_check:
                src.next_check = min(src.next_check, src.last_check + src.interval_s)
            self._save()
        self._wake.set()
        return src

    def delete(self, source_id: str) -> None:
        with self._lock:
            self.get(source_id)
            del self.sources[source_id]
            self._observed.pop(source_id, None)
            self._save()

    @staticmethod
    def _default_name(src: Source) -> str:
        if src.kind == "folder":
            return Path(src.target).name or src.target
        parsed = urlparse(src.target)
        return (parsed.netloc + parsed.path).rstrip("/")[:80]

    # --- checking -----------------------------------------------------------------------------

    def tick(self, now: float | None = None) -> int:
        """Run every check that is due. Returns how many ran."""
        now = self.clock() if now is None else now
        with self._lock:
            due = sorted((s for s in self.sources.values() if s.enabled and s.next_check <= now),
                         key=lambda s: s.next_check)
        ran = 0
        for src in due:
            if self._stop.is_set():
                break
            self.check(src.id, now=now)
            ran += 1
        self._flush_if_due()
        return ran

    def preview(self, data: dict, limit: int = 20) -> dict:
        """List what a source would contain, without saving it, queueing anything or marking it seen."""
        values = self._validate(data)
        values.setdefault("name", "")
        src = Source(id="preview", created=self.clock(), **values)
        items, note = self._list_items(src, self.clock(), preview=True)
        return {"items": [i.public() for i in items[:limit]], "total": len(items), "note": note}

    def check(self, source_id: str, now: float | None = None, manual: bool = False, dry_run: bool = False) -> dict:
        """One check of one source. Never raises for fetch problems: they are recorded on the source."""
        now = self.clock() if now is None else now
        src = self.get(source_id)
        with self._check_lock:
            if manual and not dry_run and src.last_check and now - src.last_check < MANUAL_COOLDOWN_S:
                raise WatchError("watch_too_soon", f"Checked {int(now - src.last_check)} s ago; wait a moment.", 429)
            entry = {"time": now, "ok": True, "found": 0, "queued": 0, "error": None, "manual": manual, "dry_run": dry_run}
            queued: list[dict] = []
            settling = False
            was_baselined, had_error = src.baselined, src.last_error is not None
            previous = list(self._observed.get(src.id, {}))
            try:
                items, note = self._list_items(src, now)
                entry["found"] = len(items)
                if note:
                    entry["note"] = note
                queued = self._process(src, items, now, dry_run)
                entry["queued"] = len(queued)
                # a new file is never stable at its first sighting: look again as soon as it may be, not a whole
                # interval later (an interval of a minute made a dropped file wait one to two minutes). Only for files
                # first seen in THIS scan: one that stayed unstable (empty, still growing) waits the normal interval.
                if src.kind == "folder" and not dry_run:
                    fresh = [i for i in items if not i.stable and not src.is_seen(i.keys)]
                    if fresh:
                        before = {str(Path(p).resolve()) for p in previous}
                        settling = any(i.source not in before for i in fresh)
                if dry_run:
                    entry["items"] = [i["title"] for i in queued]
            except WatchError as exc:
                entry.update(ok=False, error=exc.message, code=exc.code)
            except Exception as exc:  # noqa: BLE001
                log.exception("watch check of %s failed", src.id)
                entry.update(ok=False, error=f"{type(exc).__name__}: {exc}", code="watch_failed")
            with self._lock:
                if not dry_run:
                    self._record(src, entry, now)
                    if settling and entry["ok"]:
                        src.next_check = min(src.next_check, now + max(self.stable_s, 1.0) + 1.0)
                # a quiet successful check changes only times and history: write the file now and then, not every
                # minute (with thousands of remembered keys it is a large file); anything that matters is saved at once
                quiet = (entry["ok"] and not queued and not dry_run and was_baselined and src.baselined
                         and not had_error and not settling)
                if quiet:
                    self._save_lazily()
                else:
                    self._save()
            result = dict(entry, queued_items=queued)
            return result

    def _record(self, src: Source, entry: dict, now: float) -> None:
        src.last_check = now
        src.history.insert(0, {k: v for k, v in entry.items() if k != "items"})
        del src.history[MAX_HISTORY:]
        if entry["ok"]:
            src.fails, src.last_ok, src.last_error, src.last_error_code = 0, now, None, None
            src.next_check = now + src.interval_s
        else:
            src.fails += 1
            src.last_error, src.last_error_code = entry["error"], entry.get("code")
            delay = min(src.interval_s * (2 ** min(src.fails, 10)), max(MAX_BACKOFF_S, src.interval_s))
            src.next_check = now + delay

    def _list_items(self, src: Source, now: float, preview: bool = False) -> tuple[list[Item], str | None]:
        note = None
        if src.kind == "folder":
            if not self.settings.local_paths_allowed:  # the setting may have changed since the source was added
                raise WatchError("local_paths_disabled", "Transcribing files on the server's disk is disabled.", 403)
            if unc_blocked(self.settings, src.target):
                raise WatchError("unc_path_disabled", UNC_MESSAGE, 403)
            observed = self._observed.get(src.id, {})
            items, seen_now, truncated = fetchers.scan_folder(src.target, src.recursive, observed, self.stable_s, now,
                                                              self._hash_cache)
            if not preview:
                self._observed[src.id] = seen_now
            if truncated:
                note = f"Only the first {fetchers.MAX_FOLDER_FILES} files were examined."
            return items, note
        if privacy.offline(self.settings):  # feeds and channels are fetched from the internet
            raise WatchError("offline_url", "Offline mode: feeds and YouTube channels are not checked.", 403)
        cache = {} if preview else src.cache
        wanted = max(src.last_n, src.max_per_check) if not src.baselined or preview else src.max_per_check
        if src.kind == "rss":
            body = fetchers.http_get(src.target, cache, client=self.client, guard=self.guard)
            items = [] if body is None else fetchers.parse_feed(body)
        else:
            items = fetchers.list_youtube(src.target, cache, wanted if src.baselined or preview else max(wanted, src.last_n),
                                          client=self.client, guard=self.guard)
        if all(i.published for i in items):
            items.sort(key=lambda i: i.published, reverse=True)
        return items, note

    def _process(self, src: Source, items: list[Item], now: float, dry_run: bool) -> list[dict]:
        """Decide which items are new, queue up to ``max_per_check`` of them, remember the rest as seen/unseen."""
        if not dry_run:
            with self._lock:
                src.touch(*(k for i in items for k in i.keys))
                src.prune(protect={k for i in items for k in i.keys})
        fresh = [i for i in items if not src.is_seen(i.keys)]
        to_mark: list[Item] = []
        baseline = not src.baselined
        if src.mode == "new" and (baseline or src.kind != "folder"):
            # what already existed when the source was added is not "new". Not only at the baseline: a later check may
            # list items the baseline never saw (a different listing source or depth), and those older, never-seen
            # items must not be queued. Undated items can only be told apart by the seen-set, i.e. they are old at the
            # baseline and new afterwards. Folders are exempt after the baseline: a file copied in later
            # keeps its old mtime and is still a new arrival.
            old = [i for i in fresh if (i.published is None and baseline)
                   or (i.published is not None and i.published <= src.created)]
            to_mark.extend(old)
            fresh = [i for i in fresh if i not in old]
        elif baseline:
            head = fresh[: src.last_n]
            to_mark.extend(i for i in fresh if i not in head)
            fresh = head
        ready = [i for i in fresh if i.stable]
        # oldest first, so a backlog is processed in publishing order and nothing is skipped
        ready = sorted(ready, key=lambda i: i.published) if all(i.published for i in ready) else ready[::-1]
        room = max(0, self.max_queued - self._active_jobs())
        take = ready[: min(src.max_per_check, room)]
        leftovers = len(fresh) > len(take)
        queued = []
        blocked: list[Item] = []
        if src.kind != "folder":
            # defence in depth: a feed must never make the server read a file (same rule as the jobs API) or fetch from
            # the local network (enclosure http://192.168.1.1/... or the cloud metadata address). Outside the lock: DNS.
            allowed = []
            for item in take:
                try:
                    if not is_url(item.source):
                        raise WatchError("watch_blocked_address", "not a web address")
                    if self.guard and urlparse(item.source).scheme in ("http", "https"):
                        self.guard(item.source)
                except WatchError:
                    blocked.append(item)
                else:
                    allowed.append(item)
            take = allowed
        if dry_run:
            return [{"title": i.title, "source": i.source, "key": i.key} for i in take]
        with self._lock:
            src.remember(*(k for i in blocked for k in i.keys))  # never retried: the item itself is the problem
            for item in take:
                job = self.queue.add(self._request(src, item))
                src.remember(*item.keys)
                src.pending[str(job.id)] = item.key
                src.items.insert(0, {"key": item.key, "title": item.title, "source": item.source, "queued": now,
                                     "job": job.id, "status": "queued", "result": None, "error": None})
                queued.append({"title": item.title, "source": item.source, "job": job.id})
            del src.items[MAX_ITEMS:]
            src.remember(*(k for i in to_mark for k in i.keys))
            if baseline:
                src.baselined = True
            if leftovers:  # more waiting than we may queue now: do not let a 304 hide them next time
                src.cache.pop("etag", None), src.cache.pop("modified", None)
        return queued

    def _active_jobs(self) -> int:
        try:
            return sum(1 for j in self.queue.listing() if j.get("status") in ACTIVE)
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _request(src: Source, item: Item) -> TranscriptionRequest:
        return TranscriptionRequest(source=item.source, language=src.language, speakers=src.speakers,
                                    engine=src.engine, model=src.model, folder=src.folder or None)

    # --- after the job ------------------------------------------------------------------------

    def _settle(self, src: Source, job_id: str, status: str, result: str | None, error: str | None) -> dict | None:
        key = src.pending.pop(job_id, None)
        entry = next((i for i in src.items if str(i.get("job")) == job_id), None)
        if entry:
            entry.update(status=status, result=result, error=error)
        if status == "error":
            src.last_error, src.last_error_code = f"{entry['title'] if entry else key}: {error}", "watch_job_failed"
        return entry

    def on_job_finished(self, job) -> None:
        """JobQueue hook: record the outcome, optionally start an LLM analysis and call the webhook."""
        job_id = str(job.id)
        with self._lock:
            src = next((s for s in self.sources.values() if job_id in s.pending), None)
            if not src:
                return
            status = "done" if job.status == "done" else "error"
            entry = self._settle(src, job_id, status, job.result, job.error or (job.status if status == "error" else None))
            self._save()
        if status == "done" and job.result:
            self._record_origin(src, job.result)
        if status == "done" and job.result and getattr(job, "no_speech", False):
            self._skip_empty(src, entry, job)
        elif status == "done" and src.analyze and job.result and self.llm:
            try:
                self.llm.start(job.result)
                job.add_log("[analysis started by watch source]")
            except Exception as exc:  # noqa: BLE001
                job.add_log(f"[analysis not started: {getattr(exc, 'message', exc)}]")
        if status == "done" and src.translate_to and job.result and self.llm and entry is not None                 and not getattr(job, "no_speech", False):
            self._translate(src, entry, job)
        if src.webhook_url:  # own thread: up to ~23 s of retries must not block the queue's other hooks
            threading.Thread(target=self._notify, args=(src, entry or {}, job, status), name="watch-webhook",
                             daemon=True).start()

    def _skip_empty(self, src: Source, entry: dict | None, job) -> None:
        """No speech in the recording: no analysis and no translation. Recorded on the item, not an error."""
        skipped = [name for name, wanted in (("analysis", src.analyze), ("translation", src.translate_to)) if wanted]
        job.add_log("[no speech was found in the recording" + (f": {', '.join(skipped)} skipped]" if skipped else "]"))
        if entry is not None:
            with self._lock:
                entry["no_speech"] = True
                if src.translate_to:
                    entry["translation"] = {"language": src.translate_to, "skipped": "no_speech"}
                self._save()

    def _record_origin(self, src: Source, transcript_id: str) -> None:
        """Store which watch source made the transcript in its metadata (``extra.watch``), so the detail page
        still knows it after the source's item list is trimmed or the source is removed."""
        if not self.library:
            return
        try:
            meta = self.library.metadata(transcript_id)
            meta.extra["watch"] = {"id": src.id, "name": src.name, "kind": src.kind}
            self.library.save_metadata(transcript_id, meta)
        except Exception as exc:  # noqa: BLE001 - bookkeeping only; the transcript itself is done
            log.warning("origin of %s not stored: %s", transcript_id, exc)

    def origin_of(self, transcript_id: str) -> dict | None:
        """The watch source a transcript came from, found in the items (for transcripts made before
        ``extra.watch`` was stored). ``{id, name, kind}`` or None."""
        with self._lock:
            for src in self.sources.values():
                if any(i.get("result") == transcript_id for i in src.items):
                    return {"id": src.id, "name": src.name, "kind": src.kind}
        return None

    def _translate(self, src: Source, entry: dict, job) -> None:
        """Queue the translation of a finished transcript the same way the detail page does; the outcome
        (job id, or why not) is recorded on the item. Same language as the transcript is not an error."""
        lang = src.translate_to
        note: dict | None
        try:
            queued = enqueue_translation(self.queue, self.llm, self.llm.library, job.result, lang)
            note = {"language": lang, "job": queued.get("id")}
            job.add_log(f"[translation to {lang} queued by watch source]")
        except LLMError as exc:
            if exc.code == "translation_same_language":
                note = None
            else:
                note = {"language": lang, "error": exc.message}
                job.add_log(f"[translation to {lang} not queued: {exc.message}]")
        except Exception as exc:  # noqa: BLE001 - a failed translation must not break the item's bookkeeping
            log.warning("translation for %s failed to queue: %s", src.id, exc)
            note = {"language": lang, "error": str(exc)}
            job.add_log(f"[translation to {lang} not queued: {exc}]")
        if note:
            with self._lock:
                entry["translation"] = note
                self._save()

    def _notify(self, src: Source, entry: dict, job, status: str) -> None:
        if privacy.offline(self.settings):  # a webhook would send titles and sources out
            log.info("offline mode: webhook of %s not sent", src.id)
            return
        payload = {
            "event": "transcription.done" if status == "done" else "transcription.failed",
            "source": {"id": src.id, "name": src.name, "kind": src.kind},
            "item": {"title": entry.get("title") or job.title, "source": entry.get("source") or job.request.source},
            "job": {"id": job.id, "status": job.status, "error": job.error},
            "transcript": {"id": job.result, "title": job.title} if job.result else None,
            "time": self.clock(),
        }
        for attempt in range(2):
            try:
                if self.guard:
                    self.guard(src.webhook_url)
                resp = (self.client or httpx).post(src.webhook_url, json=payload, timeout=10,
                                                   headers={"User-Agent": fetchers.USER_AGENT})
                if resp.status_code < 500:
                    return
            except (httpx.HTTPError, WatchError) as exc:
                log.warning("webhook of %s failed: %s", src.id, exc)
            time.sleep(1 + attempt)
        log.warning("webhook of %s gave up", src.id)
