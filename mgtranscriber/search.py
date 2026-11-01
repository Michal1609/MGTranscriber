"""Full-text search over every transcript in the library (SQLite FTS5).

The index lives in ``data/search.sqlite3`` and is derived data: it can be deleted at
any time and is rebuilt by ``sync()``. One row per segment; the tokenizer folds case
and diacritics, so "prepis" finds "přepis".

Keeping it current:
- ``sync()`` compares each transcript's file stamps (prepis.json, speakers.json,
  metadata.json) with the indexed ones and reindexes what changed; it runs at server
  start and (throttled) before searches, so transcripts made by the CLI show up;
- ``reindex(id)`` updates one transcript right away; the server calls it when a job
  finishes and the library calls it (via ``Library.listeners``) when segments, speaker
  names or metadata are saved, or a transcript is deleted (it then drops out);
- a search never returns hits of a transcript whose folder is gone, even before the
  next sync notices.

Locking: files are read without holding the database lock, which is only held for each
transcript's short write, so searches keep working while the first sync builds the index.

Query syntax: words are AND-ed and prefix-matched (``prep`` finds "přepis"),
``"quoted text"`` is an exact phrase. Other FTS5 syntax is not exposed.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Collection

from .llm.translations import TRANSLATIONS_DIR, translations_stamp
from .library import METADATA_FILE, SEGMENTS_FILE, SPEAKERS_FILE, InvalidTranscriptId, Library, TranscriptNotFound

SCHEMA_VERSION = 2  # 2: segments.lang (rows of translations)
SYNC_INTERVAL = 5.0  # seconds; searches within this window skip the file scan
EMPTY_RESCAN_INTERVAL = 0.2  # a search without hits scans again unless the last scan is this recent
MARK_START, MARK_END = "\x02", "\x03"
SNIPPET_TOKENS = 32
SNIPPET_CHARS = 240

SCHEMA = f"""
CREATE TABLE transcripts (id TEXT PRIMARY KEY, title TEXT NOT NULL, stamp TEXT NOT NULL);
CREATE TABLE segments (
    rowid INTEGER PRIMARY KEY, transcript_id TEXT NOT NULL, seg INTEGER NOT NULL,
    t_start REAL NOT NULL, t_end REAL NOT NULL, speaker TEXT, speaker_name TEXT, text TEXT NOT NULL,
    lang TEXT NOT NULL DEFAULT ''
);
CREATE INDEX segments_transcript ON segments (transcript_id);
CREATE VIRTUAL TABLE segments_fts USING fts5 (
    text, content='segments', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts (rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts (segments_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
PRAGMA user_version = {SCHEMA_VERSION};
"""

TERM = re.compile(r'"([^"]*)"?|([^\s"]+)')
WORD = re.compile(r"\w+")


@dataclass
class Hit:
    transcript_id: str
    title: str
    segment: int  # index in prepis.json
    start: float
    end: float
    speaker: str | None  # raw id, e.g. SPEAKER_00
    speaker_name: str | None  # display name from speakers.json, if set
    text: str  # whole segment text
    snippet: str  # excerpt around the match (plain text)
    highlights: list[list[int]]  # [start, end) ranges of matches in ``snippet``, in characters (code points)
    lang: str = ""  # "" = the original text; else the language of the translation that matched

    def to_dict(self) -> dict:
        return asdict(self)


def fts_query(text: str) -> str:
    """User input -> safe FTS5 query. Empty string when there is nothing to search for."""
    parts = []
    for phrase, word in TERM.findall(text):
        if phrase:
            words = WORD.findall(phrase)
            if words:
                parts.append('"' + " ".join(words) + '"')
        else:
            parts.extend(f'"{w}"*' for w in WORD.findall(word))
    return " ".join(parts)


def fold(text: str | None) -> str:
    """Case and diacritics folded, for comparing speaker names ("jana novakova" = "Jana Nováková")."""
    decomposed = unicodedata.normalize("NFKD", (text or "").strip().casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def split_marks(marked: str) -> tuple[str, list[list[int]]]:
    """Snippet with MARK_START/MARK_END -> plain text plus highlight ranges."""
    out, ranges, start = [], [], None
    length = 0
    for ch in marked:
        if ch == MARK_START:
            start = length
        elif ch == MARK_END:
            if start is not None:
                ranges.append([start, length])
            start = None
        else:
            out.append(ch)
            length += 1
    return "".join(out), ranges


class SearchIndex:
    def __init__(self, library: Library, path: Path):
        self.library = library
        self.path = Path(path)
        self._lock = threading.RLock()  # the connection; held only for short reads and writes
        self._sync_lock = threading.Lock()  # one sync at a time
        self._conn: sqlite3.Connection | None = None
        self._last_sync = 0.0

    # --- storage ----------------------------------------------------------------------

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.create_function("fold", 1, fold, deterministic=True)
            if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                for (kind, name) in conn.execute(
                        "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'trigger') "
                        "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'segments_fts_%'").fetchall():
                    conn.execute(f'DROP {kind.upper()} IF EXISTS "{name}"')
                conn.executescript(SCHEMA)
            self._conn = conn
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _stamp(self, transcript_id: str) -> str:
        """Change marker of one transcript. One directory scan (not an exists + stat per file); the translations
        folder is only looked into when there is one. Same text as before, so existing indexes stay valid."""
        folder = self.library.path(transcript_id)
        stamps: dict[str, int] = {}
        translations = False
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.name in (SEGMENTS_FILE, SPEAKERS_FILE, METADATA_FILE):
                    stamps[entry.name] = entry.stat().st_mtime_ns
                elif entry.name == TRANSLATIONS_DIR:
                    translations = True
        files = ",".join(str(stamps.get(f, 0)) for f in (SEGMENTS_FILE, SPEAKERS_FILE, METADATA_FILE))
        return f"{files},{translations_stamp(folder) if translations else ''}"  # translations/<lang>.json: indexed with a language tag

    # --- updating -----------------------------------------------------------------------

    def reindex(self, transcript_id: str) -> bool:
        """(Re)index one transcript; removes it from the index when it no longer exists."""
        try:  # the stamp first: if the files change while being read, the next sync sees a stale stamp
            stamp = self._stamp(transcript_id)
            title = self.library.metadata(transcript_id).title
            if self._only_other_metadata_changed(transcript_id, stamp, title):
                return True  # e.g. the folder was changed: nothing the index stores is different
            segments = self.library.segments(transcript_id)
            names = self.library.names(transcript_id)
            translated = self._translations(transcript_id)
        except (TranscriptNotFound, OSError, ValueError, KeyError, TypeError):
            self.remove(transcript_id)
            return False
        with self._lock:
            db = self._db()
            with db:
                db.execute("DELETE FROM segments WHERE transcript_id = ?", (transcript_id,))
                db.execute("INSERT OR REPLACE INTO transcripts (id, title, stamp) VALUES (?, ?, ?)",
                           (transcript_id, title, stamp))
                db.executemany(
                    "INSERT INTO segments (transcript_id, seg, t_start, t_end, speaker, speaker_name, text) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(transcript_id, i, s.start, s.end, s.speaker, names.get(s.speaker or "UNKNOWN"), s.text.strip())
                     for i, s in enumerate(segments) if s.text.strip()])
                db.executemany(
                    "INSERT INTO segments (transcript_id, seg, t_start, t_end, speaker, speaker_name, text, lang) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [(transcript_id, i, s["start"], s["end"], s.get("speaker"), names.get(s.get("speaker") or "UNKNOWN"),
                      s["text"].strip(), lang)
                     for lang, segs in translated for i, s in enumerate(segs) if s["text"].strip()])
            return True

    def _translations(self, transcript_id: str) -> list[tuple[str, list[dict]]]:
        """(language, segments) of every stored translation; unreadable files are skipped."""
        out = []
        for path in sorted((self.library.path(transcript_id) / TRANSLATIONS_DIR).glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                segs = [{"start": float(s["start"]), "end": float(s["end"]), "text": str(s.get("text", "")),
                         "speaker": s.get("speaker")} for s in data["segments"]]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            out.append((path.stem, segs))
        return out

    def _only_other_metadata_changed(self, transcript_id: str, stamp: str, title: str) -> bool:
        """Segments and speaker names are as indexed and so is the title; only metadata.json differs (its
        folder, custody, ...). The stamp is refreshed so that ``sync`` does not look again, and the segments
        are not rewritten: moving many transcripts between folders stays cheap."""
        with self._lock:
            db = self._db()
            row = db.execute("SELECT title, stamp FROM transcripts WHERE id = ?", (transcript_id,)).fetchone()
            key = (0, 1, 3)  # prepis.json, speakers.json, translations (not metadata.json)
            if not row or row[0] != title or [row[1].split(",")[i] for i in key] != [stamp.split(",")[i] for i in key]:
                return False
            with db:
                db.execute("UPDATE transcripts SET stamp = ? WHERE id = ?", (stamp, transcript_id))
            return True

    def remove(self, transcript_id: str) -> None:
        with self._lock, self._db() as db:
            db.execute("DELETE FROM segments WHERE transcript_id = ?", (transcript_id,))
            db.execute("DELETE FROM transcripts WHERE id = ?", (transcript_id,))

    def sync(self) -> dict:
        """Bring the index in line with the library. Returns counts of what changed."""
        with self._sync_lock:
            return self._sync()

    def _sync(self) -> dict:
        with self._lock:
            indexed = dict(self._db().execute("SELECT id, stamp FROM transcripts").fetchall())
        current = set(self.library.ids())
        changed = removed = 0
        for transcript_id in current:
            try:
                stamp = self._stamp(transcript_id)
            except (TranscriptNotFound, OSError):
                continue
            if indexed.get(transcript_id) != stamp:
                changed += self.reindex(transcript_id)
        for transcript_id in indexed.keys() - current:
            self.remove(transcript_id)
            removed += 1
        self._last_sync = time.monotonic()
        return {"indexed": len(current), "changed": changed, "removed": removed}

    def sync_if_stale(self) -> None:
        if time.monotonic() - self._last_sync <= SYNC_INTERVAL:
            return
        if not self._sync_lock.acquire(blocking=False):
            return  # a sync is running (e.g. the first build at start): search what is indexed so far
        try:
            self._sync()
        finally:
            self._sync_lock.release()

    # --- querying -----------------------------------------------------------------------

    def search(self, text: str, limit: int = 20, transcript_id: str | None = None, offset: int = 0,
               speaker: str | None = None, transcript_ids: Collection[str] | None = None) -> dict:
        """See ``_search``. A search that finds nothing looks at the files once more (ignoring the
        ``SYNC_INTERVAL`` throttle): a transcript another process (CLI, web UI) wrote moments after the last
        scan would otherwise stay unfindable for up to 5 s, and "no hits" is the moment that costs the user."""
        found = self._search(text, limit, transcript_id, offset, speaker, transcript_ids)
        if found["total"] == 0 and self._rescan_after_miss():
            found = self._search(text, limit, transcript_id, offset, speaker, transcript_ids)
        return found

    def _rescan_after_miss(self) -> bool:
        if time.monotonic() - self._last_sync < EMPTY_RESCAN_INTERVAL:
            return False
        if not self._sync_lock.acquire(blocking=False):
            return False  # a sync is running right now: its result is what we have
        try:
            result = self._sync()
            return result["changed"] + result["removed"] > 0
        finally:
            self._sync_lock.release()

    def _search(self, text: str, limit: int = 20, transcript_id: str | None = None, offset: int = 0,
                speaker: str | None = None, transcript_ids: Collection[str] | None = None) -> dict:
        """Best matches first. ``total`` counts all matching segments. ``speaker`` keeps only segments
        of a speaker with this display name (case and diacritics ignored) or this raw label
        (``SPEAKER_01``); with it the query may be empty and the segments come in archive order.
        ``transcript_ids`` limits the search to those transcripts (a folder: the caller resolves
        it, the index does not know about folders, so moving a transcript never reindexes it)."""
        query = fts_query(text)
        who = fold(speaker)
        if not query and not who:
            return {"query": text, "total": 0, "results": []}
        if transcript_ids is not None and not transcript_ids:
            return {"query": text, "total": 0, "results": []}
        if not query:
            return self._list_speaker(text, who, limit, transcript_id, offset, transcript_ids)
        self.sync_if_stale()
        where = "segments_fts MATCH ?" + (" AND s.transcript_id = ?" if transcript_id else "")
        params: list = [query] + ([transcript_id] if transcript_id else [])
        where, params = self._restrict(where, params, transcript_ids)
        if who:
            where += " AND (fold(s.speaker_name) = ? OR fold(s.speaker) = ?)"
            params += [who, who]
        with self._lock:
            db = self._db()
            try:
                total = db.execute(
                    f"SELECT count(*) FROM segments_fts JOIN segments s ON s.rowid = segments_fts.rowid WHERE {where}",
                    params).fetchone()[0]
                rows = db.execute(
                    "SELECT s.transcript_id, t.title, s.seg, s.t_start, s.t_end, s.speaker, s.speaker_name, s.text, s.lang, "
                    f"snippet(segments_fts, 0, ?, ?, '…', {SNIPPET_TOKENS}) "
                    "FROM segments_fts JOIN segments s ON s.rowid = segments_fts.rowid "
                    f"JOIN transcripts t ON t.id = s.transcript_id WHERE {where} "
                    "ORDER BY bm25(segments_fts), s.transcript_id, s.seg LIMIT ? OFFSET ?",
                    [MARK_START, MARK_END, *params, limit, offset]).fetchall()
            except sqlite3.OperationalError:  # should not happen with fts_query(), but never 500 on odd input
                return {"query": text, "total": 0, "results": []}
        gone = {row[0] for row in rows if not self._exists(row[0])}
        if gone:  # deleted since the last sync: drop them now (and count again) rather than link to nothing
            for tid in gone:
                self.remove(tid)
            return self.search(text, limit, transcript_id, offset, speaker, transcript_ids)
        results = []
        for tid, title, seg, start, end, speaker, speaker_name, full, lang, marked in rows:
            snippet, highlights = split_marks(marked)
            results.append(Hit(tid, title, seg, start, end, speaker, speaker_name, full, snippet, highlights, lang).to_dict())
        return {"query": text, "total": total, "results": results}

    @staticmethod
    def _restrict(where: str, params: list, transcript_ids: Collection[str] | None) -> tuple[str, list]:
        """``AND s.transcript_id IN (...)``; the ids travel as one JSON array (no SQL variable limit)."""
        if transcript_ids is None:
            return where, params
        return where + " AND s.transcript_id IN (SELECT value FROM json_each(?))", [*params, json.dumps(sorted(transcript_ids))]

    def _list_speaker(self, text: str, who: str, limit: int, transcript_id: str | None, offset: int,
                      transcript_ids: Collection[str] | None = None) -> dict:
        """Everything one speaker said (no text query): by transcript, then position."""
        self.sync_if_stale()
        where = "s.lang = '' AND (fold(s.speaker_name) = ? OR fold(s.speaker) = ?)" + (
            " AND s.transcript_id = ?" if transcript_id else "")
        params: list = [who, who] + ([transcript_id] if transcript_id else [])
        where, params = self._restrict(where, params, transcript_ids)
        with self._lock:
            db = self._db()
            total = db.execute(f"SELECT count(*) FROM segments s WHERE {where}", params).fetchone()[0]
            rows = db.execute(
                "SELECT s.transcript_id, t.title, s.seg, s.t_start, s.t_end, s.speaker, s.speaker_name, s.text "
                f"FROM segments s JOIN transcripts t ON t.id = s.transcript_id WHERE {where} "
                "ORDER BY s.transcript_id, s.seg LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
        gone = {row[0] for row in rows if not self._exists(row[0])}
        if gone:
            for tid in gone:
                self.remove(tid)
            return self._list_speaker(text, who, limit, transcript_id, offset, transcript_ids)
        results = [Hit(tid, title, seg, start, end, spk, name, full, full[:SNIPPET_CHARS], []).to_dict()
                   for tid, title, seg, start, end, spk, name, full in rows]
        return {"query": text, "total": total, "results": results}

    def speaker_names(self) -> list[str]:
        """Every display name and raw label that appears as a speaker of an original (untranslated) segment."""
        self.sync_if_stale()
        with self._lock:
            rows = self._db().execute(
                "SELECT DISTINCT speaker_name, speaker FROM segments WHERE lang = ''").fetchall()
        return sorted({v for row in rows for v in row if v})

    def _exists(self, transcript_id: str) -> bool:
        try:
            self.library.path(transcript_id)
            return True
        except (TranscriptNotFound, InvalidTranscriptId):
            return False
