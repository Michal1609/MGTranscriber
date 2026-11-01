"""Transcript editing: save edited segments with version history, keeping word times where possible.

The editor (``static/js/components/editor.js``) sends whole segments (``start, end, speaker, text,
words``). On save every segment's words are re-aligned to its text so that words and the
whitespace-separated tokens of the text correspond 1:1 (``realign_words``):

* a token that matches an old word (longest common subsequence, ``lcs_pairs``, the same algorithm
  and size limit as editor.js) keeps the word's times, score, speaker and ``redacted`` flag;
* in a changed stretch the new tokens take the old words' times one for one (a corrected
  spelling), the rest get times spread over the gap between their timed neighbours in proportion
  to their length, and no score (estimated);
* redaction errs on the safe side: if a changed stretch had a redacted word, all its new tokens
  are redacted, and new tokens next to a redacted word are redacted too;
* deleted tokens drop their words.

Saving goes through ``Library.save_segments`` (listeners reindex the search). Derived files:
the LLM analysis is not deleted, it shows "outdated" through its stamp; voiceprints
(``voices.json``) and speaker names stay, because a text fix does not change the voices; the stored
exports (``prepis.txt``, ``titulky.srt``) are written again. Before a save the previous
``prepis.json`` is copied to ``versions/`` (at most one copy per ``VERSION_EVERY_S`` while editing,
``MAX_VERSIONS`` kept), so any earlier state can be restored.
"""

from __future__ import annotations

import difflib
import math
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .fsutil import file_lock
from .library import SEGMENTS_FILE, SPEAKER_KEY
from .models import UNKNOWN_SPEAKER, Segment, Word

if TYPE_CHECKING:
    from .config import Settings
    from .library import Library

VERSIONS_DIR = "versions"
VERSION_RE = re.compile(r"^prepis\.(\d{8}T\d{6}\d{3}Z)(?:\.([a-z]+))?\.json$")
VERSION_EVERY_S = 300
MAX_VERSIONS = 50
MAX_SEGMENTS = 50_000
MAX_TEXT = 20_000
OVERLAP_S = 0.5  # segments of one speaker may overlap this much; other speakers may overlap freely (separated voices)


class EditError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


# --- word alignment -------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w]+", re.UNICODE)
LCS_LIMIT = 400_000  # cells of the LCS table; the same limit as realign() in editor.js


def _key(token: str) -> str:
    return _PUNCT.sub("", token).lower() or token


def tokens(text: str) -> list[str]:
    return text.split()


def lcs_pairs(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Pairs ``(i, j)`` with ``a[i] == b[j]`` of a longest common subsequence. The common prefix and
    suffix are matched first; a middle larger than ``LCS_LIMIT`` cells is left unmatched (one
    changed stretch). Same algorithm and tie-breaking as ``lcsPairs`` in editor.js."""
    n, m = len(a), len(b)
    lo = 0
    while lo < n and lo < m and a[lo] == b[lo]:
        lo += 1
    hi = 0
    while hi < n - lo and hi < m - lo and a[n - 1 - hi] == b[m - 1 - hi]:
        hi += 1
    pairs = [(k, k) for k in range(lo)]
    ra, rb = a[lo:n - hi], b[lo:m - hi]
    rn, rm = len(ra), len(rb)
    if rn and rm and rn * rm <= LCS_LIMIT:
        dp = [[0] * (rm + 1) for _ in range(rn + 1)]
        for i in range(rn - 1, -1, -1):
            row, below = dp[i], dp[i + 1]
            for j in range(rm - 1, -1, -1):
                row[j] = below[j + 1] + 1 if ra[i] == rb[j] else max(below[j], row[j + 1])
        i = j = 0
        while i < rn and j < rm:
            if ra[i] == rb[j]:
                pairs.append((lo + i, lo + j))
                i += 1
                j += 1
            elif dp[i + 1][j] >= dp[i][j + 1]:
                i += 1
            else:
                j += 1
    pairs += [(n - hi + k, m - hi + k) for k in range(hi)]
    return pairs


def realign_words(old: list[Word], text: str, start: float, end: float, speaker: str | None) -> list[Word]:
    """Words for ``text`` (one per whitespace token), reusing what ``old`` knows (see module doc).

    Redaction errs on the safe side: when a changed stretch replaces a redacted word, all its new
    tokens are redacted, and tokens inserted next to a redacted word are redacted too."""
    new_tokens = tokens(text)
    if [w.word for w in old] == new_tokens:
        return old
    pairs = lcs_pairs([_key(w.word) for w in old], [_key(t) for t in new_tokens])
    words: list[Word | None] = [None] * len(new_tokens)
    estimated: set[int] = set()
    inserted: list[tuple[int, int]] = []  # runs of new tokens that take the place of no old word
    pi = pj = 0
    for mi, mj in [*pairs, (len(old), len(new_tokens))]:
        # changed stretch old[pi:mi] -> new[pj:mj]: tokens take the place of the old words one by one
        # (a corrected spelling keeps the times), the rest are new
        gap = old[pi:mi]
        redacted = True if any(w.redacted for w in gap) else None
        paired = min(mi - pi, mj - pj)
        for k in range(paired):
            w = gap[k]
            words[pj + k] = Word(new_tokens[pj + k], w.start, w.end, None, w.speaker or speaker, redacted)
        for j in range(pj + paired, mj):
            words[j] = Word(new_tokens[j], None, None, None, speaker, redacted)
            estimated.add(j)
        if pj + paired < mj:
            inserted.append((pj + paired, mj))
        if mi < len(old):
            w = old[mi]
            words[mj] = Word(new_tokens[mj], w.start, w.end, w.score, w.speaker or speaker, w.redacted)
        pi, pj = mi + 1, mj + 1
    out = [w for w in words if w is not None]
    for j1, j2 in inserted:  # typed next to a redacted word (inside a name, after it...): redacted
        if (j1 > 0 and out[j1 - 1].redacted) or (j2 < len(out) and out[j2].redacted):
            for j in range(j1, j2):
                out[j].redacted = True
    _spread_times(out, estimated, start, end)
    return out


def _spread_times(words: list[Word], estimated: set[int], start: float, end: float) -> None:
    """Give the ``estimated`` words times between their timed neighbours, proportional to length."""
    i = 0
    while i < len(words):
        if i not in estimated:
            i += 1
            continue
        j = i
        while j < len(words) and j in estimated:
            j += 1
        left = next((w.end for w in reversed(words[:i]) if w.end is not None), start)
        right = next((w.start for w in words[j:] if w.start is not None), end)
        if right < left:
            right = left
        weights = [len(w.word) + 1 for w in words[i:j]]
        total = sum(weights)
        t = left
        for w, weight in zip(words[i:j], weights):
            w.start = round(t, 3)
            t += (right - left) * weight / total
            w.end = round(t, 3)
        i = j


def normalize(segment: Segment) -> Segment:
    """The segment with words 1:1 to its text tokens (older transcripts may differ slightly)."""
    for w in segment.words:  # a NaN/inf score written by an older version would break the JSON answer
        if w.score is not None and not math.isfinite(w.score):
            w.score = None
    segment.words = realign_words(segment.words, segment.text, segment.start, segment.end, segment.speaker)
    return segment


# --- validation -----------------------------------------------------------------------------


def _number(value, field: str) -> float | None:
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or value > 1e7):
        raise EditError("invalid_segments", f"Invalid {field}.", 422)
    return round(float(value), 3)


def _score(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EditError("invalid_segments", "Invalid word score.", 422)
    return float(value)


def _clamp(value: float | None, start: float, end: float) -> float | None:
    return None if value is None else min(max(value, start), end)


def parse_segments(items, allow_self_overlap: bool = False) -> list[Segment]:
    """Segments from the editor's JSON, validated; words re-aligned to the text. ``allow_self_overlap``
    (transcripts of separated voices) lets a relabelled interjection overlap its new speaker's segment."""
    if not isinstance(items, list) or len(items) > MAX_SEGMENTS:
        raise EditError("invalid_segments", "Expected a list of segments.", 422)
    out: list[Segment] = []
    last_end: dict[str | None, float] = {}  # per speaker: where its previous segment ended
    for n, item in enumerate(items):
        if not isinstance(item, dict):
            raise EditError("invalid_segments", f"Segment {n} is not an object.", 422)
        text = item.get("text", "")
        if not isinstance(text, str) or len(text) > MAX_TEXT:
            raise EditError("invalid_segments", f"Segment {n}: invalid text.", 422)
        text = " ".join(text.split())
        if not text:
            continue  # an emptied segment is removed
        start, end = _number(item.get("start"), "start"), _number(item.get("end"), "end")
        if start is None or end is None or end < start:
            raise EditError("invalid_segments", f"Segment {n}: invalid start/end.", 422)
        speaker = item.get("speaker")
        if speaker in (None, "", UNKNOWN_SPEAKER):
            speaker = None
        elif not isinstance(speaker, str) or not SPEAKER_KEY.fullmatch(speaker):
            raise EditError("invalid_segments", f"Segment {n}: invalid speaker.", 422)
        if out and start < out[-1].start:
            raise EditError("invalid_segments", f"Segment {n}: out of order.", 422)
        if not allow_self_overlap and start < last_end.get(speaker, 0.0) - OVERLAP_S:  # a speaker does not overlap themselves
            raise EditError("invalid_segments", f"Segment {n}: overlapping the previous segment of the same speaker.", 422)
        words = []
        raw_words = item.get("words") or []
        if not isinstance(raw_words, list) or len(raw_words) > 5000:
            raise EditError("invalid_segments", f"Segment {n}: invalid words.", 422)
        for w in raw_words:
            if not isinstance(w, dict) or not isinstance(w.get("word", ""), str):
                raise EditError("invalid_segments", f"Segment {n}: invalid word.", 422)
            word_speaker = w.get("speaker")
            if word_speaker in (None, ""):
                word_speaker = speaker
            elif not isinstance(word_speaker, str) or not SPEAKER_KEY.fullmatch(word_speaker):
                raise EditError("invalid_segments", f"Segment {n}: invalid word speaker.", 422)
            # word times stay inside their segment (an end before the start becomes the start)
            w_start = _clamp(_number(w.get("start"), "word start"), start, end)
            w_end = _clamp(_number(w.get("end"), "word end"), start, end)
            if w_start is not None and w_end is not None and w_end < w_start:
                w_end = w_start
            words.append(Word(w.get("word", "")[:200], w_start, w_end, _score(w.get("score")), word_speaker,
                              True if w.get("redacted") else None))
        seg = Segment(start, end, " " + text, speaker, words)
        seg.words = realign_words(words, text, start, end, speaker)
        out.append(seg)
        last_end[speaker] = max(end, last_end.get(speaker, 0.0))
    return out


# --- versions -------------------------------------------------------------------------------


def _stamp_now() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"


def versions_dir(library: "Library", transcript_id: str) -> Path:
    return library.path(transcript_id, must_exist=False) / VERSIONS_DIR


def list_versions(library: "Library", transcript_id: str) -> list[dict]:
    folder = versions_dir(library, transcript_id)
    out = []
    if folder.is_dir():
        for path in folder.iterdir():
            m = VERSION_RE.match(path.name)
            if m and path.is_file():
                when = datetime.strptime(m.group(1)[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                out.append({"id": path.name, "time": when.isoformat(), "reason": m.group(2) or "edit",
                            "size": path.stat().st_size})
    return sorted(out, key=lambda v: v["id"], reverse=True)


def backup(library: "Library", transcript_id: str, reason: str = "edit", force: bool = True) -> str | None:
    """Copy the current prepis.json to ``versions/``. Without ``force`` only when the newest copy is
    older than ``VERSION_EVERY_S`` (one copy per editing burst). Returns the version id."""
    source = library.path(transcript_id, must_exist=False) / SEGMENTS_FILE
    if not source.is_file():
        return None
    folder = versions_dir(library, transcript_id)
    existing = list_versions(library, transcript_id)
    if not force and existing and time.time() - (folder / existing[0]["id"]).stat().st_mtime < VERSION_EVERY_S:
        return None
    folder.mkdir(exist_ok=True)
    name = f"prepis.{_stamp_now()}.{reason}.json"
    shutil.copy2(source, folder / name)
    (folder / name).touch()  # mtime = time of the copy (copy2 keeps the source's)
    for old in list_versions(library, transcript_id)[MAX_VERSIONS:]:
        (folder / old["id"]).unlink(missing_ok=True)
    return name


def version_segments(library: "Library", transcript_id: str, version: str) -> list[Segment]:
    import json

    if not VERSION_RE.match(version):
        raise EditError("version_not_found", "Version not found.", 404)
    path = versions_dir(library, transcript_id) / version
    if not path.is_file():
        raise EditError("version_not_found", "Version not found.", 404)
    with open(path, encoding="utf-8") as f:
        return [Segment.from_dict(s) for s in json.load(f)]


# --- saving ---------------------------------------------------------------------------------


def edit_lock(library: "Library", transcript_id: str):
    return file_lock(library.path(transcript_id, must_exist=False) / SEGMENTS_FILE)


def save(library: "Library", transcript_id: str, segments: list[Segment], *, base: str | None,
         settings: "Settings | None" = None, event: str = "edited", details: dict | None = None,
         version_reason: str = "edit", force_version: bool = False) -> dict:
    """Save edited segments. ``base`` is the ``segments_stamp`` the editor started from; another
    value means the transcript changed meanwhile (409)."""
    from . import exporters, privacy

    with edit_lock(library, transcript_id):
        current = library.segments_stamp(transcript_id)
        if base is not None and base != current:
            raise EditError("edit_conflict", "The transcript was changed elsewhere. Reload it.", 409)
        old = library.segments(transcript_id)
        changed = _changes(old, segments)
        # unchanged, also when only the words of an older transcript were made 1:1 (as GET shows them)
        if [normalize(Segment.from_dict(s.to_dict())).to_dict() for s in old] == [s.to_dict() for s in segments]:
            return {"stamp": current, "changed": changed, "version": None}
        version = backup(library, transcript_id, version_reason, force=force_version)
        library.save_segments(transcript_id, segments)
        folder = library.path(transcript_id)
        names = {}  # stored exports are written without names, like after a transcription
        outputs = [SEGMENTS_FILE]
        for exporter in exporters.list_exporters():
            if exporter.stored_as:
                content = exporter.render(segments, names)
                target = folder / exporter.stored_as
                (target.write_bytes(content) if isinstance(content, bytes)
                 else target.write_text(content, encoding="utf-8"))
                outputs.append(exporter.stored_as)
        meta = library.metadata(transcript_id)
        meta.speakers = sorted({s.speaker for s in segments if s.speaker})
        meta.extra["edited"] = privacy.now_iso()
        meta.extra["custody"] = {**privacy.custody_record(library, transcript_id, outputs, with_source=False),
                                 **({"source": meta.extra["custody"]["source"]}
                                    if (meta.extra.get("custody") or {}).get("source") else {})}
        library.save_metadata(transcript_id, meta)
        stamp = library.segments_stamp(transcript_id)
        privacy.audit(library, transcript_id, event, **(details or {}), **changed, version=version,
                      sha256=meta.extra["custody"]["outputs"].get(SEGMENTS_FILE))
        return {"stamp": stamp, "changed": changed, "version": version}


def _changes(old: list[Segment], new: list[Segment]) -> dict:
    """Rough counts for the audit log (no text: the log must not repeat what was redacted)."""
    old_keys = [(s.text.strip(), s.speaker) for s in old]
    new_keys = [(s.text.strip(), s.speaker) for s in new]
    matcher = difflib.SequenceMatcher(None, old_keys, new_keys, autojunk=False)
    edited = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in matcher.get_opcodes() if op != "equal")
    redacted_old = sum(1 for s in old for w in s.words if w.redacted)
    redacted_new = sum(1 for s in new for w in s.words if w.redacted)
    times = sum(1 for a, b in zip(old, new) if (a.start, a.end) != (b.start, b.end))
    return {"segments_changed": edited, "segments": len(new), "redacted_words": redacted_new,
            "redactions_changed": redacted_new != redacted_old or _redaction_moved(old, new),
            "times_changed": times}


def _redaction_moved(old: list[Segment], new: list[Segment]) -> bool:
    def marks(segments):
        return [(round(w.start or 0, 2), w.word) for s in segments for w in s.words if w.redacted]
    return marks(old) != marks(new)


def restore(library: "Library", transcript_id: str, version: str, settings: "Settings | None" = None) -> dict:
    """Make an earlier version current again (the current one is kept as a version first)."""
    segments = version_segments(library, transcript_id, version)
    return save(library, transcript_id, segments, base=None, settings=settings, event="restored",
                details={"from_version": version}, version_reason="restore", force_version=True)
