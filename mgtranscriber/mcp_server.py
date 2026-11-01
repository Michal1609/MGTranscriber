"""MCP server: lets AI assistants transcribe, search and read the transcript library.

Built on the official MCP Python SDK (``mcp`` 2.x, ``MCPServer``). Two transports:

- stdio: ``mgtranscriber mcp`` (Claude Desktop, Claude Code, Cursor, VS Code start it);
- Streamable HTTP: mounted at ``/mcp`` in the web server (``mgtranscriber serve``).

Reading tools (list, search, get, rename) work on the library files directly. Jobs go
through a ``JobsBackend``: inside the web server that is the server's own queue; for
stdio (``AutoJobs``) it is the running web server's HTTP API whenever one answers at
submit time (so two processes do not fight over the GPU), otherwise a queue inside the
stdio process.

Long transcriptions: ``transcribe`` returns a job id at once; ``job_status`` polls it.
With ``wait=true`` it blocks (with progress notifications) up to ``WAIT_LIMIT`` seconds.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import quote, unquote

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import APP_NAME, __version__, netguard, privacy, separated, voices
from .config import Settings
from .folders import FolderError, FolderStore
from .library import InvalidTranscriptId, Library, TranscriptNotFound, metadata_source_key
from .llm import LLMError
from .llm.translations import list_translations, load_translation, translation_segments
from .pathguard import UNC_MESSAGE, unc_blocked
from .models import UNKNOWN_SPEAKER, Segment, speaker_of, speaker_stats
from .pipeline import TranscriptionRequest
from .search import SearchIndex
from .sources import is_url
from .timefmt import clock

log = logging.getLogger(__name__)

WAIT_LIMIT = 600.0  # seconds ``transcribe(wait=true)`` blocks at most
POLL_INTERVAL = 2.0
MAX_CHARS = 20_000  # default page size of get_transcript
MAX_SPANS = 20  # time spans per transcript in find_speaker
MAX_SPANS_TOTAL = 150  # ... and in one find_speaker answer (output is also capped at MAX_CHARS)
MAX_FIND = 25  # transcripts per find_speaker answer
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
LANGUAGE = re.compile(r"auto|[a-z]{2,3}")  # same rule as POST /api/jobs
MIN_PAGE_CHARS = 1000  # smallest page get_transcript serves (a smaller max_chars is raised to this)
YOUTUBE_ID = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/|/embed/|/live/)([\w-]{11})")
# rough share of the whole job per phase, for a single progress bar
PHASE_SPAN = {"queued": (0, 0), "starting": (0, 2), "loading": (2, 5), "download": (5, 15),
              "transcribe": (15, 70), "align": (70, 85), "diarize": (85, 98), "writing": (98, 100)}

INSTRUCTIONS = f"""{APP_NAME} transcribes videos (YouTube and anything yt-dlp supports) and local audio/video
files on the user's own GPU, with speaker labels, and keeps a searchable library of transcripts.
- Find what was said: search_transcripts (full text, diacritics-insensitive; all words must be in the
  same segment, so use 1-2 keywords), then get_transcript with start/end around the hit. Cite the time
  (hh:mm:ss) and the transcript title. Every "url" field is a link into the web UI; it is null when the
  web server is not running (not an error).
- New transcription: transcribe returns a job id immediately; poll job_status until status is
  "done" and use its transcript_id. Transcription takes roughly 1/10 to 1/3 of the media length.
- Transcripts can be organised in folders ("Podcasts/Interviews"): list_folders shows them; list_transcripts and
  search_transcripts take folder to look only there; transcribe can put the new transcript into a folder.
- Speakers are SPEAKER_00, SPEAKER_01, ...; rename_speakers gives them names for all outputs.
- Voice memory remembers people across transcripts: list_speakers shows who is known, find_speaker(name)
  lists the transcripts and times where that person speaks (status "suggest" is an unverified guess, not
  proof); search_transcripts(q, speaker=name) searches only what they said."""


def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def overall_percent(phase: str, percent: float | None) -> float:
    lo, hi = PHASE_SPAN.get(phase, (0, 0))
    return lo + (hi - lo) * (percent or 0) / 100


# --- jobs backends ----------------------------------------------------------------------


class JobsBackend(Protocol):
    def submit(self, request: TranscriptionRequest) -> dict: ...
    def status(self, job_id: int) -> dict | None: ...


class LocalJobs:
    """A ``JobQueue`` in this process (the web server's, or the stdio server's own)."""

    def __init__(self, queue):
        self.queue = queue

    def submit(self, request: TranscriptionRequest) -> dict[str, Any]:
        return self.queue.public(self.queue.add(request))

    def status(self, job_id: int) -> dict | None:
        job = self.queue.get(job_id)
        if not job:
            return None
        return {**self.queue.public(job), "log_tail": job.log_since(max(job.log_lines - 5, 0))}


class ServerUnreachable(ToolError):
    """No (compatible) web server answers at the URL."""


class ServerRejectedKey(ToolError):
    """A web server answers but refuses our API key: never fall back to a second GPU queue then."""


class RemoteJobs:
    """The job API of a running web server (``/api/jobs``)."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _call(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                return json.loads(res.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            try:
                detail = json.loads(e.read()).get("detail")
            except (ValueError, AttributeError):
                detail = None
            if e.code in (401, 403):
                raise ServerRejectedKey(
                    f"The web server at {self.base_url} is running but rejected the request ({e.code}: "
                    f"{detail or e.reason}). Set the same {APP_NAME.upper()}_API_KEY for the MCP server.") from None
            raise ToolError(f"Server error {e.code}: {detail or e.reason}") from None
        except (OSError, ValueError) as e:  # refused, timeout, reset; or an answer that is not JSON
            raise ServerUnreachable(f"The web server at {self.base_url} is not reachable: {e}") from None

    def submit(self, request: TranscriptionRequest) -> dict[str, Any]:
        return self._call("POST", "/api/jobs", {
            "source": request.source, "language": request.language, "speakers": request.speakers,
            "no_diarize": request.diarize is False, "engine": request.engine, "model": request.model,
            "folder": request.folder, "separate_speakers": request.separate_speakers})

    def status(self, job_id: int) -> dict | None:
        job = self._call("GET", f"/api/jobs/{job_id}")
        if job is not None:
            job["log_tail"] = job.pop("log", [])[-5:]
        return job

    @classmethod
    def probe(cls, base_url: str, api_key: str | None) -> "RemoteJobs | None":
        """The backend when a web server answers at ``base_url``, else None.
        Raises ``ServerRejectedKey`` when it answers but refuses the key."""
        backend = cls(base_url, api_key, timeout=2.0)
        try:
            info = backend._call("GET", "/api/info")
        except ServerUnreachable:
            return None
        if not isinstance(info, dict) or info.get("name") != APP_NAME:
            return None
        backend.timeout = 10.0
        return backend


class AutoJobs:
    """stdio: the web server's queue when one answers, checked on every submit, else a queue in this process.

    Checking each time (not only at start) means a web server started after the MCP server gets the new
    jobs, and a server that refuses the key is an error instead of a silent second GPU queue.
    """

    def __init__(self, base_url: str, api_key: str | None, make_local: Callable[[], LocalJobs]):
        self.base_url = base_url
        self.api_key = api_key
        self.make_local = make_local
        self.local: LocalJobs | None = None
        self.owner: dict[int, JobsBackend] = {}  # job id -> backend that has it

    def _backend(self) -> JobsBackend:
        remote = RemoteJobs.probe(self.base_url, self.api_key)
        if remote:
            return remote
        if self.local is None:
            self.local = self.make_local()
        return self.local

    def submit(self, request: TranscriptionRequest) -> dict[str, Any]:
        backend = self._backend()
        job = backend.submit(request)
        self.owner[job["id"]] = backend
        return job

    def status(self, job_id: int) -> dict | None:
        # a job from before a restart of this process: whichever queue is there now
        return (self.owner.get(job_id) or self._backend()).status(job_id)


# --- formatting -------------------------------------------------------------------------


def job_view(job: dict) -> dict[str, Any]:
    keys = ("id", "status", "phase", "percent", "position", "title", "result", "error", "source", "log_tail")
    view = {k: job.get(k) for k in keys if k in job}
    view["job_id"] = view.pop("id")
    view["transcript_id"] = view.pop("result", None)
    if job.get("status") in ("queued", "running"):
        view["hint"] = "Call job_status with this job_id again in a little while."
    elif job.get("status") == "done":
        view["hint"] = "Read it with get_transcript(transcript_id) or search it with search_transcripts."
    return view


def speaker_name(speaker: str | None, names: dict[str, str]) -> str | None:
    speaker = speaker or UNKNOWN_SPEAKER
    if speaker == UNKNOWN_SPEAKER and speaker not in names:
        return None
    return names.get(speaker) or speaker


def render_text(segments: list[Segment], names: dict[str, str]) -> str:
    """Speaker turns: ``[00:01:02] Eva: text``."""
    out: list[str] = []
    last = object()
    for seg in segments:
        spk = speaker_of(seg)
        if spk != last:
            label = speaker_name(spk, names)
            out.append(f"\n[{clock(seg.start)}]" + (f" {label}:" if label else "") + f" {seg.text.strip()}")
            last = spk
        else:
            out[-1] += " " + seg.text.strip()
    return "\n".join(line.strip("\n") for line in out).strip()


def speech_spans(segments: list[Segment], label: str, gap: float = 2.0) -> list[dict[str, Any]]:
    """Time spans where speaker ``label`` talks; segments closer than ``gap`` seconds are merged."""
    spans: list[dict[str, Any]] = []
    for seg in segments:
        if speaker_of(seg) != label:
            continue
        if spans and seg.start - spans[-1]["end"] <= gap:
            spans[-1]["end"] = seg.end
        else:
            spans.append({"start": seg.start, "end": seg.end})
    return [{"start": round(sp["start"], 1), "end": round(sp["end"], 1),
             "time": f"{clock(sp['start'])}-{clock(sp['end'])}"} for sp in spans]


# --- server -----------------------------------------------------------------------------


def build_server(settings: Settings, library: Library, index: SearchIndex, jobs: JobsBackend,
                 allow_local_paths: bool, web_url: str | None = None, public_urls_only: bool = False) -> MCPServer:
    """The MCP server with all tools and resources; transport-independent.
    ``public_urls_only``: refuse URLs on this computer or the local network (see ``netguard``)."""
    # log_level: the SDK calls logging.basicConfig with it (stderr); INFO would log every request
    mcp = MCPServer(APP_NAME, title="Local transcripts", version=__version__, instructions=INSTRUCTIONS,
                    website_url=web_url, log_level="WARNING")

    def link(transcript_id: str, seconds: float | None = None) -> str | None:
        if not web_url:
            return None
        url = f"{web_url}/#/t/{quote(transcript_id, safe='')}"
        return url if seconds is None else f"{url}?t={round(seconds, 1):g}"

    folders = FolderStore(library, settings.data_dir / "folders.json")

    def folder_ids(folder: str, include_subfolders: bool) -> list[str]:
        """Ids of the transcripts in a folder; an unknown folder is an error (the model should see the names)."""
        try:
            path = folders.canonical(folder)
            if path and not folders.exists(path):
                raise ToolError(f"No folder '{folder}'. Use list_folders.")
            return folders.ids_in(path, include_subfolders)
        except FolderError as e:
            raise ToolError(e.message) from None

    def check_id(transcript_id: str) -> None:
        try:
            library.path(transcript_id)
        except (TranscriptNotFound, InvalidTranscriptId):
            raise ToolError(f"No transcript with id '{transcript_id}'. Use list_transcripts or search_transcripts.") from None

    def existing_transcript(source: str) -> str | None:
        """Id of the transcript already made from this URL or file, if any (a URL is matched without a
        network probe: by the stored source/webpage URL or, for YouTube, by the video id)."""
        if not is_url(source):
            return library.find_by_source("file:" + os.path.normcase(str(Path(source).resolve())))
        match = YOUTUBE_ID.search(source)
        video = match.group(1) if match else None
        for transcript_id in library.ids():
            try:
                meta = library.metadata(transcript_id)
            except (OSError, ValueError, KeyError, TypeError, LookupError):
                continue
            info = (meta.extra or {}).get("source_info") or {}
            if source in (meta.source, info.get("webpage_url")):
                return transcript_id
            key = meta.source_key or metadata_source_key(meta.to_dict()) or ""
            if video and key.lower().startswith("youtube:") and key.split(":", 1)[1] == video:
                return transcript_id
        return None

    @mcp.tool(title="Transcribe a video or file")
    async def transcribe(url_or_path: str, ctx: Context, language: str | None = None, speakers: int | None = None,
                         wait: bool = False, folder: str | None = None, replace: bool = True,
                         separate_speakers: bool = False) -> dict[str, Any]:
        """Start transcribing a video URL (YouTube etc.) or a local audio/video file on this computer.

        language: ISO 639 code of the spoken language such as "en" or "cs", or "auto" to detect it (default:
        the server setting, "auto" unless changed). Anything else is an error at once, not later in the queue.
        speakers: exact number of speakers if known (improves speaker labels).
        wait: block until done, with progress, for at most 10 minutes. When that limit passes the job is NOT
        cancelled: the answer then has wait_timed_out=true and status queued/running; keep polling
        job_status. Without wait the call returns at once with the job id.
        folder: library folder for the new transcript, e.g. "Podcasts/Interviews" (created when new; see
        list_folders). Omitted: unfiled, or the transcript's current folder when it is transcribed again.
        replace: a source that is already in the library is transcribed AGAIN and the new transcript
        REPLACES the existing one, including speaker names set by hand. The answer then has
        existing_transcript_id and a warning. With replace=false such a request is an error that names
        the existing transcript (read it with get_transcript instead).
        separate_speakers: for recordings where people talk over each other: split the audio into one track
        per voice and transcribe each (much slower; local engines only; needs the voice-separation models).
        Returns the job (job_id, status, phase, percent, ...); when status is "done", transcript_id is the
        transcript to read.
        """
        source = url_or_path.strip()
        if language is not None:
            language = language.strip()
            if not LANGUAGE.fullmatch(language):
                raise ToolError(f"Invalid language '{language}': use an ISO 639 code such as 'en' or 'cs', or 'auto'.")
        if not source or source.startswith("-"):
            raise ToolError("Give a video URL or a path to an audio/video file.")
        try:
            privacy.check_source(settings, source)  # offline mode: local files only
        except privacy.PrivacyError as e:
            raise ToolError(e.message) from None
        if not is_url(source):
            if not allow_local_paths:
                raise ToolError("This server does not transcribe files from its disk; give a URL.")
            if unc_blocked(settings, source):  # before any file system call: touching it sends credentials
                raise ToolError(UNC_MESSAGE)
            if not Path(source).is_file():
                raise ToolError(f"File not found: {source}")
            source = str(Path(source).resolve())
        if public_urls_only and is_url(source):
            try:
                await asyncio.to_thread(netguard.require_public_url, source)
            except netguard.BlockedAddress as e:
                raise ToolError(str(e)) from None
            except OSError:
                raise ToolError(f"Cannot resolve the host of {source}.") from None
        if speakers is not None and not 1 <= speakers <= 20:
            raise ToolError("speakers must be between 1 and 20.")
        if separate_speakers and (reason := await asyncio.to_thread(separated.unavailable_reason, settings,
                                                                    settings.engine)):
            raise ToolError(reason)
        target = None
        if folder is not None:
            try:
                target = (await asyncio.to_thread(folders.ensure, folder)) or ""
            except FolderError as e:
                raise ToolError(e.message) from None
        existing = await asyncio.to_thread(existing_transcript, source)
        if existing and not replace:
            raise ToolError(f"This source is already transcribed as '{existing}'. Read it with "
                            f"get_transcript('{existing}'), or call again with replace=true to transcribe it again.")
        job = await asyncio.to_thread(jobs.submit, TranscriptionRequest(source, language=language, speakers=speakers,
                                                                        folder=target,
                                                                        separate_speakers=separate_speakers))
        extra: dict[str, Any] = {}
        if existing:
            extra = {"existing_transcript_id": existing,
                     "warning": f"This source is already in the library as '{existing}'. When the job finishes the "
                                f"new transcript REPLACES it, including speaker names set by hand."}
        if not wait:
            return {**job_view(job), **extra}
        deadline = time.monotonic() + WAIT_LIMIT
        while time.monotonic() < deadline:
            job = await asyncio.to_thread(jobs.status, job["id"]) or job
            if job["status"] not in ("queued", "running"):
                break
            await ctx.report_progress(overall_percent(job["phase"], job.get("percent")), 100,
                                      f"{job['phase']}" + (f" {job['percent']:.0f}%" if job.get("percent") else ""))
            await asyncio.sleep(POLL_INTERVAL)
        view = {**job_view(job), **extra}
        if job["status"] in ("queued", "running"):
            view["wait_timed_out"] = True
            view["hint"] = (f"Waited {WAIT_LIMIT:.0f} s; the job keeps running. Call job_status with this "
                            f"job_id again in a little while.")
        return view

    @mcp.tool(title="Job status", annotations=READ_ONLY)
    async def job_status(job_id: int) -> dict[str, Any]:
        """State of a transcription job: status (queued, running, done, error, cancelled), phase, percent,
        queue position, error, the last log lines and, when done, transcript_id."""
        job = await asyncio.to_thread(jobs.status, job_id)
        if job is None:
            raise ToolError(f"No job with id {job_id}.")
        return job_view(job)

    @mcp.tool(title="List transcripts", annotations=READ_ONLY)
    def list_transcripts(query: str | None = None, limit: int = 20, folder: str | None = None,
                         include_subfolders: bool = True) -> dict[str, Any]:
        """Transcripts in the library, newest first. query filters by title or source (case and
        diacritics-insensitive); to search inside the text use search_transcripts. folder limits the
        list to a library folder ("" = unfiled; names from list_folders), by default with its subfolders.
        url is a link into the web UI, null when the web server is not running."""
        items = library.list()
        if folder is not None:
            wanted = set(folder_ids(folder, include_subfolders))
            items = [t for t in items if t["id"] in wanted]
        if query:
            q = fold(query)
            items = [t for t in items if q in fold(t["title"]) or q in fold(t.get("source") or "")]
        limit = max(1, min(limit, 200))
        return {
            "total": len(items),
            "transcripts": [{
                "id": t["id"], "title": t["title"], "source": t.get("source"),
                "date": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t["date"])),
                "duration": clock(t["duration"]), "duration_seconds": round(t["duration"], 1),
                "language": t.get("language"), "folder": t.get("folder") or "",
                "speakers": [{"id": s, "name": speaker_name(s, t["names"]), "seconds": round(sec)}
                             for s, sec in t["speakers"].items()],
                "url": link(t["id"]),
            } for t in items[:limit]],
        }

    def resolve_speaker(who: str) -> str:
        """The speaker filter as stored: an exact name or label as is; else the one close name; else an error
        that lists candidates, so a typo is never mistaken for "this person said nothing"."""
        known = {fold(n): n for n in index.speaker_names()}
        key = fold(who)
        if key in known:
            return who
        if voices.enabled(settings):
            try:
                if any(fold(p["name"]) == key for p in voices.open_db(settings).list()):
                    return who  # a known person without named segments: zero hits is the true answer
            except (OSError, ValueError, KeyError):
                pass
        near = [n for k, n in known.items() if key in k or (len(k) >= 3 and k in key)]
        if not near:
            near = [known[k] for k in difflib.get_close_matches(key, list(known), n=3, cutoff=0.75)]
        if len(near) == 1:
            return near[0]
        if near:
            raise ToolError(f"Unknown speaker '{who}'. Did you mean: {', '.join(sorted(near)[:5])}? "
                            "See list_speakers for people from the voice memory.")
        raise ToolError(f"Unknown speaker '{who}': no transcript has a speaker with this name or label. "
                        "See list_speakers (voice memory) or get_transcript (speaker names of one transcript).")

    @mcp.tool(title="Search transcripts", annotations=READ_ONLY)
    def search_transcripts(q: str = "", limit: int = 10, transcript_id: str | None = None,
                           speaker: str | None = None, folder: str | None = None,
                           include_subfolders: bool = True) -> dict[str, Any]:
        """Full-text search over all transcripts, hits ordered by relevance (best first). ALL words must
        occur in the same segment (a sentence or two): AND, not OR; so use 1-2 distinctive keywords, not a
        whole question, or a "quoted phrase". Words are prefix-matched ("prep" finds "přepis"); case and
        diacritics are ignored. Each hit has the transcript, the time (hh:mm:ss and seconds), the speaker
        and a snippet with **matches** in bold. Cite hits as "title @ time". Translations are searched too:
        a hit with lang set (e.g. "cs") is in that translation, not the original (read it with
        get_transcript(..., lang=)); lang null is the original text. url is null without a web server.
        speaker: only what this person said (a name from list_speakers or from the transcripts, or a label
        like SPEAKER_01). A close but not exact name is used when it is unambiguous (speaker_matched says
        which); an unknown name is an error. With an empty q it lists their segments in archive order.
        transcript_id: search one transcript. folder: search only in that library folder ("" = unfiled),
        by default including its subfolders. total counts all matching segments, hits only the first limit."""
        if not q.strip() and not (speaker or "").strip():
            raise ToolError("Give a search text q, a speaker, or both.")
        who = (speaker or "").strip() or None
        matched = None
        if who:
            resolved = resolve_speaker(who)
            if resolved != who:
                matched = who = resolved
        ids = None if folder is None else folder_ids(folder, include_subfolders)
        found = index.search(q, limit=max(1, min(limit, 100)), transcript_id=transcript_id,
                             speaker=who, transcript_ids=ids)
        hits = []
        for r in found["results"]:
            snippet, offset = r["snippet"], 0
            for a, b in r["highlights"]:
                snippet = snippet[:a + offset] + "**" + snippet[a + offset:b + offset] + "**" + snippet[b + offset:]
                offset += 4
            hits.append({
                "transcript_id": r["transcript_id"], "title": r["title"], "time": clock(r["start"]),
                "start": r["start"], "end": r["end"],
                "speaker": r["speaker_name"] or (r["speaker"] if r["speaker"] not in (None, UNKNOWN_SPEAKER) else None),
                "snippet": snippet, "citation": f"{r['title']} @ {clock(r['start'])}", "url": link(r["transcript_id"], r["start"]),
                # "" = the original text; else the language of the translation that matched (read it with lang=)
                "lang": r.get("lang") or None,
            })
        out: dict[str, Any] = {"query": q, "total": found["total"], "hits": hits}
        if matched:
            out["speaker_matched"] = matched
        words = len(re.findall(r"\w+", q))
        if not hits and words > 1:
            out["hint"] = ("No segment contains all of these words together (every word must be in the same "
                           "segment). Try 1-2 distinctive keywords, or a single word, instead of the whole question.")
        return out

    @mcp.tool(title="List folders", annotations=READ_ONLY)
    def list_folders() -> dict[str, Any]:
        """The library's folders as paths like "Podcasts/Interviews", with how many transcripts are directly in
        each (count) and in it and its subfolders (total), plus how many are unfiled. Pass a path as folder to
        list_transcripts, search_transcripts or transcribe."""
        tree = folders.tree()
        return {"total_transcripts": tree["total"], "unfiled": tree["unfiled"],
                "folders": [{k: f[k] for k in ("path", "count", "total")} for f in tree["folders"]]}

    def voice_db() -> voices.VoiceDB:
        if not voices.enabled(settings):
            raise ToolError("Voice memory is switched off on this server (setting voice_memory).")
        return voices.open_db(settings)

    @mcp.tool(title="List known speakers", annotations=READ_ONLY)
    def list_speakers() -> dict[str, Any]:
        """People in the voice memory (speakers the user has taught the app to recognise). Per person:
        transcripts and speech_seconds count the SURE appearances in the whole library (confirmed, auto or
        named by hand: the same rows and numbers as find_speaker with include_suggestions=false).
        suggested_transcripts and suggested_seconds are extra, unverified guesses and are NOT included in
        those two (see find_speaker include_suggestions). voice_sample_seconds is only the length of the
        voice samples the person was enrolled from. Use find_speaker for where."""
        if not voices.enabled(settings):
            return {"enabled": False, "total": 0, "people": []}
        db = voices.open_db(settings)
        stats = voices.appearance_stats(library, db)
        empty = {"transcripts": 0, "seconds": 0.0, "suggested_transcripts": 0, "suggested_seconds": 0.0}
        people = []
        for p in db.list():
            st = stats.get(p["id"], empty)
            people.append({"person_id": p["id"], "name": p["name"], "transcripts": st["transcripts"],
                           "speech_seconds": st["seconds"], "suggested_transcripts": st["suggested_transcripts"],
                           "suggested_seconds": st["suggested_seconds"],
                           "voice_sample_seconds": p.get("speech_s"), "samples": len(p.get("samples", []))})
        people.sort(key=lambda p: (-p["transcripts"], fold(p["name"])))
        return {"enabled": True, "total": len(people), "people": people}

    @mcp.tool(title="Find where a person speaks", annotations=READ_ONLY)
    def find_speaker(name: str | None = None, person_id: str | None = None, limit: int = 20,
                     include_suggestions: bool = False, include_spans: bool = True) -> dict[str, Any]:
        """Transcripts where a person from the voice memory speaks, newest first. Give name (case and
        diacritics ignored) or person_id from list_speakers. Per transcript: first_time, speech_seconds
        (the person's speech there; the measure list_speakers sums), span_count and, with include_spans,
        the time spans of their speech (hh:mm:ss and seconds; at most 20 per transcript and 150 per answer,
        spans_truncated says some are left out). include_spans=false gives only the summary, which is
        usually enough. status: confirmed (the user confirmed the voice), auto (recognised by voice) and
        manual (named by hand) are sure. suggest (only with include_suggestions=true) is an UNVERIFIED guess
        of the voice matcher: needs_confirmation is true then; do not present it as fact. At most 25
        transcripts per answer: transcripts_truncated says more exist (total is the real count). url is the
        web UI link to the first span, null when the web server is not running. Read what was said with
        get_transcript(transcript_id, start, end) or search_transcripts(q, speaker=name)."""
        if bool(name and name.strip()) == bool(person_id):
            raise ToolError("Give exactly one of name or person_id (see list_speakers).")
        db = voice_db()
        people = {p["id"]: p["name"] for p in db.list()}
        if person_id:
            if person_id not in people:
                raise ToolError(f"No person with id '{person_id}'. Use list_speakers.")
            pid = person_id
        else:
            key = fold(name)
            matches = [i for i, n in people.items() if fold(n) == key] or [i for i, n in people.items() if key in fold(n)]
            if not matches:
                raise ToolError(f"No person named '{name}' in the voice memory. Use list_speakers.")
            if len(matches) > 1:
                raise ToolError("More people match: " + ", ".join(f"{people[i]} ({i})" for i in matches)
                                + ". Call again with person_id.")
            pid = matches[0]
        try:
            found = voices.appearances(library, db, pid, include_suggest=include_suggestions)
        except voices.VoiceError as e:
            raise ToolError(e.message) from None
        limit = max(1, min(limit, MAX_FIND))
        out = []
        budget = MAX_SPANS_TOTAL
        size = 0
        transcripts_truncated = len(found) > limit
        spans_truncated = False
        for row in found[:limit]:
            try:
                spans = speech_spans(library.segments(row["transcript_id"]), row["label"])
            except (OSError, ValueError):
                spans = []
            shown = spans[:min(MAX_SPANS, budget)] if include_spans else []
            entry = {
                "transcript_id": row["transcript_id"], "title": row["title"], "label": row["label"],
                "status": row["status"], "needs_confirmation": row["status"] == "suggest",
                "speech_seconds": row.get("speech_s"),
                "first_time": clock(spans[0]["start"]) if spans else None, "span_count": len(spans),
                "url": link(row["transcript_id"], spans[0]["start"] if spans else None),
            }
            if include_spans:
                entry["spans"] = shown
            size += len(json.dumps(entry, ensure_ascii=False))
            if out and size > MAX_CHARS:  # the answer stays readable; ask again with a narrower question
                transcripts_truncated = True
                break
            budget -= len(shown)
            spans_truncated = spans_truncated or (include_spans and len(shown) < len(spans))
            out.append(entry)
        suggested = sum(1 for r in found if r["status"] == "suggest")
        sure_seconds = sum(r.get("speech_s") or 0 for r in found if r["status"] != "suggest")
        return {"person_id": pid, "name": people[pid], "total": len(found), "returned": len(out),
                "sure_total": len(found) - suggested, "suggested_total": suggested,
                "sure_speech_seconds": round(sure_seconds, 1),
                "transcripts_truncated": transcripts_truncated, "spans_truncated": spans_truncated,
                "appearances": out}

    @mcp.tool(title="Read a transcript", annotations=READ_ONLY)
    def get_transcript(transcript_id: str, format: str = "text", start: float | None = None,
                       end: float | None = None, max_chars: int = MAX_CHARS,
                       lang: str | None = None) -> dict[str, Any]:
        """Transcript content. format "text": speaker turns with [hh:mm:ss] times; "segments": a list
        with start/end seconds, speaker and text. start/end (seconds) select a time range, which is widened
        to whole segments: every segment that overlaps it is returned, so the text may begin a little before
        start and end after end (range shows what was returned). Long transcripts come in pages of about
        max_chars characters (default 20000, minimum 1000): if truncated is true, call again with
        start=next_start. lang (e.g. "cs") reads the stored translation of that language instead of the
        original (same segments, translated text); translation_lang echoes it (null = the original),
        transcript_language is the language of the original recording and translations lists the
        translations that exist. url is the web UI link, null when the web server is not running."""
        if format not in ("text", "segments"):
            raise ToolError('format must be "text" or "segments".')
        if max_chars <= 0:
            raise ToolError(f"max_chars must be positive (default {MAX_CHARS}, minimum {MIN_PAGE_CHARS}).")
        if (start is not None and start < 0) or (end is not None and end < 0):
            raise ToolError("start and end are seconds from the beginning and cannot be negative.")
        if start is not None and end is not None and start >= end:
            raise ToolError(f"start ({start:g} s) must be before end ({end:g} s).")
        check_id(transcript_id)
        summary = library.summary(transcript_id)
        if start is not None and start >= summary["duration"]:
            raise ToolError(f"start ({start:g} s) is after the end of this transcript "
                            f"({summary['duration']:.0f} s, {clock(summary['duration'])}).")
        names = library.names(transcript_id)
        source = library.segments(transcript_id)
        translations = [t["lang"] for t in list_translations(library, transcript_id)]
        if lang:
            try:
                data = load_translation(library, transcript_id, lang)
            except LLMError as exc:
                raise ToolError(exc.message) from None
            if data is None:
                raise ToolError(f"There is no '{lang}' translation of this transcript. "
                                f"Available: {', '.join(translations) or 'none'}.")
            source = translation_segments(data)
        segments = [s for s in source if (start is None or s.end > start) and (end is None or s.start < end)]
        budget = max(MIN_PAGE_CHARS, min(max_chars, 200_000))
        warnings = []
        if budget != max_chars:
            warnings.append(f"max_chars {max_chars} was changed to {budget} (allowed: {MIN_PAGE_CHARS}-200000).")
        page, used, next_start = [], 0, None
        for seg in segments:
            size = len(seg.text) + 30
            if page and used + size > budget:
                next_start = seg.start
                break
            page.append(seg)
            used += size
        body: dict[str, Any] = {
            "id": transcript_id, "title": summary["title"], "duration": clock(summary["duration"]),
            "speakers": {s: speaker_name(s, names) for s in summary["speakers"]},
            "range": {"start": page[0].start if page else start, "end": page[-1].end if page else end},
            "truncated": next_start is not None, "next_start": next_start, "url": link(transcript_id),
            "transcript_language": summary.get("language"),
            "translation_lang": lang.lower() if lang else None, "translations": translations,
        }
        if warnings:
            body["warnings"] = warnings
        if format == "text":
            body["text"] = render_text(page, names)
        else:
            body["segments"] = [{"start": s.start, "end": s.end, "speaker": speaker_of(s),
                                 "speaker_name": speaker_name(s.speaker, names), "text": s.text.strip()} for s in page]
        return body

    @mcp.tool(title="Rename speakers", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                                                   idempotentHint=True, openWorldHint=False))
    def rename_speakers(transcript_id: str, mapping: dict[str, str]) -> dict[str, Any]:
        """Give speakers display names, e.g. {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}. Keys are speaker
        ids from get_transcript; an empty name removes the name. Other names are kept."""
        check_id(transcript_id)
        known = set(speaker_stats(library.segments(transcript_id)))
        unknown = set(mapping) - known
        if unknown:
            raise ToolError(f"Unknown speaker ids: {', '.join(sorted(unknown))}. Known: {', '.join(sorted(known))}.")
        with voices.transcript_lock(library, transcript_id):  # same lock as voice identification
            names = {**library.names(transcript_id), **mapping}
            saved = library.save_names(transcript_id, {k: v for k, v in names.items() if v.strip()})
        index.reindex(transcript_id)
        return {"id": transcript_id, "speakers": {s: speaker_name(s, saved) for s in sorted(known)}}

    @mcp.resource(f"{APP_NAME}://transcripts", name="transcripts", mime_type="application/json",
                  description="All transcripts in the library (id, title, date, duration, speakers).")
    def transcripts_resource() -> str:
        return json.dumps(list_transcripts(limit=200), ensure_ascii=False)

    @mcp.resource(f"{APP_NAME}://transcripts/{{transcript_id}}", name="transcript", mime_type="text/plain",
                  description="One transcript as text with times and speaker names.")
    def transcript_resource(transcript_id: str) -> str:
        transcript_id = unquote(transcript_id)
        check_id(transcript_id)
        meta = library.metadata(transcript_id)
        text = render_text(library.segments(transcript_id), library.names(transcript_id))
        return f"# {meta.title}\n\n{text}\n"

    return mcp


def web_url_for(settings: Settings) -> str:
    host = "127.0.0.1" if settings.host in ("0.0.0.0", "::", "") else settings.host
    return f"http://{'[' + host + ']' if ':' in host else host}:{settings.port}"


def run_stdio(settings: Settings, server_url: str | None = None, local: bool = False) -> None:
    """``mgtranscriber mcp``: serve MCP over stdin/stdout."""
    from .jobs import JobQueue

    library = Library(settings.output_dir)
    index = SearchIndex(library, settings.search_file)
    library.listeners.append(index.reindex)
    url = server_url or web_url_for(settings)
    queues = []

    def make_local() -> LocalJobs:
        # own queue and store: never share data/jobs.json with a web server process
        queue = JobQueue(settings, store=settings.data_dir / "mcp-jobs.json")
        queue.on_finished.append(lambda job: job.result and index.reindex(job.result))
        queues.append(queue)
        return LocalJobs(queue)

    try:
        remote = None if local else RemoteJobs.probe(url, settings.api_key)
    except ServerRejectedKey as e:
        log.warning("%s", e)  # every transcribe call reports it too
        remote = None
    if remote:
        log.info("jobs go to the web server at %s", url)
    jobs: JobsBackend = make_local() if local else AutoJobs(url, settings.api_key, make_local)
    server = build_server(settings, library, index, jobs, allow_local_paths=True,
                          web_url=url if remote else None)
    try:
        index.sync()
        server.run("stdio")
    finally:
        for queue in queues:
            queue.stop()
        index.close()
