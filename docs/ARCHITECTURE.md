# Architecture

`mgtranscriber` (working name, see [Renaming](#renaming)) is a Python package with a
FastAPI web UI. A transcription goes **source → engine → library**. The web server
runs transcriptions through a persistent **job queue**, one at a time, each one in a
worker subprocess so that GPU memory is released when the job ends.

```
mgtranscriber/
  __init__.py        APP_NAME, __version__
  cli.py             `mgtranscriber transcribe|serve|mcp` (also `python -m mgtranscriber`)
  config.py          Settings: defaults < settings.json < .env / environment
  settings_store.py  the whitelist of settings the web UI may change (settings.json) + the HF token in .env
  diagnostics.py     system checks (GPU, ffmpeg, Deno/Node, HuggingFace, Ollama, ...); `python -m mgtranscriber.diagnostics`
  leveling.py        level(): lifts quiet passages before VAD / recognition (setting level_audio)
  models.py          Segment, Word, TranscriptionResult, speaker helpers
  progress.py        Phase names, Progress(phase, percent, message, data), ProgressCallback
  sources.py         Media; probe() and fetch_audio() for URLs (yt-dlp) and local files
  pipeline.py        TranscriptionRequest; run() = one transcription into the library
  tasks.py           job task types: "transcribe" (pipeline.run) or a registered task
  library.py         Library: transcript folders, metadata.json, speakers.json, listing
  folders.py         virtual library folders ("projects"): path normalisation, FolderStore (data/folders.json)
  search.py          SearchIndex: SQLite FTS5 full-text index over all transcripts
  mcp_server.py      MCP server (tools, resources, job backends); stdio entry point
  fsutil.py          replace_file() (atomic replace, retried while Windows locks the file)
  timefmt.py         clock / SRT / VTT time formatting
  engines/           Engine interface + registry; whisperx_local.py
  exporters/         export format registry; txt.py, srt.py, json_export.py, vtt.py, ...
  jobs/              JobQueue (persistent), runners (subprocess / in-process), worker.py
  watch/             watched folders / RSS / YouTube sources: scheduler that adds jobs (see WATCH.md)
  server/            create_app(), deps.py, errors.py, security.py (host/origin/API key),
                     mcp_http.py (/mcp), routers/ (one module per feature)
static/              web UI (no build step, no CDN): index.html, app.css, js/
tests/               pytest; mock_engine.py; GPU test behind the `gpu` marker
tools/               make_demo_data.py (invented demo library), qa_*.py (Playwright UI QA), readme_shots.py, make_icons.py
transcribe.py, server.py, start.bat, web.bat   thin compatibility wrappers
```

## Settings

`config.load_settings(root)` merges, in increasing priority: dataclass defaults,
`settings.json` in the data folder (`data/`; `MGTRANSCRIBER_SETTINGS_FILE` overrides the path; an older one in the project root is moved there at the first start), `.env`, the process environment. Environment
keys are `MGTRANSCRIBER_<FIELD>` (e.g. `MGTRANSCRIBER_OUTPUT_DIR`, `MGTRANSCRIBER_MODEL`,
`MGTRANSCRIBER_DIARIZE=false`); `HF_TOKEN` has no prefix. Unknown keys in
`settings.json` end up in `Settings.extra`, so a feature can read its own options
(`settings.extra.get("ollama_url")`) without changing `Settings`. Relative paths
resolve against the project root. `data_dir` (default `data/`) holds app state such
as `jobs.json`, `job-logs/` and `search.sqlite3`; `output_dir` (default `output/`) is
the library. Both are git-ignored.

### Settings the web can change

`settings_store.SPECS` is the whitelist (`language`, `engine`, `model`, `diarize`, `llm_provider`, `llm_model`,
`llm_language`, `llm_auto`, `voice_memory`, `watch_enabled`); `PUT /api/settings` validates all keys, writes
`settings.json` atomically (unknown keys are kept) and applies the change to the live objects: `Settings` is frozen
but shared by the queue, the watcher and the LLM layer, so the shared object itself is updated: all fields and a modified
copy of `extra` go in with ONE `__dict__.update` call (atomic under the GIL), so a reader never sees a new engine with the old
model. A key set by a `MGTRANSCRIBER_*` variable is `locked`. **Secrets are never editable through
the web**, with one deliberate exception, the HuggingFace token (`set_hf_token`) and, under the same rules, the cloud API keys (`set_api_key`, write-only; readers look them up in the environment, then `.env`, on every use): the token is validated with HuggingFace,
written to `.env` (other lines, line endings and permissions kept; a non-UTF-8 file is refused), accepted only while the server
listens on loopback and the request comes from a loopback address, saved without verification when HuggingFace cannot be
asked (offline mode), and never returned. A new feature that adds
a user-facing option adds one `SPECS` entry plus its label/help keys in `static/js/i18n/system.js`.

`diagnostics.run_checks()` returns `{id, group, status, code, params, detail}` per check; the UI translates
`system.items.<id>.<code>`. Add a check: a function returning `_check(...)`, one line in `run_checks`, texts in
`i18n/system.js`. Slow results (HuggingFace) are cached for minutes; checks run in parallel with timeouts.

## Data model

Segments are the JSON shape WhisperX writes (and old `prepis.json` files contain):

```json
{"start": 6.6, "end": 12.1, "text": " Hello ...", "speaker": "SPEAKER_00",
 "words": [{"word": "Hello", "start": 6.6, "end": 6.9, "score": 0.88, "speaker": "SPEAKER_00"}]}
```

`speaker` is optional (no diarization); the API reports a missing speaker as
`"UNKNOWN"` (`models.UNKNOWN_SPEAKER`). Word `start`/`end` can be missing for tokens
the aligner could not place (numbers, symbols).

## Library

One folder per transcript in `output_dir`; the folder name is the transcript **id**:
`library.transcript_id_for(title, source_key)` = ASCII slug of the title + 8 hex chars
of a hash of the source (`sources.source_key()`: extractor + video id for URLs, the
absolute path for files). Two sources with the same title never share a folder;
transcribing the *same* source again reuses its folder. `create()` first looks the
source up by `metadata.json` (`source_key`, or for older folders the stored video
id/extractor or file path), so a renamed video or a folder from an older version
named after the title is reused rather than duplicated. Folders from older versions
keep working unchanged. The title shown everywhere comes from `metadata.json`.

**Derived files.** Features keep per-transcript files computed from `prepis.json`
(`speakers.json`, `voices.json`, `analysis.json`, `translations/<lang>.json` (strategy 2), ...). A new run of the same source
changes the segments and speaker labels, so each such file uses one of two strategies:

1. **Delete on re-transcription.** List the file in `library.STALE_ON_RETRANSCRIBE`;
   `commit_run()` removes it (and the old `audio.*`) once the new run has succeeded, never
   before: a cancelled or failed run leaves the old transcript as it was. Use this for data
   that is cheap to recompute or meaningless for new speakers (speaker names,
   voiceprint matches). The list lives in `library.py` because the pipeline runs in the
   job worker process, where feature modules may not be imported.
2. **Mark as outdated.** Store `Library.segments_stamp(id)` (sha1 of `prepis.json`) in
   the file and compare it when reading. The feature then shows "outdated, run again"
   instead of losing an expensive result (the LLM analysis does this). This also
   catches edits of the text, not only re-transcription.

| file | content |
|---|---|
| `prepis.json` | segments; the source of truth. A folder counts as a transcript iff this exists |
| `metadata.json` | `title, source, date (ISO 8601), duration, language, engine, model, speakers[], extra{}` |
| `speakers.json` | `{"SPEAKER_00": "Alice"}` display names set in the UI |
| `prepis.txt`, `titulky.srt` | written by exporters with `stored_as` at the end of a run |
| `audio.*` | downloaded audio (URL sources), streamed to the web player |
| `versions/` | earlier `prepis.json` states kept by the editor and before a re-transcription ([EDITOR.md](EDITOR.md)) |
| `audit.jsonl` | append-only, hash-chained log of what was done with the transcript ([PRIVACY.md](PRIVACY.md)) |

`versions/` and `audit.jsonl` are **not** derived data: never list them in
`STALE_ON_RETRANSCRIBE`. Words may carry `"redacted": true` (set in the editor); exports apply it
only with `?redact=true` (`privacy.redact_segments`). The editor saves through
`save_segments()` like everything else, so the listeners and the stamps of derived files keep
working.

Folders without `metadata.json` (from the original tool) get metadata derived from
`prepis.json` and the file's mtime. Features that need to store per-transcript data
should either put small values in `Metadata.extra` (unknown keys in `metadata.json`
are preserved there) or add their own file in the folder (e.g. `summary.json`) and
read it through a `Library` method. `Library.summary()` is cached by the mtimes of
`prepis.json` and `metadata.json`.

`Library.listeners` are called with the transcript id after `save_segments`,
`save_names` or `save_metadata` (the search index subscribes; a failing listener is
logged and never fails the save). A future transcript editor only has to save through
the library and search stays current. Writes go to a temp file and are moved into place
with `fsutil.replace_file`.

## Library folders (projects)

A transcript's **virtual folder** is the path string `Metadata.folder` (`"Podcasts/Cílek"`, `""` = unfiled) in its
`metadata.json`; nothing moves on disk, so ids, links, search, voices and MCP are unaffected. `folders.py` holds the
pure rules (`normalize_path`: trimmed names, `/` separators, at most 5 levels, 60 characters per name, no control
characters; `in_folder`) and `FolderStore(library, data/folders.json)`, which persists the folders the user created
(so that empty ones exist; written atomically under a lock, and re-read when another process changed the file). The
tree is that list plus every folder (and ancestor) a transcript is in, so a transcript is never in an invisible
folder. `FolderStore.move/delete/assign` change each affected transcript through `Library.set_folder` (the metadata
save runs `Library.listeners`) under the editor's per-transcript lock, so a concurrent text edit cannot undo a move.
Deleting a folder removes its subfolders and moves their transcripts to the parent or to unfiled; transcripts are
never deleted. `Library.folder_of()` / `folders_map()` read only `metadata.json` (cached by mtime) so listing and
counting stay cheap.

A re-transcription keeps the folder (`pipeline.run` reads it under the lock) unless `TranscriptionRequest.folder` names
one. The folder is **not** in the search index: `SearchIndex.search(..., transcript_ids=...)` takes an id filter, the
API/MCP resolve a folder to ids per request (passed as one JSON array, no SQL variable limit). That way moving 500
transcripts never rewrites 500 indexes of segments.

## Search

`search.SearchIndex(library, settings.search_file)`: one row per segment in a regular
table plus an external-content FTS5 table (`unicode61 remove_diacritics 2`). The index is
derived data and can be deleted at any time.

- `reindex(id)` rewrites one transcript, or removes it when it is gone.
- `sync()` compares file mtimes (prepis.json, speakers.json, metadata.json) and
  reindexes what changed. The server runs it at start; searches run it at most every 5 s.
- `search(q, limit, transcript_id, offset)`: user input goes through `fts_query()`.
  Words become quoted prefix terms and `"..."` stays a phrase, so FTS5 syntax can never
  break a query. Snippet highlights come back as character ranges, not HTML.

In the server the queue's `on_finished` hook and `Library.listeners` call `reindex()`.
API: `GET /api/search` (see [API.md](API.md)).

## Engines

```python
class Engine(ABC):
    name: ClassVar[str]
    needs_audio: ClassVar[bool] = True       # False: works from the URL (e.g. YouTube captions)
    supports_diarization: ClassVar[bool] = False
    default_model: ClassVar[str | None] = None
    def __init__(self, settings: Settings): ...
    def check(self, options: TranscribeOptions) -> None: ...   # fail fast, raise EngineError
    def transcribe(self, media: Media, options: TranscribeOptions,
                   progress: ProgressCallback) -> TranscriptionResult: ...
```

`TranscribeOptions(language|None=auto, model, speakers, diarize, batch_size, extra)`.
`Media(source, kind "url"|"file", title, audio_path, info, workdir)`; when `needs_audio` is
true the pipeline has already put the audio in `media.audio_path`. An engine that needs
audio only sometimes (YouTube captions + speakers, or its fallback) sets `needs_audio = False`
and calls `sources.fetch_audio(media, media.workdir, progress)` itself.
Return `TranscriptionResult(segments, language, duration, model, speaker_embeddings)`.

| engine | module | how |
|---|---|---|
| `whisperx` | `whisperx_local.py` | faster-whisper → wav2vec2 align → pyannote |
| `parakeet`, `canary` | `onnx_asr_local.py` | onnx-asr (ONNX Runtime CUDA) + Silero VAD (≤ 20 s pieces); Parakeet times words itself, Canary goes through align; then pyannote |
| `youtube` | `youtube_captions.py` | json3 captions via yt-dlp (manual → `<lang>-orig`); optional pyannote on the downloaded audio; fallback engine without captions |
| `elevenlabs`, `deepgram` | `cloud.py` | upload the audio file (httpx), words + speakers from the reply |
| `gemini` | `gemini.py` | local VAD pieces → Gemini text per piece (JSON) → local align + pyannote |

`engines/common.py` holds the shared steps: `env_value()` (API keys: environment, then `.env`),
`load_audio()` (decodes with ffmpeg and, with setting `level_audio`, levels the loudness, see below), `align()` (falls back to spreading words when there is no wav2vec2 model for the
language), `diarize(..., voices=dict)` (pyannote + word assignment; fills raw voice embeddings),
`split_sentences()` (timed words → segments at sentence ends, speaker changes, max length).

**Loudness leveling** (`leveling.py`, setting `level_audio`, default on; `MGTRANSCRIBER_LEVEL_AUDIO=false` or
`settings.json` turns it off). VADs have absolute thresholds, so a passage far quieter than the rest of a recording
(FLEURS en: 50 dB, peak 0.002) between loud ones was skipped by every engine. A global peak normalization cannot fix
that, so `level()` works locally: per 2 s window (hop 0.5 s) the speech level is the 90th percentile of the 25 ms
frame RMS, the gain is the distance to −23 dBFS (only upwards, at most +48 dB, not when the window is already within
12 dB of the target, never past 0.97 peak) and is interpolated between window centres. Windows without speech
(below −80 dBFS or with a flat envelope: silence, stationary noise) get no gain of their own and take the one of the
nearest speech, so noise is never pumped up by itself and exact silence stays silent. Loud audio is returned
unchanged (the same array). `common.load_audio_pair()` returns `(audio, original)`; whisperx, parakeet, canary
and gemini use the levelled `audio` for the VAD, recognition and alignment and the **original** for pyannote
(same length, so word times fit both). Reason: leveling evens out the loudness of the speakers, which the
diarization uses as a cue. On AMI it raised the speaker confusion from 2.1 to 9.4 % in one meeting and added a
speaker in another (DER +4 points for both engines); with the original audio the DER is back within one point.
The youtube engine diarizes the unchanged audio. The audio file in the library is not modified, the cloud
engines upload the original file, and voice-memory embeddings (`voices.py`) come from the original audio.

**Add an engine:** create `mgtranscriber/engines/<name>.py` with an `Engine` subclass and
register it in `engines/__init__.py`:

```python
register_engine("myengine", ".myengine:MyEngine", "My engine", description="...",
                cloud=False, models=("default-model", "other"), languages=(), diarization=True,
                needs_language=False, env=("MY_API_KEY",), packages=("my_sdk",))
```

The keyword details go to `GET /api/info` for the new-job form: `models` fill the model select,
`cloud=True` shows the "audio leaves the computer" warning, and `env`/`packages` are checked
without importing the engine; a missing one is reported as `missing: {code, detail}` and the
engine is shown disabled with the reason. `available=False` marks a planned engine.
Imports are lazy (only `get_engine()` imports the module), so heavy dependencies
belong inside the module, never at the top of `engines/__init__.py`. Out-of-tree engines: `MGTRANSCRIBER_ENGINES="name=package.module:Class;..."`
(the tests use this for `tests.mock_engine:MockEngine`). Report progress with
`progress(Progress(Phase.TRANSCRIBE, percent, message))`; `message` becomes a log line,
`percent` drives the progress bar. Use the phases in `progress.Phase` where they fit;
the UI maps phases to steps in `static/js/components/jobs.js` (`STEPS`).

## Pipeline

`pipeline.run(request, settings, progress, library) -> transcript_id`:
engine → `check()` → `sources.probe()` (emits `data={"title"}`) → `library.begin_run()` (a staging folder `.run-*` in the transcript folder) →
`fetch_audio()` into the stage → `engine.transcribe()` → under the editor lock: backup of the old
`prepis.json`, `commit_run()` (the old audio and stale files move to `stage/old`, the new audio moves in;
nothing is deleted yet), save segments, stored exports, metadata → `finish_run()` drops the stage. Any error
or cancellation ends in `abort_run()`, which puts the old files back and removes the stage, so the old
transcript is byte-identical to before. A killed worker leaves its stage: the next run of the source restores
an interrupted commit (after 10 min of quiet) and sweeps stages untouched for 24 h; progress reports touch
the stage so a long run is never mistaken for a dead one. Secure delete refuses (409) while a run is active.
The old audio stays when the new run staged none (captions only). The run ends with the final progress event with `data={"result": id}`. It is used by the CLI, the job worker and
tests alike. Post-processing features (summaries, voice memory) should run *after*
`run()` returns — e.g. as a `JobQueue.on_finished` hook or their own job type — rather
than being wedged into the pipeline.

## Exporters

```python
Exporter(name, label, extension, media_type, render, stored_as=None, applies_names=True)
render(segments: list[Segment], names: dict[str, str]) -> str | bytes
```

**Add a format:** `mgtranscriber/exporters/vtt.py` calling `register(Exporter("vtt", "VTT",
"vtt", "text/vtt; charset=utf-8", render))`, then add it to the import line at the bottom
of `exporters/__init__.py` (import order = button order). It automatically appears in
`GET /api/info`, as a download button in the detail page, and at
`GET /api/transcripts/{id}/export/vtt?names=true`. Set `stored_as` only if the file
should also be written into the folder after each transcription. A document format that needs
the title, source or date sets `takes_context=True` and gets
`render(segments, names, ExportContext(title, source, date, duration, language, redacted, options))`;
call exporters through `exporters.render(exporter, segments, names, context)` (DOCX does this).

**Offline mode** (`privacy.offline(settings)`, [PRIVACY.md](PRIVACY.md)): a new engine with
`cloud=True` (or one listed in `privacy.NETWORK_ENGINES`) and a new LLM provider with
`"cloud": True` are refused automatically; a feature that contacts the network itself must call
`privacy.check_source()` or check `privacy.offline()`.

## Jobs

`JobQueue(settings, runner=SubprocessRunner())` runs one job at a time and persists
the job list to `data/jobs.json` (small, no logs); each job's log is appended to
`data/job-logs/<id>.log` and the last 2000 lines are kept in memory. A failed save is
logged and retried on the next change, it never stops the queue. After a restart,
queued jobs stay queued. `stop()` (server shutdown) kills the running worker and
queues the job again; a job that was running when the server *died* is queued again
unless that already happened `MAX_ATTEMPTS` (2) times, then it becomes `interrupted`.
On Windows all workers belong to a kill-on-close Job Object (`jobs/winjob.py`), so
they die with the server process and never hold the GPU after a crash. History keeps
the last 100 finished jobs.

`job.done` is a `threading.Event` set when the job finished (`queue.wait(job, timeout)`).
`queue.on_finished` hooks run one at a time, in order, on a daemon thread after that: they never
delay the next job, and a slow hook (an HTTP call) does not hold up shutdown.

**Task types.** `TranscriptionRequest.task` is `"transcribe"` by default. Other GPU work
on an existing transcript (e.g. voiceprints) uses the same queue, so it never competes
with a transcription for the GPU:

```python
# mgtranscriber/tasks.py, under "Built-in tasks" (the worker process imports this module)
register_task("voiceprints", "mgtranscriber.voices:compute_task", "Voiceprints")
# fn(request, settings, progress, library) -> transcript_id; request.source = transcript id
queue.add(TranscriptionRequest(transcript_id, task="voiceprints"))
```

Both runners call `tasks.execute()`. Tests register extra tasks via
`MGTRANSCRIBER_TASKS="name=module:function"`.

The worker (`python -m mgtranscriber.jobs.worker`, request + settings as JSON on stdin)
writes **only JSON events to stdout**; everything libraries print goes to stderr and
becomes the (filtered) raw log:

```
{"event": "progress", "phase": "align", "percent": 50.0, "message": null, "data": null}
{"event": "done", "result": "<transcript id>"}
{"event": "error", "message": "..."}
```

Job JSON (`GET /api/jobs`, SSE `state` events): `id, source, url (=source), language,
speakers, no_diarize, engine, model, status (queued|running|done|error|cancelled|interrupted),
phase, percent, title, result, error, created, started, finished, position, log_lines`.
`InProcessRunner` runs the pipeline in a thread (tests, debugging; cancel is cooperative).

## Server

`server.create_app(settings=None, queue=None, runner=None)` builds the app; shared
services are on `app.state` and injected with `deps.SettingsDep`, `LibraryDep`,
`QueueDep`. Errors that the UI should translate are raised as
`ApiError(status, code, english_message)` → `{"detail": ..., "code": ...}`; the
frontend shows `errors.<code>` from its dictionary when present.

`server/security.py` (pure ASGI middleware, applies to every route) answers only to
allowed `Host` headers (setting `allowed_hosts`; default: loopback names, plus this
computer's names and addresses when bound to `0.0.0.0`) against DNS rebinding, and
rejects state-changing requests (POST/PUT/PATCH/DELETE) whose `Origin`/`Sec-Fetch-Site`
shows another website. `allow_local_paths` (`auto` = only when bound to loopback)
controls whether API clients may transcribe files from the server's disk. With
`api_key` set it also requires `Authorization: Bearer`, `X-API-Key` or the web UI's
session cookie (`routers/auth.py`; random token checked against `security.Sessions`,
which expire and end on logout) on `/api` (except `/api/auth/*`), `/v1` and `/mcp`.
`server/limits.py` answers bodies over `max_upload_mb` with 413, and URLs from clients go
through `netguard.require_public_url` when the server is not on loopback (SSRF). `ApiError` and these errors render as `{"detail", "code"}`, or in
OpenAI's `{"error": {...}}` shape under `/v1` (`errors.error_body`).

Besides `/api`, the server hosts:

- `routers/openai_compat.py`: `/v1/audio/transcriptions` and `/v1/models`. Uploads
  become queue jobs and the request awaits `job.done`.
- `mcp_http.py`: the MCP server's Streamable HTTP endpoint at `/mcp`, stateless, run
  inside the app's lifespan.

Both are documented for users in [API.md](API.md) and [MCP.md](MCP.md). The MCP tools
live in `mcp_server.build_server(settings, library, index, jobs_backend, ...)`, which
is independent of the transport:

- `LocalJobs(queue)` is used inside the web server, or for stdio without a server.
- `RemoteJobs(url)` is used by `mgtranscriber mcp` when a web server answers, so the two
  processes never share a queue.

Add an MCP tool there with `@mcp.tool()` and a typed return (`dict[str, Any]` gives
structured output).

Endpoints: `GET /api/info` · `GET|POST /api/jobs` · `POST /api/jobs/upload` (multipart; `server/uploads.py` is shared
with `/v1`: streaming to `data/uploads/`, name sanitising, keeping the file as `audio.*`; a normal persistent job whose
file is kept or deleted by a queue hook, `finish_job`) · `GET|DELETE /api/jobs/{id}` ·
`GET /api/jobs/{id}/events` (SSE: `log`, `state`, `end`) · `GET /api/jobs/summary` ·
`POST /api/jobs/{id}/retry` · `DELETE /api/jobs/{id}/history` · `POST /api/jobs/clear` (`JobQueue.page/summary/remove/clear`;
`GET /api/jobs` pages with `limit`/`offset`/`status`, a plain list without) · `GET /api/transcripts` ·
`GET /api/transcripts/{id}` · `PUT /api/transcripts/{id}/speakers` · `/api/folders` (+ `PUT /api/transcripts/{id}/folder`) ·
`GET /api/transcripts/{id}/export/{fmt}` · `GET /api/transcripts/{id}/files/{name}`
(audio, plus old `prepis.txt`/`titulky.srt`/`prepis.json` links) · `GET /api/search` ·
`/api/auth/{status,login,logout}` · `/api/recordings` (below). OpenAPI docs at `/api/docs`.

**Recordings.** `static/js/components/recorder.js` records with MediaRecorder (5 s timeslice, webm/Opus; Safari
mp4/AAC) and PUTs every chunk to `/api/recordings/{id}/chunks/{n}` while recording; its state is a module singleton,
so it survives navigation inside the app. `server/recordings.py` keeps a session as `data/recordings/<id>/`
(`session.json` + `NNNNNN.part`; ids validated as 32 hex chars before any path is built). `finish` concatenates the
chunks, remuxes with ffmpeg when installed, hands the file to `uploads.adopt_file()` (so it is an ordinary upload
from there: `finish_job` keeps it as `audio.*`, `sweep` clears leftovers) and queues the job; unfinished sessions are
listed for recovery and swept after 7 days. The whole recording counts against `max_upload_mb`. User docs:
[USER_GUIDE.md](USER_GUIDE.md) ("Recording"), [API.md](API.md).

**Old server behind a new page.** The frontend is served from disk, so an update without a restart leaves the old
Python code behind new JavaScript. `server/build.py` fingerprints the package's `.py` files at process start;
`GET /api/info` returns `build` and `restart_needed` (the files on disk differ from that fingerprint, re-checked at
most every 5 s). `static/js/components/stale.js` shows a banner when `restart_needed` is true, when `build` is missing
(a server older than this feature) or when `api()` gets a bare 404/422 (no error `code`) on a GET of `/api/`. Every
error the server raises itself carries a `code`, so only an unknown endpoint triggers the last rule.

**Add a router:** create `mgtranscriber/server/routers/<feature>.py`:

```python
from fastapi import APIRouter
from ..deps import LibraryDep

router = APIRouter(prefix="/api/<feature>", tags=["<feature>"])

@router.get("")
def overview(library: LibraryDep): ...
```

and add one line to `ROUTERS` in `server/routers/__init__.py`. Non-`/api` prefixes are
fine too (see `openai_compat.py`); the static frontend is mounted last at `/`, so
routers always win. Long-lived components belong in `create_app` and its lifespan.

## Frontend

Plain ES modules, no build step, no CDN (works offline).

```
static/index.html          shell: top bar (brand, #nav, language select), <main id="view">, #toast
static/js/main.js          imports pages, sets up the language select, starts the router
static/js/core/router.js   registerPage(), href(path, query), navigate(), isCurrent()
static/js/core/i18n.js     t(), applyI18n(), setLang(), onLangChange(), LANGUAGES
static/js/core/api.js      api() (JSON fetch, translated errors), getInfo() (cached /api/info)
static/js/core/util.js     $, esc, time/date formatting, toast, copyText, speaker helpers
static/js/i18n/{en,cs}.js  dictionaries; i18n/system.js adds the Settings / first-run texts via extendLanguages()
static/extra.css           styles of the Settings page and the first-run guide, plus responsive/accessibility fixes
static/js/pages/*.js       one module per page (home, jobs, detail, voices, voice_person, watch, system = Settings)
static/js/core/jobsummary.js   one shared poll of /api/jobs/summary (home strip, nav badge); nav `badge: true` + components/navbadge.js
static/js/components/*.js  pieces used by pages (job cards, library tiles + search, login, status icons and the token form, the first-run guide `setup.js`)
```

**Add a page:** create `static/js/pages/search.js`:

```js
import { registerPage } from "../core/router.js";
import { applyI18n, t } from "../core/i18n.js";

registerPage({
  id: "search",
  path: "/search",                          // params: "/t/:id"
  nav: { label: "nav.search", order: 20 },  // optional top-bar link
  title: () => t("search.title"),
  mount(el, params) {
    el.innerHTML = `<h1 data-i18n="search.title"></h1>`;
    applyI18n(el);
    return () => {};                        // optional cleanup (timers, EventSource, audio)
  },
});
```

then `import "./pages/search.js";` in `main.js` and add the keys to both dictionaries.
Routes are hash based (`#/search`). A query string after the path is passed to
`mount(el, params, query)` as an object; e.g. search hits link to `#/t/<id>?t=385.8`,
and the detail page scrolls to that segment, highlights it and seeks the player there.
When the server has an API key, `api()` opens the login dialog (`components/login.js`)
on `401 auth_required`. The top-bar nav appears once there are two or more
pages with `nav`. Changing the language re-mounts the current page, so render all text
with `t()`/`data-i18n` at mount time. After an `await`, check `isCurrent(id, params)`
before touching the DOM.

## i18n

Dictionaries are nested objects; keys are dotted (`t("jobs.status.done")`). Plurals use
`Intl.PluralRules` categories: `{ one, few, many, other }` selected by `params.n`
(Czech needs `one/few/other`). Numbers in params are locale-formatted. Missing keys fall
back to English, then to the key itself. In HTML templates use `data-i18n`,
`data-i18n-placeholder`, `data-i18n-title`, `data-i18n-aria-label` and call
`applyI18n(el)`. The language is chosen from `localStorage.lang`, then the browser
(`cs`/`sk` → Czech, otherwise English). Add a language: `static/js/i18n/<code>.js` and
one entry in `LANGUAGES` in `core/i18n.js`. Backend messages stay English; anything the
user should see translated needs an error `code` or a frontend key.

## Tests

```
uv run pytest            # fast unit tests, no GPU (mock engine, TestClient, subprocess worker)
uv run pytest -m gpu     # + real WhisperX on a 40 s clip (tiny model; MGTRANSCRIBER_TEST_MODEL=small to change)
```

The GPU test cuts `tests/fixtures/clip.wav` from a local transcript's audio with
ffmpeg (`uv run python -m tests.make_clip`); the clip is third-party content and is
git-ignored. Fixtures in `tests/conftest.py`: `settings` (temp output/data dirs, engine
`mock`), `library`, `legacy_transcript` (folder without metadata.json), `media_file`
(dummy input; a name containing `fail`/`slow`/`silent` changes the mock engine's
behavior), `live_server(**overrides)` (the real app under uvicorn on a free port, for the
`openai` and MCP SDK clients). `test_mcp.py` drives the MCP server in-process, as a
`mgtranscriber mcp` subprocess over stdio, and over Streamable HTTP; `test_openai.py` uses
the official `openai` client.

## Renaming

The product name is `APP_NAME` in `mgtranscriber/__init__.py` (CLI prog name, env prefix,
API info, FastAPI title). Package-internal imports are relative and the worker module
path is derived from `__package__`, so renaming means: rename the directory, update
`[project] name`, `[project.scripts]` and `[tool.hatch.build.targets.wheel] packages`
in `pyproject.toml`, the imports in `transcribe.py`/`server.py`/`tests/`, and the
UI name in `static/js/i18n/*.js` (`app.name`).
