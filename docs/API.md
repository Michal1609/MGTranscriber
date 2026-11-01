# HTTP API

`mgtranscriber serve` (or `web.bat`) serves three APIs on the same port (default
`http://127.0.0.1:8000`):

| prefix | for | format |
|---|---|---|
| `/api/...` | the web UI and scripts | JSON, errors `{"detail": "...", "code": "..."}` |
| `/v1/...` | OpenAI-compatible clients | OpenAI's formats, errors `{"error": {"message", "type", "param", "code"}}` |
| `/mcp` | AI assistants (Model Context Protocol) | see [MCP.md](MCP.md) |

Interactive OpenAPI docs: `/api/docs`. The architecture behind it is in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Access

By default the server listens on `127.0.0.1` only and needs no key. To reach it from
other computers, set an API key and listen on all interfaces:

```
# .env
MGTRANSCRIBER_API_KEY=some-long-random-string
MGTRANSCRIBER_HOST=0.0.0.0
```

`serve` prints a warning when the host is not loopback and no key is set.

With a key, every `/api` (except `/api/auth/*`), `/v1` and `/mcp` request needs one of:

- `Authorization: Bearer <key>` (OpenAI clients send this with `api_key=<key>`),
- `X-API-Key: <key>`,
- the web UI's session cookie. The UI asks for the key once and `POST /api/auth/login`
  turns it into an HttpOnly, SameSite=Strict cookie with a random session token (never
  the key). The server keeps an HMAC of each token in `data/sessions.json`; a session
  lasts 30 days, `POST /api/auth/logout` ends it on the server, and changing the API
  key ends all of them.

Missing or wrong key: `401` with `code: "auth_required"` and `WWW-Authenticate: Bearer`.

**Use a long random key** (for example `python -c "import secrets; print(secrets.token_urlsafe(32))"`),
not a word. Wrong keys, in a header or in the login form, are counted per client address:
from the third miss on that address gets `429` (`code: "too_many_attempts"`, `Retry-After`)
for 2, 4, 8 ... up to 300 seconds, even with the right key. Other addresses are not affected.
Behind a reverse proxy every client may share the proxy's address, so one guesser can pause
all of them; the long key is the real protection.
The static frontend stays public, so the UI can show its login form.

Always on, key or not:

- **Host check.** Requests whose `Host` header is not allowed get `400 invalid_host`.
  This blocks DNS rebinding. Allowed by default: loopback names, plus this computer's
  name and addresses when listening on `0.0.0.0`. Override with
  `MGTRANSCRIBER_ALLOWED_HOSTS=host1,host2`, or `*` for any host.
- **Origin check.** POST/PUT/PATCH/DELETE from a page of another origin get
  `403 cross_origin`. The same goes for GET/HEAD under `/api`, `/v1` and `/mcp` (a foreign page's
  `<img src="/api/...">` must not trigger exports or audio processing): `Sec-Fetch-Site: cross-site` or an `Origin`
  that is not this server is refused. Clients that send neither header (curl, SDKs) are not affected.
- **Browser headers.** Every response has `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer` and a `Content-Security-Policy` (own origin only, `frame-ancestors 'none'`), so the
  web UI cannot be framed by another page. `/api/docs` (Swagger UI) loads its script and style from a CDN
  (jsdelivr) and is the only page that does; see [PRIVACY.md](PRIVACY.md).
- **No server paths.** An upload is shown as `uploads/<name>` (`url`, `source`; `upload: true`), never the absolute
  path. `GET /api/system` (`storage`) and `GET /api/voices` (`file`) show the full path only to a client on this
  computer, others get the folder or file name. Validation errors (422) do not echo the rejected values.
- **Local paths.** Transcribing files from the server's disk (`source` that is not a
  URL) is allowed only when the server listens on loopback.
  Override with `MGTRANSCRIBER_ALLOW_LOCAL_PATHS=true|false`. Otherwise the request
  gets `403 local_paths_disabled`. Uploads through `/v1` are not affected.
  **Network (UNC) paths are refused** (`403 unc_path_disabled`) in every spelling (`\\server\share`,
  `//server/share`, `\\?\UNC\...`, mixed separators), before the server touches them: opening one makes Windows
  send the user's login credentials to that host. `\\?\C:\...` (a long local path) is fine.
  `MGTRANSCRIBER_ALLOW_UNC_PATHS=true` allows them (for a NAS you trust). The same applies to the MCP `transcribe`
  tool and to watch folders.
- **URLs on the local network (SSRF).** yt-dlp fetches whatever URL a job names. When
  the server listens on another address than loopback, `POST /api/jobs` and the MCP
  `transcribe` tool refuse URLs whose host resolves to a loopback, private, link-local
  or otherwise non-public address (`403 blocked_address`), so clients cannot use the
  server to reach your router, NAS or other internal services. The check runs once
  before the job starts; redirects followed by yt-dlp and DNS rebinding are not
  covered, so keep an API key set as well. On loopback there is no check: the client
  is you, on this computer.
- **Request size.** Request bodies (uploads to `/v1` and `/api/jobs/upload`) larger than
  `MGTRANSCRIBER_MAX_UPLOAD_MB` (default 4096) get `413 request_too_large`, checked
  while the upload arrives. An upload briefly takes up to twice its size on disk: the
  web framework's temporary file plus the copy in `data/uploads/`.

| endpoint | |
|---|---|
| `GET /api/auth/status` | `{"required": bool, "authenticated": bool}` |
| `POST /api/auth/login` `{"key": "..."}` | sets the session cookie; `401 invalid_key` on a wrong key (answered after a 1 s delay; logins are checked one at a time, so parallel guesses are no faster) |
| `POST /api/auth/logout` | ends the session and clears the cookie |

**Behind a reverse proxy with TLS** (Caddy, nginx, a tunnel): the cookie gets the
`Secure` flag when the request arrived over HTTPS. uvicorn learns that from the
proxy's `X-Forwarded-Proto` header, which it trusts only from loopback by default.
A proxy on the same computer works as is; for a proxy on another machine set
`FORWARDED_ALLOW_IPS=<proxy address>` in the environment of `mgtranscriber serve`.
Without that the cookie works but is not marked `Secure`.

## Full-text search

`GET /api/search?q=<text>&limit=20&offset=0&transcript=<id>&folder=<path>&recursive=true`

Searches every transcript in the library (SQLite FTS5 in `data/search.sqlite3`).

- Case and diacritics are ignored: `prepis` finds "přepis".
- Words are AND-ed and prefix-matched.
- `"quoted words"` is an exact phrase.
- Other FTS5 syntax is treated as plain text, so odd input never fails.
- `folder` limits the search to a [library folder](#folders) (`folder=` empty = unfiled); `recursive` (default
  `true`) includes its subfolders. The folder is resolved to transcript ids per request and applied as a filter,
  so moving transcripts never reindexes them. An unknown folder simply finds nothing.

```json
{
  "query": "safe AI",
  "total": 30,
  "results": [
    {
      "transcript_id": "the-ai-safety-expert-1a2b3c4d",
      "title": "The AI Safety Expert ...",
      "segment": 141,
      "start": 385.84, "end": 387.1,
      "speaker": "SPEAKER_00",
      "speaker_name": "Roman",
      "text": "We can make safe AI.",
      "snippet": "We can make safe AI.",
      "highlights": [[12, 16], [17, 19]]
    }
  ]
}
```

- Results are best match first; `total` counts all matching segments.
- `speaker_name` is the display name from `speakers.json`, or `null`.
- `highlights` are `[start, end)` offsets into `snippet` in characters (Unicode code
  points, as Python counts). JavaScript strings count UTF-16 units, so slice
  `Array.from(snippet)` there, or emoji and similar characters shift the ranges.
- The web UI links each hit to `#/t/<id>?t=<seconds>`. The detail page scrolls to that
  segment, highlights it and sets the player there.

The index updates itself:

- when a job finishes;
- when speaker names, segments or metadata are saved through the server, or a
  transcript is deleted (`Library.changed(id)`); hits of a folder deleted by hand are
  dropped at the next search;
- at server start (it is built if missing);
- before a search, at most every 5 s, by checking file timestamps. This picks up
  transcripts made by the CLI.

Deleting the file is safe; it is rebuilt.

## Translations

A translation is a stored language version of a transcript (`translations/<lang>.json`), made by a queued job
(task `translate`, same queue as transcriptions). Segments keep `start`, `end` and `speaker`; only `text` is translated
and there are no `words`.

| endpoint | |
|---|---|
| `GET /api/transcripts/{id}/translations` | `{source_lang, translations: [{lang, source_lang, provider, model, created, stale, segments, untranslated}], jobs: [active translate jobs], languages}` |
| `GET /api/transcripts/{id}/translations/{lang}` | the translation: `segments`, `stale` (the original changed since), `segments_stamp`, `untranslated` (segment indexes that kept the original text) |
| `POST /api/transcripts/{id}/translations` | `{"language": "cs", "provider"?, "model"?}` → `202` with the job. Provider errors (`llm_unreachable`, `llm_model_missing`, `llm_missing_key`, offline mode) are returned at once; `409 translation_same_language`; an identical waiting job is returned instead of a new one |
| `DELETE /api/transcripts/{id}/translations/{lang}` | deletes it and cancels a waiting job for that language |
| `GET /api/transcripts/{id}/export/{fmt}?lang=cs` | `txt`, `srt`, `vtt`, `md`, `docx`, `json` of the translation (`404 translation_not_found`; `409 translation_not_redactable` together with `redact=true`) |

Search hits (`GET /api/search`) have a `lang` field: `""` for the original, else the translation's language. Job fields
for a translate job: `task: "translate"`, `source` = transcript id, `language` = target, `engine` = provider,
`model` = model.

**Translate on submit.** `POST /api/jobs`, `POST /api/jobs/upload` (form field) and `POST /api/recordings/{id}/finish`
take an optional `translate_to` (`cs`, `de`, `pt-BR` ...; one of the languages of `GET /api/info` → `translate_languages`,
a regional code is stored as its base language). When the transcription finishes, a `translate` job is queued exactly as
`POST /api/transcripts/{id}/translations` does (same provider check, so offline mode applies). An unsupported value is
refused at once with `400 invalid_language`. The job JSON carries `translate_to`; it survives a server restart and a retry.
Nothing fails afterwards: if the transcript is already in that language nothing is queued, and if the translation cannot
be queued (no LLM provider, offline mode ...) the transcription stays done and the reason is stored on the job as
`translation: {language, error}` (`{language, job}` when it was queued; also a line in its log,
`[translation to cs not queued: ...]`). The response that creates the job already carries `translation_warning` (the
reason) when the provider check fails at that moment. Watched sources use their own `translate_to` and are not affected.
MCP `get_transcript` takes an optional `lang` and lists `translations`; `search_transcripts` hits carry `lang`.

## Transcripts and jobs

| endpoint | |
|---|---|
| `GET /api/info` | name, version, defaults, engines, export formats |
| `GET /api/transcripts[?folder=<path>&recursive=true]` | library, newest first; every item has `folder`. With `folder`: only that [folder](#folders) (empty = unfiled), `recursive` (default `true`, as in search and MCP; `false` = only that folder) includes its subfolders. `limit` / `offset` return one page (still an array; the total is in the `X-Total-Count` header), without them the whole list |
| `GET /api/transcripts/{id}` | summary + `metadata` + `segments` |
| `PUT /api/transcripts/{id}/speakers` | `{"SPEAKER_00": "Alice"}` display names |
| `GET /api/transcripts/{id}/export/{fmt}?names=true` | `txt`, `srt`, `vtt`, `json`, `md`, `docx` (see `/api/info`); `redact=true` = redacted version, `layout=table` = DOCX as a table |
| `GET` · `PUT /api/transcripts/{id}/segments` | editor: segments with words and `stamp` · save `{base, segments}` (see [EDITOR.md](EDITOR.md)) |
| `GET /api/transcripts/{id}/versions` · `POST …/versions/{v}/restore` | earlier versions · restore one |
| `GET` · `POST /api/transcripts/{id}/custody` | hashes, check and audit log · hash now (see [PRIVACY.md](PRIVACY.md)) |
| `GET /api/transcripts/{id}/redacted/audio?beep=true` | MP3 with redacted words beeped or silenced |
| `DELETE /api/transcripts/{id}?confirm=<id>` | delete for good (folder, voice samples from it, search rows) |
| `GET /api/privacy` | `{"offline": true}` when offline mode is on |
| `GET /api/transcripts/{id}/files/{name}` | audio (supports Range), plus the original file names `prepis.txt`, `titulky.srt`, `prepis.json` |
| `GET /api/jobs` · `POST /api/jobs` | list (see [paging](#job-list-and-history)) · create `{"source": "<url or path>", "language": "en", "speakers": 2, "no_diarize": false, "engine": null, "model": null, "folder": null}` (`folder`: target [folder](#folders), created when new; omitted: unfiled, or the transcript's folder when it is transcribed again; `""` = unfiled). Optional `"separate_speakers": true` runs the "separate voices" mode: the recording is split into one track per speaker, each track is transcribed alone and the words are merged by time, so overlapping speech is kept (segments of different speakers may overlap). Only for the local engines `whisperx`, `parakeet`, `canary` and only when the separation models are available: otherwise `400 separation_unavailable`. The job JSON has `separate_speakers`; the transcript metadata `extra.separation` (`used`, `tracks`, `speakers`) |
| `POST /api/jobs/upload` | transcribe an uploaded file (multipart, see below) |
| `GET /api/jobs/{id}` · `DELETE /api/jobs/{id}` | detail with log · cancel |
| `GET /api/jobs/{id}/events?since=0` | Server-Sent Events: `log`, `state`, `end` |
| `GET /api/jobs/summary` | `{"running": job or null, "queued": n, "active": n, "failed_recent": n}` (failures of the last 24 h) |
| `POST /api/jobs/{id}/retry` | queue a failed, cancelled or interrupted job again with the same request (`201`, the new job; the old entry is replaced). `409 job_not_retryable`, `409 source_gone` (the file is gone, e.g. an upload) |
| `DELETE /api/jobs/{id}/history` | remove a finished job from the history; the transcript is not touched. `409 job_active` for a queued or running job |
| `POST /api/jobs/clear?status=finished` | remove finished jobs from the history (`finished` (default) · `done` · `failed`); `{"removed": n}` |

### Job list and history

`GET /api/jobs?status=&limit=&offset=`. `status`: `active` (running + queued), `done`, `failed` (error, cancelled,
interrupted), `finished`, `all` (default), or raw statuses, comma separated (`done,error`); unknown values are
`400 invalid_status`. Without `limit` the answer is a plain JSON array, newest first, as before (so scripts and the
MCP server keep working). With `limit` (1–200) it is a page:

```json
{"items": [ {job}, … ], "total": 53, "limit": 20, "offset": 0,
 "counts": {"all": 53, "active": 3, "done": 47, "failed": 3, "finished": 50}}
```

`total` counts the jobs matching `status`; `counts` is always the whole history. Active jobs come running first, then
in queue order (`position`); the rest newest first. Finished jobs are kept up to 100 (older ones are dropped).

### `POST /api/jobs/upload` (multipart/form-data)

What the web UI's file picker and drag & drop use. One file per request; several files are several requests,
each one its own job.

| field | |
|---|---|
| `file` | required: an audio or video file. The extension must be one of `upload.extensions` of `GET /api/info` and the declared content type `audio/*`, `video/*` or generic (`application/octet-stream`, none) |
| `language` | `auto` or a two or three letter code; default: the server setting |
| `speakers` | 1-20 |
| `no_diarize` | `true` = no speaker labels |
| `separate_speakers` | `true` = split the audio into one track per voice, transcribe each and merge (overlapping speech; much slower; local engines only; see `POST /api/jobs`) |
| `engine`, `model`, `folder` | as in `POST /api/jobs` (an empty `folder` field means none chosen) |

The answer is the job JSON (`201`), like `POST /api/jobs`. `GET /api/info` reports `upload.max_mb` (0 = no
limit) and `upload.extensions`.

- **Not a path and not a link.** It works with `allow_local_paths` off and in offline mode. The Origin check
  (a browser page of another site gets `403 cross_origin`) and the API key apply as for every `POST`.
- **Streamed to disk** as `data/uploads/<random>/<name>`; the job waits in the normal queue and survives a
  server restart (the file stays until the job ends). When it is done the file is moved into the transcript
  folder as `audio.<ext>` (formats the player knows: webm, m4a, mp4, opus, ogg, mp3, wav, flac) and
  `metadata.source` is the original file name (`extra.uploaded_via` = `web-upload`). A failed or cancelled job
  deletes the file; leftovers are swept at start.
- **Errors.** `413 request_too_large` (over `max_upload_mb`), `415 unsupported_media_type`, `400 empty_file`,
  `400 unknown_engine`, `400 separation_unavailable`, `403 offline_engine`, `422` for invalid fields. A path inside `data/uploads` is refused
  as a source of `POST /api/jobs`.

Transcript ids of new transcripts are `<slug of the title>-<8 hex chars of the source>`.
Folders from older versions keep their title as the id.

Job `status`: `queued`, `running`, `done`, `error`, `cancelled`, or `interrupted` (the
server died during the job repeatedly). `task` is `"transcribe"` or another registered
task type, such as `"voiceprints"`.

### Recording sessions (`/api/recordings`)

What the web UI's **Record…** button uses: the browser sends the audio in chunks while it records, so little is
lost when the tab or computer dies. A session is a folder `data/recordings/<id>/`; nothing is queued until
`finish`. The same security as every other write applies (Origin check, API key), and nothing here needs the
network, so it works in offline mode and with `allow_local_paths` off.

| call | |
|---|---|
| `POST /api/recordings` | JSON `{mime, source?, slice_ms?, label?}`; `mime` is MediaRecorder's type (`audio/webm;codecs=opus`, `audio/ogg`, `audio/mp4`, ...; else `415`), `source` is `microphone` (default), `tab` or `mix`. Answer `201`: `{id, created, updated, mime, source, label, chunks, bytes, duration, contiguous}` |
| `PUT /api/recordings/{id}/chunks/{n}` | the raw bytes of chunk `n` (0, 1, 2 ... in recording order) as the body; optional `?t=<recorded ms at the end of the chunk>` (the listing's `duration`). Idempotent: a retry replaces the chunk. Answer: the session |
| `POST /api/recordings/{id}/finish` | JSON with the job options of `POST /api/jobs/upload` (`language`, `speakers`, `no_diarize`, `separate_speakers`, `engine`, `model`, `folder`) plus `chunks` (how many the client sent; omitted: up to the highest received), `allow_gaps` (default false: join even when chunks in the middle are missing) and `title`. Joins the chunks (ffmpeg remuxes them into a seekable file when it is installed), queues the job exactly like an upload (`201`, job JSON) and deletes the session. On an error the session is kept |
| `GET /api/recordings` | unfinished sessions, newest first (recovery list) |
| `GET /api/recordings/{id}` | one session |
| `DELETE /api/recordings/{id}` | discard it |

- **Ids** are 32 hex characters, anything else is `404 recording_not_found` (`422` for an invalid path value).
- **Limits.** The whole recording counts against `max_upload_mb` (`413 recording_too_large`); one chunk is at most
  64 MB (`413 request_too_large`); `400 empty_file`; at most 50 unfinished sessions (`409 too_many_recordings`).
  The first chunk (`0`) carries the container header, so `finish` without it is `409 recording_chunks_missing`
  even with `allow_gaps`; a missing middle chunk is the same error unless `allow_gaps` is true. A session without
  data is `409 recording_empty`; a second `finish` at the same time is `409 recording_busy`.
- **Sweep.** Sessions not touched for 7 days are deleted at start and whenever a session starts.
- **Format.** Browsers produce `webm`/Opus (Chrome, Edge, Firefox), `ogg`/Opus (some Firefox builds) or fragmented
  `mp4`/AAC (Safari). The chunks are one container stream cut in pieces, so joining them is a concatenation.
- The finished job is the same as an upload's: the file ends as `audio.<ext>` of the transcript and
  `metadata.extra.uploaded_via` is `web-upload`. The title (file name) is `Recording <date time>` unless `title` is set.

## Folders

Folders are **virtual**: a transcript's folder is a slash-separated path in its `metadata.json` (`"folder":
"Podcasts/Cílek"`, `""` = unfiled; one per transcript), files never move, so ids and links are stable.
`data/folders.json` additionally lists the folders the user created, so that empty ones exist; the tree is that list
plus every folder a transcript is in. Names are trimmed, a backslash counts as `/`, empty names collapse, at most
5 levels, 60 characters per name, no control characters, no `.`/`..`. Names that differ only by case are the same
folder (the existing spelling wins). Errors: `400 folder_invalid | folder_too_deep | folder_name_too_long`,
`404 folder_not_found`, `409 folder_exists`, `400 folder_cycle`.

| endpoint | |
|---|---|
| `GET /api/folders` | `{folders: [{path, name, parent, depth, count, total}], unfiled, total, limits}`: `count` transcripts are directly in a folder, `total` in it and below |
| `POST /api/folders` | `{"path": "Client X/2025"}` creates it and missing parents (`201`, `{path, created}`; an existing one is `created: false`) |
| `POST /api/folders/move` | `{"path": "A", "to": "B/A2"}` renames or moves a folder with its subfolders and every transcript below it: `{path, moved, failed}` |
| `DELETE /api/folders?path=A&move_to=parent` | deletes the folder and its subfolders; their transcripts go to the parent (`parent`, default) or to unfiled (`unfiled`). Transcripts are never deleted: `{deleted, moved, failed, target}` |
| `PUT /api/transcripts/{id}/folder` | `{"folder": "A/B"}` (created when new; `""` = unfiled) |
| `POST /api/folders/assign` | `{"ids": [...], "folder": "A/B"}` moves many (at most 5000); unknown ids come back in `missing` |

Every change goes through the library, so the search index and other listeners run. `failed` lists transcripts
that could not be written (the rest were moved; repeat the call).

## System status and settings

| endpoint | |
|---|---|
| `GET /api/system?refresh=false` | every check the app needs, see below. `refresh=true` ignores cached results (honoured at most every 10 s; one run per check at a time) |
| `GET /api/system/gpu-test` · `POST /api/system/gpu-test` | the last PyTorch GPU test (never starts one) · run it now: a subprocess that loads torch, ~10 s, one at a time, at most every 10 s, `409 gpu_busy` while a job runs |
| `PUT /api/system/hf-token` `{"token": "hf_..."}` | checks the format (ASCII, 8–200 characters), then asks HuggingFace, then writes `HF_TOKEN` to `.env`. Only when the server listens on loopback **and** the client is a loopback address (`403 local_only`). A rejected token is not saved (`400 hf_token_invalid`); when HuggingFace cannot be asked (offline mode, no connection) it is saved and the answer says `"verified": false`. `409 env_file_unreadable` for a `.env` that is not UTF-8 |
| `DELETE /api/system/hf-token` | removes it again |
| `PUT /api/system/keys/{ENV}` `{"value": "..."}` · `DELETE /api/system/keys/{ENV}` | store / remove a cloud API key (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY`, `DEEPGRAM_API_KEY`) in `.env`; effective at once. Same rules as the HuggingFace token: loopback only (`403 local_only`), `409 key_from_environment` when the process environment sets it, `400 key_invalid`, `404 key_unknown`. Write-only: answers `{"env": ..., "set": true}` |
| `POST /api/system/keys/{ENV}/test` | asks the service with the stored key (a free read-only call): `{"env", "ok", "code": "ok"\|"rejected"\|"limited"\|"unreachable"\|"no_key"}`; `403 offline_test` in offline mode |
| `GET /api/system/ollama/pull` · `POST /api/system/ollama/pull` `{"model": "gemma3:4b"}` | progress of the model download · start it (`202`; `409 pull_running`; `403 offline_download` in offline mode) |
| `GET /api/settings` · `PUT /api/settings` `{"values": {"language": "en", "llm_auto": true}}` | editable settings with value, default, source and choices · save them (`null` resets a key) |

`/api/system` returns `{version, python, checks, engines, keys, ready, pyannote_engines, hf_token_source, can_save_token, can_save_keys}`.
Every check is `{id, group, status, code, params, detail}`: `status` is `ok`, `warn`, `error` or `na` (not needed with
the current settings), `code` + `params` are stable (the UI translates them), `detail` is an English sentence with the fix.
Checks: `gpu` (nvidia-smi, VRAM, driver, torch build), `ffmpeg`, `jsruntime` (Deno/Node for yt-dlp), `hf` (token valid and
allowed to download the pyannote model), `ollama` (running, model installed), `storage`, `network`, `offline`.
`ready.transcribe` / `ready.diarize` summarise whether a transcription / speaker recognition can work now.
**No secret is ever returned:** API keys appear only as `{"env": "GEMINI_API_KEY", "set": true, "source": "environment"|".env"|"none", "url": ...}` (`source: "environment"` cannot be changed through the API), the HuggingFace token never.

`/api/settings` accepts only a whitelist: `language`, `engine`, `model`, `diarize`, `llm_provider`, `llm_model`,
`llm_language`, `llm_auto`, `voice_memory`, `watch_enabled`, `offline_mode`. Anything else (API keys, `api_key`, paths, host, port,
`llm_base_url`, ...) is refused with `400 setting_unknown` and must be edited in `.env` / `settings.json`. Values go
to `settings.json` (other keys in the file are kept) and apply immediately, no restart. A key that is set by an
environment variable (`MGTRANSCRIBER_MODEL`, `MGTRANSCRIBER_LLM_PROVIDER`, ...) is reported with `locked: true` and
refused with `409 setting_overridden`. Errors: `setting_invalid` (400), `settings_file_invalid` (409, a damaged
`settings.json` is never overwritten).

## OpenAI-compatible API

Works with the official OpenAI SDKs and anything else that speaks the Audio
Transcriptions API (setup for Open WebUI, n8n and Home Assistant: [INTEGRATIONS.md](INTEGRATIONS.md)):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="<MGTRANSCRIBER_API_KEY or anything>")
with open("talk.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1", file=f, response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
        extra_body={"diarize": True, "num_speakers": 2},  # extensions, optional
    )
print(result.text)
for seg in result.segments:
    print(seg.start, seg.model_extra["speaker"], seg.text)
```

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions -H "Authorization: Bearer $KEY" \
  -F file=@talk.mp3 -F model=whisper-1 -F response_format=srt
```

### `POST /v1/audio/transcriptions` (multipart/form-data)

| field | |
|---|---|
| `file` | required; any audio/video format ffmpeg reads |
| `model` | see below; default `whisper-1` |
| `language` | ISO-639-1 code; empty = detect |
| `response_format` | `json` (default), `text`, `srt`, `vtt`, `verbose_json` |
| `timestamp_granularities[]` | `segment` (default) and/or `word`; only with `verbose_json` |
| `prompt`, `temperature` | accepted and ignored |
| `stream` | `true` is rejected (not supported) |
| `diarize` | extension: label speakers. Default `false`, like OpenAI's Whisper. Needs `HF_TOKEN` |
| `num_speakers` | extension: exact number of speakers (1-20). Implies `diarize` |

Responses:

- `json`: `{"text": "...", "usage": {"type": "duration", "seconds": 40}}`.
- `verbose_json`: `task`, `language` (as a name, e.g. `"english"`), `duration`, `text`,
  `segments` and `usage`, plus `words` when requested.
  - Each segment carries OpenAI's fields; `tokens`, `avg_logprob` and the like are
    placeholders.
  - Segments and words get an extra `speaker` field (`null` without diarization).
  - Word times are always set: words the aligner could not place get their
    neighbours' times.
  - Extra top-level field `transcript_id`: the library entry.
- `srt`, `vtt`: subtitles. With diarization SRT lines start with `[SPEAKER_00]` and
  VTT uses `<v SPEAKER_00>` voice tags.
- `text`: plain text.

Behaviour:

- **Queueing.** The upload becomes an ordinary job in the same queue as the web UI, so
  it never competes with another transcription for the GPU. The request waits for the
  result (it shows up in the UI's job list meanwhile). If the client disconnects
  first, the job is cancelled. The upload exists only for the request, so its job is
  never resumed: stopping the server cancels it, and after a crash it is marked
  "interrupted". Ctrl+C waits at most 5 s for open requests.
- **Timeouts.** Long media can exceed a client's timeout. The OpenAI Python SDK
  defaults to 10 minutes; pass `timeout=` if needed.
- **Library.** The result is a normal library entry. The uploaded file is kept as its
  audio, so the web player works. `metadata.source` is the uploaded file name.
- **Errors** (OpenAI format):

  | status | when |
  |---|---|
  | `400` | invalid request (`param` names the field) |
  | `401` | missing or wrong key |
  | `404` | `model_not_found` |
  | `413` | `request_too_large`: the upload is over `MGTRANSCRIBER_MAX_UPLOAD_MB` |
  | `500` | `transcription_failed`, with the engine's message |

### Models

`GET /v1/models` lists `whisper-1`, the Whisper models and the available engines.
`GET /v1/models/{id}` answers for every accepted id:

| `model` | meaning |
|---|---|
| `whisper-1`, `gpt-transcribe` (Open WebUI's default), `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize` | the server default (setting `model`, e.g. `large-v3`) |
| `tiny` ... `large-v3`, `large-v3-turbo`, `turbo`, `distil-large-v3` | that Whisper model (WhisperX engine) |
| an engine name from `/api/info` | that engine with its default model |
| `<engine>/<model>` | that engine and model |
