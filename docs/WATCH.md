# Watching sources

`mgtranscriber/watch/` checks sources on a schedule and adds what is new to the normal job queue. It never
transcribes anything itself: the queue (one job at a time, GPU-friendly) decides when a job runs, and a job
started by a watch source looks like any other job in the UI.

| kind | target | how new items are found |
|---|---|---|
| `folder` | folder path | audio/video files; optionally recursive. A file is queued only when its size and mtime did not change between two checks (and it was last modified at least `watch_stable_s` seconds ago), so half-copied files are ignored. `.part`, `.tmp`, `.crdownload`… are skipped. |
| `rss` | RSS 2.0 / Atom feed URL | items with an audio/video `<enclosure>` (or Atom `rel="enclosure"` / `media:content`). |
| `youtube` | channel (`@handle`, `/channel/UC…`) or playlist URL | the channel/playlist RSS feed when it can be derived from the URL (light, no JavaScript runtime needed); otherwise yt-dlp *flat* extraction (a listing, no media download). An `@handle` is resolved once and the feed is remembered. No API key. Live and upcoming streams are skipped. |

## Per-source settings

`name`, `language` (`null` = default, `auto`, `cs`, …), `speakers`, `engine`, `model` — passed to the job;
`interval_s` — check period; `mode` — `new` (only what appears after the source was added) or `last_n` (also the
newest `last_n` existing items, 1–50); `max_per_check` (1–20, default 5); `recursive` (folders); `analyze` (start
the LLM analysis after the transcription, like the global `llm_auto` but per source); `translate_to`
(`null` = none, or a language code the translate feature supports, e.g. `cs`: after each finished transcription
a translation job is queued exactly as the "Translate to…" button does; skipped when the transcript is already
in that language; an unavailable LLM / offline mode is recorded as `translation.error` on the item, the
transcript itself is unaffected); `webhook_url`; `enabled`;
`folder` — library folder (`"Podcasts/Cílek"`, created when new; empty = unfiled) for what the source transcribes.

## Deduplication and persistence

Sources, their state and the keys of everything already handled live in `data/watch.json` (written atomically).
Keys: file → content key (size + hash of the first and last 64 KB, so renamed or moved files are not repeated);
RSS → `guid:<guid>` and `url:<enclosure>` (either match counts, so a CDN URL change does not duplicate an
episode); YouTube → `yt:<video id>`. Up to 20 000 keys per source are kept; keys the source still lists are refreshed at every check and never evicted, so long feeds are not queued again. An item is marked handled when it is
queued, so a restart never queues it twice; jobs that were still pending are reconciled on startup. A failed
job is recorded on the item and the source (`last_error`) but not retried automatically.

"Only new from now" means: items that existed (or were published before) the moment the source was created are
marked as handled at the first check without being queued. For feeds and YouTube this relies on the items' publish
dates, and it applies at every check, not only the first: an unseen item dated before the source was created is
marked handled and never queued, even if a later check lists it for the first time (entries without a date are judged
by the seen list only). Folders are the exception after the first check: a file copied in later is queued even when its
modification time is old.

## Being kind to servers

* Minimum intervals (seconds): folder 10, RSS 300, YouTube 1800. Smaller values are raised. Defaults: 60 / 3600 / 3600.
  Override in `settings.json`: `watch_min_folder_s`, `watch_min_rss_s`, `watch_min_youtube_s`.
* Checks run one at a time on one thread. Feeds are fetched with `ETag`/`Last-Modified` revalidation and a 32 MB cap. Folder scans skip junctions/mount points and stop after 5 000 files or 2 000 directories.
* On errors the next check is delayed exponentially (`interval × 2^failures`, at most 6 h, never less than the
  interval); a success resets it. A 429 is reported as rate limited and handled the same way.
* At most `max_per_check` items are queued per check, and nothing is queued while `watch_max_queued` (default 20)
  jobs are waiting or running; the rest stays for later checks. Backlogs are processed oldest first.
* A manual check of one source is refused within 30 s of the previous check (HTTP 429, `watch_too_soon`).

Other settings keys: `watch_enabled` (default true; false disables the scheduler), `watch_stable_s` (default 10).

## Security

* Feeds can only point to `http(s)` media: an enclosure such as `C:/x.wav` or `\\host\share\a.mp3` is ignored, and an
  item that is not a web URL is never queued (a feed can never make the server read a file).
* Watching folders needs `allow_local_paths` (checked again at every check; a paused source can always be resumed or deleted).
* A folder on a network share (`\\server\share`, also `//server/share`) is refused (`unc_path_disabled`) unless
  `MGTRANSCRIBER_ALLOW_UNC_PATHS=true`: opening it sends the user's Windows credentials to that server.
* When the server is reachable from the network (not bound to loopback) feed, preview, webhook and every redirect
  must resolve to **public** addresses only (no loopback, private, link-local or reserved ones), so a client cannot use
  the server to probe your LAN or cloud metadata. On loopback everything is allowed (local NAS, local Ollama). A host
  name that changes its answer between the check and the request (DNS rebinding) is not covered: when you expose the
  server, set `MGTRANSCRIBER_API_KEY`.
* Webhooks are sent from their own thread, so slow receivers never delay other queue hooks.

## Webhook (optional)

If `webhook_url` is set, a generic JSON `POST` is sent when a job of that source finishes (two attempts, 10 s
timeout; no external service involved):

```json
{"event": "transcription.done", "source": {"id": "9d381e0f", "name": "Inbox", "kind": "folder"},
 "item": {"title": "a", "source": "C:\\in\\a.wav"}, "job": {"id": 1, "status": "done", "error": null},
 "transcript": {"id": "a-f823a1e6", "title": "a"}, "time": 1791567235.9}
```

`event` is `transcription.failed` (with `transcript: null`) when the job failed or was cancelled. An LLM
analysis started by `analyze` runs afterwards and is not awaited.

## API

`/api/watch` (see also `/api/docs`). Errors carry a stable `code` (`watch_not_found`, `watch_invalid_target`,
`watch_folder_missing`, `watch_folder_forbidden`, `watch_too_soon`, `watch_rate_limited`, …). Folder sources need
`allow_local_paths` (default on loopback) and cannot point at the library or data folders.

| request | |
|---|---|
| `GET /api/watch` | `{sources: [...], limits: {min_interval_s, default_interval_s, ...}}` |
| `POST /api/watch` | create (body: the settings above plus `kind`, `target`); the first check runs within a second |
| `GET/PATCH/PUT/DELETE /api/watch/{id}` | read, update (changing `target` starts over), delete |
| `POST /api/watch/{id}/check[?dry_run=true]` | check now; a dry run lists what would be queued and changes nothing |
| `GET /api/watch/{id}/history` | last 20 checks (`time, ok, found, queued, error, manual, dry_run`) and the last 30 items with their job outcome |
| `POST /api/watch/preview` | what a not yet saved source would find |

The UI page is "Watching" (`#/watch`).

## Tests

`tests/test_watch.py` (unit + API; local fake RSS/HTTP server, temporary folders, a mock queue and an injected
clock). `uv run pytest -m gpu tests/test_watch_gpu.py` drops a clip into a watched folder of a running server and
transcribes it with a small model. A real YouTube channel can be checked without downloading anything:
`POST /api/watch/preview` with `{"kind": "youtube", "target": "https://www.youtube.com/@TED"}`.
