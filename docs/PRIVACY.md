# Privacy and sensitive data

For journalists, lawyers, doctors and anyone who transcribes recordings that must not leave
their computer or must hold up later: an **offline mode**, a **chain of custody** with an
**audit log**, **redaction** of words in the text and audio, and **deletion** of a transcript
for good.

Code: `mgtranscriber/privacy.py`, `mgtranscriber/server/routers/privacy.py`,
`static/js/components/privacy.js`. Tests: `tests/test_privacy.py`.

What is reliable and what is not is listed at the end, under [Limits](#limits). Read it before
relying on these features.

## Offline mode

Turn it on on the **Settings** page (it applies at once), in `settings.json`:

```json
{ "offline_mode": true }
```

or with the environment variable `MGTRANSCRIBER_OFFLINE_MODE=true` (it beats `settings.json` and
the Settings page, so it can also lock the mode off). The top bar shows **Offline mode** (after
a reload of the page when it was switched on the Settings page).

In offline mode the server refuses:

| what | where it is enforced | error code |
|---|---|---|
| cloud engines (ElevenLabs, Deepgram, Gemini) and the YouTube captions engine | `engines.get_engine()`: the engine is never even loaded, in the server or in a job worker. Also checked when a job is created (`POST /api/jobs`, `/v1/audio/transcriptions`) | `offline_engine` (403) |
| URL sources (YouTube, podcasts, any link) | `pipeline.run()` and `POST /api/jobs`, the MCP `transcribe` tool | `offline_url` (403) |
| cloud LLM providers (Gemini, Claude) for analysis and questions | `llm.make_provider()`, used by every LLM feature | `offline_llm` (403) |
| Ollama on another computer (`llm_base_url` not loopback) | `llm.make_provider()` | `offline_llm` |
| watched feeds and YouTube channels, webhooks of watched sources | `watch/manager.py` (checks fail with `offline_url`; webhooks are not sent) | |

The checks run where the work happens, not only in the UI, so no API client, MCP client or
watched folder can get around them. `GET /api/info` and `GET /api/privacy` report
`{"offline": true}`. In `/api/info` the refused engines have `missing: {"code": "offline"}`, so
the new-job form shows them disabled.

Local engines (WhisperX, Parakeet, Canary), local files, watched folders and Ollama on this
computer keep working.

**Model downloads.** Offline mode is about *your data*. It does not stop the first download of
model weights from Hugging Face. No audio or text is sent there, only the model is fetched.
For an air-gapped machine, download the models once and then also set `HF_HUB_OFFLINE=1`.

**API documentation page.** `/api/docs` (Swagger UI) loads its script and style from the jsdelivr CDN, so
opening that one page contacts that server (nothing from your library is sent), even in offline mode. The
web UI itself loads nothing from outside. Use `/api/openapi.json` or the docs in this repository when offline.

## Chain of custody

Every transcription stores SHA-256 hashes in `metadata.json` under `extra.custody`:

```json
"custody": {
  "algorithm": "sha256",
  "updated": "2026-10-09T18:18:37+00:00",
  "source": {"file": "interview.wav", "size": 72593316, "sha256": "f157…"},
  "outputs": {"prepis.json": "980f…", "prepis.txt": "364e…", "titulky.srt": "4ca1…"}
}
```

* `source` is the media that was transcribed: the local file, or the downloaded `audio.*` for
  a link. It is hashed once, at transcription.
* `outputs` are hashed again after every edit, restore or re-transcription.
* `GET /api/transcripts/{id}/custody` compares the stored hashes with the files now
  (`ok` / `changed` / `missing`) and reports whether the audit log is intact.
  `POST` hashes everything now, which is useful for transcripts made before this feature.
* In the UI: **Privacy → Chain of custody** on the detail page, with **Hash now**.

A hash proves that a file is bit-for-bit the one that was hashed. To prove *when*, keep the
hash somewhere you do not control alone, for example in an e-mail to yourself or a qualified
timestamp.

## Audit log

`audit.jsonl` in the transcript folder records every action, one JSON object per line, append-only:

```json
{"time":"2026-10-09T18:16:59+00:00","event":"edited","segments_changed":2,"segments":1257,"redacted_words":0,"redactions_changed":false,"times_changed":1,"version":"prepis.20261009T181659123Z.edit.json","sha256":"…","prev":"5d1c…"}
```

| event | when |
|---|---|
| `created`, `retranscribed` | a transcription finished (source, engine, model, hashes) |
| `edited`, `restored` | the editor saved, or a version was restored (counts only: **the log never contains transcript text**, so it does not leak redacted words) |
| `exported` | any download from `/export/...`, also redacted ones and the redacted audio (format, `redacted`, hash of the file handed out) |
| `hashed` | custody hashes recorded by hand |
| `deleted` | in `data/audit.jsonl` (the folder is gone): only the SHA-256 of the transcript id, the number of files and the removed voice samples, never the title |

Each line carries `prev`, the SHA-256 of the line before it. Removing, reordering or changing
a line breaks the chain, and the UI shows "The audit log was changed or shortened outside the
app". This makes tampering *evident*, not impossible: someone with write access can rewrite
the whole file consistently. Keep copies (or hashes of the log) elsewhere if that matters.

Two more safeguards, and their limits:

* **Interrupted write.** A last line without its newline (power cut, killed process) is not a
  record: reading shows it as `{"event": "incomplete"}`, the chain check ignores it, and the next
  append cuts it off and notes `recovered_bytes` in the new record. A damaged line in the
  *middle* of the log still breaks the chain.
* **Anchor.** Cutting off the *end* of a log keeps the chain valid, so after every append the
  number of records and the hash of the last one are noted in `audit.jsonl.anchor` next to the log.
  `GET .../custody` returns `audit_state: {chain, records, partial_tail, anchor}` where `anchor`
  is `ok`, `none` (a log from before this feature), `truncated` (fewer records than noted) or
  `mismatch`; `audit_intact` is false for the last two. The anchor sits in the same folder, so it
  catches accidents and careless edits, not someone who rewrites both files (a record added
  after the anchor was last written, because the process died in between, is accepted).
  Deleting the anchor makes the state `none`: keep an external copy of the count and last hash
  when this matters.

Viewing the transcript and listening to the audio are not logged.

## Redaction

In the editor, select words and press **Redact** (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>X</kbd>).
Redacted words are shown on a dark bar. They stay in `prepis.json` with `"redacted": true`, so
the original remains available to you and you can un-redact them.

Redacted versions are separate exports (**Privacy → Redacted export**):

* `GET /api/transcripts/{id}/export/{fmt}?redact=true` for every format (TXT, SRT, VTT, JSON, MD,
  DOCX). The title and the source (path or URL) may name what was redacted, so they are left
  out of every redacted export; add `title=true` to keep the title (the source is never included).
  Each run of redacted words becomes one `[REDIGOVÁNO]` (Czech transcripts),
  `[REDIGOVANÉ]` (Slovak) or `[REDACTED]` (other languages), in the text *and* in the words of
  the JSON export. The redacted Markdown leaves out the LLM analysis, because it may quote the
  redacted words. The file name ends in `.redacted.<ext>`.
* `GET /api/transcripts/{id}/redacted/audio` is an MP3 of the audio with every redacted word
  silenced (50 ms margin on both sides) and a 1 kHz beep in its place (`?beep=false` for
  silence only). Words without their own time silence the gap between their timed neighbours.
  ffmpeg must be installed. It takes about 40 s for 90 minutes of audio, and the request
  waits for it. One redacted audio is made at a time (`409 redaction_running` for another
  request meanwhile). The temporary file is created in `data/tmp/`, not in the system temp
  folder, and removed after sending, after an error and when the client disconnects; files left
  by a crash are removed at the next request after an hour. The ffmpeg run itself is not
  interrupted by a disconnect, it finishes first.

## Deleting a transcript for good

**Privacy → Delete for good**, then type the transcript id to confirm. The API is
`DELETE /api/transcripts/{id}?confirm=<id>`. It refuses while a job or an analysis works on
the transcript (`transcript_busy`). It:

1. renames the folder to a hidden tombstone (`.deleting-<random>`), so the transcript is gone from
   the library at once and a failure part-way never leaves a transcript with zeroed files;
2. overwrites every file in the tombstone with zeros (text, versions, audio, analysis,
   voiceprints, audit log, ...) and removes it;
3. removes the voice samples that were taken from this transcript from the voice database
   (people left without samples go too);
4. removes the transcript from the search index;
5. writes the `deleted` record to `data/audit.jsonl`: the transcript id as an HMAC-SHA-256 with a
   random key kept in `data/audit.key` (a plain hash of a readable id could be matched to a
   title by guessing), the number of files and voice samples, and `incomplete: true` when the
   files could not all be removed.

Steps 3 to 5 run whatever happens to the files. A tombstone that could not be removed (the
response has `incomplete: true`) is retried at the next delete (`purge_tombstones`). Until then
its content is still on disk, though overwritten as far as it got. Deleting `data/audit.key`
makes old `deleted` records unmatchable to their transcripts.

## Limits

Be honest with yourself about what software deletion and logging can do:

* **Overwriting is best effort.** On SSDs (wear levelling), copy-on-write or journaling file
  systems, cloud-synced folders (OneDrive, Dropbox), backups, Windows shadow copies and
  antivirus quarantine, old copies can survive. The real protection is **full-disk encryption**
  (BitLocker, FileVault, LUKS). Delete the keys and the data is gone.
* **The search index** (`data/search.sqlite3`) drops the rows, but SQLite may keep deleted
  content in free pages until the file is rebuilt. The index is derived data: deleting the file
  is safe and it is rebuilt at the next start.
* **Job history** (`data/jobs.json`, `data/job-logs/`) keeps the title and source of past jobs
  until they age out (last 100). Delete those files by hand if the title itself is sensitive.
* The original **local media file** you transcribed is not touched; it is yours.
* **Redaction** does not change the search index, the LLM analysis or names in
  `speakers.json`. Share only the redacted exports. Do not share the transcript folder.
* **Audit log truncation** is only detected by the anchor next to the log (see above), and the
  audit log is serialised with an OS file lock (`fsutil.file_lock`), which is not reliable on
  network shares: keep `data/` and the library on a local disk.
* **Audit log and hashes** detect changes. They do not prevent them, and do not prove *who* did
  something: the app has no user accounts.
* **Offline mode** is a setting of this server. Anyone who can edit `settings.json` or the
  environment can turn it off. Set it with `MGTRANSCRIBER_OFFLINE_MODE` in the service
  definition if that matters.
* An LLM analysis that was made **before** offline mode was turned on (with a cloud provider)
  has already left the computer.

## What is fully done and what is a basis

Done and tested (unit tests, API tests, a real browser run on a copy of an 87-minute
transcript):

* offline mode in the engine and LLM registries, the pipeline, jobs, `/v1`, MCP and watch;
* hashes at transcription and after every edit, verification, the hash-chained audit log for
  create, edit, restore, export and delete;
* word redaction in the editor, redacted text exports in every format, beeped or silenced audio;
* deletion with confirmation, voice samples and search entries included.

A basis to build on:

* no user accounts, so the log cannot say who did what;
* no signed or externally timestamped hashes;
* no redaction of speaker names or of the LLM analysis, and no automatic detection of
  personal data (names, numbers);
* deletion cannot guarantee physical erasure (see above), and job history is not cleaned
  automatically.
