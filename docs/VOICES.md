# Voice memory

Recognises known speakers across transcripts by their voice: remember a speaker once
("this is Jana"), and later transcripts name her automatically. Everything runs
locally. The thresholds below come from measurements on real recordings.

## How it works

1. **Diarization** (pyannote `speaker-diarization-community-1`) returns one 256-dim
   embedding per speaker label (`return_embeddings=True`, WeSpeaker ResNet34; no extra cost).
   `engines.common.diarize(..., voices=dict)` collects them, engines return them as
   `TranscriptionResult.speaker_embeddings` (whisperx, parakeet, canary, gemini, youtube with
   diarization; ElevenLabs/Deepgram have none).
2. **Pipeline** (`pipeline.run`, last step) calls `voices.after_transcription()`: stores the
   embeddings in `voices.json` in the transcript folder, identifies them against the voice
   database and writes **only automatic matches** into `speakers.json`. It never raises.
3. **The user** confirms or rejects suggestions, or remembers a voice, in the transcript's
   speaker panel. The "Voices" page lists, renames and deletes people and samples.

Vectors are L2-normalised and compared by cosine similarity. A person's score is
`max(cosine to the mean of their samples, cosine to the nearest sample)`.

## Measured (see the spike)

| | same person | different people |
|---|---|---|
| same podcast, different 10-min parts (74 / 314 pairs) | 0.935 – 0.992 | ≤ 0.337 |
| different podcasts: the same person as a host and as a guest of other podcasts (3 / 37 pairs) | 0.845 – 0.916 | ≤ 0.222 |
| 10 s of speech in the new recording | ≥ 0.734 | ≤ 0.298 |
| enrolled from only 20 s | ≥ 0.755 | ≤ 0.169 |

Pitch-shifting a voice by 3 semitones makes it a different person (< 0.34) for the model.

## Decisions

| constant (`voices.py`) | value | meaning |
|---|---|---|
| `AUTO_THRESHOLD` | 0.60 | from here, with `MARGIN`, the name is set automatically |
| `MARGIN` | 0.10 | lead over the second-best person needed for `auto` |
| `SUGGEST_THRESHOLD` | 0.45 | from here the person is only suggested in the UI |
| `MIN_IDENTIFY_S` | 10 s | labels with less speech stay unknown |
| `MIN_ENROLL_S` | 20 s | a voice is remembered only from at least 20 s of speech |

* **1:1 within a transcript:** two labels never get the same person automatically; the weaker
  one gets a suggestion. Two unknown voices therefore cannot merge into one known person.
* **Unknown voices are never enrolled automatically.** Enrolment is always a user action.
* **Names typed by the user are never replaced**; automatic names fill only empty labels.
* **Rejected** suggestions are remembered per label and not suggested again.
* **Enrol guard:** a new sample that does not resemble the person's existing samples
  (< 0.45) is refused with `voice_mismatch` unless forced (the UI asks).
* Accepting a suggestion also adds the voice as another sample (more recordings = more robust).
  A label with less than 20 s of speech is only named and marked as confirmed, without a sample.
  A confirmed match survives "identify again" as long as the person exists.
* Rejecting a match removes the label's name only when voice memory set it automatically;
  a name the user typed stays. A label whose voice is already saved cannot be saved twice.
* Concurrency: the server threads and job workers change the database and `voices.json` under
  one lock per file (a thread lock plus an OS lock on `<file>.lock`); writes go to a unique,
  fsynced temp file that is then moved into place. A damaged database answers
  `voice_db_corrupt`; a damaged `voices.json` counts as "no voiceprints".

## Files

**Voice database**: `data/voiceprints.json` (setting `voices_file` in `settings.json` or env
`MGTRANSCRIBER_VOICES_FILE`; turn the feature off with `voice_memory: false` /
`MGTRANSCRIBER_VOICE_MEMORY=false`). `data/` is git-ignored.

```json
{"version": 1, "people": {"3f2a…": {"name": "Jana Nováková", "created": "2026-10-09T18:00:00+00:00",
  "samples": [{"id": "a1b2…", "vector": "<base64 float32 ×256, L2-normalised>",
               "model": "pyannote/speaker-diarization-community-1", "dim": 256, "speech_s": 139.2,
               "source": {"transcript": "<id>", "label": "SPEAKER_01", "title": "…"},
               "added": "…"}]}}}
```

* `version` is the file format; a newer version is refused (never overwritten).
* `model` is stored **per sample**: after a diarization model change old samples are ignored
  for recognition and shown as "old model" on the Voices page; remember the voice again.

**Per transcript** `voices.json` (removed on re-transcription via `library.STALE_ON_RETRANSCRIBE`):

```json
{"version": 1, "model": "…", "speakers": {"SPEAKER_00": {"vector": "<b64>", "speech_s": 389.0,
  "match": {"status": "auto|suggest|unknown", "person_id": "…", "name": "…", "score": 0.91, "margin": 0.7},
  "rejected": ["<person id>"], "enrolled": {"person_id": "…", "sample_id": "…"}}}}
```

`speakers.json` stays the library contract `{label: name}`.

## API (`/api/voices`)

| | |
|---|---|
| `GET /api/voices` | people with samples (no vectors), file path, thresholds |
| `PATCH /api/voices/people/{id}` `{name}` | rename |
| `DELETE /api/voices/people/{id}` · `…/samples/{sample}` | delete a person / one sample |
| `GET /api/voices/export` | download the database (JSON) |
| `DELETE /api/voices` | delete the database **and** every transcript's `voices.json` (also temp copies); cancels queued voiceprint jobs; voice memory stays on |
| `GET /api/voices/transcripts/{id}` | per label: speech, live match against the database, pending job |
| `POST …/transcripts/{id}/enroll` `{label, name, person_id?, force?}` | "Remember voice as…" |
| `POST …/transcripts/{id}/accept` · `…/reject` `{label}` | confirm / reject the match |
| `POST …/transcripts/{id}/identify` | identify again and fill empty names with automatic matches |
| `POST …/transcripts/{id}/compute` | queue a `voiceprints` job for an older transcript |
| `POST /api/voices/scan` `{person_id?, compute_missing?}` | queue the archive scan (job task `voice_scan`; `source` is `all` or `person:<id>`); an identical waiting scan is returned instead. `GET /api/voices` shows `scan_job` and `appearances` per person |
| `GET /api/voices/people/{id}/appearances[?suggestions=false]` | `{person_id, name, transcripts, speech_s, appearances: [{transcript_id, title, label, speech_s, status, score, date, audio}]}`, newest first |
| `GET /api/voices/people/{id}/suggestions` | unconfirmed matches (suggest/auto) with `score`, `start`, `end` (5 s) and `snippet`, best first |
| `POST /api/voices/people/{id}/suggestions/accept` · `…/reject` `{items: [{transcript, label}], min_score?}` | confirm / reject in bulk; accept without `items` takes everything with score ≥ `min_score` (default 0.60). Returns `{accepted, failed: [{transcript, label, code}]}` |
| `POST /api/voices/people/{id}/merge` `{other, name?}` | merge person `other` into this one (not undoable) |
| `GET /api/search?q=&speaker=&person=` | full-text search limited to a speaker name/label or a person |

Errors carry codes `voice_*` (translated in the UI), among them `voice_changed` and `voice_merge_same`.

## Remembering a voice from the UI

- "Remember voice" is one step. A speaker that already has a name is saved at once (toast, tag "voice saved as X", the
  result is announced in an `aria-live` region). Without a name a small form opens with the field focused and a
  "Save voice" button; Enter saves, Esc cancels. Errors (e.g. less than 20 s of speech) are shown as text next to the speaker.
- After the user renames a speaker (not yet remembered, at least 20 s of speech) an inline prompt asks "Remember the
  voice of X?" with Remember / Not now. It never enrolls by itself (voiceprints are biometric data). "Not now" is kept
  per transcript + speaker label in the browser (`localStorage` key `voices.not_now`).
- With an empty voice memory the Voices page lists the steps (open a transcript, name a speaker, Remember voice, scan),
  `POST /api/voices/scan` returns `empty: true`, and the scan job's summary says "Voice memory is empty" instead of zero counts.

## The whole archive

Voice memory also works backwards: once people are remembered, the existing archive can be searched for them.

* **Scan the archive** (Voices page, or a single person's page): a background job (task `voice_scan`, in the
  normal job queue, with progress and cancel). It goes through every transcript that has voiceprints and runs
  *identify* again, storing the result exactly as after a transcription: automatic names fill empty labels only,
  suggestions are stored, names typed by the user and confirmed matches stay. The embeddings already exist, so it
  is CPU-only and quick (no GPU). The option "also queue voiceprints for transcripts that have none" adds
  `voiceprints` jobs (GPU) for older transcripts; they identify themselves when they finish.
  A scan for one person only touches transcripts where a label scores at least 0.45 against that person (or is
  already matched to them).
* **Person page** (`#/voices/<id>`): *To confirm* lists every unconfirmed match to the person across the archive
  (transcript, label, score, an excerpt and **Play 5 s**), best score first, with Yes / No per row and
  "Accept all with a score from X" (default 0.60). A confirmation names the label; the voice is added as a sample
  only while the person has fewer than 10 samples (`BULK_SAMPLE_CAP`), so confirming hundreds of matches does not
  bloat the database. "Accept all" skips labels that already have a different name (typed by you); only choosing the row yourself replaces it. A bulk action never confirms somebody else: if the label's current match is another person,
  it is skipped (`voice_changed`).
* **Where this person speaks**: every transcript with the label, seconds of speech, status and date. Status:
  `confirmed` (you confirmed or remembered it), `auto`, `suggest`, `manual` (the name you typed equals the
  person's). An automatic match that contradicts the name you typed for the label counts as `suggest`.
* **Merge people**: duplicates (the same voice remembered as "J. Novak" and "Jana Novakova") are merged into one
  person: samples, stored matches and enrolments in transcripts move over, and names in `speakers.json` that came
  from the person (automatic or confirmed) are renamed; a name you typed for a label that has no relation to the
  person stays. Order: transcripts, then the database (one atomic write), then the transcripts once more (a scan
  running meanwhile may have matched the old person again). An interrupted merge is finished by calling it again with the same two people (it reports `resumed`).
  Splitting a person is not supported (delete the wrong sample instead).
* **Search by speaker**: `GET /api/search?q=...&speaker=<name or label>` or `&person=<person id>` keeps only the
  segments of that speaker (case and diacritics ignored); with an empty `q` it lists everything they said.

## Older transcripts

Transcripts made before voice memory (or by an engine without embeddings) have no
`voices.json`. The detail page offers "Compute voiceprints", which queues a job of task
`voiceprints` (`tasks.py`, runs in the worker process like a transcription, so it never
competes for the GPU). It does **not** re-run diarization: it runs the diarization model's
embedding network on each existing label's speech spans (word speakers, ≥ 2 s, ≤ 10 s each)
and averages them robustly, so labels and names stay as they are. Needs the audio file
(`audio.*` in the folder, or the original local file) and `HF_TOKEN`.

## Privacy

Voiceprints are biometric data (GDPR art. 9). They stay on this computer: no upload, no
telemetry. Only explicit user actions create samples; the Voices page shows where the
database lives, exports it, and deletes everything (database and per-transcript files)
in one step. Speaker names in `speakers.json` are kept on delete (they are not biometric).

## Risks and limits

* Measured on studio podcasts (English, a few people across different podcasts and guest appearances). Phone audio, noise, music
  and other languages are not measured; the suggestion band is the safety net.
* The model is pitch-sensitive: a cold or a very different speaking style can lower scores.
* A diarization label that mixes two people has a blended embedding; enrolling it poisons a
  voice. The 20 s minimum and the mismatch guard reduce that risk.
* `pipeline._embedding` of pyannote is a private attribute (used for older transcripts).
