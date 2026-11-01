# Transcript editor

The detail page of a transcript has an **Edit** button. The editor replaces the reading view
until **Done**. Each segment is one row: its start time (click to play from there), its
speaker and its text, which you edit in place while the audio plays. Changes are saved
automatically about a second after you stop typing. Before the edits, the previous state is
kept as a version that can be restored.

Code: `static/js/components/editor.js` (UI), `mgtranscriber/editor.py` (re-alignment, versions,
saving), `mgtranscriber/server/routers/editor.py` (API). Tests: `tests/test_editor.py`.

## Using it

| key | action |
|---|---|
| <kbd>Ctrl</kbd>+<kbd>Space</kbd> | play / pause |
| <kbd>Ctrl</kbd>+<kbd>,</kbd> / <kbd>Ctrl</kbd>+<kbd>.</kbd> | back / forward 5 s |
| <kbd>Enter</kbd>, <kbd>Alt</kbd>+<kbd>↓</kbd> / <kbd>Alt</kbd>+<kbd>↑</kbd> | next / previous segment; the audio jumps there |
| <kbd>Alt</kbd>+<kbd>Shift</kbd>+<kbd>S</kbd> | choose the speaker of the segment (the select); <kbd>Esc</kbd> returns to the text |
| <kbd>Ctrl</kbd>+<kbd>Enter</kbd> | split the segment at the cursor (at the end of the word the cursor is in) |
| <kbd>Backspace</kbd> at the start of a segment | merge it with the previous one |
| <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>X</kbd> | redact the selected words, or un-redact them when all are redacted |
| <kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Y</kbd> (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd>) | undo / redo |

The keys are layout-independent (`KeyboardEvent.code`), so <kbd>,</kbd> and <kbd>.</kbd> are
the keys next to <kbd>M</kbd> on Czech keyboards too. On a phone the toolbar buttons (Undo,
Redo, Split, Merge, Redact) do the same, and the player plays and pauses. The **Shortcuts**
button shows the table above; **Versions** lists the stored versions with **Restore**.

* **Speaker** of a segment: the select next to the text. "+ New speaker" adds the next free
  label (`SPEAKER_02`, ...), which you can name in the speaker panel as usual. The words of the
  segment get the new speaker too.
* **Keyboard access.** The whole segment list is one tab stop: the text of the row you were in last
  (roving `tabindex`; the speaker selects are left out of the tab order and reached with
  <kbd>Alt</kbd>+<kbd>Shift</kbd>+<kbd>S</kbd>). In the reading view the segments are one tab stop too:
  <kbd>↑</kbd>/<kbd>↓</kbd> (or <kbd>←</kbd>/<kbd>→</kbd>), <kbd>Home</kbd> and <kbd>End</kbd> move between
  them and <kbd>Enter</kbd> or <kbd>Space</kbd> plays from the segment (only when the transcript has audio).
* **Typing** inside a redacted passage keeps the new words redacted.
* **Undo** keeps 200 steps for the editing session (typing in one segment is one step until a
  pause of 1.5 s). Every undo and redo is saved like any other change.
* While the audio plays, the spoken row is highlighted and scrolled into view, unless you are
  typing.
* Leaving the page with unsaved changes asks for confirmation. If the transcript was changed
  elsewhere meanwhile (another tab, a re-transcription), saving stops with "Changed elsewhere"
  and nothing is overwritten. Click the status to reload.
* `#/t/<id>?edit=1` opens the editor directly. A reload stays in the editor.

## Word times

`prepis.json` stores word times (`words[].start/end`) that the player, SRT/VTT export, search
links, voice memory and redaction rely on. The editor keeps them as well as it can. A
segment's words always correspond 1:1 to the whitespace-separated tokens of its text. After
each change the words are **re-aligned** to the new text (in the browser for the live view,
`realign()`; on the server again, `editor.realign_words()`, for API clients):

1. A longest-common-subsequence match of the old words and the new tokens (compared without
   punctuation and case) keeps the matched words with their times, score and redaction. A common
   prefix and suffix are matched first; the middle is matched by LCS only when its table is at most
   `LCS_LIMIT` = 400,000 cells (`editor.LCS_LIMIT` in Python, the same constant in `editor.js`).
   A larger middle is not matched at all and counts as one changed stretch (step 2). Both
   implementations use the same algorithm, tie-breaking and limit, so the browser and the server
   produce the same words for the same edit.
2. Within a changed stretch, new tokens replace the old words one for one: a corrected spelling
   keeps the time of the word it replaces. The score is dropped, because the word is no longer
   the model's.
3. Further new tokens get times spread between their timed neighbours, in proportion to their
   length (estimated, no score). Deleted tokens drop their words.

**Split** cuts the words at the cursor. The first part ends at its last timed word, the second
starts at its first timed word. Without word times the split point is proportional to the text.
**Merge** concatenates the words.

Words that WhisperX could not align (numbers, symbols) keep having no time unless they are
retyped.

## Saving and versions

`PUT /api/transcripts/{id}/segments` with `{"base": <stamp>, "segments": [...]}`:

* `base` is the `stamp` (sha1 of `prepis.json`) that `GET /api/transcripts/{id}/segments`
  returned. A different current stamp returns `409 edit_conflict`. `null` overwrites.
* Segments are validated (times, speaker labels `[\w.-]{1,40}`, sizes). Emptied segments are
  removed. A save that changes nothing writes nothing.
* Before writing, the current `prepis.json` is copied to `versions/prepis.<UTC time>.<reason>.json`,
  at most once per 5 minutes of editing (`VERSION_EVERY_S`) and up to 50 copies. Reasons are
  `edit`, `restore` and `retranscribe`. The pipeline also keeps a copy before a new run of the
  same source replaces an edited transcript.
* `GET /api/transcripts/{id}/versions` lists them.
  `POST /api/transcripts/{id}/versions/{version}/restore` makes one current again. The current
  state is kept as a version first, so a restore can be undone.

Each save goes through `Library.save_segments()`, so the search index is updated by its
listener. It then:

| derived data | what happens | why |
|---|---|---|
| search index | reindexed (library listener) | must find the new text |
| `prepis.txt`, `titulky.srt` | written again | stored exports of the transcript |
| `analysis.json` (LLM) | kept, shown as **outdated** (`segments_stamp`) | expensive; the user decides whether to run it again |
| `voices.json`, `speakers.json` | kept | a text fix does not change the voices. Moving segments between speakers leaves each label's voiceprint as it was computed. That is good enough for recognition, and voiceprints can be recomputed from the voice panel |
| `metadata.json` | `speakers` updated, `extra.edited`, `extra.custody` (new hashes) | see [PRIVACY.md](PRIVACY.md) |
| `audit.jsonl` | `edited` / `restored` record with counts, never the text | see [PRIVACY.md](PRIVACY.md) |

Saves of one transcript are serialised with `fsutil.file_lock` on `prepis.json`, which also
covers other processes.

## API summary

| endpoint | |
|---|---|
| `GET /api/transcripts/{id}/segments` | segments with words (1:1 to the text), `stamp`, `names`, `language`, `versions` |
| `PUT /api/transcripts/{id}/segments` | save: `{base, segments}` → `{stamp, changed, version, versions}` |
| `GET /api/transcripts/{id}/versions` | `[{id, time, reason, size}]`, newest first |
| `POST /api/transcripts/{id}/versions/{version}/restore` | restore, returns what `GET .../segments` returns |

Errors: `edit_conflict` (409), `invalid_segments` (422), `version_not_found` (404), `edit_busy` (503).

## Not done yet

* No collaborative editing: two editors on one transcript conflict (409) and the second one must
  reload.
* No find-and-replace across the transcript, and no editing of word times by hand (no waveform).
* Speaker changes apply to whole segments. To give part of a segment to another speaker, split it
  first.
* Very long transcripts (thousands of segments) render all rows at once. This was tested with
  1,256 segments (87 min) and works smoothly, but there is no virtualisation.
