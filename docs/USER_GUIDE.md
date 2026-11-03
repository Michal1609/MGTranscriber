# User guide

A step-by-step guide for everyday use of MGTranscriber, from installation to export. Details live in the
topic documents, linked where they matter.

Contents: [Install](#1-install) · [Hugging Face token](#2-set-up-the-hugging-face-token) ·
[First transcript](#3-your-first-transcript) · [Editor](#4-correct-the-text-in-the-editor) ·
[Speakers and voice memory](#5-speakers-and-voice-memory) · [Summary and questions](#6-summary-and-questions) ·
[Folders and search](#7-folders-and-search) · [Exports](#8-exports) · [Watching sources](#9-watch-folders-podcasts-and-channels) ·
[Offline mode](#10-offline-and-sensitive-recordings) · [Problems](#11-common-problems)

## 1. Install

1. Download the project from GitHub (**Code → Download ZIP**) and unpack it somewhere permanent, for example
   `C:\Tools\MGTranscriber`.
2. Run **`install.bat`**. It checks for an NVIDIA GPU, installs what is missing (uv, ffmpeg, Deno, the Python
   packages and PyTorch; 3-5 GB on the first run), and creates a desktop and Start menu shortcut. No
   administrator rights are needed.
3. When it asks, optionally install [Ollama](https://ollama.com). It powers summaries and questions
   ([section 6](#6-summary-and-questions)).

Running the installer again only fixes what is missing. `install.bat -Repair` reinstalls the packages and
`install.bat -Uninstall` removes the environment and shortcuts; your transcripts (`output\`), data (`data\`) and
`.env` stay. The installer's log is in `%LOCALAPPDATA%\MGTranscriber\logs`.

Other ways to install (one-line PowerShell, Docker, manual with uv): [README](../README.md#install).
Docker, NAS and GPU passthrough: [DOCKER.md](DOCKER.md).

**Start the app** with the app's shortcut (Desktop or Start menu) or `web.bat`. A console window stays open while it runs, and
your browser opens <http://127.0.0.1:8000>. Close the console window to stop it.

## 2. Set up the Hugging Face token

Telling speakers apart uses pyannote models from Hugging Face. They are free but you must accept their terms once:

1. Create an account at <https://huggingface.co>.
2. Open <https://huggingface.co/pyannote/speaker-diarization-community-1> and click **Agree**.
3. Create a token of type **Read** at <https://huggingface.co/settings/tokens>.
4. In the app, open **Settings** (top bar), find **Hugging Face** and paste the token. The app checks it with
   Hugging Face before saving it into `.env` and never shows it again.

The Settings page lists everything the app needs (GPU, ffmpeg, Deno/Node for YouTube, token, Ollama, engines) with a
green, yellow or red mark and says what to do for each red one. If the token or the consent is missing when you
start your first transcript, a short wizard walks you through these steps and offers **Continue without
speakers**.

Saving the token in the web UI works only when the server runs for this computer alone. Otherwise put
`HF_TOKEN=hf_...` into `.env` and restart.

## 3. Your first transcript

1. On the home page, in **New transcript**, paste a link (YouTube, a podcast episode or another page yt-dlp
   understands) **or add a recording from your computer**: press **Choose files…** or drag files onto the card
   (see below). You can do both at once.
2. Choose the **language** of the recording (**Auto**, the default, detects it; a fixed language is more reliable when you know it, and `canary` needs one).
3. Enter the **number of speakers** if you know it. This makes the speaker split noticeably better. Leave it empty
   if you do not.
4. Optionally pick another **engine** or **model** ([engines](../README.md#engines)): `large-v3` is the most
   careful default, `large-v3-turbo` about three times faster with a small loss, `parakeet` the fastest recognition step (with
   speaker detection the whole transcript takes about as long as with turbo), `canary` best for Czech in our test (25 European languages; the language must be chosen).
5. If people talk over each other, tick **Separate voices (overlapping speech)** (see
   [below](#overlapping-speech-separate-voices)).
6. Press **Start**.

A toast confirms that the job started, and a **Running** strip under the form shows it with the current phase
(preparing, downloading, transcribing, aligning, speakers), a percentage, **Cancel** and a link to its log. Next to it
is "+N queued · M failed recently", a link to the **Jobs** page. The strip is hidden while nothing runs or waits.
Jobs run one at a time; the rest wait in the queue, and the queue survives a restart. The first run is slower because
models are downloaded (they are kept in `data\models\`).

### Overlapping speech: separate voices

When people talk at the same time, a recognition engine writes down only the loudest voice. With **Separate
voices (overlapping speech)** the recording is first split into one track per speaker, every track is transcribed on
its own and the results are merged by time. Where two people speak at once, their sentences overlap in time in the
transcript, each with its own speaker.

- It is **much slower** (the recognition runs once per voice, plus the separation) and works only with the **local
  engines** (WhisperX, Parakeet, Canary). With a cloud engine or YouTube captions the box is greyed out and the API
  answers `400 separation_unavailable`.
- It needs the separation models (the same Hugging Face access as speaker detection; the error message says
  what is missing). Setting the **number of speakers** helps.
- Words of a track that fall outside that speaker's turns (the other voice leaking into the track) are dropped.
- A separated run uses **one language for the whole recording**: it is detected on the voice that speaks most in the
  first 30 seconds and used for all voices. If nobody speaks for at least 5 seconds at the start (music, silence),
  each voice detects its own, so set the language explicitly for recordings that begin with a long intro.
- In the editor you may move a short interjection to the other speaker even when it overlaps their sentence.
- The transcript's audio stays the original recording; the detail page and the job row say that voices were separated.
- Voices with less than 3 seconds of speech in total are treated as separation noise and get no track; in a short
  recording the limit is 2 % of all speech. A long run can therefore show fewer speakers than the model first found.
- Command line: `mgtranscriber transcribe talk.mp4 --separate-speakers`. MCP: `transcribe(..., separate_speakers=true)`.

### The Jobs page

**Jobs** in the top bar (with a badge for running + queued jobs) lists every job, 20 per page:

- tabs **Active** (running first, then in queue order), **Finished**, **Failed / cancelled** and **All**; the arrow
  keys, Home and End move between the tabs;
- each row shows the title or source, engine and model, when it was created and finished, how long it took, its status
  and phase. Click the title to open the live log;
- **Cancel** (active jobs), **Retry** (failed, cancelled or interrupted: queues the same request again and replaces
  the entry; not offered for uploads, whose file is deleted when the job ends), **Open transcript** (finished) and
  **Remove from history** (finished or failed only: the transcript is never deleted);
- **Clear finished…** removes all finished, failed and cancelled entries after a confirmation. Transcripts stay.

A failed job shows the reason in one line; the technical details (a traceback) are behind **Details**.

When the job finishes, **Open transcript** shows the result: the text in blocks per speaker, in colours, with times.
If the folder has audio, a player at the bottom jumps to a time when you click it, and the spoken sentence is
highlighted.

**A recording from your computer** can be added right in the form. Press **Choose files…** or drop audio or
video files anywhere on the **New transcript** card; several at once are fine, each becomes its own job. The files
are listed under the link field (✕ removes one) and are uploaded when you press **Start**, with a progress bar per
file. Refused files (not audio or video, or larger than the limit shown next to the button, 4096 MB by default,
setting `max_upload_mb`) are named in a message and are not uploaded. This works the same with offline mode on and
when the server does not read files from its own disk (`allow_local_paths` off), because the file is uploaded, not
read from a path. The uploaded file is kept as the transcript's audio, so the player works (for formats a
browser cannot play, such as `.mkv` or `.avi`, only the text is kept).

From the command line a file is transcribed like this:

```
uv run mgtranscriber transcribe interview.mp4 --language en --speakers 2
```

(Run it from the project folder. Options: `uv run mgtranscriber transcribe --help`.) Or put the file into a
[watched folder](#9-watch-folders-podcasts-and-channels) and it is queued by itself.

Results are in `output\<title>\`: `prepis.txt`, `titulky.srt`, `prepis.json`, `metadata.json` and the audio.

**YouTube note:** transcribe content you have the right to use, and follow YouTube's Terms of Service. See the
[disclaimer in the README](../README.md#youtube-and-terms-of-service).

### Recording from the microphone or a meeting

Press **Record…** next to **Choose files…**. This records first and transcribes afterwards (it is not live
transcription). Choose what to record:

- **Microphone**: pick the device if you have several.
- **Meeting audio from a tab or screen**: for Teams, Zoom or Meet running **in the browser**. The browser asks what
  to share; pick the browser tab with the meeting and tick **Share tab audio**. Only the sound is recorded, not the
  picture. This works in Chrome and Edge. Firefox and Safari cannot share tab audio, so they offer only the microphone. A
  desktop app (Teams or Zoom as a program) cannot be captured this way: share the whole screen with system audio
  (Windows, Chrome/Edge) or use the microphone.
- **Microphone + tab audio**: your voice and the others on the call in one recording.

**Start recording** shows a red indicator, the running time and a level meter; the browser tab title also shows
`● 12:34`. **Pause** and **Resume** keep one recording, **Stop and transcribe** sends it to the queue with the form's
language, speakers, engine and folder, like an upload, and **Discard** throws it away (after a confirmation).
Recording people usually requires their consent: tell them.

**Long meetings are safe.** The recording is uploaded in 5-second parts *while* you record, so a crashed or closed
tab, a closed laptop lid or a power cut loses at most the last few seconds. The line "Saved on the server: 37:12"
shows how far it got; if the server cannot be reached the parts wait and are sent again. After a restart or a
reload the home page lists **Unfinished recordings** ("Recording from 14:02 (37 min)"): **Transcribe** sends what was
saved, **Discard** deletes it. Unfinished recordings nobody comes back to are removed after 7 days. Keep the tab open
while recording (other pages of the app are fine) and prevent the computer from sleeping; the page asks the system
to keep the screen on where the browser allows it.

The recording is stored as `webm`/Opus (Chrome, Edge, Firefox; Safari produces `m4a`/AAC, which works but is not
tested regularly) in `data\recordings\` until it is finished, then as the transcript's audio. The total size
counts against `max_upload_mb` (Opus speech is roughly 30 MB per hour).

**"Recording needs a secure connection".** Browsers allow the microphone and screen sharing only on `localhost` or
over HTTPS. `http://localhost:8000` and `http://127.0.0.1:8000` work. Reaching the app from another computer as
`http://192.168.1.20:8000` does not, so on a phone or a second PC you need HTTPS. Options:

- put a reverse proxy with a certificate in front (Caddy: `caddy reverse-proxy --from https://transcribe.local --to
  127.0.0.1:8000`, with a certificate from its local CA or one of your own; add the name to `allowed_hosts`);
- `tailscale serve --https=443 http://127.0.0.1:8000` (valid certificate inside your tailnet);
- for a test only, in Chrome set `chrome://flags/#unsafely-treat-insecure-origin-as-secure` for that address.

Whatever you use, also set `MGTRANSCRIBER_API_KEY` when the app is reachable from the network.

## 4. Correct the text in the editor

Open a transcript and press **Edit**. Each row is a segment: a time (click to play from there), a speaker and the
text. Type to fix the text while the audio plays; changes are saved about a second after you stop typing, and the
previous state is kept as a version you can restore (**Versions**).

| key | action |
|---|---|
| `Ctrl`+`Space` | play / pause |
| `Ctrl`+`,` / `Ctrl`+`.` | back / forward 5 s |
| `Enter`, `Alt`+`↓` / `Alt`+`↑` | next / previous segment (the audio jumps there) |
| `Ctrl`+`Enter` | split the segment at the cursor |
| `Backspace` at the start of a segment | merge with the previous segment |
| `Ctrl`+`Shift`+`X` | redact (or un-redact) the selected words |
| `Ctrl`+`Z` / `Ctrl`+`Y` | undo / redo |

The **Shortcuts** button in the editor shows this table. Word times are kept as well as possible as you edit; the
rules, limits and the versions mechanism are in [EDITOR.md](EDITOR.md). Press **Done** to leave the editor.

## 5. Speakers and voice memory

The model labels speakers `SPEAKER_00`, `SPEAKER_01` and so on. In the **Speakers** panel of the transcript:

- **Rename** a speaker: type a name. It is used everywhere (screen, TXT, SRT, DOCX). The names are stored in
  `speakers.json` in the transcript's folder.
- **Remember the voice**: type the speaker's name first, then press **Remember voice**. The app stores a voiceprint of that speaker. Later transcripts then
  suggest or fill in the name by themselves. The speaker needs at least 20 seconds of speech.
- **Suggestions**: when a voice resembles a remembered person, you see the suggestion and can **confirm** or
  **reject** it. Confident matches are named automatically, but only into empty names, and a name you typed is never
  replaced.
- The **Voices** page (top bar) lists the remembered people and their samples, lets you rename or delete them and
  export or erase the whole voice database.

### Voices across the whole archive

Once a few people are remembered, you can look for them in everything you have already transcribed:

1. On the **Voices** page press **Scan the whole archive** (or **Scan the archive for …** on a person's page). It
   runs in the background as a job, uses no GPU and can be cancelled in **Jobs**. Tick "also queue voiceprints for
   transcripts that have none" to prepare older transcripts too (this one does use the GPU).
2. Open a person (**Open**). **To confirm** lists the matches found in the archive, best first, with the transcript,
   the speaker label, the score and a short quote. **Play 5 s** lets you listen to the voice, **Yes** / **No**
   decide. **Accept** with a score (default 0.60) confirms everything above it at once.
3. **Where this person speaks** lists every transcript with the speaker and the time they talk. Each row opens the
   transcript. The search box on the same page searches only what this person said; in the API,
   `/api/search?q=...&speaker=Name` does the same.
4. If one person is remembered twice (two names for the same voice), open the one you want to keep and use **Merge a
   duplicate**: the other person's samples and names in transcripts move over and the duplicate disappears. This
   cannot be undone.

Names you typed are never replaced by a scan; automatic names only fill empty speakers.

Voiceprints are biometric data. They stay on your computer, and only your own actions create them. How it works,
accuracy and limits: [VOICES.md](VOICES.md).

If two people are mixed in one speaker, or one person is split in two, set the number of speakers and transcribe
again, or fix individual segments in the editor (split a segment, then change the speaker of one part).

## 6. Summary and questions

On the **Summary** tab of a transcript, choose a provider and press **Analyse**. You get a summary, key points,
chapters with times, verbatim quotes with the speaker and time, and action items. On the **Ask** tab you ask
questions about the recording; when the transcript does not contain the answer, it says so instead of guessing.

- **Local (default):** [Ollama](https://ollama.com), nothing leaves your computer. If it is missing, install it,
  then run `ollama pull gemma3:4b` and `ollama pull bge-m3`. The Settings page has a button that downloads the
  model with progress.
- **Cloud (Gemini, Claude):** better at smaller languages such as Czech and at long recordings, but the **full transcript text is sent to
  the provider**. The app shows a warning, and you enter the keys in **Settings → Cloud API keys** (paste, Save; they go into `.env`, work at once and are never shown again; Test asks the service whether the key is accepted; a key set in the system environment is shown as locked).
- If you edit the transcript afterwards, the analysis is marked **outdated**; press **Analyse again**.
- In Settings, **automatic analysis** runs it after every transcription.

Providers, models, prices and how the result is validated: [LLM.md](LLM.md).

### Translate a transcript

If the language was left on Auto, or you simply want the text in another language, press **Translate to…** above the
transcript, choose the language (and, if you like, the provider and model; the defaults come from the LLM settings)
and press **Translate**. The translation is a job in the normal queue, so it waits for transcriptions that are
already running and shows its progress on the home page and the Jobs page.

- A switcher **Original (EN) | Čeština | …** appears. The translated view has the same speakers, colours and times:
  clicking a line plays the audio at that moment.
- A translation is **read-only**. To change the text, edit the original and press **Translate again** (the
  translation is marked *outdated* as soon as the original changes).
- **TXT, SRT, VTT, MD and DOCX** downloads follow the language you are viewing: the SRT/VTT of a translation work as
  subtitles. A redacted export is only available for the original.
- Search finds the translation too: a Czech query finds an English recording, and the result carries a language tag.
- **Delete translation** removes it; deleting the transcript for good removes its translations as well.
- With a cloud provider the whole text is sent to it (the panel warns you). In offline mode only a local Ollama is allowed.
- **Translating right away:** the *New transcript* form has **Translate to** next to *Save in folder* (default: no
  translation). Whatever you start there (link, uploaded files, a recording) is translated automatically when the
  transcription finishes, as if you had pressed Translate; the translation appears as its own job. If the transcript is
  already in that language nothing happens. If no LLM provider is set up (or offline mode forbids the one you chose) the
  transcription is kept; the form warns you when you start it, and the Jobs page shows *Translation to … was not queued: reason* under the finished job; set a provider in Settings and press
  **Translate to…** on the transcript.
- Translations have no word-level timestamps; the times are those of the original segments.

## 7. Folders and search

### Folders

Folders (also called projects) keep a growing library tidy. They are **virtual**: a transcript only remembers which
folder it is in, so nothing is moved on disk and links, search, voices and the API keep working. A transcript is
in one folder; folders can be nested up to five levels (`Podcasts/Interviews`).

- **Tree.** The sidebar on the home page lists **All transcripts**, **Unfiled** and your folders with the
  number of transcripts in each (a folder counts what is in its subfolders too). On a phone it is a drop-down.
  Opening a folder shows its transcripts and those of its subfolders; the path above the tiles is a breadcrumb.
- **Create, rename, delete.** **＋** next to *Folders* creates a folder (type `Client X/2025` to make a subfolder at
  once; an empty folder is fine). Inside a folder, **New subfolder**, **Rename** (change the whole path to move it
  elsewhere) and **Delete**. Deleting never deletes transcripts: choose whether they go to the parent folder or to
  *Unfiled*; subfolders are removed with their folder and their transcripts go to the same place.
- **Move transcripts.** Drag a tile onto a folder in the tree (a selected group moves together), or tick the
  checkbox on one or more tiles and press **Move to…**, which also works with the keyboard alone
  (`Tab` to the checkbox, `Space` to select, `Enter` on **Move to…**). A transcript page has the folder as a
  breadcrumb and its own **Move to…**.
- **Save in the right place from the start.** The new-transcription form has **Save in folder**; it follows the
  folder open in the library, and **New folder…** creates one. It applies to links and uploaded files, and a
  watched source can have its own target folder (see [WATCH.md](WATCH.md)).

Transcribing the same source again keeps the transcript in its folder.

### Deleting transcripts

- **Delete** in the library: the bin icon on a tile deletes that transcript; tick several tiles and use **Delete** in
  the selection bar to delete them together. The transcript page has a **Delete** button next to its title.
- A dialog names the transcripts and asks once ("This cannot be undone"); there is no ID to type. After deleting
  from the transcript page you land in the library.
- A transcript that a job or an analysis is still working on is not deleted (the toast says why).
- For sensitive recordings the **Privacy** panel's deletion still overwrites the files before removing them
  ([PRIVACY.md](PRIVACY.md)); the quick delete uses the same operation but never asks for the ID.

### Search

- On the home page, the field above **Finished transcripts** searches the titles **and the text of all transcripts**,
  ignoring diacritics and case. A click on a match opens the transcript at that exact time.
- Inside a transcript, **Search the transcript** highlights matches. `Enter` and `Shift`+`Enter` jump to the next
  and previous one.
- When a folder is open, **Search only in this folder** limits the text search to it (and its subfolders).

## 8. Exports

A transcript has download buttons for **TXT**, **SRT**, **VTT**, **JSON**, **MD** (Markdown) and **DOCX**
(optionally as a table). They use the current speaker names, and TXT and SRT are also saved in the transcript folder
when the transcription finishes. **Copy text** puts the transcript on the clipboard.

For recordings with sensitive passages, mark words as redacted in the editor, then use the **Privacy** panel to
export a redacted text and a beeped audio file: [PRIVACY.md](PRIVACY.md).

For other programs, the data is in `prepis.json` (segments, words with times and speakers), or use the
[API](API.md) and the [MCP server](MCP.md).

## 9. Watch folders, podcasts and channels

The **Watching** page turns the app into something that transcribes new material by itself:

1. Press **Add source** and choose its kind: a **folder** on this computer, a podcast **RSS** feed, or a
   **YouTube** channel or playlist.
2. **Show what it finds** previews the items without queueing anything. Set the language, the number of speakers, the engine, whether to analyse afterwards, and whether to take only
   new items or the latest N too.
3. Save. New items are added to the normal queue.

Files in a folder are queued only after they stop changing (so half-copied files are ignored), and the same item is
never queued twice. Sources are checked at sensible minimum intervals so that servers are not hammered.
Everything about intervals, deduplication and the optional webhook: [WATCH.md](WATCH.md).

## 10. Offline and sensitive recordings

For recordings that must not leave the computer, turn on **Offline mode** on the Settings page. The app then
refuses cloud engines, cloud LLMs, a remote Ollama and links, in the server itself and not only in the interface,
so API clients and watched folders cannot get around it. Local engines, local files and a local Ollama keep working.
The top bar shows an **Offline mode** badge.

One thing offline mode does not do: it does not block the first download of model files from Hugging Face (no audio
or text is sent there). On a machine without internet, download the models once beforehand and set
`HF_HUB_OFFLINE=1`.

The same page of documentation covers the chain of custody (hashes of the source and outputs), the audit log,
redaction and secure deletion, and, importantly, what they do **not** guarantee: [PRIVACY.md](PRIVACY.md). Read its
limits section before relying on them in a legal or medical setting.

## 11. Common problems

| problem | what to do |
|---|---|
| The page does not open | Is the console window of `web.bat` still open? The address is <http://127.0.0.1:8000>. If port 8000 is taken, run `uv run mgtranscriber serve --port 8080`. |
| A yellow banner says the server runs an older version | The page was updated but the server was not restarted (it is still running the old code, so new pages may be empty or fail). Close the console window of `web.bat`, start it again and reload the page. |
| **Settings** shows a red item | It says what to do. The same checks run from the command line: `uv run python -m mgtranscriber.diagnostics`. |
| "Agree to the terms" or a token error when starting | Do [section 2](#2-set-up-the-hugging-face-token), including **Agree** on the model page. Or continue without speakers (`--no-diarize`). |
| YouTube download fails with a message about a JavaScript runtime | Install [Deno](https://deno.com) (`winget install DenoLand.Deno`) or Node.js 22+ and restart the app. Videos with captions work without it through the `youtube` engine. |
| YouTube download fails with 403 or "sign in" | YouTube changes often. Update yt-dlp (`uv sync --upgrade-package yt-dlp`) and try again. If it is a restricted video, download the file yourself and transcribe the file. |
| Out of GPU memory (CUDA out of memory) | Close other GPU programs (games, Ollama models), or lower the load: `--batch-size 4`, or the `large-v3-turbo` or `parakeet` engine. |
| Very slow | Is it running on the CPU? Settings shows the GPU. Use `large-v3-turbo` or `parakeet`. A first run also downloads models. |
| A language comes out as a neighbouring one or with strange words (Parakeet; for example Czech written as Slovak) | Parakeet detects the language itself and cannot be forced. Use `canary` or `whisperx` with the language set, for example Czech. |
| Speakers are mixed up or too many | Enter the exact number of speakers and transcribe again, or fix it in the editor. Overlapping speech and phone audio are hard for every model. |
| Summary: "Ollama is not running" | Start Ollama (`ollama serve` or the tray app), then reload the page. Missing model: `ollama pull gemma3:4b`. |
| "Offline mode: ..." errors | Offline mode is on and the action needs the internet. Use a local file and a local engine, or turn offline mode off in Settings. (If `MGTRANSCRIBER_OFFLINE_MODE` is set in the environment, it wins over the setting.) |
| The server must be reached from another computer | Start it with `--host 0.0.0.0` **and** set `MGTRANSCRIBER_API_KEY` in `.env`. Without a key anyone on your network can use it. |
| "Changed elsewhere" when saving in the editor | The transcript was changed in another tab or by a new transcription. Click the status to reload; nothing was overwritten. |
| I want to undo an edit | **Versions** in the editor lists stored versions; **Restore** makes one current (the current state is stored first). |
| Something else | Open an issue on GitHub with the output of `uv run python -m mgtranscriber.diagnostics` (it never prints tokens) and the job's log. |

Tips: put the two things you change most often in `.env`, for example `MGTRANSCRIBER_MODEL=large-v3-turbo` and
`MGTRANSCRIBER_LANGUAGE=en`; the Settings page shows such values as locked.
