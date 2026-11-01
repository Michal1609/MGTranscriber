# MCP server

MGTranscriber includes a [Model Context Protocol](https://modelcontextprotocol.io) server.
AI assistants (Claude Desktop, Claude Code, Cursor, VS Code, ...) can use it to
transcribe videos and files on your GPU, search all your transcripts, read them and
name speakers.

It is built on the official MCP Python SDK (`mcp` 2.x, protocol 2026-07-28; older
clients that use the initialize handshake work too) and offers both recommended
transports:

| transport | start | use when |
|---|---|---|
| **stdio** | `mgtranscriber mcp` (the client starts it) | the assistant runs on the same computer; nothing else to start |
| **Streamable HTTP** | part of `mgtranscriber serve`, at `http://127.0.0.1:8000/mcp` | the web server is running anyway, or the assistant is on another computer |

The legacy HTTP+SSE transport is not offered (deprecated in the spec).

## Tools

| tool | what it does |
|---|---|
| `transcribe(url_or_path, language?, speakers?, wait?, folder?, replace?, separate_speakers?)` | Starts a job for a video URL (YouTube or anything yt-dlp supports) or a local audio/video file. Returns `job_id`, `status`, `phase`, `percent`, `position` and, when done, `transcript_id`. `language` is checked at once (ISO 639 code or `auto`; the same rule as `POST /api/jobs`). With `wait=true` it blocks up to 10 minutes and sends progress notifications; when the limit passes the job keeps running and the answer has `wait_timed_out: true` (poll `job_status`). `folder` (`"Podcasts/Interviews"`, created when new) puts the new transcript into a library folder. A source that is already in the library is transcribed again and the new transcript **replaces** the old one, including speaker names set by hand: the answer then has `existing_transcript_id` and a `warning`; with `replace=false` it is an error that names the existing transcript instead. `separate_speakers=true` transcribes every voice of an overlapping recording on its own (much slower; local engines only; an error at once when the server's engine or the separation models do not allow it). |
| `job_status(job_id)` | State of a job, plus the last log lines. |
| `list_transcripts(query?, limit?, folder?, include_subfolders?)` | The library, newest first, each with its `folder`. `query` filters by title or source, ignoring case and diacritics. `folder` limits the list to a library folder (`""` = unfiled; an unknown one is an error that points to `list_folders`), by default with its subfolders. |
| `list_folders()` | The folders as paths with `count` (directly in it) and `total` (with subfolders), plus how many transcripts are unfiled. |
| `search_transcripts(q, limit?, transcript_id?, speaker?, folder?, include_subfolders?)` | Full-text search over all transcripts (or one library `folder`). `speaker` (a name or a label such as `SPEAKER_01`, case and diacritics ignored) keeps only what that person said; with an empty `q` it lists their segments. Hits are ordered by relevance. All words must be in the **same segment** (AND, prefix-matched), so 1-2 keywords or a `"quoted phrase"` work best; a multi-word query without hits gets a `hint`. An unknown `speaker` is an error ("Unknown speaker ..., see list_speakers"); one unambiguous close name is used and reported as `speaker_matched`. Each hit has the time (`hh:mm:ss` and seconds), the speaker, a snippet with `**matches**`, a ready `citation` ("title @ time"), a link into the web UI (`url`) and `lang`: `null` for the original text, a language code when the hit is in a translation (see [LLM.md](LLM.md)) (read it with `get_transcript(..., lang=)`). |
| `get_transcript(transcript_id, format?, start?, end?, max_chars?)` | `format="text"` gives speaker turns with `[hh:mm:ss]`; `"segments"` gives a list with seconds. `start`/`end` (seconds) select a range, widened to whole segments (the text may begin a little before `start` and end after `end`; `range` says what came back). `start >= end`, a `start` after the end of the transcript, a negative value or `max_chars <= 0` is an error; `max_chars` below 1000 is raised to 1000 with a `warnings` entry. Long transcripts come in pages of about `max_chars` (default 20 000): when `truncated` is true, call again with `start=next_start`. `lang` reads a stored translation: `translation_lang` echoes it (`null` = the original) and `transcript_language` is the language of the original recording. |
| `list_speakers()` | The people in the [voice memory](VOICES.md): name, `person_id`, and over the whole library `transcripts` and `speech_seconds` of the **sure** appearances (confirmed, auto, named by hand: the numbers `find_speaker` gives by default). `suggested_transcripts`/`suggested_seconds` are unverified guesses and are never added to those; `voice_sample_seconds` is only the length of the enrolled voice samples. |
| `find_speaker(name? or person_id?, limit?, include_suggestions?, include_spans?)` | Where that person speaks, newest first: transcript, speaker label, how sure (`confirmed`, `auto`, `manual` are sure; `suggest`, only with `include_suggestions`, is an unverified guess and has `needs_confirmation: true`), a per-transcript summary (`first_time`, `speech_seconds`, `span_count`) and, unless `include_spans=false`, the time spans of their speech (`hh:mm:ss` and seconds, up to 20 per transcript and 150 per answer; `spans_truncated` says some are left out), with a link to the first one. At most 25 transcripts per answer: `transcripts_truncated` says more exist (`total` is the real count). `sure_total`/`sure_speech_seconds` match `list_speakers`. Read the words with `get_transcript(transcript_id, start, end)`. |
| `rename_speakers(transcript_id, mapping)` | `{"SPEAKER_00": "Alice"}`. Merged with the existing names; an empty name removes one. Used by the web UI, the exports and search. |

Read-only tools carry `readOnlyHint`, so clients can allow them without asking.

`url` fields (a link into the web UI) are `null` when no web server runs: over stdio the link is only
filled in when a web server answered at start (over /mcp: see "Links" below). That is not an error; cite title and time instead.

Resources:

- `mgtranscriber://transcripts`: the library as JSON.
- `mgtranscriber://transcripts/{id}`: one transcript as text (percent-encode ids with spaces).

### Long-running transcription

`transcribe` returns a job id at once and the assistant polls `job_status` (the
server instructions tell the model to do so). `wait=true` blocks up to 10 minutes and
sends progress notifications, a convenience for short media. This works with every
client.

**MCP Tasks** (the `io.modelcontextprotocol/tasks` extension of the 2026-07-28 spec,
SEP-2663: the server answers with a task, the client polls `tasks/get` and can cancel)
are *not* used, as of 10/2026:

- The official Python SDK (`mcp` 2.3.0, the one used here) does not implement them: its
  release notes list "the tasks extension (SEP-2663)" as a known gap (v2.0.0, v2.2.0),
  and only the generic extension hooks (`mcp.server.extension`) exist.
- FastMCP 4.x does (`fastmcp[tasks]`, backed by the `pydocket` job queue), but using it would mean
  replacing the server framework and adding a second job system next to our own queue.
  A tool that supports tasks runs *synchronously* for clients that do not
  negotiate the extension, which is worse than the job id for every client that exists
  today. Writing the protocol by hand on top of the SDK hooks is not worth the risk
  of drifting from a spec still moving.

Job ids map one-to-one to task ids, so Tasks can be put on top when the SDK ships
them, without changing the tools; `job_status` stays as the fallback for clients
without Tasks.

### Where jobs run

Jobs go to the web server's queue whenever one runs, so MCP transcriptions do not
compete with the web UI's for the GPU.

- **Streamable HTTP (`/mcp`):** the web server's own queue. Jobs show up in the web UI.
- **stdio (`mgtranscriber mcp`):** for every `transcribe` call it checks whether a web
  server answers at the configured host/port (`MGTRANSCRIBER_HOST`/`MGTRANSCRIBER_PORT`,
  default `127.0.0.1:8000`, with `MGTRANSCRIBER_API_KEY` if set) and then:
  - if one answers, sends the job to it (also when the web server was started after
    the MCP server);
  - if one answers but refuses the key (401/403), the call fails with "set the same
    `MGTRANSCRIBER_API_KEY` for the MCP server" instead of starting a second queue;
  - otherwise runs the job in its own queue in the MCP process, stored in
    `data/mcp-jobs.json`. That queue stops when the client closes the server; an
    unfinished job resumes the next time it starts.
  - `--server URL` sets the web server to use; `--local` always uses the own queue.

  One overlap remains: a job already running in the MCP process's own queue keeps
  running when a web server starts, so for that job's duration two transcriptions can
  share the GPU. If the web server stops during a session, `job_status` reports that
  it is not reachable.

Reading tools always work on the library files directly, so search and reading work
without the web server.

Links: over stdio the `url` fields are `null` unless a web server answered at the configured
address when the MCP server started (they point to that server). Over `/mcp` they point to
the web server's address, except when it listens on all interfaces (`0.0.0.0`): then there is
no single address to link to and they are `null` too.

Local file paths: always allowed over stdio (the assistant runs as you, on your
computer). Over HTTP they follow `MGTRANSCRIBER_ALLOW_LOCAL_PATHS`, which by default
allows them only when the server listens on loopback. Network paths (`\\server\share`) are refused in both cases
unless `MGTRANSCRIBER_ALLOW_UNC_PATHS=true`, because text the assistant reads (a web page, an e-mail) could
make it name one and so send your Windows credentials to a stranger's server. Since stdio can read any
audio your account can, prompt injection can also put such files into the library.

## Setup

In the examples below, replace `<project>` with the folder of this project, e.g.
`C:\Tools\mgtranscriber`. The command is the
installed script `<project>\.venv\Scripts\mgtranscriber.exe`; on macOS/Linux use
`<project>/.venv/bin/mgtranscriber`. It reads `.env` and `settings.json` from the
project, so `HF_TOKEN`, `MGTRANSCRIBER_MODEL` etc. apply as usual.

Diarization (speaker labels) needs `HF_TOKEN` in `.env`, as for the CLI.

For HTTP, start the web server first (`web.bat` or `mgtranscriber serve`). If you set
`MGTRANSCRIBER_API_KEY`, send it as `Authorization: Bearer <key>`.

### Claude Code

```bash
# stdio, available in every project of your user account
claude mcp add --scope user MGTranscriber -- "<project>\.venv\Scripts\mgtranscriber.exe" mcp

# or Streamable HTTP (web server running)
claude mcp add --scope user --transport http MGTranscriber http://127.0.0.1:8000/mcp
claude mcp add --scope user --transport http MGTranscriber http://127.0.0.1:8000/mcp --header "Authorization: Bearer <key>"
```

Without `--scope` the server is added with the default `local` scope: it works **only in the
directory where you ran the command** (and is invisible elsewhere). Use `--scope user` for all
your projects, or `--scope project` to write `.mcp.json` into the current project (shared
through version control). If you already added it without a scope, remove that entry and add
it again:

```bash
claude mcp remove MGTranscriber -s local
```

Check with `claude mcp list`, or `/mcp` inside Claude Code.

Non-interactive use without changing any settings:

```bash
claude -p "Search my transcripts for 'AI safety' and cite the times." \
  --mcp-config mgtranscriber-mcp.json --strict-mcp-config \
  --allowedTools "mcp__MGTranscriber__search_transcripts,mcp__MGTranscriber__get_transcript"
```

Tool names are `mcp__<server name>__<tool>`, and the server name is the key you used in
`claude mcp add` or in `mcpServers` (case matters): `MGTranscriber` in these examples, so
`mcp__MGTranscriber__search_transcripts`. Use the same spelling in `--allowedTools` and in
permission rules.

`mgtranscriber-mcp.json` has the same `mcpServers` shape as the Claude Desktop file below.

### Claude Desktop

Edit `claude_desktop_config.json` (Settings → Developer → Edit Config; on Windows
`%APPDATA%\Claude\claude_desktop_config.json`) and restart Claude Desktop:

```json
{
  "mcpServers": {
    "MGTranscriber": {
      "command": "<project>\\.venv\\Scripts\\mgtranscriber.exe",
      "args": ["mcp"]
    }
  }
}
```

Use `"env": {"MGTRANSCRIBER_PORT": "8000", "MGTRANSCRIBER_API_KEY": "<key>"}` if your web
server uses another port or a key. Claude Desktop's config file starts local (stdio)
servers; remote HTTP servers are added as connectors and must be reachable over HTTPS
(see "ChatGPT and other remote clients" below). On the same computer, stdio is the
simple choice.

### Cursor

`~/.cursor/mcp.json` (all projects) or `.cursor/mcp.json` (one project):

```json
{
  "mcpServers": {
    "MGTranscriber": {
      "command": "<project>\\.venv\\Scripts\\mgtranscriber.exe",
      "args": ["mcp"]
    }
  }
}
```

Streamable HTTP instead:

```json
{
  "mcpServers": {
    "MGTranscriber": {
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "Authorization": "Bearer <key>" }
    }
  }
}
```

### VS Code (GitHub Copilot agent mode)

`.vscode/mcp.json` in a workspace, or run **MCP: Add Server** from the command palette:

```json
{
  "inputs": [
    { "type": "promptString", "id": "mgtranscriber-key", "description": "MGTranscriber API key", "password": true }
  ],
  "servers": {
    "MGTranscriber": {
      "type": "stdio",
      "command": "<project>\\.venv\\Scripts\\mgtranscriber.exe",
      "args": ["mcp"]
    },
    "mgtranscriber-http": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "Authorization": "Bearer ${input:mgtranscriber-key}" }
    }
  }
}
```

Use one of the two entries. Drop `inputs` and `headers` when the server has no key.

### Other clients and testing

Any MCP client works with the same command (`mgtranscriber mcp`) or URL (`/mcp`). To
explore the tools by hand: `npx @modelcontextprotocol/inspector`, then connect to the
command or the URL.

From Python (official SDK):

```python
import asyncio
from mcp import Client

async def main():
    async with Client("http://127.0.0.1:8000/mcp") as client:
        hits = await client.call_tool("search_transcripts", {"q": "safe AI"})
        print(hits.structured_content["hits"][0]["citation"])

asyncio.run(main())
```

### ChatGPT and other remote clients

ChatGPT only connects to remote MCP servers over public **HTTPS**; it cannot start a
local command or reach `127.0.0.1`. That means exposing `/mcp` through a tunnel or a
reverse proxy with TLS, with an API key set (`MGTRANSCRIBER_API_KEY`) and the public
host name in `MGTRANSCRIBER_ALLOWED_HOSTS`. Check which authentication your client
supports before exposing anything. Exposing the server to the internet makes your GPU
and your transcripts reachable by anyone who gets the key. MGTranscriber does nothing
ChatGPT-specific.

## Security

- **API key.** `/mcp` uses the same key as the rest of the server (`Authorization:
  Bearer`) and the same Host check (DNS rebinding) and Origin check.
- **What tools can do.** Tools can read every transcript and start transcriptions.
  Over stdio they can also transcribe any file your user account can read.
- **URLs on your network.** When the server listens on another address than loopback,
  `transcribe` refuses URLs whose host resolves to this computer or the local network
  (SSRF guard, see [API.md](API.md#access)).
- **Prompt injection.** Transcript text comes from third-party media. Treat it as
  untrusted input in your assistant, like web pages.
