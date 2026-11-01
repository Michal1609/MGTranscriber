# LLM analysis and questions

Summary, key points, chapters, quotes, action items and "ask the transcript" for every
transcript. Output language is selectable (`auto` = the transcript's language, English, Czech); quotes stay in the original language.
Start it from the **Summary** tab of a transcript (or `POST /api/transcripts/{id}/analysis`).

## Providers

| provider | default model | where it runs | notes |
|---|---|---|---|
| `ollama` (default) | `gemma3:4b` (+ `bge-m3` embeddings for Q&A) | your machine | free, private; ~3.5 GB VRAM at 8k context |
| `gemini` | `gemini-3.5-flash-lite` | Google | needs `GEMINI_API_KEY` |
| `anthropic` | `claude-haiku-5-5` | Anthropic | needs `ANTHROPIC_API_KEY`; set `llm_final_model` (e.g. `claude-sonnet-5-5`) for the summary/chapter steps |

Local setup: install [Ollama](https://ollama.com), then `ollama pull gemma3:4b` and
`ollama pull bge-m3`. If Ollama is not running or the model is missing, the UI/API says so and
shows the command to run (`ollama serve`, `ollama pull <model>`).

Quality notes from the 88 min test interview (English audio, Czech output): `gemma3:4b` gives
usable Czech, 10-16 chapters, ~12 verbatim quotes and takes ~70-80 s on an idle RTX 5070 (several
minutes when the GPU is busy with transcription). `qwen2.5:7b-instruct` was faster but its Czech
was clearly worse. Cloud models are better at Czech and at following the output language.

## Configuration

Keys of `settings.json` (or `MGTRANSCRIBER_LLM_<NAME>` environment variables / `.env`):

| key | default | meaning |
|---|---|---|
| `llm_provider` | `ollama` | `ollama`, `gemini` or `anthropic` |
| `llm_model` | provider default | model for the chosen provider |
| `llm_final_model` | none | stronger model for reduce/chapters/translation |
| `llm_base_url` | `http://127.0.0.1:11434` | Ollama address |
| `llm_embed_model` | `bge-m3` | Ollama embedding model (Q&A retrieval) |
| `llm_num_ctx` | `8192` | Ollama context window (always sent explicitly; Ollama's default of 4096 silently truncates) |
| `llm_language` | `language` setting; `auto` = language of the transcript | language of the generated text |
| `llm_auto` | `false` | analyse automatically after each transcription |

**API keys come from the process environment or the project's `.env`**: `GEMINI_API_KEY`,
`ANTHROPIC_API_KEY`. You can enter them in **Settings → Cloud API keys** (written to `.env`, effective
immediately, no restart; a key from the real environment is locked there). They are never read from
`settings.json` and never returned by the API (`GET /api/llm` only says whether a key is set).

## Privacy

Ollama keeps everything on your computer. **Gemini and Anthropic receive the full transcript text
(and your questions)**; check their data-use terms (for example free-tier Gemini traffic may be used
to improve Google products). The UI shows a warning when a cloud provider is selected. Nothing is
sent unless you start an analysis or ask a question (or enable `llm_auto`).

## How it works

1. Segments are merged into speaker turns (`[mm:ss] Name: text`; names come from `speakers.json`).
2. Chunks of ~4500 tokens (local) or the whole transcript (cloud providers) are analysed (map).
3. Partial summaries are combined (reduce); topic markers are merged into chapters, then spread
   evenly (minimum spacing, long gaps filled from the candidates).
4. A translation pass puts chapter and action-item texts into the target language.
5. **Validation**: every time is snapped to the start of a real segment; every quote must occur in
   the transcript (similarity >= 0.85) or it is dropped, and gets its real time and speaker.

Questions use the whole transcript when it fits in 60 % of the model's context, otherwise the
overlapping 220-word windows closest to the question (`bge-m3` embeddings, computed once per
transcript). An answer is returned only if the model cites passages that really occur in the
transcript; otherwise the result is `found: false` ("The transcript does not say."). For retrieval,
a question whose best window scores below 0.44 cosine is rejected without calling the model, and each
citation must itself be related to the question.

Result file: `analysis.json` in the transcript folder (`provider`, `model`, `language`, `created`,
summary, key points, chapters, quotes, action items, validation stats, token usage).

## Translation

"Translate to…" on the detail page (or `POST /api/transcripts/{id}/translations`) creates a **queued job** (task
`translate`), so it never competes with a transcription for the GPU. Defaults: provider/model from the LLM settings
(`llm_final_model`, when set, for the configured provider).

1. Only segments with text are sent, in batches of up to 8 (local) or 30 (cloud) segments / ~450 (local) or ~1500 tokens.
   Each request carries the speaker names, the title, 3 already translated lines before and 2 original lines after the
   batch as context.
2. Structured output `{"items": [{"id", "text"}]}`. A reply is accepted only if it has exactly the requested ids, once
   each, with non-empty text (order may differ). Otherwise the batch is tried once more, then split in halves, down to
   single segments. Text is never moved between segments. A segment that never works keeps its original text and is
   listed in `untranslated` (the UI says so).
3. Progress follows the share of translated characters.

Result: `translations/<lang>.json` = `{version, source_lang, target_lang, provider, model, created, segments_stamp,
segments: [{start, end, speaker, text}], untranslated, usage, seconds}`. **Word timestamps are dropped** (a
translation has no word alignment; nothing in the exports needs them). The `segments_stamp` of `prepis.json` is the
outdated check, as for `analysis.json`: an edit or re-transcription marks the translation `stale` instead of removing
it. Redaction marks live on words, so a translation cannot be exported redacted (409). The translation is read-only in
the UI; the editor edits the original only. It is indexed by the search with a language tag, removed by secure delete,
and every translation writes a `translated` line in the audit log.

Quality (gemma3:4b, 24 s of demo interview, Czech → English): fluent and correct for most lines in 11 s; it can still
get a meaning wrong ("less water, regularly" became "more water, less frequently"). Cloud models or `gemma3:12b`-class
models are noticeably safer for anything that matters.

## API

* `GET /api/llm` - configured defaults and providers (availability, installed Ollama models).
* `POST /api/transcripts/{id}/analysis` `{provider?, model?, language?}` - starts in the background (202);
  errors such as `llm_unreachable`, `llm_model_missing`, `llm_missing_key` are returned at once.
* `GET /api/transcripts/{id}/analysis` - `{run: {status, percent, message, error}, analysis}`.
* `DELETE /api/transcripts/{id}/analysis` - cancels a running analysis, otherwise deletes the stored one.
* `POST /api/transcripts/{id}/ask` `{question, provider?, model?, language?}` -
  `{found, answer, citations: [{time, speaker, text}], retrieval}`.
* `GET /api/transcripts/{id}/export/md` - Markdown with the analysis (when present) and the transcript.

Analyses run in background threads of the server, outside the transcription queue and the GPU
job worker (Ollama itself shares the GPU, so transcription and analysis slow each other down).

## Prices (verified 2026-10-09 from the providers' pricing pages; they change)

Per 1M tokens, input / output. A 88 min transcript is roughly 25-40k input tokens per pass.

| model | input | output |
|---|---|---|
| `gemini-3.5-flash-lite` | $0.30 | $2.50 |
| `gemini-3.1-flash-lite` | $0.25 | $1.50 |
| `gemini-3.8-flash` | $0.75 | $3.75 (until 2026-12-31, then $1.50 / $7.50) |
| `claude-haiku-5-5` | $0.10 | $0.50 (higher above 100K-token prompts) |
| `claude-sonnet-5-5` | $2 | $10 |
| `claude-opus-5-5` | $4 | $20 |

A full analysis of a long recording costs cents with the Flash-Lite / Haiku models. Batch APIs are
50 % cheaper but are not used here.

## Adding a provider

Subclass `mgtranscriber.llm.base.LLMProvider` (`check`, `generate_json`, optional `embed`,
`with_model`), set `max_context`/`chunk_tokens`, and register it in `mgtranscriber/llm/__init__.py`
(`PROVIDERS`, `make_provider`).

## Tests

`uv run pytest` runs the unit tests (scripted fake provider, mocked Gemini/Anthropic SDK clients, a fake
Ollama HTTP server). `uv run pytest -m ollama` runs the integration tests against a real Ollama.
