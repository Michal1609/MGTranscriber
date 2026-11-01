# Integrations (OpenAI-compatible API)

MGTranscriber answers the OpenAI Audio Transcriptions API at `/v1/audio/transcriptions`
(see [API.md](API.md#openai-compatible-api) for all fields), so anything that can talk
to "OpenAI Whisper" can use your own GPU instead. This page shows how to point common
tools at it.

What you need in every case:

| setting | value |
|---|---|
| **Base URL** | `http://<host>:8000/v1` (with `/v1`). `<host>` must be reachable **from the tool**: from a Docker container `127.0.0.1` is the container itself, use the computer's address or `host.docker.internal`. |
| **API key** | the value of `MGTRANSCRIBER_API_KEY` if you set one; otherwise any non-empty string (clients refuse an empty key). |
| **Model** | `whisper-1` always works. `gpt-transcribe`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe` also map to the server default model; `GET /v1/models` lists what else is accepted (`large-v3`, an engine name, `<engine>/<model>`). |

The server must listen on an address the tool can reach (`MGTRANSCRIBER_HOST=0.0.0.0`
plus a key and `MGTRANSCRIBER_ALLOWED_HOSTS`, see [API.md](API.md#access)). Every upload
becomes a normal job in the queue and ends up in the library.

Extensions (optional, not part of OpenAI's API): `diarize=true` labels speakers,
`num_speakers=N` fixes their number. Most tools cannot send extra fields; with
`response_format=verbose_json`, `srt` or `vtt` the labels appear in the output.

Timeouts: transcription takes a fraction of the media length. Tools with short
HTTP timeouts (30-60 s) fail on long files; raise the timeout where the tool lets you.

## curl

```bash
KEY=your-key   # MGTRANSCRIBER_API_KEY
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $KEY" \
  -F file=@talk.mp3 -F model=whisper-1
# {"text":"...","usage":{"type":"duration","seconds":40}}

# subtitles with speaker labels
curl http://127.0.0.1:8000/v1/audio/transcriptions -H "Authorization: Bearer $KEY" \
  -F file=@talk.mp3 -F model=whisper-1 -F response_format=srt -F diarize=true -F num_speakers=2
```

Verified against a running server (mock engine): `json` and `srt` with diarization,
401 without the key.

## Python (`openai` package)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="your-key", timeout=1800)
with open("talk.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1", file=f, response_format="verbose_json",
        timestamp_granularities=["segment"],
        extra_body={"diarize": True},   # optional extension
    )
print(result.text)
for seg in result.segments:
    print(seg.start, seg.model_extra["speaker"], seg.text)
```

Verified with `openai` 3.27 against a running server (mock engine): `models.list()`,
`verbose_json` with speakers, `text`.

## Open WebUI

Speech-to-text of the chat microphone and voice calls. Open WebUI's backend sends the
audio to the engine, so the URL must be reachable **from the Open WebUI container**.

Admin Panel → Settings → Audio:

1. **Speech-to-Text Engine**: `OpenAI`
2. **API Base URL**: `http://host.docker.internal:8000/v1` (or the computer's LAN address)
3. **API Key**: your `MGTRANSCRIBER_API_KEY` (any text if you set none)
4. **STT Model**: `whisper-1` (Open WebUI's default `gpt-transcribe` is accepted too)
5. Save.

Or with environment variables (applied at start; the admin panel overrides them later):

```yaml
environment:
  - AUDIO_STT_ENGINE=openai
  - AUDIO_STT_OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1
  - AUDIO_STT_OPENAI_API_KEY=your-key
  - AUDIO_STT_MODEL=whisper-1
```

Source: [Open WebUI: Using OpenAI for Speech-to-Text](https://docs.openwebui.com/features/chat-conversations/audio/speech-to-text/openai-stt-integration).
Not run here (the Open WebUI container is heavy); the request it sends is the standard
multipart upload shown under curl. Note the model name: older Open WebUI versions send
`whisper-1`, current ones `gpt-transcribe`; both work.

## n8n

**HTTP Request node** (works everywhere and exposes every field):

- Method `POST`, URL `http://<host>:8000/v1/audio/transcriptions`
- Authentication: *Generic Credential Type → Header Auth*, name `Authorization`, value `Bearer your-key`
- Send Body: on, Body Content Type `Form-Data`, parameters:
  - `file`: type *n8n Binary File*, input data field name `data`
  - `model`: `whisper-1`
  - optionally `response_format` = `verbose_json`, `diarize` = `true`
- Options → Timeout: raise it (milliseconds) for long media.

The transcript is `{{ $json.text }}` in the next node.

**OpenAI node** (Audio → *Transcribe a Recording*): the OpenAI credential has an optional
**Base URL** field; set it to `http://<host>:8000/v1`
([n8n credential docs](https://docs.n8n.io/integrations/builtin/credentials/openai/)).
Whether the node's transcription call honours the credential's base URL depends on the
n8n version; if the request still goes to api.openai.com, use the HTTP Request node.
Not verified here.

## Home Assistant

Home Assistant itself has no built-in OpenAI-compatible STT with a custom URL. The
community integration **[einToast/openai_stt_ha](https://github.com/einToast/openai_stt_ha)**
(HACS, Apache-2.0) provides a `stt` platform for Assist with an `api_url` setting that
takes any compatible server (README as of 10/2026). In `configuration.yaml`:

```yaml
stt:
  - platform: openai_stt
    api_key: your-key
    api_url: http://<host>:8000/v1
    model: whisper-1
    realtime: false      # the realtime API is not implemented by MGTranscriber
```

Then choose it as the speech-to-text engine of your Assist pipeline. Use a small, fast
model on the server (the server default `model` setting, e.g. `small` or `large-v3-turbo`)
because Assist waits for the answer. Not verified in a real Home Assistant; the request
is the plain transcription call from the curl section.
