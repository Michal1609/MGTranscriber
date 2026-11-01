# Docker

The image runs the web server (`mgtranscriber serve`): web UI, job queue, OpenAI-compatible API at
`/v1` and MCP at `/mcp`, all on port 8000.

| | |
|---|---|
| base | `ubuntu:24.04`; the CUDA runtime (cuBLAS, cuDNN, NCCL) comes with the PyTorch wheels |
| size | CUDA build: 4.9 GB download, 14 GB on disk (PyTorch and NVIDIA libraries); CPU build: several GB smaller (no CUDA libraries). Models are downloaded later, into `/data` |
| user | uid 1000 (`user:` in compose to change it) |
| volumes | `/output` transcripts, `/data` queue, search index, voice memory, model caches |
| healthcheck | `GET /api/info` from inside the container |

## Requirements

- **GPU build:** an NVIDIA driver on the host with CUDA 12.8 support (R570 or newer, needed for RTX 50xx;
  older GPUs work from R528) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  On Windows, Docker Desktop with the WSL2 backend passes the GPU through without anything else.
  Check: `docker run --rm --gpus all ubuntu:24.04 nvidia-smi`.
- **CPU build:** nothing. Transcription is much slower (use `-m large-v3-turbo`, Parakeet, or the
  `youtube` engine for videos with captions).
- x86-64 for the GPU build. The CPU build also builds on arm64 (Apple Silicon, Graviton).

## Start

```
cp .env.example .env     # HF_TOKEN=hf_... (speaker recognition), see the README
docker compose up -d     # builds the image the first time, then http://localhost:8000
docker compose logs -f
```

No NVIDIA GPU:

```
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

Without compose:

```
docker build -t mgtranscriber .                             # or: --build-arg TORCH=cpu
docker run -d --gpus all -p 127.0.0.1:8000:8000 --env-file .env \
  -v ./output:/output -v ./data:/data --name mgtranscriber mgtranscriber
```

A transcription from the command line inside the running container:

```
docker compose exec mgtranscriber mgtranscriber transcribe "https://www.youtube.com/watch?v=..." -s 2
```

**Docker Desktop on Windows:** bind mounts of Windows folders (`./data:/data`) go through WSL2 and are
slow for the model files in `/data`. Use a named volume for `/data` (`- mgtranscriber-data:/data` plus a
top-level `volumes: {mgtranscriber-data: {}}`), or keep the project inside the WSL2 file system. `/output`
can stay a Windows folder.

## Configuration

Everything is set with environment variables (in `.env` or `environment:` in compose); see
`docs/ARCHITECTURE.md#settings`. The image sets `MGTRANSCRIBER_HOST=0.0.0.0`, `MGTRANSCRIBER_DATA_DIR=/data`,
`MGTRANSCRIBER_OUTPUT_DIR=/output` and `HF_HOME=/data/cache/huggingface`.

| variable | why |
|---|---|
| `HF_TOKEN` | speaker recognition (pyannote); accept the model conditions first. Put it in `.env`: the container listens on `0.0.0.0`, so the web UI refuses to store a token (it only accepts one on a loopback server) |
> **Renamed from localscribe.** The variables used to start with `LOCALSCRIBE_`. The application still reads the old names
> (with a warning), but `docker compose` does not: in `.env` and compose files use the `MGTRANSCRIBER_` names, and the
> image and service are now called `mgtranscriber`.

| `MGTRANSCRIBER_API_KEY` | **required** as soon as other computers can reach the port |
| `MGTRANSCRIBER_PUBLISH` | compose only: where the port is published, default `127.0.0.1:8000`; `0.0.0.0:8000` for the whole network |
| `MGTRANSCRIBER_ALLOWED_HOSTS` | host names the browser uses, e.g. `localhost,nas,nas.local,192.168.1.10` (or `*` behind a proxy that checks the host). Requests with other `Host` headers get 403. |
| `MGTRANSCRIBER_LLM_BASE_URL` | Ollama address; compose sets `http://ollama:11434`. Ollama on the Docker host: `http://host.docker.internal:11434` |
| `MGTRANSCRIBER_MODEL`, `MGTRANSCRIBER_LANGUAGE`, ... | defaults for new jobs |

`settings.json` (the Settings page saves to it) lives in the data folder, `/data/settings.json`, which is a volume,
so it survives recreating the container. `MGTRANSCRIBER_SETTINGS_FILE` puts it somewhere else.

**Local LLM:** `docker compose --profile ollama up -d`, then
`docker compose exec ollama ollama pull gemma3:4b` and `... ollama pull bge-m3`. Ollama shares the GPU;
with 8 GB VRAM a transcription and an analysis at the same time may not fit. Offline mode only allows an
Ollama on loopback, so with the compose `ollama` service keep offline mode off.

**Behind a reverse proxy (TLS):** publish on `127.0.0.1` only, put the proxy's host name in
`MGTRANSCRIBER_ALLOWED_HOSTS`, set an API key, and set `FORWARDED_ALLOW_IPS` to the proxy's address so
that cookies get the `Secure` flag (see docs/API.md). Uploads through `/v1` can be large: raise the
proxy's body size limit (nginx `client_max_body_size`).

## Unraid

1. Install **Nvidia Driver** from Community Applications and note the GPU UUID (Settings → Nvidia Driver).
2. Docker → Add Container (or Compose Manager with `docker-compose.yml`):
   - Repository: an image you built from this repository (`docker compose build`, tag `mgtranscriber:latest`, or `mgtranscriber:cpu`);
     no prebuilt image is published yet
   - Extra Parameters: `--runtime=nvidia`
   - Variables: `NVIDIA_VISIBLE_DEVICES=<GPU UUID>`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`,
     `HF_TOKEN`, `MGTRANSCRIBER_API_KEY`, `MGTRANSCRIBER_ALLOWED_HOSTS=<server name>,<server IP>`
   - Paths: `/output` → `/mnt/user/appdata/mgtranscriber/output` (or a share), `/data` → `/mnt/user/appdata/mgtranscriber/data`
   - Port 8000
   - User: Unraid shares belong to `nobody:users` (99:100). Either add `--user 99:100` to Extra
     Parameters, or `chown -R 1000:1000` the two folders.
3. Put `/data` on the cache pool (SSD): models and the search index are read often.

## TrueNAS SCALE

- Apps → Discover → **Custom App** (or install via YAML with `docker-compose.yml`).
- Image `mgtranscriber:latest` built from this repository (no prebuilt image is published yet); port 8000; environment as above.
- **GPU:** in *Resources*, allocate the NVIDIA GPU to the app (TrueNAS 24.10+ ships the NVIDIA driver;
  enable "Install NVIDIA Drivers" in Apps settings on 25.04+).
- Storage: two host-path datasets for `/output` and `/data`. Apps run as `apps` (568:568) by default;
  set the container user to 568 or give uid 1000 write access to the datasets.

## Synology / other NAS

Without an NVIDIA GPU use the CPU image. Many NAS CPUs are slow: start with `large-v3-turbo` or smaller
models and expect transcription slower than real time.

## Building

```
docker build -t mgtranscriber .                        # CUDA 12.8 (default)
docker build -t mgtranscriber:cpu --build-arg TORCH=cpu .
docker build --build-arg UV_VERSION=0.12.24 --build-arg DENO_VERSION=2.9.7 .
```

The image is built from `pyproject.toml`, `uv.lock`, `mgtranscriber/` and `static/` only (`.dockerignore`),
with `uv sync --frozen`, so it has exactly the locked versions.
