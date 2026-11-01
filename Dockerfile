# syntax=docker/dockerfile:1
#
# mgtranscriber web server (web UI, OpenAI-compatible /v1, MCP /mcp) for a Linux host with an NVIDIA GPU.
#   docker compose up -d                                    # see docker-compose.yml
#   docker build -t mgtranscriber .                           # CUDA 12.8 build of PyTorch (default)
#   docker build -t mgtranscriber:cpu --build-arg TORCH=cpu . # no GPU
#
# The CUDA runtime (cuBLAS, cuDNN, ...) comes with the PyTorch wheels, so the base image is plain Ubuntu;
# the host only needs the NVIDIA driver and the NVIDIA Container Toolkit. See docs/DOCKER.md.

ARG UV_VERSION=0.12.24
ARG DENO_VERSION=2.9.7

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM denoland/deno:bin-${DENO_VERSION} AS deno

# --- build: Python and the virtual environment -------------------------------------------
FROM ubuntu:24.04 AS build
ARG TORCH=cu128
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# dependencies first: this layer is reused as long as pyproject.toml and uv.lock do not change
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TORCH" = "cpu" ]; then groups="--no-group cu128 --group cpu"; else groups=""; fi; \
    uv sync --frozen --no-dev --no-install-project $groups
COPY mgtranscriber ./mgtranscriber
COPY static ./static
COPY LICENSE THIRD_PARTY_NOTICES.md ./
# editable install: the app finds static/ next to the package in /app
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TORCH" = "cpu" ]; then groups="--no-group cu128 --group cpu"; else groups=""; fi; \
    uv sync --frozen --no-dev $groups

# --- runtime -----------------------------------------------------------------------------
FROM ubuntu:24.04
ARG TORCH=cu128
LABEL org.opencontainers.image.title="mgtranscriber" \
      org.opencontainers.image.description="Local transcription with speaker diarization: web UI, OpenAI-compatible API, MCP" \
      org.opencontainers.image.licenses="Apache-2.0"
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=deno /deno /usr/local/bin/deno
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app /app
RUN mkdir -p /data /output && chown 1000:1000 /data /output
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MGTRANSCRIBER_HOST=0.0.0.0 \
    MGTRANSCRIBER_PORT=8000 \
    MGTRANSCRIBER_DATA_DIR=/data \
    MGTRANSCRIBER_OUTPUT_DIR=/output \
    HF_HOME=/data/cache/huggingface \
    DENO_DIR=/data/cache/deno \
    MGTRANSCRIBER_TORCH_BUILD=${TORCH}
WORKDIR /app
# uid 1000 = the "ubuntu" user of the base image; override with `user:` in compose (Unraid: 99:100)
USER 1000:1000
VOLUME ["/data", "/output"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request, urllib.error\ntry:\n    urllib.request.urlopen('http://127.0.0.1:8000/api/info', timeout=4)\nexcept urllib.error.HTTPError as e:\n    sys.exit(e.code >= 500)\nexcept Exception:\n    sys.exit(1)"]
ENTRYPOINT ["tini", "--", "mgtranscriber"]
CMD ["serve"]
