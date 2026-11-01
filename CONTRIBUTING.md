# Contributing

Thanks for helping! Bug reports, docs fixes and pull requests are welcome.

## Setup

```
uv sync                          # CUDA 12.8 build of PyTorch (NVIDIA GPU)
uv sync --no-group cu128 --group cpu   # or: CPU build
uv run pytest                    # fast tests: no GPU, no Ollama, no network
uv run pytest -m gpu             # end-to-end on a real GPU (needs HF_TOKEN and tests/make_clip.py)
uvx ruff check .                 # lint (bug rules only)
```

`uv sync` and `uv run` include the `dev` group (pytest, httpx, openai, playwright) by default, so
`uv run pytest` works right after a plain `uv sync`. The end-user installers (`install.ps1`, `web.bat`,
`start.bat`) pass `--no-dev`; if you develop in a folder set up by `install.bat`, run a plain `uv sync`
once to add the dev tools back. On macOS and Linux arm64 only the CPU group installs
(`uv sync --no-group cu128 --group cpu`).

Windows users can also run `install.bat`. Project structure and how to add an engine, exporter,
API router or page: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Pull requests

- One topic per PR; describe what changes for the user and how you tested it.
- Add or update tests in `tests/` (the mock engine in `tests/mock_engine.py` keeps them GPU-free).
- Keep the web UI free of build steps and CDNs (plain ES modules in `static/`), and add UI strings to
  both `static/js/i18n/cs.js` and `en.js`.
- Do not commit transcripts, audio, voiceprints, tokens or anything from `output/` and `data/`.
  Test fixtures must be synthetic or under a free license.
- Heavy dependencies must be imported lazily (inside the engine/feature module), so the server starts
  quickly and optional features stay optional.
- New dependencies: explain why; prefer permissive licenses (MIT/BSD/Apache). Update `uv.lock`
  (`uv lock`) and, for models or bundled software, `THIRD_PARTY_NOTICES.md`.

By submitting a contribution you agree that it is licensed under the Apache License 2.0
(see section 5 of the [LICENSE](LICENSE)).

## Reporting bugs

Use the issue form. Include the version (`uv run mgtranscriber --version`), Windows/Linux/Docker, GPU
and driver (`nvidia-smi`), the engine and the relevant part of the job log. Remove names, URLs and
transcript text you do not want to make public. Security problems: see [SECURITY.md](SECURITY.md).

## AI-assisted contributions

Welcome, but you are responsible for the result: read and test what you submit. Note that yt-dlp does
not accept AI-generated reports, so do not forward problems to them without checking them yourself.
