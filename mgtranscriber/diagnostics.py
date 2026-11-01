"""System diagnostics for the Settings / System status page and the first-run guide.

Every check returns ``{id, group, status, code, params, detail}``:

* ``status``: ``ok`` | ``warn`` | ``error`` | ``na`` (not needed with the current settings)
* ``code`` + ``params``: stable, translated by the UI (``system.items.<id>.<code>``)
* ``detail``: the English fallback sentence, including the concrete fix

No check ever returns a secret. Checks that talk to the network (HuggingFace, Ollama) have
short timeouts; slow results are cached for a few minutes (``refresh=True`` bypasses the cache).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from . import APP_NAME, __version__, jsruntime, privacy
from .config import Settings, is_loopback_host
from .engines import list_engines
from .engines.common import DIARIZATION_MODEL, env_value

HF_MODEL_URL = f"https://huggingface.co/{DIARIZATION_MODEL}"
HF_TOKENS_URL = "https://huggingface.co/settings/tokens"
MIN_DRIVER_CUDA128 = 570  # NVIDIA driver branch that supports CUDA 12.8 (the PyTorch build we install)
MIN_VRAM_MB = 6000  # large-v3 + alignment + diarization comfortably fit in ~6 GB
LOW_FREE_VRAM_MB = 3500
LOW_DISK_BYTES = 5 * 1024**3

#: engines that need the local pyannote model for speakers (the cloud services diarize themselves)
SERVICE_DIARIZATION = {"elevenlabs", "deepgram"}
#: keys that a cloud feature reads from the environment, and who uses them
CLOUD_KEYS = {
    "GEMINI_API_KEY": ["gemini engine", "Gemini summaries"],
    "ANTHROPIC_API_KEY": ["Claude summaries"],
    "ELEVENLABS_API_KEY": ["elevenlabs engine"],
    "DEEPGRAM_API_KEY": ["deepgram engine"],
}

_CACHE_TTL = {"ok": 600.0, "error": 60.0}
_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()
_inflight: set[str] = set()  # keys being computed right now: a second request gets ``busy`` instead of a second run
_last_refresh: dict[str, float] = {}
MIN_REFRESH_S = 10.0  # ``refresh`` cannot be used to hammer HuggingFace, Ollama or a GPU subprocess


def _check(id: str, group: str, status: str, code: str, detail: str, **params) -> dict:
    return {"id": id, "group": group, "status": status, "code": code, "params": params, "detail": detail}


def refresh_allowed(kind: str) -> bool:
    """True at most once per ``MIN_REFRESH_S`` for each kind of refresh (``system``, ``gpu``)."""
    now = time.monotonic()
    with _cache_lock:
        if now - _last_refresh.get(kind, -MIN_REFRESH_S) < MIN_REFRESH_S:
            return False
        _last_refresh[kind] = now
        return True


def _cached(key: str, refresh: bool, fn: Callable[[], dict], busy: dict | None = None) -> dict:
    """Cached result of ``fn``. Only one computation per key runs at a time; a concurrent caller gets the stale
    value, or ``busy`` (the check's own "did not finish" result), instead of starting another one."""
    with _cache_lock:
        hit = _cache.get(key)
        if hit and not refresh and hit[0] > time.time():
            return hit[1]
        if key in _inflight:
            return hit[1] if hit else (busy or _check(key, "app", "warn", "timeout", "The check is still running."))
        _inflight.add(key)
    try:
        value = fn()
    finally:
        with _cache_lock:
            _inflight.discard(key)
    ttl = _CACHE_TTL["ok"] if value["status"] in ("ok", "na") else _CACHE_TTL["error"]
    with _cache_lock:
        _cache[key] = (time.time() + ttl, value)
    return value


def cached_result(key: str) -> dict | None:
    with _cache_lock:
        hit = _cache.get(key)
    return hit[1] if hit else None


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


# --- hardware ---------------------------------------------------------------------------------


def nvidia_smi() -> list[dict] | None:
    """GPUs as ``[{name, total_mb, used_mb, driver}]``, or None when nvidia-smi is absent or fails."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                gpus.append({"name": parts[0], "total_mb": int(float(parts[1])), "used_mb": int(float(parts[2])),
                             "driver": parts[3]})
            except ValueError:
                continue
    return gpus or None


def torch_build() -> dict:
    """Installed torch version and whether it is a CUDA build; reads package metadata, never imports torch."""
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None, "cuda": False}
    return {"installed": True, "version": version, "cuda": "+cu" in version}


def check_gpu() -> dict:
    gpus, torch = nvidia_smi(), torch_build()
    if not gpus:
        return _check("gpu", "hardware", "warn", "no_nvidia",
                      "No NVIDIA GPU found (nvidia-smi is missing or failed). Transcription then runs on the CPU, "
                      "which is many times slower. Install the NVIDIA driver (https://www.nvidia.com/drivers), "
                      "or use a cloud engine.")
    gpu = gpus[0]
    free = gpu["total_mb"] - gpu["used_mb"]
    base = dict(name=gpu["name"], total_gb=round(gpu["total_mb"] / 1024, 1), free_gb=round(free / 1024, 1),
                driver=gpu["driver"], torch=torch["version"] or "")
    if torch["installed"] and not torch["cuda"]:
        return _check("gpu", "hardware", "error", "cpu_torch",
                      f"{gpu['name']} found, but PyTorch {torch['version']} is a CPU-only build. Reinstall the "
                      "dependencies with `uv sync` (the project pins the CUDA 12.8 build).", **base)
    try:
        driver_major = int(gpu["driver"].split(".")[0])
    except ValueError:
        driver_major = MIN_DRIVER_CUDA128
    if driver_major < MIN_DRIVER_CUDA128:
        return _check("gpu", "hardware", "error", "old_driver",
                      f"NVIDIA driver {gpu['driver']} is too old for CUDA 12.8 (needs {MIN_DRIVER_CUDA128} or newer). "
                      "Update it at https://www.nvidia.com/drivers.", **base)
    if gpu["total_mb"] < MIN_VRAM_MB:
        return _check("gpu", "hardware", "warn", "low_vram",
                      f"{gpu['name']} has only {base['total_gb']} GB of VRAM. Choose the 'medium' or 'small' model "
                      "or the Parakeet engine.", **base)
    if free < LOW_FREE_VRAM_MB:
        return _check("gpu", "hardware", "warn", "vram_busy",
                      f"Only {base['free_gb']} GB of VRAM is free: another program is using the GPU. Close it "
                      "before transcribing (games, other AI tools, a loaded Ollama model).", **base)
    return _check("gpu", "hardware", "ok", "ok",
                  f"{gpu['name']}, {base['total_gb']} GB VRAM ({base['free_gb']} GB free), driver {gpu['driver']}.",
                  **base)


def torch_cuda_test(refresh: bool = False) -> dict:
    """Ask PyTorch itself (in a subprocess, so the server never loads CUDA) whether it can use the GPU."""

    def run() -> dict:
        code = "import torch;print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
        try:
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            return _check("gpu_torch", "hardware", "warn", "test_failed", f"The PyTorch GPU test did not finish: {exc}")
        words = out.stdout.strip().split(" ", 2)
        if out.returncode != 0 or len(words) < 2:
            return _check("gpu_torch", "hardware", "warn", "test_failed",
                          "PyTorch could not be started: " + (out.stderr.strip().splitlines() or ["unknown error"])[-1])
        if words[0] == "True":
            return _check("gpu_torch", "hardware", "ok", "ok",
                          f"PyTorch can use the GPU (CUDA {words[1]}).", cuda=words[1])
        return _check("gpu_torch", "hardware", "error", "no_cuda",
                      "PyTorch cannot use the GPU. Update the NVIDIA driver and reinstall with `uv sync`.")

    busy = _check("gpu_torch", "hardware", "warn", "test_running", "A PyTorch GPU test is already running; wait for it.")
    return _cached("gpu_torch", refresh, run, busy)


# --- tools ------------------------------------------------------------------------------------


def check_ffmpeg() -> dict:
    exe = shutil.which("ffmpeg")
    if not exe:
        return _check("ffmpeg", "tools", "error", "missing",
                      "ffmpeg was not found. It decodes all audio. Install it (Windows: `winget install Gyan.FFmpeg`, "
                      "macOS: `brew install ffmpeg`, Linux: your package manager) and restart the app.")
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10).stdout
        version = out.split()[2] if out.startswith("ffmpeg version") else ""
    except (OSError, subprocess.SubprocessError, IndexError):
        version = ""
    return _check("ffmpeg", "tools", "ok", "ok", f"ffmpeg {version}".strip() + ".", version=version)


def check_js_runtime(offline: bool = False) -> dict:
    if offline:
        return _check("jsruntime", "tools", "na", "offline", "Offline mode: URL sources are disabled, so YouTube is not used.")
    found = jsruntime.detect()
    if found:
        return _check("jsruntime", "tools", "ok", "ok", f"{found[0].capitalize()} {found[1]} (needed by yt-dlp for YouTube).",
                      name=found[0].capitalize(), version=found[1])
    return _check("jsruntime", "tools", "warn", "missing", jsruntime.HELP + " Local files work without it.")


# --- HuggingFace --------------------------------------------------------------------------------


def _hf_cached_locally() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(DIARIZATION_MODEL, "config.yaml"), (str, Path))
    except Exception:  # noqa: BLE001
        return False


def hf_status(token: str | None, refresh: bool = False) -> dict:
    """Does the token exist, is it valid, and may it download the pyannote model? Never contains the token."""
    if not token:
        return _check("hf", "models", "error", "no_token",
                      f"No HuggingFace token. Speaker recognition needs one: create a free 'Read' token at {HF_TOKENS_URL} "
                      "and paste it here (or put HF_TOKEN=... into .env).", model_url=HF_MODEL_URL, tokens_url=HF_TOKENS_URL)
    key = "hf:" + hashlib.sha256(token.encode()).hexdigest()[:16]

    def run() -> dict:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, OfflineModeIsEnabled, RepositoryNotFoundError

        links = dict(model_url=HF_MODEL_URL, tokens_url=HF_TOKENS_URL)
        try:
            api = HfApi()
            user = api.whoami(token=token).get("name", "")
            api.auth_check(DIARIZATION_MODEL, token=token)
            return _check("hf", "models", "ok", "ok", f"Token of '{user}' may download {DIARIZATION_MODEL}.", user=user, **links)
        except OfflineModeIsEnabled:
            if _hf_cached_locally():
                return _check("hf", "models", "ok", "offline_cached", "Offline mode: the speaker model is already downloaded.", **links)
            return _check("hf", "models", "warn", "offline_missing",
                          "Offline mode is on and the speaker model is not downloaded yet. Go online once.", **links)
        except (GatedRepoError, RepositoryNotFoundError):
            return _check("hf", "models", "error", "no_access",
                          f"The token is valid but has no access to the speaker model. Open {HF_MODEL_URL}, log in and "
                          "accept the terms ('Agree and access repository'). A fine-grained token also needs the "
                          "permission to read gated repositories.", **links)
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                return _check("hf", "models", "error", "invalid_token",
                              f"HuggingFace rejected the token (expired or mistyped). Create a new one at {HF_TOKENS_URL}.", **links)
            if status == 403:
                return _check("hf", "models", "error", "no_access", f"The token may not read the speaker model. Accept the terms at {HF_MODEL_URL}.", **links)
            return _check("hf", "models", "warn", "unreachable", f"HuggingFace answered with an error ({status or exc.__class__.__name__}). Try again later.", **links)
        except Exception as exc:  # noqa: BLE001 - network errors come in many classes
            return _check("hf", "models", "warn", "unreachable",
                          f"Cannot reach huggingface.co ({exc.__class__.__name__}). The token could not be verified.", **links)

    busy = _check("hf", "models", "warn", "timeout", "The HuggingFace check is still running.")
    return _cached(key, refresh, run, busy)


def check_hf(settings: Settings, refresh: bool = False) -> dict:
    """The HuggingFace check; in offline mode nothing is contacted: only whether the model is already on disk."""
    if privacy.offline(settings):
        links = dict(model_url=HF_MODEL_URL, tokens_url=HF_TOKENS_URL)
        if _hf_cached_locally():
            return _check("hf", "models", "ok", "offline_cached", "Offline mode: the speaker model is already downloaded.", **links)
        return _check("hf", "models", "warn", "offline_missing",
                      "Offline mode is on and the speaker model is not downloaded yet. Switch offline mode off once to download it.", **links)
    return hf_status(settings.hf_token, refresh)


# --- Ollama -----------------------------------------------------------------------------------


def ollama_url(settings: Settings) -> str:
    from .llm import load_config
    from .llm.ollama import DEFAULT_URL

    return (load_config(settings).base_url or DEFAULT_URL).rstrip("/")


def _get_json(url: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def check_ollama(settings: Settings) -> dict:
    from .llm import PROVIDERS, load_config
    from .llm.ollama import same_model

    config = load_config(settings)
    base = ollama_url(settings)
    wanted = config.model or PROVIDERS["ollama"]["model"] if config.provider == "ollama" else PROVIDERS["ollama"]["model"]
    used = config.provider == "ollama"
    if privacy.offline(settings) and not is_loopback_host(urlparse(base if "//" in base else "//" + base).hostname or ""):
        return _check("ollama", "services", "warn" if used else "na", "offline_remote",
                      f"Offline mode: the Ollama server {base} is not on this computer, so it is not used. Point llm_base_url at localhost.",
                      url=base, model=wanted)
    try:
        version = _get_json(base + "/api/version").get("version", "")
        installed = [m["name"] for m in _get_json(base + "/api/tags").get("models", [])]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return _check("ollama", "services", "warn" if used else "na", "not_running",
                      f"Ollama does not answer at {base}. Summaries and questions need it: install it from https://ollama.com "
                      "and start it (open the Ollama app or run `ollama serve`).", url=base, model=wanted)
    params = dict(url=base, version=version, model=wanted, installed=installed)
    if any(same_model(wanted, m) for m in installed):
        return _check("ollama", "services", "ok", "ok", f"Ollama {version} is running; model {wanted} is installed.", **params)
    return _check("ollama", "services", "warn" if used else "na", "model_missing",
                  f"Ollama {version} is running, but the model {wanted} is not downloaded. Download it here or run `ollama pull {wanted}`.",
                  **params)


# --- application ------------------------------------------------------------------------------


def check_storage(settings: Settings) -> dict:
    out = settings.output_dir
    try:
        out.mkdir(parents=True, exist_ok=True)
        writable = os.access(out, os.W_OK)
        free = shutil.disk_usage(out).free
    except OSError as exc:
        return _check("storage", "app", "error", "unwritable", f"The library folder {out} is not usable: {exc}", path=str(out))
    if not writable:
        return _check("storage", "app", "error", "unwritable", f"The library folder {out} is not writable.", path=str(out))
    free_gb = round(free / 1024**3, 1)
    if free < LOW_DISK_BYTES:
        return _check("storage", "app", "warn", "low_disk", f"Only {free_gb} GB free next to the library ({out}). Audio and models need space.",
                      path=str(out), free_gb=free_gb)
    return _check("storage", "app", "ok", "ok", f"Library: {out} ({free_gb} GB free).", path=str(out), free_gb=free_gb)


def check_network(settings: Settings) -> dict:
    if settings.is_loopback:
        return _check("network", "app", "ok", "local", f"The server listens on {settings.host} (this computer only).", host=settings.host)
    if settings.api_key:
        return _check("network", "app", "ok", "protected", f"The server listens on {settings.host}; access needs the API key.", host=settings.host)
    return _check("network", "app", "warn", "open",
                  f"The server listens on {settings.host} without an API key: anyone on your network can read your transcripts. "
                  "Set MGTRANSCRIBER_API_KEY in .env.", host=settings.host)


def check_offline(settings: Settings) -> dict | None:
    if privacy.offline(settings):
        return _check("offline", "app", "ok", "mode", "Offline mode is on: no cloud engines or LLMs, no URL sources, Ollama only on this computer.")
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on"):
        return _check("offline", "app", "ok", "on", "HF_HUB_OFFLINE is set: models are loaded from disk only.")
    return None


def engine_report(settings: Settings) -> list[dict]:
    rows = []
    for e in list_engines():
        if not e.available:
            continue
        missing = e.missing(settings)
        if missing is None and privacy.offline(settings) and privacy.needs_network(e):
            missing = {"code": "offline", "detail": ""}
        rows.append({"name": e.name, "label": e.label, "cloud": e.cloud, "missing": missing,
                     "uses_pyannote": e.diarization and e.name not in SERVICE_DIARIZATION})
    return rows


KEY_URLS = {
    "GEMINI_API_KEY": "https://aistudio.google.com/apikey",
    "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
    "ELEVENLABS_API_KEY": "https://elevenlabs.io/app/settings/api-keys",
    "DEEPGRAM_API_KEY": "https://console.deepgram.com/",
}
#: name -> (url, headers) of a cheap authenticated GET that costs nothing and changes nothing
_KEY_TESTS = {
    "GEMINI_API_KEY": ("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1", lambda k: {"x-goog-api-key": k}),
    "ANTHROPIC_API_KEY": ("https://api.anthropic.com/v1/models?limit=1",
                          lambda k: {"x-api-key": k, "anthropic-version": "2023-06-01"}),
    "ELEVENLABS_API_KEY": ("https://api.elevenlabs.io/v1/models", lambda k: {"xi-api-key": k}),
    "DEEPGRAM_API_KEY": (None, lambda k: {"Authorization": f"Token {k}"}),  # host: the engine's (EU default)
}


def key_report(settings: Settings) -> list[dict]:
    """Which keys are set and where from (``environment`` = cannot be changed here). Never contains a value."""
    return [{"env": name, "set": bool(env_value(settings, name)), "used_by": used, "url": KEY_URLS[name],
             "source": "environment" if os.environ.get(name) else ".env" if env_value(settings, name) else "none"}
            for name, used in CLOUD_KEYS.items()]


def test_key(settings: Settings, name: str) -> dict:
    """Ask the service whether the stored key works: ``{"ok": bool, "code": ok|rejected|unreachable|no_key}``.
    Only the status is reported, never anything from the key or the response."""
    value = env_value(settings, name)
    if not value:
        return {"ok": False, "code": "no_key"}
    url, headers = _KEY_TESTS[name]
    if url is None:
        from .engines.cloud import DEEPGRAM_URL

        url = (env_value(settings, "DEEPGRAM_API_URL") or DEEPGRAM_URL).rstrip("/") + "/v1/projects"
    req = urllib.request.Request(url, headers={**headers(value), "User-Agent": f"{APP_NAME}/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=15):
            return {"ok": True, "code": "ok"}
    except urllib.error.HTTPError as exc:
        # 403: the key is known but may not do this check (a restricted key): not the same as a wrong key
        return {"ok": False, "code": "rejected" if exc.code in (400, 401) else "limited" if exc.code == 403 else "unreachable"}
    except Exception:  # noqa: BLE001 - never put the exception text (a URL, a header) into the answer
        return {"ok": False, "code": "unreachable"}


# --- everything -------------------------------------------------------------------------------


def run_checks(settings: Settings, refresh: bool = False) -> dict:
    """The full report. Checks run in parallel; one that crashes or hangs never breaks the others."""
    jobs: list[tuple[str, str, Callable[[], dict]]] = [
        ("gpu", "hardware", check_gpu),
        ("ffmpeg", "tools", check_ffmpeg),
        ("jsruntime", "tools", lambda: check_js_runtime(privacy.offline(settings))),
        ("hf", "models", lambda: check_hf(settings, refresh)),
        ("ollama", "services", lambda: check_ollama(settings)),
        ("storage", "app", lambda: check_storage(settings)),
        ("network", "app", lambda: check_network(settings)),
    ]
    pool = ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="diagnostics")
    futures = [(id, group, pool.submit(fn)) for id, group, fn in jobs]
    checks = []
    for id, group, future in futures:
        try:
            checks.append(future.result(timeout=12))
        except FutureTimeout:
            checks.append(_check(id, group, "warn", "timeout", "The check did not finish in time."))
        except Exception as exc:  # noqa: BLE001
            checks.append(_check(id, group, "warn", "check_failed", f"The check failed: {exc}"))
    pool.shutdown(wait=False)
    if offline := check_offline(settings):
        checks.append(offline)
    by_id = {c["id"]: c for c in checks}
    engines = engine_report(settings)
    ffmpeg_ok = by_id["ffmpeg"]["status"] == "ok"
    return {
        "name": APP_NAME,
        "version": __version__,
        "python": sys.version.split()[0],
        "checks": checks,
        "engines": engines,
        "keys": key_report(settings),
        "hf_token_source": hf_token_source(settings),
        "can_save_token": settings.is_loopback,
        "can_save_keys": settings.is_loopback,
        "ready": {
            "transcribe": ffmpeg_ok,
            "diarize": ffmpeg_ok and by_id["hf"]["status"] == "ok",
        },
        "pyannote_engines": [e["name"] for e in engines if e["uses_pyannote"]],
    }


def hf_token_source(settings: Settings) -> str:
    """Where the active token comes from (``none`` | ``environment`` | ``.env``), for the UI's hint."""
    if not settings.hf_token:
        return "none"
    return "environment" if os.environ.get("HF_TOKEN") else ".env"


# --- Ollama model download ----------------------------------------------------------------------


class OllamaPull:
    """Downloads one Ollama model in a background thread; ``state()`` feeds the UI's progress bar."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict = {"status": "idle", "model": None, "percent": 0.0, "message": "", "error": None}

    def state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def start(self, base_url: str, model: str) -> bool:
        """False when a download is already running."""
        with self._lock:
            if self._state["status"] == "running":
                return False
            self._state = {"status": "running", "model": model, "percent": 0.0, "message": "starting", "error": None}
        threading.Thread(target=self._run, args=(base_url, model), name="ollama-pull", daemon=True).start()
        return True

    def _set(self, **values) -> None:
        with self._lock:
            self._state.update(values)

    def _run(self, base_url: str, model: str) -> None:
        layers: dict[str, list[int]] = {}
        try:
            body = json.dumps({"model": model, "stream": True}).encode()
            req = urllib.request.Request(base_url + "/api/pull", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                for raw in resp:
                    if not raw.strip():
                        continue
                    event = json.loads(raw)
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    if event.get("digest") and event.get("total"):
                        layers[event["digest"]] = [event["total"], event.get("completed", 0)]
                    total = sum(t for t, _ in layers.values())
                    percent = 100 * sum(c for _, c in layers.values()) / total if total else 0.0
                    self._set(percent=round(min(percent, 99.9), 1), message=event.get("status", ""))
            self._set(status="done", percent=100.0, message="success")
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            self._set(status="error", error=str(exc) or exc.__class__.__name__)


# --- command line ---------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m mgtranscriber.diagnostics``: print the checks; exit 0 when the essentials work.

    Essentials are ffmpeg, a usable library folder and a GPU setup that is not broken; a missing NVIDIA GPU is
    "CPU mode" (a warning), a missing HuggingFace token only disables speaker recognition.
    """
    import argparse

    from .config import load_settings

    parser = argparse.ArgumentParser(prog="python -m mgtranscriber.diagnostics", description=main.__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)
    report = run_checks(load_settings())
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        marks = {"ok": "[ OK ]", "warn": "[warn]", "error": "[FAIL]", "na": "[ -- ]"}
        for check in report["checks"]:
            print(f"{marks[check['status']]} {check['id']:<9} {check['detail']}")
        if not report["ready"]["diarize"]:
            print("Note: speaker recognition is not ready (HuggingFace token / model access); transcription still works without it.")
    essential = {"ffmpeg", "storage", "gpu"}
    return 1 if any(c["status"] == "error" for c in report["checks"] if c["id"] in essential) else 0


if __name__ == "__main__":
    sys.exit(main())
