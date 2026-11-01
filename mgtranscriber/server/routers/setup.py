"""System status (what is needed to run), the HuggingFace token and the editable settings.

* ``GET /api/system``: all checks (GPU, ffmpeg, Deno/Node, HF token, Ollama, ...). Nothing secret is ever returned.
* ``GET /api/system/gpu-test``: asks PyTorch itself whether it can use the GPU (slow, cached).
* ``PUT|DELETE /api/system/keys/{ENV}``, ``POST .../test``: the same for the cloud API keys.
* ``PUT|DELETE /api/system/hf-token``: validates and stores the HuggingFace token in ``.env`` (loopback only).
* ``GET|POST /api/system/ollama/pull``: download an Ollama model with progress.
* ``GET|PUT /api/settings``: the whitelisted settings of ``settings_store`` (``settings.json``).
"""

import copy
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ... import diagnostics, privacy, settings_store
from ...config import is_loopback_host
from ..deps import QueueDep, SettingsDep
from ..errors import ApiError

router = APIRouter(prefix="/api/system", tags=["system"])
settings_router = APIRouter(prefix="/api/settings", tags=["settings"])

OLLAMA_MODEL = re.compile(r"^[\w.\-:/]{1,100}$")


def guarded(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except settings_store.SettingError as exc:
        raise ApiError(exc.status, exc.code, exc.message) from exc


@router.get("")
def system_status(request: Request, settings: SettingsDep, refresh: bool = False):
    # ``refresh`` is a GET that costs network calls: honoured at most once every few seconds
    refresh = refresh and diagnostics.refresh_allowed("system")
    if refresh:
        diagnostics.clear_cache()
    report = diagnostics.run_checks(settings, refresh=refresh)
    if client_is_local(request):  # the Settings page shows where the library is, to the user at this computer
        return report
    report = copy.deepcopy(report)  # the report may be cached
    for check in report.get("checks", []):  # other clients learn no user name or install path
        full = check.get("params", {}).get("path")
        if full:
            short = Path(full).name
            check["params"]["path"] = short
            check["detail"] = check["detail"].replace(full, short)
    return report


@router.get("/gpu-test")
def gpu_test_result():
    """The last PyTorch GPU test, without running one (``POST`` runs it)."""
    return diagnostics.cached_result("gpu_torch") or {
        "id": "gpu_torch", "group": "hardware", "status": "na", "code": "not_run", "params": {},
        "detail": "The PyTorch GPU test has not been run yet."}


@router.post("/gpu-test")
def gpu_test_run(queue: QueueDep):
    """Run the test now: a subprocess that loads torch and CUDA, so it is an action (POST, origin-checked), runs one
    at a time, is rate limited, and is refused while a transcription is using the GPU."""
    if any(job.get("status") == "running" for job in queue.listing()):
        raise ApiError(409, "gpu_busy", "A job is using the GPU right now. Try again when it has finished.")
    return diagnostics.torch_cuda_test(refresh=diagnostics.refresh_allowed("gpu"))


class Token(BaseModel):
    token: str  # no length limit here: a 422 would echo the secret; settings_store checks the format


def client_is_local(request: Request) -> bool:
    """The peer of the socket is this computer. Bind address alone is not enough: a reverse proxy or a tunnel on this
    computer forwards other people's requests from 127.0.0.1 (their Host/Origin then do not match, but be strict)."""
    host = request.client.host if request.client else ""
    return is_loopback_host(host)


@router.put("/hf-token")
def save_hf_token(body: Token, request: Request, settings: SettingsDep):
    """Check the token's format, ask HuggingFace whether it works, save it to ``.env``.

    A rejected token is not saved. A valid one without access to the speaker model is saved (the user then accepts
    the model's terms). When HuggingFace cannot be asked (offline mode, no connection) the token is saved WITHOUT
    verification and the answer says so: ``verified: false``."""
    token = guarded(settings_store.check_token_format, body.token)  # before any network or file access
    if not settings.is_loopback or not client_is_local(request):
        raise ApiError(403, "local_only", "The token can only be saved from this computer while the server listens on this computer only.")
    if privacy.offline(settings):
        result, verified = diagnostics.check_hf(settings), False
    else:
        result = diagnostics.hf_status(token, refresh=True)
        if result["code"] == "invalid_token":
            raise ApiError(400, "hf_token_invalid", result["detail"])
        verified = result["code"] in ("ok", "no_access")
    guarded(settings_store.set_hf_token, settings, token, client_local=True)
    return {**result, "saved": True, "verified": verified}


@router.delete("/hf-token")
def delete_hf_token(request: Request, settings: SettingsDep):
    guarded(settings_store.set_hf_token, settings, None, client_local=client_is_local(request))
    return {"removed": True}


class KeyIn(BaseModel):
    value: str  # no length limit here: a 422 would echo the secret; settings_store checks the format


def _key_name(name: str) -> str:
    if name not in diagnostics.CLOUD_KEYS:
        raise ApiError(404, "key_unknown", "That is not a key this program uses.")
    return name


@router.put("/keys/{name}")
def save_api_key(name: str, body: KeyIn, request: Request, settings: SettingsDep):
    """Store a cloud API key in ``.env`` (write-only: no endpoint returns it). Same rules as the HuggingFace token."""
    guarded(settings_store.set_api_key, settings, _key_name(name), body.value, client_local=client_is_local(request))
    return {"env": name, "set": True}


@router.delete("/keys/{name}")
def delete_api_key(name: str, request: Request, settings: SettingsDep):
    guarded(settings_store.set_api_key, settings, _key_name(name), None, client_local=client_is_local(request))
    return {"env": name, "set": False}


@router.post("/keys/{name}/test")
def test_api_key(name: str, settings: SettingsDep):
    """Ask the service whether the stored key is accepted (a free read-only call). Not in offline mode."""
    if privacy.offline(settings):
        raise ApiError(403, "offline_test", "Offline mode: nothing is sent to cloud services, so keys cannot be tested.")
    return {"env": name, **diagnostics.test_key(settings, _key_name(name))}


class PullIn(BaseModel):
    model: str | None = Field(None, max_length=100)


def _pull(request: Request) -> diagnostics.OllamaPull:
    state = request.app.state
    if not hasattr(state, "ollama_pull"):
        state.ollama_pull = diagnostics.OllamaPull()
    return state.ollama_pull


@router.get("/ollama/pull")
def pull_state(request: Request):
    return _pull(request).state()


@router.post("/ollama/pull", status_code=202)
def pull_start(body: PullIn, request: Request, settings: SettingsDep):
    from ...llm import PROVIDERS, load_config

    model = (body.model or load_config(settings).model or PROVIDERS["ollama"]["model"]).strip()
    if not OLLAMA_MODEL.fullmatch(model):
        raise ApiError(400, "invalid_model", "Invalid model name.")
    base = diagnostics.ollama_url(settings)
    if privacy.offline(settings):  # a download is network traffic, and the server must be this computer's
        raise ApiError(403, "offline_download", "Offline mode: models are not downloaded. Switch offline mode off first.")
    try:
        privacy.check_llm(settings, "ollama", False, base)
    except privacy.PrivacyError as exc:
        raise ApiError(exc.status, exc.code, exc.message) from exc
    if not _pull(request).start(base, model):
        raise ApiError(409, "pull_running", "A model download is already running.")
    diagnostics.clear_cache()
    return _pull(request).state()


# --- settings ---------------------------------------------------------------------------------------


@settings_router.get("")
def read_settings(settings: SettingsDep):
    return settings_store.describe(settings)


class SettingsIn(BaseModel):
    values: dict[str, Any]  # validated strictly by settings_store (no coercion of 1 to true)


@settings_router.put("")
def write_settings(body: SettingsIn, request: Request, settings: SettingsDep):
    result = guarded(settings_store.update, settings, body.values)
    if "watch_enabled" in body.values:
        watch = request.app.state.watch
        watch.start() if settings_store.effective(settings, "watch_enabled") else watch.stop()
    return result
