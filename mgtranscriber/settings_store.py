"""The settings a user may change from the web UI (a whitelist), stored in ``settings.json``.

Only harmless preferences are editable. Secrets (API keys, the server's own ``api_key``), paths, host/port and
anything that changes what the server may access are **not** in the whitelist: they stay in ``.env`` /
``settings.json`` edited by hand. The exceptions are the HuggingFace token and the cloud API keys (``set_hf_token``, ``set_api_key``), see there.

A value from the process environment (``MGTRANSCRIBER_*``) beats ``settings.json``; such a key is reported as
``source: "env"`` and ``locked``, and refused on write, so the UI never pretends to change something it cannot.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from dataclasses import fields
from pathlib import Path

from dotenv import dotenv_values

from . import privacy, voices
from .config import ENV_PREFIX, SETTINGS_FILE, Settings
from .engines import list_engines
from .fsutil import replace_file
from .llm import PROVIDERS, load_config

LANGUAGES = ("cs", "en", "auto", "sk", "de", "pl", "fr", "es", "it", "ru", "uk")
MODEL_RE = re.compile(r"^[\w.\-:/]{1,100}$")
_lock = threading.Lock()


class SettingError(Exception):
    """Invalid or refused change; ``code`` is stable (translated by the UI)."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


# key -> (kind, group, in Settings as a real field?)
SPECS: dict[str, tuple[str, str, bool]] = {
    "language": ("choice", "transcription", True),
    "engine": ("choice", "transcription", True),
    "model": ("text", "transcription", True),
    "diarize": ("bool", "transcription", True),
    "llm_provider": ("choice", "llm", False),
    "llm_model": ("text", "llm", False),
    "llm_language": ("choice", "llm", False),
    "llm_auto": ("bool", "llm", False),
    "voice_memory": ("bool", "features", False),
    "watch_enabled": ("bool", "features", False),
    "offline_mode": ("bool", "features", False),
}
FIELD_DEFAULTS = {f.name: f.default for f in fields(Settings) if f.name in SPECS and SPECS[f.name][2]}
EXTRA_DEFAULTS = {"llm_provider": "ollama", "llm_model": "", "llm_language": "", "llm_auto": False,
                  "voice_memory": True, "watch_enabled": True, "offline_mode": False}


def _truthy(value) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}


def env_name(key: str) -> str:
    return ENV_PREFIX + key.upper()  # llm_provider -> MGTRANSCRIBER_LLM_PROVIDER, voice_memory -> MGTRANSCRIBER_VOICE_MEMORY


def env_locked(settings: Settings, key: str) -> bool:
    """Is the key set by the environment (or, for real Settings fields, ``.env``)?"""
    if os.environ.get(env_name(key)) not in (None, ""):
        return True
    if SPECS[key][2]:
        env_file = Path(settings.root) / ".env"
        return env_file.is_file() and dotenv_values(env_file).get(env_name(key)) not in (None, "")
    return False


def _read_file(settings: Settings) -> dict:
    path = settings.settings_path
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SettingError("settings_file_invalid", f"{SETTINGS_FILE} cannot be read ({exc}). Fix or delete it first.", 409) from exc
    if not isinstance(data, dict):
        raise SettingError("settings_file_invalid", f"{SETTINGS_FILE} must contain a JSON object.", 409)
    return data


def effective(settings: Settings, key: str):
    if SPECS[key][2]:
        return getattr(settings, key)
    if key.startswith("llm_"):
        cfg = load_config(settings)
        return {"llm_provider": cfg.provider, "llm_model": cfg.model or "", "llm_language": cfg.language,
                "llm_auto": cfg.auto}[key]
    if key == "voice_memory":
        return voices.enabled(settings)
    if key == "offline_mode":
        return privacy.offline(settings)
    value = os.environ.get(env_name("watch_enabled"))  # the environment wins, like offline_mode
    if value in (None, ""):
        value = settings.extra.get("watch_enabled", True)
    return _truthy(value)


def choices(key: str) -> list[str] | None:
    if key == "language":
        return list(LANGUAGES)
    if key == "engine":
        return [e.name for e in list_engines() if e.available and e.name != "mock"]
    if key == "llm_provider":
        return list(PROVIDERS)
    if key == "llm_language":
        return ["auto", "en", "cs"]
    return None


def suggestions(settings: Settings) -> dict[str, list[str]]:
    """Model names to offer in the free-text model fields."""
    cfg = load_config(settings)
    engine = {e.name: e for e in list_engines()}.get(settings.engine)
    return {"model": list(engine.models) if engine else [],
            "llm_model": list(PROVIDERS.get(cfg.provider, PROVIDERS["ollama"])["models"]),
            "by_engine": {e.name: list(e.models) for e in list_engines() if e.available},
            "by_provider": {name: list(spec["models"]) for name, spec in PROVIDERS.items()}}


def describe(settings: Settings) -> dict:
    stored = _read_file(settings)
    items = []
    for key, (kind, group, is_field) in SPECS.items():
        locked = env_locked(settings, key)
        in_file = key in stored
        default = FIELD_DEFAULTS.get(key) if is_field else EXTRA_DEFAULTS[key]
        items.append({"key": key, "kind": kind, "group": group, "value": effective(settings, key), "default": default,
                      "choices": choices(key), "locked": locked,
                      "source": "env" if locked else "settings.json" if in_file else "default"})
    return {"settings": items, "suggestions": suggestions(settings), "file": SETTINGS_FILE,
            "env_prefix": ENV_PREFIX}


def _validate(key: str, value):
    kind = SPECS[key][0]
    if kind == "bool":
        if not isinstance(value, bool):
            raise SettingError("setting_invalid", f"{key} must be true or false.")
        return value
    if not isinstance(value, str):
        raise SettingError("setting_invalid", f"{key} must be text.")
    value = value.strip()
    if kind == "choice":
        allowed = choices(key) or []
        if value not in allowed and not (key == "language" and re.fullmatch(r"[a-z]{2,3}", value)):
            raise SettingError("setting_invalid", f"{key} must be one of: {', '.join(allowed)}.")
        return value
    if value and not MODEL_RE.fullmatch(value):
        raise SettingError("setting_invalid", f"{key} contains characters that are not allowed.")
    if key == "model" and not value:
        raise SettingError("setting_invalid", "The model cannot be empty.")
    return value


def update(settings: Settings, changes: dict) -> dict:
    """Validate and save ``{key: value}`` (``None`` resets a key to its default), then apply it to the running
    server. Nothing is written when any key is invalid or locked."""
    if not changes:
        raise SettingError("setting_invalid", "Nothing to change.")
    clean: dict = {}
    for key, value in changes.items():
        if key not in SPECS:
            raise SettingError("setting_unknown", f"'{key}' cannot be changed here.")
        if env_locked(settings, key):
            raise SettingError("setting_overridden",
                               f"{key} is set by the environment variable {env_name(key)}; change or remove it there.", 409)
        clean[key] = None if value is None else _validate(key, value)
    with _lock:
        data = _read_file(settings)
        for key, value in clean.items():
            if value is None or (not SPECS[key][2] and value == "" and key in ("llm_model", "llm_language")):
                data.pop(key, None)
            else:
                data[key] = value
        path = settings.settings_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        replace_file(tmp, path)
    apply(settings, clean)
    return describe(settings)


def apply(settings: Settings, changes: dict) -> None:
    """Make a saved change live. ``Settings`` is frozen and shared by the queue, the watcher and the LLM layer, so
    the shared object itself is updated. All fields change in ONE ``__dict__.update`` call (atomic under the GIL), so
    a reader never sees a new engine with the old model; ``extra`` is replaced by a modified copy in the same call."""
    fields_new: dict = {}
    extra = dict(settings.extra)
    for key, value in changes.items():
        if SPECS[key][2]:
            fields_new[key] = FIELD_DEFAULTS[key] if value is None else value
        elif value is None or (value == "" and key in ("llm_model", "llm_language")):
            extra.pop(key, None)
        else:
            extra[key] = value
    settings.__dict__.update({**fields_new, "extra": extra})


# --- the HuggingFace token ----------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{8,200}", re.ASCII)


def check_token_format(token: str) -> str:
    """The stripped token, or ``SettingError`` before anything else (network, files) happens."""
    token = token.strip()
    if not TOKEN_RE.fullmatch(token):
        raise SettingError("hf_token_invalid", "That does not look like a HuggingFace token.")
    return token


def _rewrite_env(path: Path, token: str | None, name: str = "HF_TOKEN") -> None:
    """Set or remove ``name`` (``HF_TOKEN`` or a cloud API key) in ``.env``, touching nothing else: other lines, the line endings and (POSIX) the
    permissions are kept; a file that is not UTF-8 is refused instead of being damaged; removing a token never creates
    a file and does not rewrite one that has no token."""
    raw = path.read_bytes() if path.is_file() else None
    if raw is None and token is None:
        return
    try:
        lines = raw.decode("utf-8").splitlines() if raw is not None else []
    except UnicodeDecodeError as exc:
        raise SettingError("env_file_unreadable", ".env is not UTF-8 text; edit it there by hand.", 409) from exc
    eol = "\r\n" if raw is not None and b"\r\n" in raw else "\n"
    kept = [ln for ln in lines if not re.match(rf"\s*(export\s+)?{re.escape(name)}\s*=", ln)]
    if token is None and len(kept) == len(lines):
        return
    if token:
        kept.append(f"{name}={token}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes((eol.join(kept) + (eol if kept else "")).encode("utf-8"))
    if raw is not None:
        shutil.copymode(path, tmp)
    elif os.name == "posix":
        os.chmod(tmp, 0o600)  # a new file with a secret: owner only
    replace_file(tmp, path)


def set_hf_token(settings: Settings, token: str | None, *, client_local: bool = True) -> None:
    """Write (or remove with ``None``) ``HF_TOKEN`` in the project's ``.env`` and apply it to the running server.

    Decision: this is the only secret the web UI may store. Without it a new user has to edit a hidden file before
    the first transcription with speakers works, the most common reason people give up. It is safe enough because
    (1) it is accepted only while the server listens on loopback AND the request comes from a loopback address (not
    through a reverse proxy or tunnel), and the security middleware rejects other origins; (2) it goes to ``.env``,
    the file the docs already name, not ``settings.json``; (3) its format is checked first and it is checked with
    HuggingFace before it is saved (unless that is impossible: offline mode, no connection), and it is never
    returned by any endpoint. Cloud API keys follow the same rules (``set_api_key``).
    """
    if not settings.is_loopback or not client_local:
        raise SettingError("local_only", "The token can only be saved from this computer while the server listens on this computer only.", 403)
    if os.environ.get("HF_TOKEN"):
        raise SettingError("token_from_environment", "HF_TOKEN is set in the process environment; change it there.", 409)
    if token is not None:
        token = check_token_format(token)
    with _lock:
        _rewrite_env(Path(settings.root) / ".env", token)
    settings.__dict__["hf_token"] = token or None


# --- cloud API keys -------------------------------------------------------------------------------

KEY_RE = re.compile(r"[A-Za-z0-9_\-.~+/=:]{8,300}", re.ASCII)


def set_api_key(settings: Settings, name: str, value: str | None, *, client_local: bool = True) -> None:
    """Write (or remove with ``None``) a cloud API key in ``.env``: the same rules as ``set_hf_token`` (loopback
    only, refused when the process environment has it, never returned). Nothing needs applying: every reader
    (engines, LLM providers, job workers) looks the key up in the environment and then in ``.env`` on each use."""
    from .diagnostics import CLOUD_KEYS

    if name not in CLOUD_KEYS:
        raise SettingError("key_unknown", "That is not a key this program uses.", 404)
    if not settings.is_loopback or not client_local:
        raise SettingError("local_only", "Keys can only be saved from this computer while the server listens on this computer only.", 403)
    if os.environ.get(name):
        raise SettingError("key_from_environment", f"{name} is set in the process environment; change it there.", 409)
    if value is not None:
        value = value.strip()
        if not KEY_RE.fullmatch(value):
            raise SettingError("key_invalid", "That does not look like an API key (8-300 characters: letters, digits and - _ . ~ + / = : only).")
    with _lock:
        _rewrite_env(Path(settings.root) / ".env", value, name)
