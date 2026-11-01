"""Settings: defaults < settings file (JSON) < environment (.env + process env).

Environment variables use the ``<APP_NAME>_`` prefix, e.g. ``MGTRANSCRIBER_OUTPUT_DIR``.
``HF_TOKEN`` is read without prefix because that is what HuggingFace tooling uses.
"""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from dotenv import dotenv_values

from . import APP_NAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PREFIX = APP_NAME.upper() + "_"
LEGACY_ENV_PREFIX = "LOCALSCRIBE_"  # the project's former name; still honoured, with a warning
SETTINGS_FILE = "settings.json"


def migrate_legacy_env(environ: dict | None = None) -> list[str]:
    """Copy ``LOCALSCRIBE_X`` to ``MGTRANSCRIBER_X`` (when the new name is unset) in ``environ`` (default: os.environ).

    Returns the legacy names that were used and warns once on stderr, so an old ``.env`` keeps working."""
    target = os.environ if environ is None else environ
    used = []
    for key in list(target):
        if key.startswith(LEGACY_ENV_PREFIX) and target[key] not in (None, ""):
            new = ENV_PREFIX + key[len(LEGACY_ENV_PREFIX):]
            if target.get(new) in (None, ""):
                target[new] = target[key]
                used.append(key)
    if used and not _legacy_warned.get("done"):
        _legacy_warned["done"] = True
        print(f"warning: {', '.join(sorted(used))} use the old name; rename to {ENV_PREFIX}* "
              f"(the old names still work for now).", file=sys.stderr)
    return used


_legacy_warned: dict[str, bool] = {}


LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}


def is_loopback_host(host: str) -> bool:
    """A loopback address (127.0.0.0/8, ::1) or "localhost". Not names like "127.evil.com", which
    any DNS server can point anywhere."""
    host = host.strip("[]").lower().rstrip(".")
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Settings:
    root: Path = PROJECT_ROOT
    output_dir: Path = PROJECT_ROOT / "output"  # transcript library
    data_dir: Path = PROJECT_ROOT / "data"  # job store and other app state
    hf_token: str | None = None
    engine: str = "whisperx"
    model: str = "large-v3"
    language: str = "auto"  # "auto" = detect from the audio; or an ISO code such as "en", "cs"
    batch_size: int = 8
    diarize: bool = True
    # Lift quiet passages before VAD and recognition (leveling.py); the audio file itself is not changed.
    level_audio: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    # Host headers the server answers to (comma-separated, "*" = any). Empty = loopback names,
    # plus this computer's name and addresses when listening on all interfaces.
    allowed_hosts: str = ""
    # Whether API clients may transcribe files on the server's disk: "auto" = only when
    # the server listens on loopback, "true" or "false".
    allow_local_paths: str = "auto"
    # Network (UNC) paths as sources and watch folders: refused unless true (opening one sends the user's
    # Windows credentials to that server). MGTRANSCRIBER_ALLOW_UNC_PATHS=true.
    allow_unc_paths: bool = False
    # Optional API key for /api, /v1 and /mcp (Bearer token, X-API-Key header or the web UI's
    # login cookie). Unset = no authentication; fine on loopback, warned about otherwise.
    api_key: str | None = None
    # Largest request body (uploads to /v1 and the like) in MB; bigger ones get 413. 0 = no limit.
    max_upload_mb: int = 4096
    # Where settings.json lives (MGTRANSCRIBER_SETTINGS_FILE). None = settings.json in data_dir, which a container keeps.
    settings_file: Path | None = None
    extra: dict = field(default_factory=dict)  # unknown keys from the settings file, for features

    @property
    def settings_path(self) -> Path:
        return Path(self.settings_file) if self.settings_file else self.data_dir / SETTINGS_FILE

    @property
    def jobs_file(self) -> Path:
        return self.data_dir / "jobs.json"

    @property
    def search_file(self) -> Path:
        return self.data_dir / "search.sqlite3"

    @property
    def is_loopback(self) -> bool:
        return is_loopback_host(self.host)

    @property
    def local_paths_allowed(self) -> bool:
        value = str(self.allow_local_paths).strip().lower()
        return self.is_loopback if value in ("", "auto") else value in {"1", "true", "yes", "on"}

    def with_overrides(self, **values) -> "Settings":
        return replace(self, **{k: v for k, v in values.items() if v is not None})

    def to_dict(self) -> dict:
        """JSON-safe form, used to hand the exact settings to job worker processes."""
        return {f.name: str(v) if isinstance(v := getattr(self, f.name), Path) else v for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        paths = {f.name for f in fields(cls) if "Path" in str(f.type)}
        return cls(**{k: Path(v) if k in paths and v is not None else v for k, v in data.items() if k in cls.__dataclass_fields__})


def _coerce(value, current, root: Path):
    if isinstance(current, bool):
        return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(value)
    if isinstance(current, Path):
        path = Path(value)
        return path if path.is_absolute() else root / path
    return value


def migrate_settings_file(old: Path, new: Path) -> bool:
    """Move a settings.json from the project root (older versions) to its place in the data folder, once. Nothing is
    overwritten; if it cannot be moved the old file is simply not used (it is never deleted without a copy)."""
    if old == new or not old.is_file() or new.exists():
        return False
    try:
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, new)
        old.unlink()
    except OSError:
        return False
    return True


def load_settings(root: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    """Build settings for a project root. ``env`` replaces os.environ (for tests)."""
    root = (root or PROJECT_ROOT).resolve()
    base = Settings(root=root, output_dir=root / "output", data_dir=root / "data")
    known = {f.name: getattr(base, f.name) for f in fields(Settings) if f.name not in {"root", "extra", "settings_file"}}

    dotenv = dict(dotenv_values(root / ".env"))
    migrate_legacy_env(dotenv)
    if env is None:
        migrate_legacy_env()
    else:
        env = dict(env)
        migrate_legacy_env(env)
    if env is None:
        # Features other than Settings fields (offline mode, LLM options, voice memory) read MGTRANSCRIBER_* from the
        # process environment only. Exporting the .env values (never overriding a real variable) makes `.env` work
        # for them too, as documented, and the job workers inherit them.
        for key, value in dotenv.items():
            if key.startswith(ENV_PREFIX) and value not in (None, "") and key not in os.environ:
                os.environ[key] = value
    environ = {**dotenv, **(os.environ if env is None else env)}

    # The settings file is found from the environment only (it cannot say where it is itself): the data folder, which a
    # container keeps, or an explicit MGTRANSCRIBER_SETTINGS_FILE. Older versions kept it in the project root.
    settings_file = _coerce(environ.get(ENV_PREFIX + "SETTINGS_FILE"), Path(), root) if environ.get(ENV_PREFIX + "SETTINGS_FILE") else None
    data_env = environ.get(ENV_PREFIX + "DATA_DIR")
    settings_path = settings_file or (_coerce(data_env, Path(), root) if data_env else base.data_dir) / SETTINGS_FILE
    migrate_settings_file(root / SETTINGS_FILE, settings_path)

    values: dict = {}
    extra: dict = {}
    if settings_path.is_file():
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        for key, value in data.items():
            (values if key in known else extra)[key] = value

    for key in known:
        if (value := environ.get(ENV_PREFIX + key.upper())) not in (None, ""):
            values[key] = value
    if environ.get("HF_TOKEN"):
        values["hf_token"] = environ["HF_TOKEN"]

    coerced = {k: _coerce(v, known[k], root) for k, v in values.items()}
    return replace(base, **coerced, settings_file=settings_file, extra=extra)
