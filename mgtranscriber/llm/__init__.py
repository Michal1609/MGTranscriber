"""LLM features over transcripts: analysis (summary, chapters, quotes, action items) and Q&A.

Providers (``ollama`` local, ``gemini`` and ``anthropic`` cloud) share the ``LLMProvider``
interface. API keys are read from the environment only (``GEMINI_API_KEY``,
``ANTHROPIC_API_KEY``, also from the project's ``.env``); they are never stored or returned.
Heavy modules are imported lazily so importing this package stays cheap.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from ..config import ENV_PREFIX, Settings
from .base import (BadModelOutput, Cancelled, LLMError, LLMProvider, MissingApiKey, ModelMissing,
                   ProviderUnavailable)

__all__ = ["LLMConfig", "LLMError", "LLMProvider", "Cancelled", "ModelMissing", "ProviderUnavailable", "MissingApiKey",
           "BadModelOutput", "PROVIDERS", "load_config", "make_provider", "provider_status", "api_key"]

log = logging.getLogger(__name__)
KEY_ENVS = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

#: name -> (label, default model, models offered in the UI)
PROVIDERS: dict[str, dict] = {
    "ollama": {"label": "Ollama (local)", "model": "gemma3:4b", "models": ["gemma3:4b", "qwen2.5:7b-instruct"],
               "cloud": False},
    "gemini": {"label": "Gemini (cloud)", "model": "gemini-3.5-flash-lite",
               "models": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"], "cloud": True},
    "anthropic": {"label": "Claude (cloud)", "model": "claude-haiku-5-5",
                  "models": ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"], "cloud": True},
}


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "ollama"
    model: str | None = None  # None = the provider's default
    final_model: str | None = None  # optional stronger model for reduce/chapters/translation
    base_url: str | None = None  # Ollama only
    embed_model: str | None = None  # Ollama only
    num_ctx: int = 8192  # Ollama only
    language: str = "auto"  # language of the generated text; "auto" = the transcript's language
    auto: bool = False  # analyse automatically after each transcription

    def public(self) -> dict:
        return {"provider": self.provider, "model": self.model or PROVIDERS[self.provider]["model"],
                "final_model": self.final_model, "base_url": self.base_url, "language": self.language,
                "auto": self.auto}


def _truthy(value) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}


def _num_ctx(value, default: int = 8192) -> int:
    try:
        number = int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        number = 0
    if not 512 <= number <= 1_000_000:
        log.warning("Ignoring invalid llm_num_ctx %r; using %d", value, default)
        return default
    return number


def load_config(settings: Settings, env: dict[str, str] | None = None) -> LLMConfig:
    """``llm_*`` keys of settings.json (``settings.extra``), overridden by ``MGTRANSCRIBER_LLM_*`` variables."""
    environ = os.environ if env is None else env
    raw = {k[4:]: v for k, v in settings.extra.items() if k.startswith("llm_")}
    for key in ("provider", "model", "final_model", "base_url", "embed_model", "num_ctx", "language", "auto"):
        if (value := environ.get(f"{ENV_PREFIX}LLM_{key.upper()}")) not in (None, ""):
            raw[key] = value
    provider = str(raw.get("provider") or "ollama").lower()
    if provider not in PROVIDERS:
        provider = "ollama"
    return LLMConfig(
        provider=provider, model=raw.get("model") or None, final_model=raw.get("final_model") or None,
        base_url=raw.get("base_url") or None, embed_model=raw.get("embed_model") or None,
        num_ctx=_num_ctx(raw.get("num_ctx")), language=str(raw.get("language") or settings.language or "auto"),
        auto=_truthy(raw.get("auto", False)))


def api_key(provider: str, settings: Settings, env: dict[str, str] | None = None) -> str | None:
    """The key from the process environment or the project's .env. Never from settings.json."""
    name = KEY_ENVS.get(provider)
    if not name:
        return None
    environ = os.environ if env is None else env
    value = environ.get(name) or dotenv_values(Path(settings.root) / ".env").get(name)
    return value.strip() if value and value.strip() else None


def make_provider(cfg: LLMConfig, settings: Settings, provider: str | None = None, model: str | None = None, *,
                  env: dict[str, str] | None = None) -> LLMProvider:
    """Build a provider. ``provider``/``model`` override the configured ones (request options)."""
    name = (provider or cfg.provider).lower()
    if name not in PROVIDERS:
        raise LLMError(f"Unknown LLM provider '{name}'. Available: {', '.join(PROVIDERS)}.", "unknown_llm_provider")
    from ..privacy import PrivacyError, check_llm

    try:  # offline mode: no cloud provider, Ollama only on this computer
        check_llm(settings, name, PROVIDERS[name]["cloud"], cfg.base_url if name == "ollama" else None)
    except PrivacyError as exc:
        error = LLMError(exc.message, exc.code)
        error.status = exc.status
        raise error from None
    same = name == cfg.provider
    chosen = model or (cfg.model if same else None)
    if name == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(chosen, cfg.base_url, num_ctx=cfg.num_ctx, embed_model=cfg.embed_model)
    if name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(chosen, api_key(name, settings, env))
    from .claude import AnthropicProvider
    return AnthropicProvider(chosen, api_key(name, settings, env))


def provider_status(cfg: LLMConfig, settings: Settings, env: dict[str, str] | None = None) -> list[dict]:
    """What the UI needs to offer a choice: never contains keys, only whether one is set."""
    from ..privacy import PrivacyError, check_llm

    out = []
    for name, spec in PROVIDERS.items():
        item = {"name": name, "label": spec["label"], "cloud": spec["cloud"],
                "default_model": cfg.model if name == cfg.provider and cfg.model else spec["model"],
                "models": list(spec["models"])}
        try:
            check_llm(settings, name, spec["cloud"], cfg.base_url if name == "ollama" else None)
        except PrivacyError:
            out.append({**item, "available": False, "running": False, "installed": [], "offline": True})
            continue
        if name == "ollama":
            try:
                from .ollama import OllamaProvider
                installed = OllamaProvider(None, cfg.base_url).installed_models()
                item.update(available=True, running=True, installed=installed,
                            models=sorted(set(spec["models"]) | set(installed)))
            except LLMError:
                item.update(available=False, running=False, installed=[])
        else:
            item.update(available=api_key(name, settings, env) is not None, key_env=KEY_ENVS[name])
        out.append(item)
    return out
