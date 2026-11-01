"""Engine registry.

Engines are registered by name with an import target (``"module:Class"``) and imported
lazily, so listing engines never loads torch. Extra engines can be plugged in without
touching this file through the ``<APP>_ENGINES`` env variable: ``name=package.module:Class;...``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from ..config import ENV_PREFIX
from .base import Engine, EngineError, TranscribeOptions

if TYPE_CHECKING:
    from ..config import Settings

__all__ = ["Engine", "EngineError", "EngineInfo", "TranscribeOptions", "get_engine", "list_engines", "register_engine"]


@dataclass(frozen=True)
class EngineInfo:
    name: str
    target: str  # "module:Class"; a leading dot resolves against this package
    label: str
    available: bool = True  # False = planned: listed, but not runnable yet
    description: str = ""
    cloud: bool = False  # True = audio or the URL leaves the computer (the UI warns)
    models: tuple[str, ...] = ()  # choices for the UI; the first one is the default
    languages: tuple[str, ...] = ()  # empty = any language Whisper knows
    diarization: bool = False
    needs_language: bool = False  # cannot auto-detect the language
    env: tuple[str, ...] = ()  # required environment variables (API keys), checked for the UI
    packages: tuple[str, ...] = ()  # required importable modules, checked for the UI

    def missing(self, settings: "Settings | None" = None) -> dict | None:
        """Why the engine cannot run here: ``{"code": "planned"|"missing_key"|"missing_package", "detail": ...}``."""
        if not self.available:
            return {"code": "planned", "detail": ""}
        if settings is not None:
            from ..privacy import needs_network, offline

            if needs_network(self) and offline(settings):
                return {"code": "offline", "detail": ""}
        for module in self.packages:
            if importlib.util.find_spec(module) is None:
                return {"code": "missing_package", "detail": module}
        if settings is not None:
            from .common import env_value

            for key in self.env:
                if not env_value(settings, key):
                    return {"code": "missing_key", "detail": key}
        return None

    def public(self, settings: "Settings | None" = None) -> dict:
        data = asdict(self)
        del data["target"], data["packages"]
        data["missing"] = self.missing(settings)
        return data


_REGISTRY: dict[str, EngineInfo] = {}


def register_engine(name: str, target: str, label: str | None = None, available: bool = True, description: str = "",
                    **details) -> None:
    _REGISTRY[name] = EngineInfo(name, target, label or name, available, description, **details)


EU25 = ("bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl",
        "pt", "ro", "ru", "sk", "sl", "sv", "uk")

register_engine("whisperx", ".whisperx_local:WhisperXEngine", "WhisperX (local GPU)",
                description="faster-whisper + wav2vec2 alignment + pyannote diarization",
                models=("large-v3", "large-v3-turbo", "medium", "small"), diarization=True)
register_engine("parakeet", ".onnx_asr_local:ParakeetEngine", "Parakeet TDT v3 (local GPU)",
                description="NVIDIA Parakeet-TDT-0.6B-v3 via ONNX Runtime: the fastest recognition step (speaker detection takes the same time as for every engine), 25 European languages (detects the language itself), own word timestamps",
                models=("parakeet-tdt-0.6b-v3",), languages=EU25, diarization=True, packages=("onnx_asr",))
register_engine("canary", ".onnx_asr_local:CanaryEngine", "Canary 1B v2 (local GPU)",
                description="NVIDIA Canary-1B-v2 via ONNX Runtime: accurate multilingual recognition for 25 European languages (the language must be chosen, it is not detected)",
                models=("canary-1b-v2",), languages=EU25, diarization=True, needs_language=True, packages=("onnx_asr",))
register_engine("youtube", ".youtube_captions:YouTubeCaptionsEngine", "YouTube captions",
                description="Existing captions of the video (manual, then automatic); falls back to WhisperX when there are none",
                diarization=True)
register_engine("elevenlabs", ".cloud:ElevenLabsEngine", "ElevenLabs Scribe v2 (cloud)",
                description="Uploads the audio to ElevenLabs. Speakers and word timestamps from the service.",
                cloud=True, models=("scribe_v2",), diarization=True, env=("ELEVENLABS_API_KEY",))
register_engine("deepgram", ".cloud:DeepgramEngine", "Deepgram Nova-3 (cloud)",
                description="Uploads the audio to Deepgram (EU endpoint by default). Speakers and word timestamps from the service.",
                cloud=True, models=("nova-3",), diarization=True, env=("DEEPGRAM_API_KEY",))
register_engine("gemini", ".gemini:GeminiEngine", "Gemini + local alignment (cloud)",
                description="Uploads the audio to Google Gemini for the text; word timestamps and speakers are computed locally",
                cloud=True, models=("gemini-2.5-flash", "gemini-3.8-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"), diarization=True,
                env=("GEMINI_API_KEY",), packages=("google.genai",))


def _load_env_engines() -> None:
    for item in os.environ.get(ENV_PREFIX + "ENGINES", "").split(";"):
        if "=" in item:
            name, target = (part.strip() for part in item.split("=", 1))
            if name not in _REGISTRY:
                register_engine(name, target)


def list_engines() -> list[EngineInfo]:
    _load_env_engines()
    return list(_REGISTRY.values())


def get_engine(name: str, settings: "Settings") -> Engine:
    _load_env_engines()
    info = _REGISTRY.get(name)
    if info is None:
        raise EngineError(f"Unknown engine '{name}'. Available: {', '.join(_REGISTRY)}")
    if not info.available:
        raise EngineError(f"Engine '{name}' is not implemented yet.")
    from ..privacy import PrivacyError, check_engine

    try:
        check_engine(settings, name)  # offline mode: never even load a cloud engine
    except PrivacyError as exc:
        raise EngineError(exc.message) from None
    module_name, _, class_name = info.target.partition(":")
    module = importlib.import_module(module_name, package=__name__ if module_name.startswith(".") else None)
    return getattr(module, class_name)(settings)
