"""Job task types. A job runs ``pipeline.run`` (task "transcribe") or a registered task.

Other GPU work (e.g. recomputing voiceprints of an existing transcript) goes through
the same queue, so it never competes with a transcription for the GPU::

    register_task("voiceprints", "mgtranscriber.voices:compute_task", "Voiceprints")

``fn(request, settings, progress, library) -> transcript_id``; for tasks on an existing
transcript ``request.source`` is the transcript id. Targets are imported lazily, in the
worker process.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

from .config import Settings
from .library import Library
from .pipeline import TranscriptionRequest, run
from .progress import ProgressCallback, no_progress

TRANSCRIBE = "transcribe"


@dataclass(frozen=True)
class TaskInfo:
    name: str
    target: str  # "package.module:function"
    label: str


_TASKS: dict[str, TaskInfo] = {}


def register_task(name: str, target: str, label: str | None = None) -> None:
    _TASKS[name] = TaskInfo(name, target, label or name)


def list_tasks() -> list[TaskInfo]:
    return list(_TASKS.values())


def get_task(name: str) -> Callable:
    try:
        info = _TASKS[name]
    except KeyError:
        raise KeyError(f"Unknown task '{name}'. Available: {', '.join([TRANSCRIBE, *_TASKS])}") from None
    module, _, attr = info.target.partition(":")
    return getattr(importlib.import_module(module), attr)


def execute(request: TranscriptionRequest, settings: Settings, progress: ProgressCallback = no_progress,
            library: Library | None = None) -> str:
    """Run the job's task; returns the transcript id."""
    if request.task == TRANSCRIBE:
        return run(request, settings, progress, library)
    return get_task(request.task)(request, settings, progress, library or Library(settings.output_dir))


def _register_from_env() -> None:
    """``<APP>_TASKS="name=package.module:function;..."``: tasks from outside the package (tests, plugins)."""
    import os

    from .config import ENV_PREFIX

    for item in os.environ.get(ENV_PREFIX + "TASKS", "").split(";"):
        name, _, target = item.partition("=")
        if name.strip() and ":" in target:
            register_task(name.strip(), target.strip())


# Built-in tasks: register them here (this module is imported in the worker process too).
register_task("voiceprints", "mgtranscriber.voices:compute_task", "Voiceprints")
register_task("voice_scan", "mgtranscriber.voices:scan_task", "Voice scan")
register_task("translate", "mgtranscriber.llm.translate:translate_task", "Translation")
_register_from_env()
