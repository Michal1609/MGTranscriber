"""Subprocess entry point for one job: ``python -m <package>.jobs.worker`` with a JSON payload on stdin.

Payload: ``{"request": {...TranscriptionRequest...}, "settings": {...Settings.to_dict()...}}``.

Protocol: stdout carries only JSON events, one per line::

    {"event": "progress", "phase": "...", "percent": 12.5, "message": "...", "data": {...}}
    {"event": "done", "result": "<transcript id>"}
    {"event": "error", "message": "..."}

Everything else that libraries print (warnings, tqdm bars, native code) is redirected
to stderr, which the parent treats as the raw log. A separate process means all GPU
memory is released when the job ends.
"""

from __future__ import annotations

import json
import os
import sys
import traceback


def main() -> int:
    # Keep the real stdout for events; send anything else written to fd 1 to stderr.
    events = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def emit(**event) -> None:
        events.write(json.dumps(event, ensure_ascii=False) + "\n")
        events.flush()

    from ..config import Settings
    from ..engines import EngineError
    from ..pipeline import TranscriptionRequest
    from ..tasks import execute

    payload = json.loads(sys.stdin.read())
    settings = Settings.from_dict(payload["settings"])
    request = TranscriptionRequest.from_dict(payload["request"])

    try:
        result = execute(request, settings, progress=lambda p: emit(event="progress", **p.to_dict()))
    except (EngineError, FileNotFoundError) as e:
        emit(event="error", message=str(e))
        return 1
    except Exception as e:
        traceback.print_exc()
        emit(event="error", message=f"{type(e).__name__}: {e}")
        return 1
    emit(event="done", result=result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
