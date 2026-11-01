"""Command line: ``<app> transcribe SOURCE ...`` and ``<app> serve ...``."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from . import APP_NAME, __version__
from .config import ENV_PREFIX, load_settings, migrate_legacy_env
from .progress import Progress

SHUTDOWN_GRACE = 5  # seconds serve waits for open requests after Ctrl+C


def console_progress() -> callable:
    """Print messages as lines and percentages as one updating line (on a terminal)."""
    state = {"inline": False}
    tty = sys.stdout.isatty()

    def report(p: Progress) -> None:
        if p.message:
            if state["inline"]:
                print()
                state["inline"] = False
            print(p.message, flush=True)
        elif p.percent is not None and tty:
            print(f"\r  {p.phase}: {p.percent:5.1f}%", end="", flush=True)
            state["inline"] = True

    return report


def cmd_transcribe(args: argparse.Namespace) -> int:
    from .engines import EngineError
    from .library import Library
    from .models import speaker_stats
    from .pipeline import TranscriptionRequest, run
    from .timefmt import clock

    settings = load_settings()
    if args.output:
        settings = settings.with_overrides(output_dir=Path(args.output).resolve())
    library = Library(settings.output_dir)
    request = TranscriptionRequest(
        source=args.source, language=args.language, speakers=args.speakers, diarize=False if args.no_diarize else None,
        engine=args.engine, model=args.model, batch_size=args.batch_size,
        separate_speakers=args.separate_speakers,
    )
    try:
        transcript_id = run(request, settings, console_progress(), library)
    except (EngineError, FileNotFoundError) as e:
        print(e)
        return 1

    print("\nSpeaking time:")
    for speaker, seconds in speaker_stats(library.segments(transcript_id)).items():
        print(f"  {speaker}: {clock(seconds)}")
    print(f"\n{args.done_prefix} {library.path(transcript_id)}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    from .mcp_server import web_url_for

    settings = load_settings().with_overrides(host=args.host, port=args.port)
    app = create_app(settings)
    url = web_url_for(settings)
    print(f"Web UI: {url}  (stop with Ctrl+C)")
    print(f"OpenAI-compatible API: {url}/v1   MCP (Streamable HTTP): {url}/mcp")
    if not settings.is_loopback and not settings.api_key:
        print(f"WARNING: the server listens on {settings.host}, so other computers can reach it, and no API key is set.\n"
              f"         Anyone on the network can use it. Set {ENV_PREFIX}API_KEY in .env.")
    if args.open:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    # ws="none": no WebSocket routes; keeps uvicorn from loading the websockets package (and locking its DLL).
    # timeout_graceful_shutdown: Ctrl+C does not wait for requests that block on a job (/v1 uploads, MCP
    # wait=true); after it they are cancelled and the queue stops (an upload's job is cancelled, others resume).
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning", ws="none",
                timeout_graceful_shutdown=SHUTDOWN_GRACE)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    import logging

    from .mcp_server import run_stdio

    # stdout is the MCP channel: everything else must go to stderr
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    run_stdio(load_settings(), server_url=args.server, local=args.local)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=APP_NAME, description="Local transcription with speaker diarization")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="transcribe a URL or a local audio/video file")
    t.add_argument("source", help="video URL (YouTube etc.) or path to a local audio/video file")
    t.add_argument("--language", "-l", help="language code (cs, en, ...) or 'auto' (detect); default from settings (auto)")
    t.add_argument("--speakers", "-s", type=int, help="exact number of speakers, if known")
    t.add_argument("--model", "-m", help="Whisper model (large-v3, large-v3-turbo, medium, small, tiny, ...)")
    t.add_argument("--engine", "-e", help="transcription engine (default: whisperx)")
    t.add_argument("--output", "-o", help="library root folder (default: output)")
    t.add_argument("--batch-size", type=int, help="lower = less VRAM (default 8)")
    t.add_argument("--no-diarize", action="store_true", help="skip speaker diarization (no HF token needed)")
    t.add_argument("--separate-speakers", action="store_true",
                   help="separate overlapping voices into one track per speaker and transcribe each (slower; local engines)")
    t.add_argument("--done-prefix", default="Done:", help=argparse.SUPPRESS)  # transcribe.py prints "Hotovo:" for scripts
    t.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("serve", help="run the web UI")
    s.add_argument("--host", help="0.0.0.0 = reachable from your network (set an API key!)")
    s.add_argument("--port", type=int)
    s.add_argument("--open", action="store_true", help="open the browser after start")
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("mcp", help="run the MCP server over stdio (for Claude Desktop, Claude Code, Cursor, ...)")
    m.add_argument("--server", help="web server URL for jobs (default: from settings, used when it answers)")
    m.add_argument("--local", action="store_true", help="always run jobs in this process, never via the web server")
    m.set_defaults(func=cmd_mcp)
    return p


def main(argv: list[str] | None = None) -> int:
    migrate_legacy_env()  # LOCALSCRIBE_* -> MGTRANSCRIBER_* for the readers that use os.environ directly
    args = build_parser().parse_args(argv)
    return args.func(args)
