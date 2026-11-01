import json
import os
import time
from pathlib import Path

import pytest

from mgtranscriber.config import ENV_PREFIX, PROJECT_ROOT, Settings
from mgtranscriber.engines import register_engine
from mgtranscriber.library import Library

MOCK_TARGET = "tests.mock_engine:MockEngine"
register_engine("mock", MOCK_TARGET)
# subprocess workers pick the mock engine up from the environment
os.environ[ENV_PREFIX + "ENGINES"] = f"mock={MOCK_TARGET}"
os.environ[ENV_PREFIX + "TASKS"] = "mock-task=tests.mock_engine:mock_task"

LEGACY_SEGMENTS = [
    {"start": 0.0, "end": 1.5, "text": " Ahoj.", "speaker": "SPEAKER_00",
     "words": [{"word": "Ahoj.", "start": 0.0, "end": 1.5, "score": 0.9, "speaker": "SPEAKER_00"}]},
    {"start": 2.0, "end": 3.0, "text": " Nazdar,", "speaker": "SPEAKER_01", "words": []},
    {"start": 3.0, "end": 4.25, "text": " jak se máš?", "speaker": "SPEAKER_01", "words": []},
]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(root=PROJECT_ROOT, output_dir=tmp_path / "output", data_dir=tmp_path / "data",
                    engine="mock", model="", language="en", allowed_hosts="testserver")


@pytest.fixture
def library(settings: Settings) -> Library:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return Library(settings.output_dir)


@pytest.fixture
def legacy_transcript(library: Library) -> str:
    """A transcript folder as the original tool wrote it: no metadata.json."""
    folder = library.root / "Old video"
    folder.mkdir()
    (folder / "prepis.json").write_text(json.dumps(LEGACY_SEGMENTS, ensure_ascii=False), encoding="utf-8")
    (folder / "audio.webm").write_bytes(b"\x1aE\xdf\xa3fake")
    return folder.name


@pytest.fixture
def media_file(tmp_path: Path):
    """Factory for dummy input files; the name steers the mock engine."""

    def make(name: str = "clip.wav") -> Path:
        path = tmp_path / "in" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"RIFF fake")
        return path

    return make


def wait_for(predicate, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


@pytest.fixture
def live_server(settings):
    """Factory: run the real app with uvicorn on a free port (in-process jobs); yields the base URL."""
    import threading

    import uvicorn

    from mgtranscriber.jobs import InProcessRunner
    from mgtranscriber.server import create_app

    servers = []

    def start(**overrides) -> str:
        app = create_app(settings.with_overrides(**overrides), runner=InProcessRunner())
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on", ws="none"))
        threading.Thread(target=server.run, daemon=True).start()
        wait_for(lambda: server.started, timeout=20)
        servers.append(server)
        port = server.servers[0].sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    yield start
    for server in servers:
        server.should_exit = True
