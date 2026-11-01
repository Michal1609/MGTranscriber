"""A watched folder in a really running server: drop in a clip, it is queued and transcribed by a small model.
Run with: uv run pytest -m gpu tests/test_watch_gpu.py"""

import os
import shutil
import time

import httpx
import pytest

from tests.conftest import wait_for
from tests.make_clip import ensure_clip

pytestmark = pytest.mark.gpu
MODEL = os.environ.get("MGTRANSCRIBER_TEST_MODEL", "tiny")


def test_clip_dropped_into_watched_folder_is_transcribed(live_server, settings, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        clip = ensure_clip()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    watched = tmp_path / "inbox"
    watched.mkdir()
    base = live_server(engine="whisperx", model=MODEL, language="en", diarize=False,
                       extra={"watch_stable_s": 1, "watch_min_folder_s": 1})
    with httpx.Client(base_url=base, timeout=30) as http:
        source = http.post("/api/watch", json={"kind": "folder", "target": str(watched), "interval_s": 1}).json()
        sid = source["id"]
        wait_for(lambda: http.get(f"/api/watch/{sid}").json()["baselined"])
        shutil.copy(clip, watched / "dropped.wav")  # arrives after the source was added
        wait_for(lambda: http.get(f"/api/watch/{sid}/history").json()["items"], timeout=30)
        wait_for(lambda: http.get(f"/api/watch/{sid}/history").json()["items"][0]["status"] in ("done", "error"), timeout=300)
        item = http.get(f"/api/watch/{sid}/history").json()["items"][0]
        assert item["status"] == "done", item
        text = " ".join(s["text"] for s in http.get(f"/api/transcripts/{item['result']}").json()["segments"]).lower()
        assert "ai safety" in text or "safe ai" in text
        time.sleep(3)  # and it is not queued again
        assert len(http.get("/api/jobs").json()) == 1
