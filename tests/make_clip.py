"""Cut the short test clip for the GPU end-to-end test out of a local transcript's audio.

The clip is third-party content, so it is generated on demand and git-ignored:

    uv run python -m tests.make_clip            # writes tests/fixtures/clip.wav
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "tests" / "fixtures" / "clip.wav"
SOURCE_GLOB = "output/The AI Safety Expert*/audio.*"
SECONDS = 40  # the first 40 s contain three speaker changes


def find_source() -> Path | None:
    return next(iter(sorted(ROOT.glob(SOURCE_GLOB))), None)


def ensure_clip() -> Path:
    """Return the clip path, creating it if needed. Raises FileNotFoundError when it cannot be made."""
    if CLIP.is_file():
        return CLIP
    source = find_source()
    if source is None:
        raise FileNotFoundError(f"No source audio matching {SOURCE_GLOB}")
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg is not in PATH")
    CLIP.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-t", str(SECONDS), "-ac", "1", "-ar", "16000", str(CLIP)],
        check=True,
    )
    return CLIP


if __name__ == "__main__":
    try:
        print(ensure_clip())
    except FileNotFoundError as e:
        sys.exit(str(e))
