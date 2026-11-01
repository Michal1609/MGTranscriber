"""
Compatibility wrapper for `mgtranscriber transcribe` (used by start.bat).

    uv run transcribe.py "https://www.youtube.com/watch?v=..." --speakers 2
    uv run transcribe.py recording.mp4 --language en
"""

import sys

from mgtranscriber.cli import main

if __name__ == "__main__":
    # Scripts can find the result folder on a line starting with "Hotovo:".
    sys.exit(main(["transcribe", *sys.argv[1:], "--done-prefix", "Hotovo:"]))
