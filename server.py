"""
Compatibility wrapper for `mgtranscriber serve` (used by web.bat).

    uv run server.py                  # http://127.0.0.1:8000
    uv run server.py --open           # and open the browser
    uv run server.py --host 0.0.0.0   # reachable from your network (no authentication!)
"""

import sys

from mgtranscriber.cli import main

if __name__ == "__main__":
    sys.exit(main(["serve", *sys.argv[1:]]))
