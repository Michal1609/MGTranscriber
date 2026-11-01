r"""Network (UNC) paths as sources.

Touching ``\\host\share\file`` (even ``Path.is_file()``) makes Windows open an SMB session and send the
logged-in user's NTLM response to that host, so a path an attacker slips in (a prompt-injected MCP call, a
rogue API client) leaks credentials before anything is read. Sources and watch folders of that form are
refused before any file system call, unless ``MGTRANSCRIBER_ALLOW_UNC_PATHS=true`` (setting ``allow_unc_paths``).
The check looks at the text only.
"""

from __future__ import annotations

import re

# \\?\C:\dir (a long path on a local drive) is not a network path; everything else starting with two
# separators is: \\host\share, \\?\UNC\host\share, \\.\pipe\x, \\?\GLOBALROOT\..., \\?\Volume{...}
_LOCAL_LONG = re.compile(r"^\\\\[?.]\\[A-Za-z]:(\\|$)")
_NT_LOCAL = re.compile(r"^\\\?\?\\[A-Za-z]:(\\|$)")

UNC_MESSAGE = (r"Network paths (\\server\share) are refused: opening one sends this user's Windows "
               "credentials to that server. Set MGTRANSCRIBER_ALLOW_UNC_PATHS=true to allow them.")


def is_unc_path(path: str) -> bool:
    r"""True for a network/device path in any spelling: backslashes, forward slashes, mixed, ``\\?\UNC\``,
    NT-style ``\??\UNC\``, with leading blanks or quotes."""
    text = str(path).strip().strip("\"'").strip().replace("/", "\\")
    if text.startswith("\\??\\"):  # NT object namespace: \??\UNC\host\share, \??\C:\x
        return _NT_LOCAL.match(text) is None
    if not text.startswith("\\\\"):
        return False
    return _LOCAL_LONG.match(text) is None


def unc_blocked(settings, path: str) -> bool:
    return is_unc_path(path) and not getattr(settings, "allow_unc_paths", False)
