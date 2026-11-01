"""Local transcription toolkit: WhisperX on your GPU, a job queue, a transcript library and a web UI.

The product name lives in ``APP_NAME`` only. Everything user-visible (CLI name, env prefix,
web title) derives from it; internal imports are relative, so renaming the package means
renaming this directory and the ``[project]`` entries in pyproject.toml.
"""

APP_NAME = "mgtranscriber"          # package, command, env prefix, cookie
APP_DISPLAY_NAME = "MGTranscriber"  # shown to people (web UI, installer, docs)
__version__ = "0.2.0"
