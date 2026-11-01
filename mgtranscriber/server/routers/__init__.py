"""Feature routers. A new feature adds its own module here and one line to ROUTERS."""

from . import (analysis, auth, editor, folders, jobs, openai_compat, privacy, recordings, search, setup, system, transcripts,
               translations, voices, watch)

ROUTERS = [
    system.router,
    setup.router,
    setup.settings_router,
    jobs.router,
    analysis.router,  # before transcripts: its export/md route adds the analysis
    transcripts.router,
    translations.router,
    folders.router,
    folders.transcript_router,
    editor.router,
    privacy.router,
    search.router,
    auth.router,
    openai_compat.router,
    voices.router,
    watch.router,
    recordings.router,
]
