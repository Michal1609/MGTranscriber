"""App info for the frontend: name, version, engines and export formats."""

from fastapi import APIRouter

from ... import APP_NAME, __version__, privacy
from ...engines import list_engines
from ...exporters import list_exporters
from ...llm.translate import LANGUAGES
from .. import build, uploads
from ..deps import SettingsDep

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/info")
def info(settings: SettingsDep):
    return {
        "name": APP_NAME,
        "version": __version__,
        "build": build.STARTED,  # fingerprint of the running code (an old server has none)
        "restart_needed": build.restart_needed(),  # the files on disk changed since this server started
        "defaults": {"engine": settings.engine, "model": settings.model, "language": settings.language,
                     "diarize": settings.diarize},
        "engines": [e.public(settings) for e in list_engines()],
        "exporters": [e.public() for e in list_exporters()],
        "privacy": privacy.status(settings),
        "translate_languages": LANGUAGES,  # the choices of "Translate to" (code -> English name)
        # what POST /api/jobs/upload accepts (max_mb 0 = no limit)
        "upload": {"max_mb": settings.max_upload_mb, "extensions": list(uploads.MEDIA_EXTENSIONS)},
    }
