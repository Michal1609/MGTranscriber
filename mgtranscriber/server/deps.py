"""FastAPI dependencies giving routers access to the shared services on ``app.state``."""

from typing import Annotated

from fastapi import Depends, Request

from ..config import Settings
from ..folders import FolderStore
from ..jobs import JobQueue
from ..library import Library
from ..search import SearchIndex


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_library(request: Request) -> Library:
    return request.app.state.library


def get_queue(request: Request) -> JobQueue:
    return request.app.state.queue


def get_folders(request: Request) -> FolderStore:
    return request.app.state.folders


def get_search(request: Request) -> SearchIndex:
    return request.app.state.search


SettingsDep = Annotated[Settings, Depends(get_settings)]
SearchDep = Annotated[SearchIndex, Depends(get_search)]
LibraryDep = Annotated[Library, Depends(get_library)]
FoldersDep = Annotated[FolderStore, Depends(get_folders)]
QueueDep = Annotated[JobQueue, Depends(get_queue)]
