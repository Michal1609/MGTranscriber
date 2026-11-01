"""Virtual library folders: the tree, create / rename / move / delete, and putting transcripts in them."""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from ...folders import MAX_ASSIGN, FolderError
from ..deps import FoldersDep, LibraryDep
from ..errors import ApiError

router = APIRouter(prefix="/api/folders", tags=["folders"])
transcript_router = APIRouter(prefix="/api/transcripts", tags=["folders"])


class PathIn(BaseModel):
    path: str = Field(..., max_length=1000)


class MoveIn(BaseModel):
    path: str = Field(..., max_length=1000)
    to: str = Field(..., max_length=1000)


class FolderIn(BaseModel):
    folder: str = Field("", max_length=1000)  # "" = unfiled; a new folder is created


class AssignIn(FolderIn):
    ids: list[str] = Field(..., max_length=MAX_ASSIGN)


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except FolderError as e:
        raise ApiError(e.status, e.code, e.message) from None


@router.get("")
def list_folders(folders: FoldersDep):
    """``{folders: [{path, name, parent, depth, count, total}], unfiled, total}``: ``count`` transcripts are
    directly in a folder, ``total`` in it and below. ``unfiled`` have no folder, ``total`` is all."""
    return folders.tree()


@router.post("", status_code=201)
def create_folder(body: PathIn, folders: FoldersDep):
    """Create a folder (and missing parents). Creating an existing one is not an error (``created: false``)."""
    path, created = call(folders.create, body.path)
    return {"path": path, "created": created}


@router.post("/move")
def move_folder(body: MoveIn, folders: FoldersDep):
    """Rename or move a folder, with its subfolders and all their transcripts. ``to`` is the new full path."""
    return call(folders.move, body.path, body.to)


@router.delete("")
def delete_folder(folders: FoldersDep, path: str = Query(..., max_length=1000),
                  move_to: str = Query("parent", pattern=r"^(parent|unfiled)$")):
    """Delete a folder and its subfolders. The transcripts in them go to the parent folder (``move_to=parent``)
    or to unfiled; transcripts are never deleted."""
    return call(folders.delete, path, move_to)


@router.post("/assign")
def assign(body: AssignIn, folders: FoldersDep):
    """Put many transcripts into one folder (created when new; empty = unfiled). Unknown ids come back in
    ``missing`` and do not fail the request."""
    return call(folders.assign, body.ids, body.folder)


@transcript_router.put("/{transcript_id}/folder")
def set_folder(transcript_id: str, body: FolderIn, folders: FoldersDep, library: LibraryDep):
    library.path(transcript_id)  # 404 for unknown transcripts
    result = call(folders.assign, [transcript_id], body.folder)
    return {"id": transcript_id, "folder": result["folder"]}
