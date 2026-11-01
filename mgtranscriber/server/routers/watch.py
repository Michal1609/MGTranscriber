"""Watched sources: CRUD, manual check (also as dry run) and preview."""

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ...folders import FolderError
from ...llm.translations import load_translation
from ...watch import WatchError, WatchManager
from ..errors import ApiError

router = APIRouter(prefix="/api/watch", tags=["watch"])


class SourceIn(BaseModel):
    kind: str | None = Field(None, pattern=r"^(folder|rss|youtube)$")
    target: str | None = Field(None, min_length=1, max_length=2000)
    name: str | None = Field(None, max_length=120)
    enabled: bool | None = None
    language: str | None = Field(None, pattern=r"^(auto|[a-z]{2,3})$")
    speakers: int | None = Field(None, ge=1, le=20)
    engine: str | None = Field(None, max_length=40)
    model: str | None = Field(None, max_length=100)
    interval_s: int | None = Field(None, ge=1)
    mode: str | None = Field(None, pattern=r"^(new|last_n)$")
    last_n: int | None = Field(None, ge=1, le=50)
    recursive: bool | None = None
    analyze: bool | None = None
    translate_to: str | None = Field(None, max_length=12)  # target language of an automatic translation
    webhook_url: str | None = Field(None, max_length=2000)
    max_per_check: int | None = Field(None, ge=1, le=20)
    folder: str | None = Field(None, max_length=1000)  # library folder for the transcripts; created when new

    def values(self) -> dict:
        return self.model_dump(exclude_unset=True)


def manager(request: Request) -> WatchManager:
    return request.app.state.watch


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except WatchError as exc:
        raise ApiError(exc.status, exc.code, exc.message) from exc


def canonical_folder(request: Request, values: dict) -> dict:
    """The folder spelled like an existing one that differs only by case ("podcasts" -> "Podcasts")."""
    if values.get("folder"):
        try:
            values["folder"] = request.app.state.folders.canonical(values["folder"])
        except FolderError as exc:
            raise ApiError(exc.status, exc.code, exc.message) from exc
    return values


def register_folder(request: Request, folder: str) -> None:
    """A source's target folder shows up in the folder list at once, even before its first transcript."""
    if folder:
        try:
            request.app.state.folders.create(folder)
        except FolderError as exc:
            raise ApiError(exc.status, exc.code, exc.message) from exc


@router.get("")
def list_sources(request: Request):
    m = manager(request)
    return {"sources": m.list(), "limits": m.limits()}


@router.post("", status_code=201)
def create_source(body: SourceIn, request: Request):
    source = call(manager(request).create, canonical_folder(request, body.values()))
    register_folder(request, source.folder)
    return source.public()


@router.post("/preview")
def preview(body: SourceIn, request: Request):
    """What a source would contain right now. Nothing is saved, queued or marked as processed."""
    return call(manager(request).preview, body.values())


@router.get("/{source_id}")
def get_source(source_id: str, request: Request):
    return call(manager(request).get, source_id).public()


@router.patch("/{source_id}")
@router.put("/{source_id}")
def update_source(source_id: str, body: SourceIn, request: Request):
    source = call(manager(request).update, source_id, canonical_folder(request, body.values()))
    register_folder(request, source.folder)
    return source.public()


@router.delete("/{source_id}")
def delete_source(source_id: str, request: Request):
    call(manager(request).delete, source_id)
    return {"deleted": source_id}


@router.post("/{source_id}/check")
def check_now(source_id: str, request: Request, dry_run: bool = False):
    """Check now. ``dry_run=true`` lists what would be queued without queueing or remembering anything."""
    m = manager(request)
    result = call(m.check, source_id, manual=True, dry_run=dry_run)
    return {"result": result, "source": m.get(source_id).public()}


@router.get("/{source_id}/history")
def history(source_id: str, request: Request):
    src = call(manager(request).get, source_id)
    queue, library = request.app.state.queue, request.app.state.library
    items = []
    for item in src.items:
        tr = item.get("translation")
        if tr and tr.get("job") is not None:  # the live state of the translate job
            job = queue.get(tr["job"])
            if job:
                status = job.status
            else:  # cleared from the queue: only the stored translation proves it succeeded
                found = item.get("result") and load_translation(library, item["result"], tr["language"])
                status = "done" if found else "unknown"
            item = {**item, "translation": {**tr, "status": status}}
        items.append(item)
    return {"history": src.history, "items": items}
