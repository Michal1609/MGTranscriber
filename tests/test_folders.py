"""Virtual library folders: normalisation, the store, the API, jobs, search and the MCP tools."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from mcp import Client

from mgtranscriber.folders import FolderError, FolderStore, in_folder, normalize_path
from mgtranscriber.jobs import InProcessRunner, JobQueue
from mgtranscriber.mcp_server import LocalJobs, build_server
from mgtranscriber.search import SearchIndex
from mgtranscriber.server import create_app

from .conftest import wait_for

SEGMENTS = [{"start": 0.0, "end": 2.0, "text": " Ahoj světe.", "speaker": "SPEAKER_00", "words": []}]


def add(library, tid: str, text: str = "Ahoj světe", folder: str | None = None) -> str:
    path = library.root / tid
    path.mkdir(parents=True, exist_ok=True)
    seg = [{**SEGMENTS[0], "text": f" {text}."}]
    (path / "prepis.json").write_text(json.dumps(seg, ensure_ascii=False), encoding="utf-8")
    meta = {"title": tid, "date": "2025-01-01T00:00:00+00:00", "duration": 2.0}
    if folder is not None:
        meta["folder"] = folder
    (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return tid


@pytest.fixture
def store(settings, library):
    return FolderStore(library, settings.data_dir / "folders.json")


def folder_paths(store) -> list[str]:
    return [f["path"] for f in store.tree()["folders"]]


# --- normalisation ------------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("", ""), (None, ""), ("  ", ""), ("/", ""),
    ("Podcasts/Cílek", "Podcasts/Cílek"),
    ("  a //  b / ", "a/b"),
    ("a\\b\\c", "a/b/c"),
    ("/a/b/", "a/b"),
    ("é", "é"),  # NFC
])
def test_normalize(raw, expected):
    assert normalize_path(raw) == expected


@pytest.mark.parametrize("raw, code", [
    ("a/b/c/d/e/f", "folder_too_deep"),
    ("x" * 61, "folder_name_too_long"),
    ("/".join(["y" * 52] * 5), "folder_name_too_long"),
    ("a/../b", "folder_invalid"),
    ("a/./b", "folder_invalid"),
    ("bad\x00name", "folder_invalid"),
    ("tab\tname", "folder_invalid"),
    ("new\nline", "folder_invalid"),
])
def test_normalize_refuses(raw, code):
    with pytest.raises(FolderError) as e:
        normalize_path(raw)
    assert e.value.code == code


def test_in_folder():
    assert in_folder("a/b", "a/b") and in_folder("a/b/c", "a", recursive=True)
    assert not in_folder("a/b/c", "a") and not in_folder("ab", "a", recursive=True)
    assert in_folder("", "") and not in_folder("a", "", recursive=True)  # unfiled is only unfiled


# --- the store ----------------------------------------------------------------------------------


def test_empty_folders_persist(store, settings, library):
    assert store.create("Client X/2025") == ("Client X/2025", True)
    assert store.create("client x") == ("Client X", False)  # case-insensitive: the existing spelling wins
    assert folder_paths(store) == ["Client X", "Client X/2025"]
    saved = json.loads((settings.data_dir / "folders.json").read_text(encoding="utf-8"))
    assert saved["folders"] == ["Client X", "Client X/2025"]
    again = FolderStore(library, settings.data_dir / "folders.json")  # a new process
    assert folder_paths(again) == ["Client X", "Client X/2025"]


def test_tree_counts_and_implicit_folders(store, library):
    add(library, "t1", folder="Podcasts")
    add(library, "t2", folder="Podcasts/Cílek")
    add(library, "t3", folder="Podcasts/Cílek")
    add(library, "t4")
    add(library, "t5", folder="Only/Implicit")  # never created through the store
    store.create("Empty")
    tree = store.tree()
    by_path = {f["path"]: f for f in tree["folders"]}
    assert tree["unfiled"] == 1 and tree["total"] == 5
    assert (by_path["Podcasts"]["count"], by_path["Podcasts"]["total"]) == (1, 3)
    assert (by_path["Podcasts/Cílek"]["count"], by_path["Podcasts/Cílek"]["total"]) == (2, 2)
    assert by_path["Empty"]["total"] == 0
    assert by_path["Only/Implicit"]["parent"] == "Only" and by_path["Only"]["count"] == 0
    assert folder_paths(store) == ["Empty", "Only", "Only/Implicit", "Podcasts", "Podcasts/Cílek"]


def test_rename_moves_the_subtree(store, library):
    add(library, "t1", folder="A")
    add(library, "t2", folder="A/B")
    add(library, "t3", folder="A/B/C")
    add(library, "t4", folder="Other")
    store.create("A/Empty")
    result = store.move("A", "Z/Renamed")
    assert result["path"] == "Z/Renamed" and result["moved"] == 3 and result["failed"] == []
    assert library.folders_map() == {"t1": "Z/Renamed", "t2": "Z/Renamed/B", "t3": "Z/Renamed/B/C", "t4": "Other"}
    assert folder_paths(store) == ["Other", "Z", "Z/Renamed", "Z/Renamed/B", "Z/Renamed/B/C", "Z/Renamed/Empty"]


def test_move_errors_and_case_rename(store, library):
    add(library, "t1", folder="a")
    store.create("B")
    with pytest.raises(FolderError) as e:
        store.move("a", "a/inner")
    assert e.value.code == "folder_cycle"
    with pytest.raises(FolderError) as e:
        store.move("a", "b")  # "B" exists (case-insensitive)... but the name "b" is canonicalised onto it
    assert e.value.code == "folder_exists"
    with pytest.raises(FolderError) as e:
        store.move("missing", "x")
    assert e.value.code == "folder_not_found" and e.value.status == 404
    with pytest.raises(FolderError) as e:
        store.move("a", "x/y/z/a/b/c")
    assert e.value.code == "folder_too_deep"
    assert store.move("a", "A")["path"] == "A"  # only the case changes
    assert library.folder_of("t1") == "A"


def test_delete_moves_transcripts_to_the_parent(store, library):
    add(library, "t1", folder="P")
    add(library, "t2", folder="P/Sub")
    add(library, "t3", folder="P/Sub/Deep")
    store.create("P/Sub/Empty")
    result = store.delete("P/Sub", "parent")
    assert result["moved"] == 2 and result["target"] == "P"
    assert result["deleted"] == ["P/Sub", "P/Sub/Deep", "P/Sub/Empty"]
    assert library.folders_map() == {"t1": "P", "t2": "P", "t3": "P"}
    assert folder_paths(store) == ["P"]
    assert len(library.list()) == 3  # nothing deleted


def test_delete_to_unfiled_and_top_level(store, library):
    add(library, "t1", folder="P")
    add(library, "t2", folder="P/Sub")
    store.delete("P/Sub", "unfiled")
    assert library.folder_of("t2") == ""
    store.delete("P")  # a top level folder: the parent is unfiled
    assert library.folders_map() == {"t1": "", "t2": ""}
    assert folder_paths(store) == []
    with pytest.raises(FolderError):
        store.delete("P")
    with pytest.raises(FolderError):
        store.delete("x", "elsewhere")


def test_bulk_assign(store, library):
    ids = [add(library, f"t{i}") for i in range(5)]
    out = store.assign(ids[:3] + ["nope", ids[0]], "New/Project")
    assert out == {"folder": "New/Project", "moved": 3, "missing": ["nope"], "failed": []}
    assert library.folders_map()["t2"] == "New/Project" and library.folders_map()["t3"] == ""
    assert "New" in folder_paths(store)  # created, with its parent
    assert store.assign(["t0"], "new/project")["folder"] == "New/Project"  # existing spelling
    assert store.assign(ids, "")["moved"] == 3  # the three that were in the folder
    assert set(library.folders_map().values()) == {""}


def test_set_folder_runs_listeners_and_keeps_other_metadata(store, library):
    add(library, "t1")
    seen = []
    library.listeners.append(seen.append)
    meta = library.metadata("t1")
    meta.extra["keep"] = 1
    library.save_metadata("t1", meta)
    seen.clear()
    store.assign(["t1"], "F")
    assert seen == ["t1"]
    after = library.metadata("t1")
    assert after.folder == "F" and after.extra["keep"] == 1 and after.title == "t1"
    assert library.summary("t1")["folder"] == "F"


def test_legacy_transcript_without_metadata_can_be_filed(store, library, legacy_transcript):
    assert library.summary(legacy_transcript)["folder"] == ""
    store.assign([legacy_transcript], "Old")
    assert library.folder_of(legacy_transcript) == "Old"
    assert library.metadata(legacy_transcript).title == "Old video"


def test_unreadable_folders_file_is_set_aside(settings, library):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "folders.json").write_text("{not json", encoding="utf-8")
    store = FolderStore(library, settings.data_dir / "folders.json")
    assert folder_paths(store) == []
    assert (settings.data_dir / "folders.json.broken").exists() or (settings.data_dir / "folders.broken").exists()
    store.create("Fresh")
    assert folder_paths(FolderStore(library, settings.data_dir / "folders.json")) == ["Fresh"]


def test_another_process_changes_are_picked_up(store, library, settings):
    other = FolderStore(library, settings.data_dir / "folders.json")
    store.create("One")
    other.create("Two")
    assert folder_paths(store) == ["One", "Two"]


def test_list_by_folder(library):
    add(library, "a", folder="X")
    add(library, "b", folder="X/Y")
    add(library, "c")
    ids = lambda **kw: sorted(t["id"] for t in library.list(**kw))  # noqa: E731
    assert ids() == ["a", "b", "c"]
    assert ids(folder="X") == ["a"]
    assert ids(folder="X", recursive=True) == ["a", "b"]
    assert ids(folder="") == ["c"]


# --- the API ------------------------------------------------------------------------------------


@pytest.fixture
def client(settings, library):
    add(library, "ep1", "první epizoda o počasí")
    add(library, "ep2", "druhá epizoda o počasí", folder="Podcasts")
    add(library, "ep3", "třetí epizoda o počasí", folder="Podcasts/Cílek")
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        yield c


def test_api_folder_lifecycle(client):
    tree = client.get("/api/folders").json()
    assert tree["unfiled"] == 1 and tree["total"] == 3
    assert [f["path"] for f in tree["folders"]] == ["Podcasts", "Podcasts/Cílek"]

    r = client.post("/api/folders", json={"path": " Client X / 2025 "})
    assert r.status_code == 201 and r.json() == {"path": "Client X/2025", "created": True}
    assert client.post("/api/folders", json={"path": "client x"}).json() == {"path": "Client X", "created": False}
    assert client.post("/api/folders", json={"path": "a/../b"}).json()["code"] == "folder_invalid"
    assert client.post("/api/folders", json={"path": "/"}).status_code == 400

    r = client.post("/api/folders/move", json={"path": "Podcasts", "to": "Shows"})
    assert r.json()["moved"] == 2
    assert client.get("/api/transcripts/ep3").json()["folder"] == "Shows/Cílek"
    assert client.post("/api/folders/move", json={"path": "Nope", "to": "x"}).status_code == 404
    assert client.post("/api/folders/move", json={"path": "Shows", "to": "Client X"}).json()["code"] == "folder_exists"

    r = client.delete("/api/folders", params={"path": "Shows/Cílek"})
    assert r.json()["moved"] == 1 and r.json()["target"] == "Shows"
    assert client.get("/api/transcripts/ep3").json()["folder"] == "Shows"
    assert client.delete("/api/folders", params={"path": "Shows", "move_to": "unfiled"}).json()["moved"] == 2
    assert client.get("/api/folders").json()["unfiled"] == 3
    assert client.delete("/api/folders", params={"path": "Shows", "move_to": "x"}).status_code == 422


def test_api_set_and_bulk_folder(client):
    r = client.put("/api/transcripts/ep1/folder", json={"folder": "Inbox"})
    assert r.json() == {"id": "ep1", "folder": "Inbox"}
    assert client.get("/api/transcripts/ep1").json()["metadata"]["folder"] == "Inbox"
    assert client.put("/api/transcripts/missing/folder", json={"folder": "x"}).status_code == 404
    assert client.put("/api/transcripts/ep1/folder", json={"folder": "a/../b"}).status_code == 400

    r = client.post("/api/folders/assign", json={"ids": ["ep2", "ep3", "ghost"], "folder": "Inbox"})
    assert r.json() == {"folder": "Inbox", "moved": 2, "missing": ["ghost"], "failed": []}
    assert client.get("/api/folders").json()["folders"][0] == {
        "path": "Inbox", "name": "Inbox", "parent": "", "depth": 1, "count": 3, "total": 3}


def test_api_listing_filter(client):
    def ids(**params):
        return sorted(t["id"] for t in client.get("/api/transcripts", params=params).json())

    assert ids() == ["ep1", "ep2", "ep3"]
    assert ids(folder="") == ["ep1"]
    assert ids(folder="Podcasts") == ["ep2", "ep3"]  # recursive by default, like search and MCP
    assert ids(folder="Podcasts", recursive="false") == ["ep2"]
    assert ids(folder="Nothing") == []
    assert client.get("/api/transcripts", params={"folder": "a/../b"}).status_code == 400


def test_api_search_in_folder(client):
    client.app.state.search.sync()  # the start-up sync runs on a thread; wait for it
    def hits(**params):
        found = client.get("/api/search", params={"q": "počasí", **params}).json()
        return sorted(h["transcript_id"] for h in found["results"]), found["total"]

    assert hits() == (["ep1", "ep2", "ep3"], 3)
    assert hits(folder="Podcasts") == (["ep2", "ep3"], 2)
    assert hits(folder="Podcasts", recursive="false") == (["ep2"], 1)
    assert hits(folder="") == (["ep1"], 1)
    assert hits(folder="Missing") == ([], 0)
    # follows a move at once (the index does not store folders)
    client.put("/api/transcripts/ep1/folder", json={"folder": "Podcasts"})
    assert hits(folder="Podcasts", recursive="false") == (["ep1", "ep2"], 2)
    # a speaker-only search is limited too
    spk = client.get("/api/search", params={"speaker": "SPEAKER_00", "folder": "Podcasts/Cílek"}).json()
    assert [h["transcript_id"] for h in spk["results"]] == ["ep3"]


def test_job_with_folder(client, media_file, settings):
    r = client.post("/api/jobs", json={"url": str(media_file("talk one.wav")), "language": "en", "folder": "Client/Calls"})
    assert r.status_code == 201 and r.json()["folder"] == "Client/Calls"
    wait_for(lambda: client.get("/api/jobs").json()[0]["status"] == "done")
    tid = client.get("/api/jobs").json()[0]["result"]
    assert client.get(f"/api/transcripts/{tid}").json()["folder"] == "Client/Calls"
    assert "Client/Calls" in [f["path"] for f in client.get("/api/folders").json()["folders"]]

    # transcribing the same source again keeps the folder unless one is given
    r = client.post("/api/jobs", json={"url": str(media_file("talk one.wav")), "language": "en"})
    wait_for(lambda: client.get(f"/api/jobs/{r.json()['id']}").json()["status"] == "done")
    assert client.get(f"/api/transcripts/{tid}").json()["folder"] == "Client/Calls"
    r = client.post("/api/jobs", json={"url": str(media_file("talk one.wav")), "language": "en", "folder": ""})
    wait_for(lambda: client.get(f"/api/jobs/{r.json()['id']}").json()["status"] == "done")
    assert client.get(f"/api/transcripts/{tid}").json()["folder"] == ""

    bad = client.post("/api/jobs", json={"url": str(media_file("x.wav")), "folder": "a/../b"})
    assert bad.status_code == 400 and bad.json()["code"] == "folder_invalid"


def test_upload_with_folder(client):
    r = client.post("/api/jobs/upload", files={"file": ("Memo.wav", b"RIFF fake", "audio/wav")},
                    data={"language": "en", "folder": "Voice memos"})
    assert r.status_code == 201
    job_id = r.json()["id"]
    wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] == "done")
    tid = client.get(f"/api/jobs/{job_id}").json()["result"]
    assert client.get(f"/api/transcripts/{tid}").json()["folder"] == "Voice memos"
    # an empty form field means "no folder chosen"
    r = client.post("/api/jobs/upload", files={"file": ("Memo2.wav", b"RIFF fake", "audio/wav")},
                    data={"language": "en", "folder": ""})
    assert r.status_code == 201 and r.json()["folder"] is None


def test_watch_source_has_a_target_folder(client, tmp_path):
    watched = tmp_path / "inbox"
    watched.mkdir()
    r = client.post("/api/watch", json={"kind": "folder", "target": str(watched), "folder": "Dropbox / Calls"})
    assert r.status_code == 201, r.text
    assert r.json()["folder"] == "Dropbox/Calls"
    assert "Dropbox/Calls" in [f["path"] for f in client.get("/api/folders").json()["folders"]]
    source = client.app.state.watch.get(r.json()["id"])
    from mgtranscriber.watch.fetchers import Item
    request = client.app.state.watch._request(source, Item(key="k", source="x.wav", title="x"))
    assert request.folder == "Dropbox/Calls"
    assert client.patch(f"/api/watch/{r.json()['id']}", json={"folder": ""}).json()["folder"] == ""


# --- MCP ----------------------------------------------------------------------------------------


def test_mcp_folder_tools(settings, library, media_file):
    add(library, "ep1", "první epizoda o počasí")
    add(library, "ep2", "druhá epizoda o počasí", folder="Podcasts")
    add(library, "ep3", "třetí epizoda o počasí", folder="Podcasts/Cílek")
    queue = JobQueue(settings, runner=InProcessRunner())
    index = SearchIndex(library, settings.search_file)
    library.listeners.append(index.reindex)
    queue.on_finished.append(lambda job: job.result and index.reindex(job.result))
    index.sync()
    server = build_server(settings, library, index, LocalJobs(queue), allow_local_paths=True)

    async def call(client, tool, **args):
        result = await client.call_tool(tool, args)
        assert not result.is_error, result.content[0].text
        return result.structured_content

    async def scenario():
        async with Client(server) as client:
            folders = await call(client, "list_folders")
            assert folders["unfiled"] == 1 and folders["total_transcripts"] == 3
            assert folders["folders"] == [{"path": "Podcasts", "count": 1, "total": 2},
                                          {"path": "Podcasts/Cílek", "count": 1, "total": 1}]
            listing = await call(client, "list_transcripts", folder="Podcasts")
            assert sorted(t["id"] for t in listing["transcripts"]) == ["ep2", "ep3"]
            assert {t["id"]: t["folder"] for t in listing["transcripts"]}["ep3"] == "Podcasts/Cílek"
            flat = await call(client, "list_transcripts", folder="Podcasts", include_subfolders=False)
            assert [t["id"] for t in flat["transcripts"]] == ["ep2"]
            assert [t["id"] for t in (await call(client, "list_transcripts", folder=""))["transcripts"]] == ["ep1"]
            hits = await call(client, "search_transcripts", q="počasí", folder="podcasts/cílek")  # spelling is forgiven
            assert [h["transcript_id"] for h in hits["hits"]] == ["ep3"]
            assert (await call(client, "search_transcripts", q="počasí"))["total"] == 3
            missing = await client.call_tool("list_transcripts", {"folder": "Nope"})
            assert missing.is_error and "list_folders" in missing.content[0].text

            job = await call(client, "transcribe", url_or_path=str(media_file("mcp folder.wav")), language="en",
                             folder="Interviews/2025", wait=True)
            assert job["status"] == "done"
            assert library.folder_of(job["transcript_id"]) == "Interviews/2025"
            assert "Interviews/2025" in [f["path"] for f in (await call(client, "list_folders"))["folders"]]
            bad = await client.call_tool("transcribe", {"url_or_path": str(media_file("x.wav")), "folder": "a/../b"})
            assert bad.is_error

    try:
        asyncio.run(scenario())
    finally:
        queue.stop()
        index.close()


def test_a_move_does_not_rewrite_the_index_rows(settings, library, store):
    add(library, "t1", "první epizoda o počasí")
    index = SearchIndex(library, settings.search_file)
    library.listeners.append(index.reindex)
    index.sync()
    rows = lambda: index._db().execute("SELECT rowid FROM segments WHERE transcript_id = 't1'").fetchall()  # noqa: E731
    before = rows()
    store.assign(["t1"], "Moved")
    assert rows() == before  # only the stamp was refreshed
    assert index.sync()["changed"] == 0
    assert index.search("počasí", transcript_ids=["t1"])["total"] == 1
    assert index.search("počasí", transcript_ids=[])["total"] == 0
    # a title change still reindexes
    meta = library.metadata("t1")
    meta.title = "Nový název"
    library.save_metadata("t1", meta)
    assert index.search("počasí")["results"][0]["title"] == "Nový název"
    index.close()


def test_a_big_move_does_not_hold_the_store_lock(store, library):
    import threading

    ids = [add(library, f"b{i}") for i in range(3)]
    started, release = threading.Event(), threading.Event()
    real = library.set_folder

    def slow(tid, folder):
        started.set()
        release.wait(10)
        real(tid, folder)

    library.set_folder = slow
    worker = threading.Thread(target=store.assign, args=(ids, "Big"))
    worker.start()
    assert started.wait(10)
    done = []
    reader = threading.Thread(target=lambda: done.append((store.tree(), store.create("Other"), store.exists("Big"))))
    reader.start()
    reader.join(5)
    release.set()
    worker.join(10)
    assert done and not reader.is_alive(), "tree/create blocked while transcripts were being written"
    assert library.folders_map() == {i: "Big" for i in ids}


def test_set_folder_patches_the_key_and_refuses_an_unreadable_file(library):
    add(library, "t1")
    meta_path = library.root / "t1" / "metadata.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["extra"] = {"custody": {"x": 1}}
    data["unknown_key"] = 5
    meta_path.write_text(json.dumps(data), encoding="utf-8")
    library.set_folder("t1", "F")
    after = json.loads(meta_path.read_text(encoding="utf-8"))
    assert after["folder"] == "F" and after["extra"] == {"custody": {"x": 1}} and after["unknown_key"] == 5

    meta_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        library.set_folder("t1", "G")
    assert meta_path.read_text(encoding="utf-8") == "{broken"  # not overwritten from the legacy defaults


def test_a_transcript_deleted_meanwhile_is_skipped(store, library):
    import shutil

    add(library, "keep")
    add(library, "gone")
    mapping = library.folders_map()
    shutil.rmtree(library.root / "gone")
    moved, failed = store._set_many({"keep": "F", "gone": "F"}, mapping)
    assert (moved, failed) == (1, [])
    assert not (library.root / "gone").exists()  # no folder with only a lock file


def test_folders_json_changes_wait_for_its_lock(store):
    import threading
    import time

    from mgtranscriber.fsutil import file_lock

    finished = threading.Event()
    with file_lock(store._mutex):  # what another process (the MCP stdio server) holds while it writes
        worker = threading.Thread(target=lambda: (store.create("A"), finished.set()))
        worker.start()
        time.sleep(0.3)
        assert not finished.is_set(), "a change went through while the file was locked"
    worker.join(10)
    assert finished.is_set()


def test_watch_folder_is_spelled_like_the_existing_one(client, tmp_path):
    client.post("/api/folders", json={"path": "Podcasts"})
    watched = tmp_path / "w"
    watched.mkdir()
    r = client.post("/api/watch", json={"kind": "folder", "target": str(watched), "folder": "podcasts/new"})
    assert r.status_code == 201 and r.json()["folder"] == "Podcasts/new"
    paths = [f["path"] for f in client.get("/api/folders").json()["folders"]]
    assert "podcasts" not in paths and "Podcasts/new" in paths
