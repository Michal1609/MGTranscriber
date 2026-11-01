import json
import shutil

import pytest
from fastapi.testclient import TestClient

from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.models import Segment
from mgtranscriber.search import SearchIndex, fts_query, split_marks
from mgtranscriber.server import create_app
from tests.conftest import wait_for

CZECH = [
    {"start": 0.0, "end": 4.0, "text": " Dobrý den, tady je první přepis rozhovoru.", "speaker": "SPEAKER_00"},
    {"start": 4.0, "end": 9.5, "text": " Umělá inteligence a bezpečnost, to je téma.", "speaker": "SPEAKER_01"},
    {"start": 70.0, "end": 75.0, "text": " Přepisování textu je nuda.", "speaker": "SPEAKER_00"},
]


def add_transcript(library, folder: str, segments: list[dict]) -> str:
    path = library.root / folder
    path.mkdir(parents=True, exist_ok=True)
    (path / "prepis.json").write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    return folder


@pytest.fixture
def index(settings, library, legacy_transcript):
    add_transcript(library, "Rozhovor", CZECH)
    idx = SearchIndex(library, settings.search_file)
    library.listeners.append(idx.reindex)
    idx.sync()
    yield idx
    idx.close()


def test_diacritics_and_prefix(index):
    hits = index.search("prepis")["results"]
    assert [(h["transcript_id"], h["segment"]) for h in hits] == [("Rozhovor", 0), ("Rozhovor", 2)] or \
           {(h["transcript_id"], h["segment"]) for h in hits} == {("Rozhovor", 0), ("Rozhovor", 2)}
    first = next(h for h in hits if h["segment"] == 0)
    assert first["start"] == 0.0 and first["end"] == 4.0 and first["speaker"] == "SPEAKER_00"
    assert first["title"] == "Rozhovor"
    a, b = first["highlights"][0]
    assert first["snippet"][a:b] == "přepis"
    assert index.search("BEZPECNOST")["total"] == 1


def test_words_are_anded_and_phrases_exact(index):
    assert index.search("umela bezpecnost")["total"] == 1
    assert index.search("umela nuda")["total"] == 0
    assert index.search('"inteligence a bezpecnost"')["total"] == 1
    assert index.search('"bezpecnost a inteligence"')["total"] == 0


@pytest.mark.parametrize("odd", ['"', "*", "AND", "OR NOT", "(", "a:b", "-x", "NEAR(", "'", "   ", "přepis\"", "^"])
def test_odd_input_never_fails(index, odd):
    result = index.search(odd)
    assert result["total"] >= 0


def test_fts_query_is_quoted():
    assert fts_query('ahoj "dva tři" svět*') == '"ahoj"* "dva tři" "svět"*'
    assert fts_query('" OR "') == '"OR"'
    assert fts_query("") == ""


def test_split_marks():
    assert split_marks("a \x02b\x03 c \x02d\x03") == ("a b c d", [[2, 3], [6, 7]])


def test_rename_updates_speaker_names(index, library):
    library.save_names("Rozhovor", {"SPEAKER_01": "Eva"})
    [hit] = index.search("bezpecnost")["results"]
    assert hit["speaker_name"] == "Eva"


def test_saving_segments_reindexes(index, library):
    segments = library.segments("Rozhovor")
    segments[1] = Segment(4.0, 9.5, " Úplně jiný obsah.", "SPEAKER_01")
    library.save_segments("Rozhovor", segments)
    assert index.search("bezpecnost")["total"] == 0
    assert index.search("uplne jiny")["total"] == 1


def test_sync_picks_up_external_changes(index, library):
    add_transcript(library, "Nový", [{"start": 1.0, "end": 2.0, "text": "Kočka na střeše."}])
    shutil.rmtree(library.root / "Rozhovor")
    assert index.sync() == {"indexed": 2, "changed": 1, "removed": 1}
    assert index.search("kocka")["results"][0]["transcript_id"] == "Nový"
    assert index.search("prepis")["total"] == 0


def test_index_survives_and_rebuilds(settings, library, index):
    index.close()
    again = SearchIndex(library, settings.search_file)
    assert again.search("prepis")["total"] == 2  # persisted
    again.close()
    settings.search_file.unlink()
    for extra in settings.data_dir.glob("search.sqlite3-*"):
        extra.unlink()
    rebuilt = SearchIndex(library, settings.search_file)
    rebuilt.sync()
    assert rebuilt.search("prepis")["total"] == 2
    rebuilt.close()


def test_filter_limit_and_offset(index):
    assert index.search("prepis", transcript_id="Old video")["total"] == 0
    page1 = index.search("prepis", limit=1)
    page2 = index.search("prepis", limit=1, offset=1)
    assert page1["total"] == 2 and len(page1["results"]) == 1
    assert page1["results"][0]["segment"] != page2["results"][0]["segment"]


def test_search_api_and_job_hook(settings, legacy_transcript, media_file):
    with TestClient(create_app(settings, runner=InProcessRunner())) as client:
        wait_for(lambda: client.get("/api/search?q=ahoj").json()["total"] == 1)
        hit = client.get("/api/search", params={"q": "ahoj"}).json()["results"][0]
        assert hit["transcript_id"] == "Old video" and hit["start"] == 0.0
        assert client.get("/api/search?q=").json() == {"query": "", "total": 0, "results": []}
        assert client.get("/api/search?q=x&limit=0").status_code == 422

        client.put("/api/transcripts/Old video/speakers", json={"SPEAKER_00": "Jana"})
        assert client.get("/api/search?q=ahoj").json()["results"][0]["speaker_name"] == "Jana"

        job = client.post("/api/jobs", json={"source": str(media_file("new.wav"))}).json()
        wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] == "done")
        wait_for(lambda: client.get("/api/search?q=hello").json()["total"] == 1, timeout=5)


def test_finished_job_is_indexed_by_the_hook(settings, legacy_transcript, media_file, monkeypatch):
    monkeypatch.setattr("mgtranscriber.search.SYNC_INTERVAL", 3600)  # no periodic sync during the test
    with TestClient(create_app(settings, runner=InProcessRunner())) as client:
        wait_for(lambda: client.get("/api/search?q=ahoj").json()["total"] == 1)
        job = client.post("/api/jobs", json={"source": str(media_file("hooked.wav"))}).json()
        wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] == "done")
        wait_for(lambda: client.get("/api/search?q=hello").json()["total"] == 1, timeout=5)


def test_a_search_without_hits_rescans_files_written_since_the_last_scan(index, library):
    import time

    assert index.search("zcela nové slovo")["total"] == 0
    add_transcript(library, "Dodatecne", [{"start": 0.0, "end": 2.0, "text": " Zcela nové slovo.", "speaker": "SPEAKER_00"}])
    time.sleep(0.3)  # past EMPTY_RESCAN_INTERVAL, but well inside SYNC_INTERVAL: only the miss triggers a scan
    # written by another process (CLI, web UI): no listener told the index
    assert index.search("zcela nove slovo")["total"] == 1
