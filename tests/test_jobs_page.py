"""The Jobs page API: filters, paging, summary, retry, remove from history and clearing."""

import pytest
from fastapi.testclient import TestClient

from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.server import create_app
from tests.conftest import wait_for


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        yield c


def run_job(client, media_file, name):
    job = client.post("/api/jobs", json={"source": str(media_file(name))}).json()
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] in ("done", "error", "cancelled"))
    return job["id"]


@pytest.fixture
def history(client, media_file):
    """Two finished, two failed jobs (ids 1-4)."""
    return [run_job(client, media_file, n) for n in ("a.wav", "fail.wav", "b.wav", "fail2.wav")]


def test_plain_list_keeps_its_shape(client, history):
    listing = client.get("/api/jobs").json()
    assert isinstance(listing, list) and [j["id"] for j in listing] == [4, 3, 2, 1]
    assert [j["id"] for j in client.get("/api/jobs?status=failed").json()] == [4, 2]


def test_paging_and_counts(client, history):
    page = client.get("/api/jobs?limit=3").json()
    assert [j["id"] for j in page["items"]] == [4, 3, 2]
    assert page["total"] == 4 and page["limit"] == 3 and page["offset"] == 0
    assert page["counts"] == {"all": 4, "active": 0, "done": 2, "failed": 2, "finished": 4}
    assert [j["id"] for j in client.get("/api/jobs?limit=3&offset=3").json()["items"]] == [1]
    failed = client.get("/api/jobs?status=failed&limit=1&offset=1").json()
    assert failed["total"] == 2 and [j["id"] for j in failed["items"]] == [2]
    assert [j["id"] for j in client.get("/api/jobs?status=done,error&limit=10").json()["items"]] == [4, 3, 2, 1]
    assert client.get("/api/jobs?status=nonsense").json()["code"] == "invalid_status"
    assert client.get("/api/jobs?limit=0").status_code == 422


def test_active_tab_orders_running_first_then_queue(client, media_file):
    slow = client.post("/api/jobs", json={"source": str(media_file("slow.wav"))}).json()
    wait_for(lambda: client.get(f"/api/jobs/{slow['id']}").json()["status"] == "running")
    queued = [client.post("/api/jobs", json={"source": str(media_file(f"q{i}.wav"))}).json()["id"] for i in range(2)]
    page = client.get("/api/jobs?status=active&limit=10").json()
    assert [j["id"] for j in page["items"]] == [slow["id"], *queued]
    assert [j["position"] for j in page["items"]] == [None, 1, 2]
    summary = client.get("/api/jobs/summary").json()
    assert summary["running"]["id"] == slow["id"] and summary["queued"] == 2 and summary["active"] == 3
    # an active job cannot be removed or cleared; cancel it first
    assert client.delete(f"/api/jobs/{queued[0]}/history").json()["code"] == "job_active"
    assert client.post("/api/jobs/clear?status=active").json()["code"] == "invalid_status"
    for job_id in (slow["id"], *queued):
        client.delete(f"/api/jobs/{job_id}")
    wait_for(lambda: client.get("/api/jobs/summary").json()["active"] == 0)


def test_summary_counts_recent_failures(client, history):
    summary = client.get("/api/jobs/summary").json()
    assert summary == {"running": None, "queued": 0, "active": 0, "failed_recent": 2}


def test_retry_requeues_with_the_same_options(client, media_file, settings):
    src = media_file("fail.wav")
    job = client.post("/api/jobs", json={"source": str(src), "language": "cs", "speakers": 3}).json()
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] == "error")
    new = client.post(f"/api/jobs/{job['id']}/retry")
    assert new.status_code == 201
    body = new.json()
    assert body["id"] != job["id"] and body["language"] == "cs" and body["speakers"] == 3 and body["source"] == str(src)
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404  # the failed entry is replaced
    wait_for(lambda: client.get(f"/api/jobs/{body['id']}").json()["status"] == "error")  # it fails again, honestly
    # done jobs are not retryable; a vanished file says so
    done = run_job(client, media_file, "ok.wav")
    assert client.post(f"/api/jobs/{done}/retry").json()["code"] == "job_not_retryable"
    src.unlink()
    assert client.post(f"/api/jobs/{body['id']}/retry").json()["code"] == "source_gone"
    assert client.post("/api/jobs/999/retry").status_code == 404


def test_remove_from_history_keeps_the_transcript(client, media_file):
    job_id = run_job(client, media_file, "keep.wav")
    transcript = client.get(f"/api/jobs/{job_id}").json()["result"]
    log_file = client.app.state.queue.get(job_id).log_file
    assert log_file.exists()
    assert client.delete(f"/api/jobs/{job_id}/history").json() == {"removed": job_id}
    assert client.get(f"/api/jobs/{job_id}").status_code == 404 and not log_file.exists()
    assert client.get(f"/api/transcripts/{transcript}").status_code == 200
    assert client.delete(f"/api/jobs/{job_id}/history").status_code == 404


def test_clear_finished_and_failed(client, history):
    assert client.post("/api/jobs/clear?status=failed").json() == {"removed": 2}
    assert [j["id"] for j in client.get("/api/jobs").json()] == [3, 1]
    assert client.post("/api/jobs/clear").json() == {"removed": 2}
    assert client.get("/api/jobs").json() == []
    assert client.post("/api/jobs/clear?status=bogus").json()["code"] == "invalid_status"
