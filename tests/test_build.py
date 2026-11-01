"""/api/info tells the UI whether the server runs the code that is on disk (the "restart the server" banner)."""

from fastapi.testclient import TestClient

from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.server import build, create_app


def info(settings):
    with TestClient(create_app(settings, runner=InProcessRunner())) as c:
        return c.get("/api/info").json()


def test_info_reports_the_build_and_no_restart_needed(settings):
    data = info(settings)
    assert data["build"] == build.STARTED and len(data["build"]) == 12
    assert data["restart_needed"] is False


def test_restart_needed_when_the_files_changed_since_start(settings, monkeypatch):
    monkeypatch.setattr(build, "source_stamp", lambda: "something-else")
    monkeypatch.setattr(build, "_checked", (0.0, False))  # forget the cached answer
    assert info(settings)["restart_needed"] is True


def test_stamp_follows_the_source_files(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(build, "PACKAGE", tmp_path)
    first = build.source_stamp()
    assert first == build.source_stamp()
    (tmp_path / "a.py").write_text("x = 22\n")
    assert build.source_stamp() != first
