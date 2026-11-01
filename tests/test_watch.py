import dataclasses
import json
import os
import threading
import time
from datetime import datetime, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.server import create_app
from mgtranscriber.watch import WatchError, WatchManager
from mgtranscriber.watch.fetchers import parse_feed, youtube_feed_url
from tests.conftest import wait_for


class FakeQueue:
    def __init__(self):
        self.requests, self.jobs = [], {}

    def add(self, request):
        job = SimpleNamespace(id=len(self.requests) + 1, request=request, status="queued", result=None, error=None,
                              title=None, log=[], add_log=lambda line: None)
        self.requests.append(request)
        self.jobs[job.id] = job
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def listing(self):
        return [{"id": j.id, "status": j.status, "task": getattr(j.request, "task", "transcribe"),
                 "source": j.request.source, "language": j.request.language} for j in self.jobs.values()]

    def public(self, job):
        return {"id": job.id, "status": job.status}

    @property
    def sources(self):
        return [r.source for r in self.requests]


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def wsettings(settings):
    return dataclasses.replace(settings, extra={"watch_stable_s": 0, "watch_min_folder_s": 1, "watch_min_rss_s": 1,
                                                "watch_min_youtube_s": 1})


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def queue():
    return FakeQueue()


@pytest.fixture
def manager(wsettings, queue, clock):
    return WatchManager(wsettings, queue, clock=clock)


def rss(items):
    body = "".join(
        f"<item><title>{t}</title><guid>{g}</guid><pubDate>{format_datetime(datetime.fromtimestamp(ts, timezone.utc))}</pubDate>"
        f'<enclosure url="{u}" type="audio/mpeg"/></item>' for t, g, u, ts in items)
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>Pod</title>{body}</channel></rss>'.encode()


@pytest.fixture
def feed_server():
    """Serves /feed.xml from ``state['body']``; ``state['status']`` forces an error; records POSTs."""
    state = {"body": rss([]), "status": 200, "hits": 0, "etag": None, "posts": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["hits"] += 1
            if state["status"] != 200:
                self.send_response(state["status"])
                self.end_headers()
                return
            if state["etag"] and self.headers.get("If-None-Match") == state["etag"]:
                self.send_response(304)
                self.end_headers()
                return
            self.send_response(200)
            if state["etag"]:
                self.send_header("ETag", state["etag"])
            self.send_header("Content-Length", str(len(state["body"])))
            self.end_headers()
            self.wfile.write(state["body"])

        def do_POST(self):
            state["posts"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


def make_file(folder, name, content=b"x" * 500, mtime=None):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(content)
    if mtime:
        os.utime(path, (mtime, mtime))
    return path


# --- folder -----------------------------------------------------------------------------------

def test_folder_ignores_existing_files_and_waits_for_stable_size(manager, queue, clock, tmp_path):
    watched = tmp_path / "in"
    make_file(watched, "old.mp3", mtime=clock.now - 100)
    src = manager.create({"kind": "folder", "target": str(watched), "interval_s": 5})
    manager.check(src.id)
    assert queue.sources == [] and src.baselined
    clock.now += 10
    new = make_file(watched, "new.mp3", b"a" * 1000, mtime=clock.now)
    make_file(watched, "half.mp3.part", b"b" * 10, mtime=clock.now)
    manager.check(src.id)
    assert queue.sources == []  # first sighting: could still be written
    clock.now += 10
    new.write_bytes(b"a" * 2000)  # still growing
    os.utime(new, (clock.now, clock.now))
    manager.check(src.id)
    assert queue.sources == []
    clock.now += 10
    manager.check(src.id)
    assert queue.sources == [str(new.resolve())]
    clock.now += 10
    manager.check(src.id)
    assert len(queue.sources) == 1  # not queued twice


def test_folder_dedupes_by_content_and_supports_recursion(manager, queue, clock, tmp_path):
    watched = tmp_path / "in"
    watched.mkdir()
    src = manager.create({"kind": "folder", "target": str(watched), "recursive": True})
    manager.check(src.id)
    clock.now += 10
    make_file(watched / "sub", "a.wav", b"same" * 200, mtime=clock.now)
    manager.check(src.id)
    clock.now += 10
    manager.check(src.id)
    assert len(queue.sources) == 1
    make_file(watched, "copy-of-a.wav", b"same" * 200, mtime=clock.now)  # identical content
    for _ in range(3):
        clock.now += 10
        manager.check(src.id)
    assert len(queue.sources) == 1

    flat = manager.create({"kind": "folder", "target": str(watched), "recursive": False})
    manager.check(flat.id)
    clock.now += 10
    make_file(watched / "sub", "b.wav", b"other" * 200, mtime=clock.now)
    for _ in range(3):
        clock.now += 10
        manager.check(flat.id)
    assert flat.pending == {}  # the flat source never looks into sub/


def test_folder_last_n_takes_the_newest_existing_files(manager, queue, clock, tmp_path):
    watched = tmp_path / "in"
    for i in range(4):
        make_file(watched, f"f{i}.mp3", bytes([i]) * 300, mtime=clock.now - 1000 + i)
    src = manager.create({"kind": "folder", "target": str(watched), "mode": "last_n", "last_n": 2})
    manager.check(src.id)
    clock.now += 10
    manager.check(src.id)
    assert sorted(os.path.basename(s) for s in queue.sources) == ["f2.mp3", "f3.mp3"]
    clock.now += 10
    manager.check(src.id)
    assert len(queue.sources) == 2


def test_state_survives_a_restart_without_requeueing(wsettings, queue, clock, tmp_path):
    watched = tmp_path / "in"
    watched.mkdir()
    m1 = WatchManager(wsettings, queue, clock=clock)
    src = m1.create({"kind": "folder", "target": str(watched)})
    m1.check(src.id)
    clock.now += 10
    make_file(watched, "n.mp3", mtime=clock.now)
    m1.check(src.id)
    clock.now += 10
    m1.check(src.id)
    assert len(queue.sources) == 1
    m2 = WatchManager(wsettings, queue, clock=clock)
    assert m2.get(src.id).baselined and m2.get(src.id).pending
    for _ in range(3):
        clock.now += 10
        m2.check(src.id)
    assert len(queue.sources) == 1


def test_folder_validation(manager, wsettings, tmp_path):
    with pytest.raises(WatchError) as e:
        manager.create({"kind": "folder", "target": str(tmp_path / "missing")})
    assert e.value.code == "watch_folder_missing"
    wsettings.output_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(WatchError) as e:
        manager.create({"kind": "folder", "target": str(wsettings.output_dir)})
    assert e.value.code == "watch_folder_forbidden"
    locked = WatchManager(dataclasses.replace(wsettings, allow_local_paths="false"), FakeQueue())
    with pytest.raises(WatchError) as e:
        locked.create({"kind": "folder", "target": str(tmp_path)})
    assert e.value.code == "local_paths_disabled"


# --- RSS --------------------------------------------------------------------------------------

def episode(n, base, ts):
    return (f"Episode {n}", f"guid-{n}", f"{base}/ep{n}.mp3", ts)


def test_rss_new_only_then_new_episode(manager, queue, clock, feed_server):
    base = feed_server["url"]
    feed_server["body"] = rss([episode(1, base, clock.now - 5000), episode(2, base, clock.now - 4000)])
    src = manager.create({"kind": "rss", "target": base + "/feed.xml", "interval_s": 60})
    manager.check(src.id)
    assert queue.sources == []
    clock.now += 100
    feed_server["body"] = rss([episode(3, base, clock.now - 10), episode(1, base, clock.now - 5000)])
    manager.check(src.id)
    assert queue.sources == [f"{base}/ep3.mp3"]
    # same GUID with a changed URL (CDN change) is still the same episode
    feed_server["body"] = rss([("Episode 3", "guid-3", f"{base}/cdn/ep3.mp3", clock.now - 10)])
    clock.now += 100
    manager.check(src.id)
    assert len(queue.sources) == 1


def test_rss_new_never_queues_items_older_than_the_source_even_when_listed_later(manager, queue, clock, feed_server):
    base = feed_server["url"]
    # the baseline saw only the 5 newest; a later check lists 15 (10 older ones never seen) - none of them is "new"
    feed_server["body"] = rss([episode(n, base, clock.now - 1000 * n) for n in range(1, 6)])
    src = manager.create({"kind": "rss", "target": base + "/feed.xml", "interval_s": 60, "max_per_check": 20})
    manager.check(src.id)
    assert queue.sources == [] and src.baselined
    clock.now += 100
    feed_server["body"] = rss([episode(n, base, src.created - 1000 * n) for n in range(1, 16)])
    manager.check(src.id)
    assert queue.sources == []
    clock.now += 100
    manager.check(src.id)
    assert queue.sources == []  # they were marked as seen, not left pending
    # a really new one is still queued
    feed_server["body"] = rss([episode(99, base, clock.now - 10)] + [episode(n, base, src.created - 1000 * n)
                                                                    for n in range(1, 16)])
    clock.now += 100
    manager.check(src.id)
    assert queue.sources == [f"{base}/ep99.mp3"]


def test_youtube_handle_baseline_via_ytdlp_then_rss_listing_deeper_queues_only_newer(manager, queue, clock, monkeypatch):
    """The real @handle path: the baseline lists 5 via yt-dlp, which also yields the channel feed; every later check
    reads that feed (all 15 entries). The 10 older entries were never seen but are not "new"."""
    from mgtranscriber.watch import fetchers

    uc = "UC" + "a" * 22

    def entry_ts(n):
        return clock.now - 3600 * n  # n hours before the source was added

    def atom(entries):
        body = "".join(f"<entry><id>yt:video:v{n}</id><yt:videoId>v{n}</yt:videoId><title>Video {n}</title>"
                       f"<published>{datetime.fromtimestamp(ts, timezone.utc).isoformat()}</published></entry>"
                       for n, ts in entries)
        return ('<feed xmlns="http://www.w3.org/2005/Atom" xmlns:yt="http://www.youtube.com/xml/schemas/2015">'
                f"{body}</feed>").encode()

    calls = {"flat": 0, "feed": 0}
    feed_entries = [(n, entry_ts(n)) for n in range(1, 16)]

    def fake_flat(target, limit):
        calls["flat"] += 1
        items = [fetchers.Item(key=f"yt:v{n}", source=f"https://www.youtube.com/watch?v=v{n}", title=f"Video {n}",
                               published=entry_ts(n)) for n in range(1, 6)]
        return items[:limit], f"{fetchers.YOUTUBE_FEED}?channel_id={uc}"

    def fake_get(url, cache, client=None, guard=None):
        calls["feed"] += 1
        assert url == f"{fetchers.YOUTUBE_FEED}?channel_id={uc}"
        return atom(feed_entries)

    monkeypatch.setattr(fetchers, "youtube_flat", fake_flat)
    monkeypatch.setattr(fetchers, "http_get", fake_get)
    src = manager.create({"kind": "youtube", "target": "https://www.youtube.com/@Handle/videos", "interval_s": 60})
    manager.check(src.id)
    assert calls == {"flat": 1, "feed": 0} and src.baselined and queue.sources == []
    clock.now += 100
    manager.check(src.id)
    assert calls["flat"] == 1 and calls["feed"] == 1  # the remembered feed, 15 entries
    assert queue.sources == []
    clock.now += 100
    manager.check(src.id)
    assert queue.sources == []  # marked as seen, not left pending
    feed_entries.insert(0, (99, clock.now - 10))  # a video published after the source was added
    clock.now += 100
    manager.check(src.id)
    assert queue.sources == ["https://www.youtube.com/watch?v=v99"]


def test_rss_new_item_without_a_date_after_the_baseline_relies_on_the_seen_set(manager, queue, clock, feed_server):
    base = feed_server["url"]

    def undated(n):
        return f'<item><title>E{n}</title><guid>u-{n}</guid><enclosure url="{base}/u{n}.mp3" type="audio/mpeg"/></item>'

    def body(*items):
        return f'<?xml version="1.0"?><rss version="2.0"><channel><title>P</title>{"".join(items)}</channel></rss>'.encode()

    feed_server["body"] = body(undated(1), undated(2))
    src = manager.create({"kind": "rss", "target": base + "/feed.xml", "interval_s": 60})
    manager.check(src.id)
    assert queue.sources == [] and src.baselined  # undated items present at the baseline are the existing ones
    clock.now += 100
    feed_server["body"] = body(undated(3), undated(1), undated(2))
    manager.check(src.id)
    assert queue.sources == [f"{base}/u3.mp3"]  # unseen and undated: cannot be proven old, so it is new
    clock.now += 100
    manager.check(src.id)
    assert len(queue.sources) == 1


def test_rss_last_n_still_looks_back_and_later_old_items_are_not_dropped(manager, queue, clock, feed_server):
    base = feed_server["url"]
    feed_server["body"] = rss([episode(n, base, clock.now - 1000 + n) for n in range(1, 4)])
    src = manager.create({"kind": "rss", "target": base + "/feed.xml", "mode": "last_n", "last_n": 2,
                          "max_per_check": 1})
    for _ in range(4):
        manager.check(src.id)
        clock.now += 100
    assert sorted(queue.sources) == [f"{base}/ep2.mp3", f"{base}/ep3.mp3"]


def test_folder_new_file_with_an_old_mtime_added_after_the_baseline_is_still_queued(manager, queue, clock, tmp_path):
    watched = tmp_path / "in"
    make_file(watched, "existing.mp3", b"e" * 500, mtime=clock.now - 100)
    src = manager.create({"kind": "folder", "target": str(watched), "interval_s": 5})
    manager.check(src.id)
    assert queue.sources == [] and src.baselined
    clock.now += 10
    copied = make_file(watched, "copied.mp3", b"c" * 700, mtime=src.created - 100000)  # a copy keeps its old mtime
    for _ in range(3):
        manager.check(src.id)
        clock.now += 10
    assert queue.sources == [str(copied.resolve())]


def test_rss_last_n_limits_per_check_and_passes_settings(manager, queue, clock, feed_server):
    base = feed_server["url"]
    feed_server["body"] = rss([episode(n, base, clock.now - 1000 + n) for n in range(1, 6)])
    src = manager.create({"kind": "rss", "target": base + "/feed.xml", "mode": "last_n", "last_n": 4,
                          "max_per_check": 2, "language": "cs", "speakers": 2, "engine": "mock"})
    manager.check(src.id)
    assert queue.sources == [f"{base}/ep2.mp3", f"{base}/ep3.mp3"]  # oldest of the newest 4 first
    assert (queue.requests[0].language, queue.requests[0].speakers, queue.requests[0].engine) == ("cs", 2, "mock")
    clock.now += 100
    manager.check(src.id)
    assert queue.sources[2:] == [f"{base}/ep4.mp3", f"{base}/ep5.mp3"]
    clock.now += 100
    manager.check(src.id)
    assert len(queue.sources) == 4  # episode 1 was outside "last 4"


def test_rss_uses_etag_and_leftovers_are_not_hidden(manager, queue, clock, feed_server):
    base = feed_server["url"]
    feed_server["etag"] = '"v1"'
    feed_server["body"] = rss([episode(n, base, clock.now - 1000 + n) for n in range(1, 4)])
    src = manager.create({"kind": "rss", "target": base + "/feed.xml", "mode": "last_n", "last_n": 3, "max_per_check": 1})
    for _ in range(3):
        manager.check(src.id)
        clock.now += 100
    assert len(queue.sources) == 3  # a 304 never swallowed the waiting episodes
    manager.check(src.id)
    assert len(queue.sources) == 3


def test_backoff_on_errors_and_recovery(manager, queue, clock, feed_server):
    base = feed_server["url"]
    feed_server["status"] = 503
    src = manager.create({"kind": "rss", "target": base + "/feed.xml", "interval_s": 60})
    delays = []
    for _ in range(4):
        r = manager.check(src.id)
        assert not r["ok"]
        delays.append(src.next_check - clock.now)
        clock.now = src.next_check
    assert delays == sorted(delays) and delays[0] >= 120 and delays[-1] > delays[0]
    assert src.fails == 4 and "503" in src.last_error
    clock.now += 10**6
    feed_server["status"] = 200
    assert manager.check(src.id)["ok"]
    assert src.fails == 0 and src.last_error is None and src.next_check == clock.now + 60


def test_tick_runs_only_due_enabled_sources(manager, queue, clock, feed_server):
    base = feed_server["url"]
    a = manager.create({"kind": "rss", "target": base + "/feed.xml", "interval_s": 100})
    b = manager.create({"kind": "rss", "target": base + "/feed.xml", "enabled": False})
    assert manager.tick() == 1 and a.last_check and not b.last_check
    assert manager.tick() == 0
    clock.now += 100
    assert manager.tick() == 1


def test_manual_check_has_a_cooldown_and_dry_run_changes_nothing(manager, queue, clock, feed_server):
    base = feed_server["url"]
    feed_server["body"] = rss([episode(1, base, clock.now - 10)])
    src = manager.create({"kind": "rss", "target": base + "/feed.xml", "mode": "last_n", "last_n": 1})
    dry = manager.check(src.id, manual=True, dry_run=True)
    assert [i["title"] for i in dry["queued_items"]] == ["Episode 1"]
    assert queue.sources == [] and not src.seen and src.last_check is None
    manager.check(src.id, manual=True)
    assert len(queue.sources) == 1
    with pytest.raises(WatchError) as e:
        manager.check(src.id, manual=True)
    assert e.value.code == "watch_too_soon"


def test_queue_backpressure_leaves_items_for_later(wsettings, clock, feed_server):
    queue = FakeQueue()
    m = WatchManager(dataclasses.replace(wsettings, extra={**wsettings.extra, "watch_max_queued": 1}), queue, clock=clock)
    base = feed_server["url"]
    feed_server["body"] = rss([episode(n, base, clock.now - 100 + n) for n in range(1, 4)])
    src = m.create({"kind": "rss", "target": base + "/feed.xml", "mode": "last_n", "last_n": 3, "max_per_check": 5})
    m.check(src.id)
    assert len(queue.sources) == 1
    m.check(src.id)
    assert len(queue.sources) == 1  # queue still busy
    queue.jobs[1].status = "done"
    m.check(src.id)
    assert len(queue.sources) == 2


def test_interval_is_clamped_to_the_minimum(wsettings, queue):
    m = WatchManager(dataclasses.replace(wsettings, extra={}), queue)
    src = m.create({"kind": "youtube", "target": "https://www.youtube.com/@x", "interval_s": 5})
    assert src.interval_s == 1800
    with pytest.raises(WatchError):
        m.create({"kind": "youtube", "target": "https://example.com/@x"})


# --- finishing --------------------------------------------------------------------------------

def test_finished_job_is_recorded_analysed_and_reported(wsettings, queue, clock, feed_server):
    started = []
    llm = SimpleNamespace(start=lambda tid: started.append(tid))
    m = WatchManager(wsettings, queue, llm, clock=clock)
    base = feed_server["url"]
    feed_server["body"] = rss([episode(1, base, clock.now - 10)])
    src = m.create({"kind": "rss", "target": base + "/feed.xml", "mode": "last_n", "last_n": 1, "analyze": True,
                    "webhook_url": base + "/hook"})
    m.check(src.id)
    job = queue.jobs[1]
    job.status, job.result, job.title = "done", "ep1-abc", "Episode 1"
    m.on_job_finished(job)
    assert started == ["ep1-abc"] and not src.pending
    assert src.items[0]["status"] == "done" and src.items[0]["result"] == "ep1-abc"
    wait_for(lambda: len(feed_server["posts"]) >= 1)  # the webhook runs on its own thread
    assert feed_server["posts"][0]["event"] == "transcription.done"
    assert feed_server["posts"][0]["transcript"]["id"] == "ep1-abc"
    clock.now += 100
    feed_server["body"] = rss([episode(2, base, clock.now)])
    m.check(src.id)
    failed = queue.jobs[2]
    failed.status, failed.error = "error", "boom"
    m.on_job_finished(failed)
    assert src.items[0]["status"] == "error" and "boom" in src.last_error
    wait_for(lambda: len(feed_server["posts"]) >= 2)
    assert feed_server["posts"][1]["event"] == "transcription.failed"


# --- translate_to -----------------------------------------------------------------------------

def _translating(wsettings, queue, clock, feed_server, translate_to, transcript_lang="en", check_error=None):
    def check():
        if check_error:
            raise check_error

    llm = SimpleNamespace(
        start=lambda tid: None,
        library=SimpleNamespace(metadata=lambda tid: SimpleNamespace(language=transcript_lang)),
        provider=lambda provider=None, model=None: SimpleNamespace(check=check))
    m = WatchManager(wsettings, queue, llm, clock=clock)
    base = feed_server["url"]
    feed_server["body"] = rss([episode(1, base, clock.now - 10)])
    src = m.create({"kind": "rss", "target": base + "/feed.xml", "mode": "last_n", "last_n": 1,
                    "translate_to": translate_to})
    m.check(src.id)
    job = queue.jobs[1]
    job.status, job.result, job.title = "done", "ep1-abc", "Episode 1"
    m.on_job_finished(job)
    return m, src


def _translations(queue):
    return [r for r in queue.requests if getattr(r, "task", "transcribe") == "translate"]


def test_translate_to_queues_a_translation_after_the_job(wsettings, queue, clock, feed_server):
    _, src = _translating(wsettings, queue, clock, feed_server, "cs")
    [req] = _translations(queue)
    assert (req.source, req.language) == ("ep1-abc", "cs")
    assert src.items[0]["translation"] == {"language": "cs", "job": 2}


def test_translate_to_same_language_or_none_queues_nothing(wsettings, queue, clock, feed_server):
    _, src = _translating(wsettings, queue, clock, feed_server, "cs", transcript_lang="cs")
    assert not _translations(queue) and "translation" not in src.items[0]


def test_translate_to_none_queues_nothing(wsettings, queue, clock, feed_server):
    _, src = _translating(wsettings, queue, clock, feed_server, None)
    assert src.translate_to is None and not _translations(queue)


def test_translate_to_unavailable_llm_is_an_item_error(wsettings, queue, clock, feed_server):
    from mgtranscriber.llm import LLMError
    _, src = _translating(wsettings, queue, clock, feed_server, "cs", check_error=LLMError("Ollama is not running."))
    assert not _translations(queue)
    assert src.items[0]["status"] == "done" and "Ollama" in src.items[0]["translation"]["error"]


def test_translate_to_is_validated_and_persisted(wsettings, queue, clock):
    m = WatchManager(wsettings, queue, clock=clock)
    for bad in ("klingon-x", "xx", "../etc"):
        with pytest.raises(WatchError):
            m.create({"kind": "youtube", "target": "https://www.youtube.com/@x", "translate_to": bad})
    src = m.create({"kind": "youtube", "target": "https://www.youtube.com/@x", "translate_to": " CS "})
    assert src.translate_to == "cs"
    assert WatchManager(wsettings, queue, clock=clock).sources[src.id].translate_to == "cs"
    assert m.update(src.id, {"translate_to": "pt-BR"}).translate_to == "pt"  # always exactly a LANGUAGES key
    assert m.update(src.id, {"translate_to": ""}).translate_to is None
    assert m.limits()["translate_languages"]["cs"] == "Czech"


def test_watch_json_without_translate_to_loads(wsettings, queue, clock):
    path = wsettings.data_dir / "watch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "sources": [{"id": "abcd", "kind": "rss", "target": "http://x/f.xml"}]}),
                    encoding="utf-8")
    assert WatchManager(wsettings, queue, clock=clock).sources["abcd"].translate_to is None


# --- parsing ----------------------------------------------------------------------------------

def test_parse_atom_and_youtube_feeds():
    atom = b"""<feed xmlns="http://www.w3.org/2005/Atom" xmlns:yt="http://www.youtube.com/xml/schemas/2015">
      <entry><id>yt:video:abc123</id><yt:videoId>abc123</yt:videoId><title>Hello</title>
        <published>2026-10-01T10:00:00+00:00</published></entry></feed>"""
    [item] = parse_feed(atom)
    assert item.key == "yt:abc123" and item.source == "https://www.youtube.com/watch?v=abc123" and item.published
    podcast = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>tag:1</id><title>E</title>
      <link rel="enclosure" type="audio/mpeg" href="http://x/e.mp3"/></entry><entry><id>2</id><title>no media</title></entry></feed>"""
    assert [i.source for i in parse_feed(podcast)] == ["http://x/e.mp3"]
    with pytest.raises(WatchError):
        parse_feed(b"<html></html>")
    with pytest.raises(WatchError):
        parse_feed(b"not xml")


def test_youtube_feed_urls():
    uc = "UC" + "a" * 22
    assert youtube_feed_url(f"https://www.youtube.com/channel/{uc}").endswith(f"channel_id={uc}")
    assert youtube_feed_url("https://www.youtube.com/playlist?list=PL123").endswith("playlist_id=PL123")
    assert youtube_feed_url("https://www.youtube.com/@someone") is None


# --- API --------------------------------------------------------------------------------------

def test_api_crud_check_and_scheduler_in_the_server(wsettings, tmp_path):
    watched = tmp_path / "in"
    watched.mkdir()
    app = create_app(wsettings, runner=InProcessRunner())
    with TestClient(app) as client:
        r = client.post("/api/watch", json={"kind": "folder", "target": str(watched), "interval_s": 1})
        assert r.status_code == 201, r.text
        sid = r.json()["id"]
        listing = client.get("/api/watch").json()
        assert [s["id"] for s in listing["sources"]] == [sid] and listing["limits"]["min_interval_s"]["folder"] == 1
        wait_for(lambda: client.get(f"/api/watch/{sid}").json()["baselined"])  # the scheduler thread ran the first check
        make_file(watched, "clip.wav", mtime=None)
        r = client.put(f"/api/watch/{sid}", json={"name": "Inbox", "language": "en"})
        assert r.json()["name"] == "Inbox" and r.json()["language"] == "en"
        wait_for(lambda: any(i["status"] == "done" for i in client.get(f"/api/watch/{sid}/history").json()["items"]),
                 timeout=30)
        assert len(client.get("/api/jobs").json()) == 1
        dry = client.post(f"/api/watch/{sid}/check?dry_run=true").json()
        assert dry["result"]["dry_run"] and dry["result"]["queued"] == 0
        missing = client.post("/api/watch", json={"kind": "folder", "target": str(tmp_path / "nope")})
        assert missing.json()["code"] == "watch_folder_missing"
        assert client.post("/api/watch", json={"kind": "bogus", "target": "x"}).status_code == 422
        assert client.get("/api/watch/zzz").json()["code"] == "watch_not_found"
        assert client.post("/api/watch/preview", json={"kind": "folder", "target": str(watched)}).json()["total"] == 1
        assert client.delete(f"/api/watch/{sid}").json() == {"deleted": sid}
        assert client.get("/api/watch").json()["sources"] == []


# --- review fixes -----------------------------------------------------------------------------

def test_feed_enclosures_must_be_web_urls():
    feed = rss([("Local", "g1", "C:/Users/Public/secret.wav", 1), ("UNC", "g2", "\\\\attacker\\s\\a.mp3", 2),
                ("File", "g3", "file:///etc/x.mp3", 3), ("Web", "g4", "https://example.com/a.mp3", 4)])
    assert [i.source for i in parse_feed(feed)] == ["https://example.com/a.mp3"]


def test_item_with_a_local_source_is_never_queued(manager, queue, clock, feed_server, monkeypatch):
    from mgtranscriber.watch import fetchers

    base = feed_server["url"]
    src = manager.create({"kind": "rss", "target": base + "/feed.xml", "mode": "last_n", "last_n": 5})
    evil = fetchers.Item(key="guid:x", source="C:/Users/Public/secret.wav", title="x", published=clock.now)
    monkeypatch.setattr(manager, "_list_items", lambda *a, **k: ([evil], None))
    manager.check(src.id)
    assert queue.sources == []


def test_network_server_refuses_private_addresses(wsettings, queue, clock, feed_server):
    from mgtranscriber.watch import fetchers

    open_settings = dataclasses.replace(wsettings, host="0.0.0.0")
    m = WatchManager(open_settings, queue, clock=clock)
    for target in (feed_server["url"] + "/feed.xml", "http://169.254.169.254/latest/", "http://[::1]/feed.xml"):
        with pytest.raises(WatchError) as err:
            m.create({"kind": "rss", "target": target})
        assert err.value.code == "watch_blocked_address"
    with pytest.raises(WatchError):
        m.create({"kind": "rss", "target": "http://93.184.216.34/f.xml", "webhook_url": "http://127.0.0.1/hook"})
    with pytest.raises(WatchError) as err:
        fetchers.http_get(feed_server["url"] + "/feed.xml", guard=fetchers.require_public)
    assert err.value.code == "watch_blocked_address"
    # on loopback the same address is fine (local NAS, local Ollama)
    assert WatchManager(wsettings, queue, clock=clock).create({"kind": "rss", "target": feed_server["url"] + "/feed.xml"})


def test_each_redirect_hop_is_checked():
    import httpx

    from mgtranscriber.watch import fetchers

    def handler(request):
        if request.url.host == "public.example":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:9/feed.xml"})
        return httpx.Response(200, content=b"<rss/>")

    seen = []

    def guard(url):
        seen.append(url)
        if "127.0.0.1" in url:
            raise WatchError("watch_blocked_address", "blocked")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(WatchError) as err:
            fetchers.http_get("http://public.example/feed.xml", client=client, guard=guard)
    assert err.value.code == "watch_blocked_address" and len(seen) == 2


def test_long_feeds_are_not_forgotten(monkeypatch):
    from mgtranscriber.watch import models

    monkeypatch.setattr(models, "MAX_SEEN", 10)
    src = models.Source(id="s", kind="rss", target="http://x")
    keys = [f"k{i}" for i in range(15)]
    src.remember(*keys[:10])
    src.touch(*keys[:3])  # still listed -> most recent
    src.remember(*keys[10:13])
    assert len(src.seen) == 10 and all(k in src.seen for k in keys[:3])
    src.seen.update({k: 0 for k in keys[13:]})
    src.prune(protect=keys[:3])
    assert all(k in src.seen for k in keys[:3]) and len(src.seen) == 10


def test_folder_source_survives_disabling_local_paths_and_can_be_paused(wsettings, queue, clock, tmp_path):
    watched = tmp_path / "in"
    watched.mkdir()
    m = WatchManager(wsettings, queue, clock=clock)
    src = m.create({"kind": "folder", "target": str(watched)})
    m.settings = dataclasses.replace(wsettings, allow_local_paths="false")
    result = m.check(src.id)
    assert not result["ok"] and result["code"] == "local_paths_disabled"
    assert m.update(src.id, {"enabled": False}).enabled is False


def test_unknown_engine_is_rejected(manager, feed_server):
    with pytest.raises(WatchError) as err:
        manager.create({"kind": "rss", "target": feed_server["url"] + "/feed.xml", "engine": "nope"})
    assert err.value.code == "unknown_engine"


def test_manual_checks_respect_the_cooldown_under_concurrency(manager, queue, clock, feed_server):
    src = manager.create({"kind": "rss", "target": feed_server["url"] + "/feed.xml"})
    manager.check(src.id)  # sets last_check
    results = []

    def run():
        try:
            manager.check(src.id, manual=True)
            results.append("ran")
        except WatchError as exc:
            results.append(exc.code)

    threads = [threading.Thread(target=run) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == ["watch_too_soon"] * 3


def test_folder_scan_limits_directories_and_skips_junctions(tmp_path, monkeypatch):
    from mgtranscriber.watch import fetchers

    root = tmp_path / "w"
    (root / "a").mkdir(parents=True)
    make_file(root / "a", "x.mp3")
    items, _, truncated = fetchers.scan_folder(str(root), True, {}, 0, 0)
    assert len(items) == 1 and not truncated
    monkeypatch.setattr(fetchers, "MAX_FOLDER_DIRS", 1)
    _, _, truncated = fetchers.scan_folder(str(root), True, {}, 0, 0)
    assert truncated
    monkeypatch.undo()
    if os.name == "nt":
        import subprocess

        link = root / "loop"
        if subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(root)], capture_output=True).returncode == 0:
            items, _, truncated = fetchers.scan_folder(str(root), True, {}, 0, 0)
            assert len(items) == 1 and not truncated


# --- periodic checks (checklist item 14) ------------------------------------------------------

def test_folder_new_file_is_rechecked_soon_after_first_sighting(wsettings, queue, clock, tmp_path):
    """First sighting is never stable, so waiting a whole interval (1 min) + the next one made a dropped file look
    unseen by the periodic check; the second look comes as soon as the file may be stable."""
    settings = dataclasses.replace(wsettings, extra={**wsettings.extra, "watch_stable_s": 10, "watch_min_folder_s": 10})
    manager = WatchManager(settings, queue, clock=clock)
    watched = tmp_path / "in"
    watched.mkdir()
    src = manager.create({"kind": "folder", "target": str(watched), "interval_s": 60})
    manager.tick()
    assert src.baselined and src.next_check == clock.now + 60
    clock.now += 60
    make_file(watched, "new.mp3", mtime=clock.now - 20)
    manager.tick()  # first sighting
    assert queue.sources == []
    assert src.next_check <= clock.now + 11  # not another 60 s
    clock.now += 11
    assert manager.tick() == 1
    assert len(queue.sources) == 1
    assert src.next_check == clock.now + 60  # nothing waiting any more: back to the interval


def test_periodic_loop_follows_the_watch_enabled_setting_at_runtime(wsettings, tmp_path):
    settings = dataclasses.replace(wsettings, extra={**wsettings.extra, "watch_enabled": False})
    watched = tmp_path / "in"
    watched.mkdir()
    app = create_app(settings, runner=InProcessRunner())
    with TestClient(app) as client:
        sid = client.post("/api/watch", json={"kind": "folder", "target": str(watched), "interval_s": 1}).json()["id"]
        time.sleep(1.5)
        assert not client.get(f"/api/watch/{sid}").json()["baselined"]  # switched off: no periodic checks
        assert client.put("/api/settings", json={"values": {"watch_enabled": True}}).status_code == 200
        wait_for(lambda: client.get(f"/api/watch/{sid}").json()["baselined"])
        assert client.put("/api/settings", json={"values": {"watch_enabled": False}}).status_code == 200
        assert client.put("/api/settings", json={"values": {"watch_enabled": True}}).status_code == 200
        make_file(watched, "clip.wav")
        wait_for(lambda: len(client.get(f"/api/watch/{sid}/history").json()["items"]) == 1, timeout=30)


def test_history_translation_state_of_jobs_gone_from_the_queue(wsettings, tmp_path, monkeypatch):
    """After "clear finished" the translate job is gone: only a stored translation proves it succeeded."""
    from mgtranscriber.server.routers import watch as watch_router
    monkeypatch.setattr(watch_router, "load_translation",
                        lambda library, tid, lang: {"segments": []} if tid == "has-cs" else None)
    watched = tmp_path / "in"
    watched.mkdir()
    app = create_app(wsettings, runner=InProcessRunner())
    with TestClient(app) as client:
        sid = client.post("/api/watch", json={"kind": "folder", "target": str(watched), "interval_s": 1}).json()["id"]
        src = app.state.watch.get(sid)
        tr = {"language": "cs", "job": 987654}
        src.items = [{"key": "a", "status": "done", "result": "has-cs", "translation": dict(tr)},
                     {"key": "b", "status": "done", "result": "lost-cs", "translation": dict(tr)},
                     {"key": "c", "status": "done", "result": "x", "translation": {"language": "cs", "error": "boom"}}]
        items = client.get(f"/api/watch/{sid}/history").json()["items"]
        assert [i["translation"].get("status") for i in items] == ["done", "unknown", None]
        assert src.items[0]["translation"] == tr  # stored state is not changed by reading


def test_start_while_the_loop_is_still_stopping_keeps_it_running(manager):
    """stop() joins for 5 s only; a check that runs longer leaves the thread alive with the stop flag set, and a
    start() right after must not leave the watcher dying silently."""
    manager._thread = SimpleNamespace(is_alive=lambda: True)
    manager._stop.set()
    manager.start()
    assert not manager._stop.is_set()


@pytest.mark.parametrize("env, running", [("false", False), ("true", True)])
def test_startup_follows_the_watch_enabled_environment_lock(wsettings, monkeypatch, env, running):
    from mgtranscriber import settings_store
    # the file says the opposite, so only the environment can decide
    settings = dataclasses.replace(wsettings, extra={**wsettings.extra, "watch_enabled": not running})
    monkeypatch.setenv("MGTRANSCRIBER_WATCH_ENABLED", env)
    assert settings_store.effective(settings, "watch_enabled") is running
    app = create_app(settings, runner=InProcessRunner())
    with TestClient(app):
        thread = app.state.watch._thread
        assert (thread is not None and thread.is_alive()) is running


def test_a_file_that_never_becomes_stable_does_not_keep_the_short_cadence(wsettings, queue, clock, tmp_path):
    settings = dataclasses.replace(wsettings, extra={**wsettings.extra, "watch_stable_s": 10, "watch_min_folder_s": 10})
    manager = WatchManager(settings, queue, clock=clock)
    watched = tmp_path / "in"
    watched.mkdir()
    src = manager.create({"kind": "folder", "target": str(watched), "interval_s": 60})
    manager.tick()
    clock.now += 60
    make_file(watched, "empty.mp3", b"", mtime=clock.now - 20)  # size 0: never stable
    manager.tick()  # first sighting: one early look
    assert src.next_check <= clock.now + 11
    clock.now += 11
    manager.tick()
    assert queue.sources == [] and src.next_check == clock.now + 60  # not 11 s forever
