"""The Watching form in a real browser: a source of every kind can be saved with the default interval."""

import socket
import threading
import time

import pytest
import uvicorn

from mgtranscriber.jobs import InProcessRunner
from mgtranscriber.server import create_app

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def server(settings, tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    settings = settings.with_overrides(extra={**settings.extra, "watch_enabled": False})
    app = create_app(settings, runner=InProcessRunner())
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(10)


@pytest.fixture
def page(server):
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="chrome", headless=True)
        except Exception as err:  # no system Chrome on this machine
            pytest.skip(f"Chrome not available: {err}")
        pg = browser.new_context(bypass_csp=True).new_page()  # the test driver evaluates strings; CSP is tested in test_security_hardening
        yield pg
        browser.close()


@pytest.mark.parametrize("kind", ["folder", "rss", "youtube"])
def test_default_and_whole_minutes_are_valid_for_every_kind(page, server, kind):
    page.goto(f"{server}/#/watch")
    page.wait_for_selector(".w-add")
    page.click(".w-add")
    page.select_option("select[name=kind]", kind)
    valid = "e => e.validity.valid"
    assert page.eval_on_selector("input[name=interval]", valid)  # the default of the kind
    # the folder minimum is 10 s = 0.17 min: whole minutes must not be rejected as a step mismatch
    for value in ("30", "31", "90"):
        page.fill("input[name=interval]", value)
        assert page.eval_on_selector("input[name=interval]", valid), (kind, value)
    page.fill("input[name=interval]", "1")  # below the minimum of a feed or a channel, fine for a folder
    assert page.eval_on_selector("input[name=interval]", valid) == (kind == "folder")


def test_folder_can_be_saved_with_default_interval(page, server, tmp_path):
    folder = tmp_path / "watched"
    folder.mkdir()
    page.goto(f"{server}/#/watch")
    page.wait_for_selector(".w-add")
    page.click(".w-add")
    page.select_option("select[name=kind]", "folder")
    page.fill("input[name=target]", str(folder))
    page.click("button[type=submit]")
    page.wait_for_selector(".w-add:visible", timeout=10000)
    sources = page.evaluate("fetch('/api/watch').then(r => r.json())")["sources"]
    assert len(sources) == 1 and sources[0]["kind"] == "folder" and sources[0]["interval_s"] == 60
