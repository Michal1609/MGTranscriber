"""The Content-Security-Policy of the web UI, enforced by a real browser: no page may trigger a violation."""

import pytest

from .test_ui_watch import server, sync_api  # noqa: F401  (fixtures)

PAGES = ["#/", "#/jobs", "#/system", "#/watch", "#/voices", "#/t/Old%20video"]

# init scripts run regardless of the page's CSP; every violation is reported to the console
WATCHER = "document.addEventListener('securitypolicyviolation', e => console.error('CSPVIOLATION ' + e.violatedDirective + ' ' + e.blockedURI));"


def test_pages_load_without_csp_violations(server, library, legacy_transcript):  # noqa: F811
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="chrome", headless=True)
        except Exception as err:  # no system Chrome on this machine
            pytest.skip(f"Chrome not available: {err}")
        page = browser.new_context().new_page()  # CSP enforced (no bypass)
        page.add_init_script(WATCHER)
        problems = []
        page.on("console", lambda m: problems.append(m.text) if "CSPVIOLATION" in m.text or "Content Security Policy" in m.text else None)
        for route in PAGES:
            page.goto(f"{server}/{route}")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(500)  # late fetches, audio element, lazy parts
        browser.close()
    assert problems == []
