"""QA of the home page's "now running" strip and the Jobs page (scenario ``jobs`` of tools.qa_ui).

The server runs with a runner that never finishes a job by itself (it reports 42 % until cancelled), a history of
47 finished, 3 failed and 1 cancelled job is seeded into the throw-away data folder, and the flow drives the real UI.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

SEEDED_DONE = 47
PAGE_SIZE = 20  # static/js/pages/jobs.js

TRACEBACK = (
    "HTTPError: HTTP Error 403: Forbidden\n"
    "Traceback (most recent call last):\n"
    '  File "mgtranscriber/sources.py", line 120, in download\n'
    "    raise HTTPError(url, 403, 'Forbidden', None, None)\n"
    "urllib.error.HTTPError: HTTP Error 403: Forbidden"
)


class HoldExecution:
    """A job that stays at 42 % until it is cancelled."""

    def __init__(self, on_event):
        self.stop = threading.Event()
        on_event({"event": "progress", "phase": "transcribe", "percent": 42.0, "message": "Transcribing (held by QA)",
                  "data": {}})

    def wait(self) -> int:
        self.stop.wait()
        return 1

    def cancel(self) -> None:
        self.stop.set()


class HoldRunner:
    def start(self, request, settings, on_event):
        return HoldExecution(on_event)


def seed(settings, transcript_ids: list[str], clip: Path) -> None:
    """Write a job history to ``settings.jobs_file`` (the app loads it at start)."""
    from mgtranscriber.jobs import JobQueue
    from mgtranscriber.pipeline import TranscriptionRequest

    queue = JobQueue(settings, autostart=False)
    now = time.time()

    def finish(job, status, **fields):
        queue.pending.remove(job.id)
        job.status = job.phase = status
        job.done.set()
        for key, value in fields.items():
            setattr(job, key, value)

    for i in range(SEEDED_DONE):
        job = queue.add(TranscriptionRequest(source=f"https://example.org/demo/video-{i}", engine="whisperx",
                                             model="large-v3-turbo", language="cs"))
        started = now - 3600 * (i + 2)
        job.add_log(f"> https://example.org/demo/video-{i}")
        job.add_log("[transcribe] done")
        finish(job, "done", title=f"Demo video {i + 1}", result=transcript_ids[i % len(transcript_ids)],
               created=started - 5, started=started, finished=started + 240)
    for i in range(3):
        job = queue.add(TranscriptionRequest(source=str(clip), engine="youtube", language="en"))
        job.add_log(f"> {clip}")
        job.add_log("Error: HTTP Error 403: Forbidden")
        finish(job, "error", title=f"Failed demo {i + 1}", error=TRACEBACK if i == 0 else "YouTube refused the download.",
               created=now - 1800 * (i + 1), started=now - 1800 * (i + 1), finished=now - 1700 * (i + 1))
    job = queue.add(TranscriptionRequest(source=str(clip)))
    finish(job, "cancelled", title="Cancelled demo", created=now - 9000, started=now - 8900, finished=now - 8800)
    queue._save()


def page_actions():
    def click(selector):
        return lambda page: (page.click(selector), page.wait_for_timeout(400))

    return {"jobs": ("#/jobs", None), "jobs-failed": ("#/jobs?tab=failed", None),
            "jobs-finished": ("#/jobs?tab=done", None), "jobs-open": ("#/jobs?tab=failed", click(".jr-toggle"))}


def jobs_flow(browser, base: str, work: Path, viewports: dict) -> list[str]:
    from tools.qa_ui import make_media

    out: list[str] = []
    clip = make_media(work, "held.wav")

    def check(ok, message):
        if not ok:
            out.append(message)

    ctx = browser.new_context(bypass_csp=True, viewport=viewports["desktop"], locale="en-GB")
    ctx.add_init_script("try{localStorage.setItem('lang','en');localStorage.setItem('setup.skip-speakers','0')}catch(e){}")
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"exception: {e}"))
    page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))

    def api(path, method="GET"):
        return page.evaluate("([p, m]) => fetch(p, {method: m}).then(r => r.json())", [path, method])

    def tab_rows():
        return page.locator("#j-list > li").count()

    # 1. nothing runs: no strip, no job list, no badge; the transcripts are there
    page.goto(base + "/#/")
    page.wait_for_selector(".tile, .library .card, #library", timeout=10000)
    page.wait_for_timeout(800)
    check(page.locator("#running").is_hidden(), "the strip is visible although nothing runs or waits")
    check(page.locator("#jobs").count() == 0 and page.locator(".job").count() == 0, "the home page still lists jobs")
    check(page.locator(".nav-badge").is_hidden(), "the nav badge is visible with no active job")

    # 2. start a job from the form: toast + strip, still on the home page
    page.fill("#url", str(clip))
    page.click("#start-btn")
    page.wait_for_selector("#toast.show", timeout=5000)
    check("started" in page.inner_text("#toast").lower(), f"toast after starting: {page.inner_text('#toast')!r}")
    page.wait_for_function("document.querySelector('#running') && !document.querySelector('#running').hidden", timeout=8000)
    page.wait_for_function("document.querySelector('.rs-phase')?.textContent.includes('42')", timeout=8000)
    check(page.evaluate("location.hash") in ("", "#/"), f"submitting moved the page to {page.evaluate('location.hash')}")
    check("held" in page.inner_text(".rs-title").lower(), f"strip title: {page.inner_text('.rs-title')!r}")
    check(page.locator(".rs-cancel").is_visible() and page.locator(".rs-log").is_visible(), "strip lacks Cancel or Log")
    for _ in range(2):
        page.evaluate("([s]) => fetch('/api/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'},"
                      " body: JSON.stringify({source: s, engine: 'whisperx'})})", [str(clip)])
    page.wait_for_function("document.querySelector('.rs-more')?.textContent.includes('+2 queued')", timeout=8000)
    more = page.inner_text(".rs-more")
    check("3 failed recently" in more, f"strip link should mention the 3 recent failures: {more!r}")
    check(page.inner_text(".nav-badge").strip().startswith("3"), f"badge: {page.inner_text('.nav-badge')!r}")

    # 3. the log link opens the job on the Jobs page, expanded
    page.click(".rs-log")
    page.wait_for_selector("#j-list > li", timeout=8000)
    page.wait_for_function("document.querySelector('.jr-toggle[aria-expanded=true]')", timeout=5000)
    page.wait_for_function("document.querySelector('.job-row.open .log')?.textContent.includes('held.wav')", timeout=8000)

    # 4. tabs: roles, arrows, active order, paging
    check(page.get_attribute(".j-tabs", "role") == "tablist", "tabs lack role=tablist")
    check(page.get_attribute("#jt-active", "aria-selected") == "true", "Active should be the open tab")
    check(tab_rows() == 3, f"active rows: {tab_rows()}")
    pills = page.eval_on_selector_all("#j-list .pill", "els => els.map(e => e.textContent)")
    check(pills == ["Running", "Queued · 1.", "Queued · 2."], f"active order: {pills}")
    page.focus("#jt-active")
    page.keyboard.press("ArrowRight")
    page.wait_for_function("document.querySelector('#jt-done').getAttribute('aria-selected') === 'true'")
    check(page.evaluate("document.activeElement.id") == "jt-done", "arrow key did not move the focus to the next tab")
    page.wait_for_function(f"document.querySelectorAll('#j-list > li').length === {PAGE_SIZE}")
    check("1–20 of 47" in page.inner_text("#j-range"), f"range: {page.inner_text('#j-range')!r}")
    check(page.is_disabled("#j-prev"), "Newer should be disabled on the first page")
    page.click("#j-next")
    page.wait_for_function("document.querySelector('#j-range').textContent.includes('21–40')")
    page.click("#j-next")
    page.wait_for_function("document.querySelector('#j-range').textContent.includes('41–47')")
    check(page.is_disabled("#j-next") and tab_rows() == 7, "last page should hold 7 rows with Older disabled")
    page.focus("#jt-done")
    page.keyboard.press("ArrowRight")  # Failed / cancelled
    page.wait_for_function("document.querySelector('#jt-failed').getAttribute('aria-selected') === 'true'")
    page.wait_for_function("document.querySelectorAll('#j-list > li').length === 4")
    check(page.locator("#j-pager").is_hidden(), "pager shown for 4 jobs")

    # 5. failed rows: readable headline, traceback behind details, Retry, Remove
    first = page.locator("#j-list > li", has_text="Failed demo 1")
    check("HTTP Error 403" in first.locator(".je-text").inner_text(), "headline of the failed job is missing")
    check(first.locator(".je-details pre").is_hidden(), "the traceback is visible before opening details")
    first.locator(".je-details summary").click()
    check("Traceback" in first.locator(".je-details pre").inner_text(), "details do not show the traceback")
    check(page.locator("#j-list > li", has_text="Failed demo 2").locator(".je-details").is_hidden(),
          "a one-line error should have no details")
    transcripts = len(api("/api/transcripts"))
    page.locator("#j-list > li", has_text="Failed demo 3").locator(".j-retry").click()
    page.wait_for_function("document.querySelectorAll('#j-list > li').length === 3")
    check(api("/api/jobs/summary")["queued"] == 3, "retry did not queue the job again")
    page.locator("#j-list > li", has_text="Cancelled demo").locator(".j-remove").click()
    page.wait_for_function("document.querySelectorAll('#j-list > li').length === 2")
    check(len(api("/api/transcripts")) == transcripts, "removing a job from the history touched the transcripts")

    # 6. clear finished: cancel keeps everything, confirm empties Finished and the failed list
    page.click("#jt-done")
    page.wait_for_function(f"document.querySelectorAll('#j-list > li').length === {PAGE_SIZE}")
    page.click("#j-clear")
    page.wait_for_selector(".fx-dialog[open]")
    page.click(".fx-dialog .fx-cancel")
    check(api("/api/jobs?status=finished&limit=1")["total"] == SEEDED_DONE + 2, "cancelling the dialog removed jobs")
    page.click("#j-clear")
    page.wait_for_selector(".fx-dialog[open]")
    page.click(".fx-dialog .fx-ok")
    page.wait_for_function("!document.querySelector('#j-empty').hidden", timeout=8000)
    check(api("/api/jobs?status=finished&limit=1")["total"] == 0, "Clear finished left jobs behind")
    check(len(api("/api/transcripts")) == transcripts, "clearing the history touched the transcripts")

    # 7. cancel the running job from the strip on the home page
    page.goto(base + "/#/")
    page.wait_for_selector(".rs-cancel", state="visible", timeout=8000)
    running = api("/api/jobs/summary")["running"]["id"]
    page.click(".rs-cancel")
    try:
        page.wait_for_function("(id) => fetch('/api/jobs/' + id).then(r => r.json()).then(j => j.status === 'cancelled')",
                               arg=running, timeout=8000)
    except Exception:
        out.append("Cancel in the strip did not cancel the running job")
    page.wait_for_timeout(500)

    # 8. mobile width: nothing overflows
    for hash_ in ("#/", "#/jobs", "#/jobs?tab=failed"):
        mobile = browser.new_context(bypass_csp=True, viewport=viewports["mobile"], locale="en-GB")
        m = mobile.new_page()
        m.goto(base + "/" + hash_)
        m.wait_for_load_state("networkidle")
        m.wait_for_timeout(600)
        over = m.evaluate("() => { const d = document.documentElement; const bad = [];"
                          "for (const el of document.querySelectorAll('body *')) { const r = el.getBoundingClientRect();"
                          "if (r.width && r.right > d.clientWidth + 1) bad.push(el.tagName + '.' + el.className + ' ' + Math.round(r.right));}"
                          "return { page: d.scrollWidth - d.clientWidth, elements: bad.slice(0, 6) }; }")
        check(over["page"] <= 1, f"{hash_} overflows at mobile width: {over}")
        mobile.close()

    out += [f"page: {e}" for e in errors]
    ctx.close()
    return out
