"""QA flow of the recorder (``tools.qa_ui --scenario record``): Chrome's fake microphone records ~10 s with a pause,
the recording becomes a job with audio, a recording interrupted by a reload is recovered / discarded, and the
panel explains an insecure context. The browser must be started with the fake media flags (qa_ui does)."""

from __future__ import annotations

CHROME_ARGS = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]

STATE = "window.__recorderState()"


def _new_page(browser, base: str, viewport: dict, init: str = ""):
    ctx = browser.new_context(bypass_csp=True, viewport=viewport, locale="en-GB", permissions=["microphone"])
    ctx.add_init_script("try{localStorage.setItem('lang','en');localStorage.setItem('setup.skip-speakers','0')}catch(e){}" + init)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"exception: {e}"))
    page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))
    page.on("dialog", lambda d: d.accept())  # beforeunload while recording
    page.goto(base + "/#/")
    page.wait_for_selector("#rec-btn")
    return ctx, page, errors


def _phase(page) -> str:
    return page.evaluate(f"{STATE}.phase")


def _wait_phase(page, phase: str, timeout: int = 20000):
    page.wait_for_function(f"{STATE}.phase === {phase!r}", timeout=timeout)


def record_flow(browser, base: str, viewports: dict) -> list[str]:
    out: list[str] = []
    ctx, page, errors = _new_page(browser, base, viewports["desktop"])
    page.click("#rec-btn")
    page.wait_for_selector("#rec-start")
    if "consent" not in page.inner_text("#recorder").lower():
        out.append("the consent notice is missing")
    if page.get_attribute("#rec-btn", "aria-expanded") != "true":
        out.append("the record button does not report aria-expanded")

    # --- record, pause, resume, stop ------------------------------------------------------------------------
    page.click("#rec-start")
    _wait_phase(page, "recording")
    page.wait_for_function("document.title.startsWith('●')", timeout=3000)
    if not page.title().startswith("●"):
        out.append(f"the tab title shows no recording indicator: {page.title()!r}")
    page.wait_for_timeout(5500)
    page.wait_for_function(f"{STATE}.acked >= 1", timeout=15000)  # chunks reach the server DURING the recording
    sessions = page.evaluate("fetch('/api/recordings').then(r => r.json())")
    if len(sessions) != 1 or sessions[0]["chunks"] < 1:
        out.append(f"no chunk on the server while recording: {sessions}")
    page.click("#rec-pause")
    _wait_phase(page, "paused")
    frozen = page.inner_text("#rec-time")
    page.wait_for_timeout(1500)
    if page.inner_text("#rec-time") != frozen:
        out.append("the timer runs while paused")
    if page.locator("#rec-resume").count() != 1:
        out.append("no resume button while paused")
    page.click("#rec-resume")
    _wait_phase(page, "recording")
    page.wait_for_timeout(4500)
    live = page.inner_text("#rec-live")
    if not live:
        out.append("nothing was announced in the live region")
    meter = page.evaluate("getComputedStyle(document.querySelector('#rec-meter-bar')).width")
    if meter in ("0px", ""):
        out.append(f"the level meter did not move with the fake tone: {meter}")
    page.fill("#speakers", "2")
    page.click("#rec-stop")
    page.wait_for_function(f"{STATE}.phase === 'idle'", timeout=30000)
    jobs = page.evaluate("fetch('/api/jobs').then(r => r.json())")
    if not jobs or jobs[0].get("speakers") != 2:
        out.append(f"the form's options did not reach the job: {jobs[:1]}")
    found = []
    for _ in range(80):  # the job runs, then its end-of-job hook keeps the file as the transcript audio
        found = page.evaluate("fetch('/api/transcripts').then(r => r.json())")
        if found and found[0].get("audio"):
            break
        page.wait_for_timeout(500)
    audio = found[0].get("audio") if found else None
    if not isinstance(audio, str) or not audio.endswith(".webm"):
        out.append(f"the transcript should have webm audio, got {found[:1]!r}")
    else:
        size = page.evaluate("(u) => fetch(u).then(r => r.arrayBuffer()).then(b => b.byteLength)",
                             f"/api/transcripts/{found[0]['id']}/files/{audio}")
        if size < 5000:
            out.append(f"the recorded audio is too small: {size} bytes")
    if page.evaluate("fetch('/api/recordings').then(r => r.json())"):
        out.append("the session was not removed after finishing")
    if page.title().startswith("●"):
        out.append("the recording indicator stays in the title")

    # --- recovery after a reload mid-recording -------------------------------------------------------------
    for action in ("recover", "drop"):
        page.click("#rec-start")
        _wait_phase(page, "recording")
        page.wait_for_timeout(6500)
        page.wait_for_function(f"{STATE}.acked >= 1", timeout=15000)
        page.reload()
        page.wait_for_selector(".rec-recover-item", timeout=15000)
        text = page.inner_text(".rec-recover-item")
        if "Recording from" not in text:
            out.append(f"the recovery list item is unclear: {text!r}")
        if _phase(page) != "idle":
            out.append("the recorder should be idle after a reload")
        jobs_before = len(page.evaluate("fetch('/api/jobs').then(r => r.json())"))
        page.click(f'.rec-recover-item [data-act="{action}"]')
        page.wait_for_function("document.querySelectorAll('.rec-recover-item').length === 0", timeout=20000)
        jobs_after = len(page.evaluate("fetch('/api/jobs').then(r => r.json())"))
        if (action == "recover") != (jobs_after == jobs_before + 1):
            out.append(f"{action}: jobs went {jobs_before} -> {jobs_after}")
        if page.evaluate("fetch('/api/recordings').then(r => r.json())"):
            out.append(f"{action}: the session is still on the server")
        page.click("#rec-btn") if page.get_attribute("#rec-btn", "aria-expanded") != "true" else None
        page.wait_for_selector("#rec-start")

    # --- discard while recording ---------------------------------------------------------------------------
    page.click("#rec-start")
    _wait_phase(page, "recording")
    page.click("#rec-discard")
    page.click("#rec-discard-yes")
    _wait_phase(page, "idle")
    page.wait_for_timeout(300)
    if page.evaluate("fetch('/api/recordings').then(r => r.json())"):
        out.append("a discarded recording is still on the server")

    # --- keeps recording while the user visits another page --------------------------------------------------
    page.click("#rec-start")
    _wait_phase(page, "recording")
    page.evaluate("location.hash = '#/jobs'")
    page.wait_for_timeout(1500)
    if _phase(page) != "recording":
        out.append("navigating to another page stopped the recording")
    page.evaluate("location.hash = '#/'")
    page.wait_for_selector("#rec-stop")
    page.click("#rec-discard")
    page.click("#rec-discard-yes")
    _wait_phase(page, "idle")

    # --- layout on a phone ----------------------------------------------------------------------------------
    page.set_viewport_size(viewports["mobile"])
    page.wait_for_selector("#rec-start")
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    if overflow > 1:
        out.append(f"the recorder overflows a phone screen by {overflow}px")
    out += [f"page: {e}" for e in errors]
    ctx.close()

    # --- an insecure context is explained -------------------------------------------------------------------
    ctx, page, errors = _new_page(browser, base, viewports["desktop"],
                                  "Object.defineProperty(window,'isSecureContext',{value:false});")
    page.click("#rec-btn")
    page.wait_for_selector("#recorder .rec-error")
    if "HTTPS" not in page.inner_text("#recorder") or page.locator("#rec-start").count() != 0:
        out.append("an insecure context is not explained / still offers recording")
    ctx.close()
    return out
