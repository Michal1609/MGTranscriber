"""QA flow of the "restart the server" banner (``tools.qa_ui --scenario empty``): a simulated old server (an /api/info
without ``build``), a changed build (``restart_needed``) and an endpoint the server does not know (422) show the
banner; a healthy server does not."""

from __future__ import annotations

import json


def _page(browser, base: str):
    ctx = browser.new_context(bypass_csp=True, viewport={"width": 1000, "height": 800}, locale="en-GB")
    ctx.add_init_script("try{localStorage.setItem('lang','en')}catch(e){}")
    return ctx, ctx.new_page()


def _patched_info(page, change):
    def handler(route):
        body = route.fetch().json()
        change(body)
        route.fulfill(json=body)

    page.route("**/api/info", handler)


def stale_flow(browser, base: str) -> list[str]:
    out: list[str] = []

    ctx, page = _page(browser, base)  # a healthy server: no banner
    page.goto(base + "/#/")
    page.wait_for_selector("#new-job")
    page.wait_for_timeout(800)
    if page.locator("#stale-banner").count():
        out.append("the banner shows although the server is current")
    ctx.close()

    cases = {
        "an old server without build": lambda b: b.pop("build", None) and b.pop("restart_needed", None),
        "restart_needed": lambda b: b.update(restart_needed=True),
    }
    for name, change in cases.items():
        ctx, page = _page(browser, base)
        _patched_info(page, change)
        page.goto(base + "/#/")
        try:
            page.wait_for_selector("#stale-banner", timeout=8000)
        except Exception:
            out.append(f"{name}: no banner")
            ctx.close()
            continue
        text = page.inner_text("#stale-banner")
        if "web.bat" not in text or page.locator("#stale-banner[role=alert]").count() != 1:
            out.append(f"{name}: the banner is unclear: {text!r}")
        page.click("#stale-hide")
        if page.locator("#stale-banner").count():
            out.append(f"{name}: the banner cannot be hidden")
        ctx.close()

    ctx, page = _page(browser, base)  # an endpoint the old server does not have: FastAPI's 422 without a code
    page.route("**/api/jobs/summary*", lambda r: r.fulfill(
        status=422, content_type="application/json",
        body=json.dumps({"detail": [{"loc": ["path", "job_id"], "msg": "value is not a valid integer", "type": "int_parsing"}]})))
    page.goto(base + "/#/jobs")
    try:
        page.wait_for_selector("#stale-banner", timeout=8000)
    except Exception:
        out.append("a 422 on /api/jobs/summary did not show the banner")
    ctx.close()
    return out
