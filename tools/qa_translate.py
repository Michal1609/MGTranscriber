"""QA of translations on the detail page (scenario ``translate`` of tools.qa_ui).

The LLM is a scripted fake (``EN: <text>``), the job runs in-process, and the flow drives the real UI: translate from the
form, the progress, the language switcher, the read-only view (no Edit button, time links, speaker colours), exports with
``?lang=``, the stale banner after the original changed, "Translate again", the search hit tag and deleting.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

LINE = re.compile(r"^(\d+) \[(.*?)\] (.*)$")


def fake_translate(user: str) -> dict:
    block = user.split("Translate these", 1)[1].split("\n\n")[0].splitlines()[1:]
    time.sleep(0.25)  # a visible progress state
    items = [(int(m.group(1)), m.group(3)) for m in map(LINE.match, block) if m]
    return {"items": [{"id": i, "text": "EN: " + text} for i, text in items]}


def install() -> None:
    """Every provider the app builds (API check and the job's worker code) is the fake."""
    import mgtranscriber.llm as llm
    from mgtranscriber.llm.service import AnalysisManager
    from tests.llm_fakes import FakeProvider

    provider = FakeProvider(translate=fake_translate, max_context=100_000)
    provider.local = True
    llm.make_provider = lambda *a, **k: provider
    AnalysisManager.provider = lambda self, p=None, m=None: provider


def click_form(page):
    page.click(".lang-open")
    page.wait_for_timeout(500)


def translate_pages(ids: dict) -> dict:
    base = f"#/t/{ids['transcripts'][0]}"
    return {"home": ("#/", None), "detail": (base, None), "detail-translate-form": (base, click_form)}


def translate_flow(browser, base: str, work: Path, viewports: dict) -> list[str]:
    out: list[str] = []

    def check(ok, message):
        if not ok:
            out.append(message)

    ctx = browser.new_context(bypass_csp=True, viewport=viewports["desktop"], locale="en-GB")
    ctx.add_init_script("try{localStorage.setItem('lang','en');localStorage.setItem('setup.skip-speakers','0')}catch(e){}")
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"exception: {e}"))
    page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))
    page.on("dialog", lambda d: d.accept())

    def api(path, method="GET"):
        return page.evaluate("([p, m]) => fetch(p, {method: m}).then(r => r.json())", [path, method])

    page.goto(base + "/")
    tid = api("/api/transcripts")[0]["id"]
    detail = f"{base}/#/t/{tid}"
    page.goto(detail)
    page.wait_for_selector(".d-blocks .seg", timeout=10000)
    original_text = page.inner_text(".d-blocks .seg")
    original_colors = page.evaluate("() => [...document.querySelectorAll('.block')].slice(0, 6).map(b => b.style.getPropertyValue('--c'))")
    check(page.locator(".lang-switch").is_hidden(), "the switcher is visible before there is a translation")
    check(page.locator(".d-edit").is_visible(), "no Edit button on the original")

    # 1. open the form and start a translation into English
    page.click(".lang-open")
    page.wait_for_selector(".lf-lang option", state="attached", timeout=5000)
    check(page.get_attribute(".lang-open", "aria-expanded") == "true", "aria-expanded is not true with the form open")
    values = page.eval_on_selector_all(".lf-lang option", "els => els.map(e => e.value)")
    check("en" in values and "cs" not in values, f"target languages should exclude the source language: {values[:6]}")
    page.select_option(".lf-lang", "en")
    page.click(".lf-start")
    page.wait_for_selector(".lang-job", timeout=8000)
    check(bool(re.search(r"Translat|waiting", page.inner_text(".lang-status"), re.I)), f"status: {page.inner_text('.lang-status')!r}")

    # 2. done: the page switches to the translation
    page.wait_for_function("document.querySelector('.lang-btn.active')?.dataset.lang === 'en'", timeout=60000)
    page.wait_for_function("document.querySelector('.d-blocks .seg')?.textContent.startsWith('EN: ')", timeout=10000)
    check(page.evaluate("document.querySelector('#view')?.classList.contains('translated-view') || document.querySelector('.translated-view') !== null"),
          "the page is not marked as a translated view")
    check(page.locator(".d-edit").is_hidden(), "the Edit button is still shown in the translated view")
    check("read-only" in page.inner_text(".lang-banner").lower(), f"banner: {page.inner_text('.lang-banner')!r}")
    links = page.eval_on_selector_all(".d-exports a", "els => els.map(a => a.getAttribute('href'))")
    check(links and all("lang=en" in h for h in links), f"export links should carry ?lang=en: {links}")
    colors = page.evaluate("() => [...document.querySelectorAll('.block')].slice(0, 6).map(b => b.style.getPropertyValue('--c'))")
    check(colors == original_colors, f"speaker colours differ in the translation: {colors} vs {original_colors}")
    check(page.evaluate("location.hash").endswith("lang=en"), f"URL does not keep the language: {page.evaluate('location.hash')}")
    if page.locator(".d-player").is_visible():  # time links and clicking a segment still seek
        times = page.evaluate("() => [...document.querySelectorAll('.d-blocks .seg')].map(s => s.dataset.i)")
        page.locator(".d-blocks .seg").nth(2).click()
        page.wait_for_timeout(300)
        seek = page.evaluate("document.querySelector('.d-audio').currentTime")
        data = api(f"/api/transcripts/{tid}/translations/en")
        check(abs(seek - data["segments"][2]["start"]) < 1.0 or not times,
              f"clicking a translated segment sought to {seek}, expected {data['segments'][2]['start']}")
        page.evaluate("document.querySelector('.d-audio').pause()")
    # 3. export and search in the translation
    srt = page.evaluate(f"fetch('/api/transcripts/{tid}/export/srt?lang=en').then(r => r.text())")
    check("EN: " in srt and "-->" in srt, "SRT in the translated language is wrong")
    hits = api(f"/api/search?q=EN&transcript={tid}")
    check(any(h.get("lang") == "en" for h in hits["results"]), "search has no hit tagged with the translation's language")

    # 4. back to the original
    page.click('.lang-btn[data-lang=""]')
    page.wait_for_function("!document.querySelector('.d-blocks .seg')?.textContent.startsWith('EN: ')", timeout=5000)
    check(page.inner_text(".d-blocks .seg") == original_text, "the original text is not back")
    check(page.locator(".d-edit").is_visible(), "no Edit button after returning to the original")

    # 5. the original changes: the translation is outdated, with a banner and a mark
    prepis = Path(work) / "root" / "output" / tid / "prepis.json"
    segs = json.loads(prepis.read_text(encoding="utf-8"))
    segs[0]["text"] = segs[0]["text"] + " (upraveno)"
    prepis.write_text(json.dumps(segs, ensure_ascii=False), encoding="utf-8")
    page.goto(detail + "?lang=en")
    page.reload()  # the same hash: a plain reload must show the language again
    page.wait_for_selector(".lang-banner:not([hidden]) .lang-stale", timeout=10000)
    check("outdated" in page.inner_text(".lang-switch").lower(), "the switcher does not mark the outdated translation")
    # 6. translate again clears it
    page.click("[data-again]")
    page.wait_for_function("!document.querySelector('.lang-banner .lang-stale') && !document.querySelector('.lang-job')", timeout=60000)
    check(page.locator(".lang-banner .lang-stale").count() == 0, "still outdated after Translate again")

    # 7. mobile width: no horizontal overflow with the switcher and the banner
    page.set_viewport_size(viewports["mobile"])
    page.wait_for_timeout(300)
    over = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    check(over <= 1, f"horizontal overflow on mobile: {over}px")
    page.set_viewport_size(viewports["desktop"])

    # 8. delete
    page.click("[data-delete]")
    page.wait_for_function("document.querySelector('.lang-switch').hidden", timeout=5000)
    check(api(f"/api/transcripts/{tid}/translations")["translations"] == [], "the translation was not deleted")
    out += errors
    ctx.close()
    return out
