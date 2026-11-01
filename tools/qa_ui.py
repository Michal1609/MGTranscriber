"""Visual and accessibility QA of the web UI with Playwright and the system Chrome.

    uv run python -m tools.qa_ui OUT_DIR [--scenario demo|empty|firstrun] [--axe PATH/axe.min.js] [--shots]

Starts the app in-process on a free port against a throw-away project root (never your output/ or data/), walks
every page at desktop (1366 px) and mobile (390 px) width, in light and dark mode, in Czech and English, and
reports: console errors, failed requests, horizontal overflow, untranslated keys, axe-core violations (when
``--axe`` points at axe.min.js) and focusable elements without a visible focus indicator. ``--shots`` also
saves a screenshot per combination. Scenarios: ``demo`` (invented data from tools.make_demo_data), ``empty``
(no transcripts, voices or sources), ``firstrun`` (no HuggingFace token, no ffmpeg) and ``upload`` (mock engine,
local paths off, upload limit 2 MB: also drives the file upload in the "New transcript" form).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import uvicorn

VIEWPORTS = {"desktop": {"width": 1366, "height": 800}, "mobile": {"width": 390, "height": 844}}
KEY_RE = re.compile(r"\b(?:nav|home|jobs|library|folders|detail|voices|watch|system|setup|common|errors|llm|engines|auth|translate)\.[a-z_]+(?:\.[a-z_]+)*\b")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(scenario: str, work: Path):
    from mgtranscriber import diagnostics
    from mgtranscriber.config import Settings
    from mgtranscriber.server import create_app
    from tools import make_demo_data

    root = work / "root"
    out, data = root / "output", root / "data"
    root.mkdir(parents=True, exist_ok=True)
    ids = {}
    if scenario in ("demo", "jobs", "translate"):
        ids = make_demo_data.build(out, data)
    else:
        out.mkdir(parents=True, exist_ok=True)
        data.mkdir(parents=True, exist_ok=True)
    token = None if scenario == "firstrun" else "hf_demo_token_not_real_000000"
    extra = {}
    runner = None
    if scenario in ("upload", "record"):  # a fake engine: the upload becomes a real job without a GPU
        from mgtranscriber.engines import register_engine
        from mgtranscriber.jobs import InProcessRunner

        register_engine("mock", "tests.mock_engine:MockEngine")
        runner = InProcessRunner()
        extra = {"engine": "mock", "model": "", "language": "en", "allow_local_paths": "false", "max_upload_mb": 2}
    settings = Settings(root=root, output_dir=out, data_dir=data, hf_token=token, allowed_hosts="127.0.0.1,localhost",
                        api_key="demo-key" if scenario == "auth" else None, **extra)

    def hf(token, refresh=False):
        if not token:
            return diagnostics._check("hf", "models", "error", "no_token", "no token", model_url=diagnostics.HF_MODEL_URL,
                                      tokens_url=diagnostics.HF_TOKENS_URL)
        return diagnostics._check("hf", "models", "ok", "ok", "ok", user="demo")

    diagnostics.hf_status = hf
    if scenario == "firstrun":
        diagnostics.check_ffmpeg = lambda: diagnostics._check("ffmpeg", "tools", "error", "missing", "missing")
    if scenario == "translate":  # a scripted LLM, jobs run in-process (tools/qa_translate.py)
        from mgtranscriber.jobs import InProcessRunner
        from tools import qa_translate

        qa_translate.install()
        runner = InProcessRunner()
    if scenario == "jobs":  # a held job and a seeded history (tools/qa_jobs.py)
        from tools import qa_jobs

        runner = qa_jobs.HoldRunner()
        qa_jobs.seed(settings, ids["transcripts"], make_media(root))
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(settings, runner=runner), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    return server, f"http://127.0.0.1:{port}", ids


# Findings that are the point of a page, not a defect: (scenario, page) -> regexes matched against the problem text.
_NOTFOUND = [r"^HTTP 404: \S+/api/transcripts/does-not-exist$", r"^console: Failed to load resource: .*404"]
EXPECTED = {
    ("demo", "notfound"): _NOTFOUND,
    ("empty", "notfound"): _NOTFOUND,
    ("firstrun", "notfound"): _NOTFOUND,
    # auth: the page is viewed without the key, so every API call of the home page is refused
    ("auth", "home"): [r"^HTTP 401: \S+/api/(info|transcripts|folders|system|jobs/summary|recordings)$", r"^console: Failed to load resource: .*401"],
}


def click(selector: str):
    return lambda page: (page.click(selector), page.wait_for_timeout(500))


def pages_for(ids: dict) -> dict[str, tuple[str, object]]:
    """Page name -> (hash, optional action run after loading, for states that need a click)."""
    pages = {"home": ("#/", None), "voices": ("#/voices", None), "watch": ("#/watch", None), "system": ("#/system", None),
             "watch-form": ("#/watch", click(".w-add")), "notfound": ("#/t/does-not-exist", None)}
    if ids.get("folders"):  # the demo library has folders (tools/qa_folders.py)
        from tools.qa_folders import page_actions

        pages.update(page_actions())
    if ids.get("transcripts"):
        base = f"#/t/{ids['transcripts'][0]}"
        pages["detail"] = (base, None)
        pages["detail-summary"] = (base, click('[data-tab="analysis"]'))
        pages["detail-ask"] = (base, click('[data-tab="ask"]'))
        pages["detail-edit"] = (base + "?edit=1", None)
        pages["detail-privacy"] = (base, click(".d-privacy summary"))
    if ids.get("people"):
        pages["voice-person"] = (f"#/voices/{ids['people']['Petr Dvořák']}", None)
    return pages


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--scenario", default="demo", choices=["demo", "empty", "firstrun", "auth", "upload", "jobs", "record", "translate"])
    parser.add_argument("--axe")
    parser.add_argument("--shots", action="store_true")
    parser.add_argument("--only", help="comma separated page names")
    parser.add_argument("--workdir", help="use this existing folder instead of a temp dir (e.g. a subst drive, so screenshots show neutral paths)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    workdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True) if not args.workdir else contextlib.nullcontext(args.workdir)
    with workdir as tmp:
        server, base, ids = start_server(args.scenario, Path(tmp))
        pages = pages_for(ids)
        if args.scenario == "jobs":
            from tools.qa_jobs import page_actions as job_pages

            pages = {"home": pages["home"], **job_pages()}
        if args.scenario == "translate":
            from tools.qa_translate import translate_pages

            pages = translate_pages(ids)
        if args.scenario in ("auth", "upload", "record"):
            pages = {"home": pages["home"]}
            if args.scenario == "upload":  # the home page with files staged: layout, focus and axe checks see the list
                pages["home-files"] = ("#/", stage_files(Path(tmp)))
        if args.only:
            pages = {k: v for k, v in pages.items() if k in args.only.split(",")}
        findings = []
        parity_done = False
        with sync_playwright() as pw:
            from tools.qa_record import CHROME_ARGS  # fake microphone for the record scenario
            browser = pw.chromium.launch(channel="chrome", headless=True, args=CHROME_ARGS if args.scenario == "record" else [])
            for vp_name, vp in VIEWPORTS.items():
                for scheme in ("light", "dark"):
                    for lang in ("cs", "en"):
                        ctx = browser.new_context(bypass_csp=True, viewport=vp, color_scheme=scheme, locale="cs-CZ" if lang == "cs" else "en-GB")
                        ctx.add_init_script(f"try{{localStorage.setItem('lang','{lang}');localStorage.setItem('setup.skip-speakers','0')}}catch(e){{}}")
                        page = ctx.new_page()
                        problems: list[str] = []
                        page.on("console", lambda m, p=problems: m.type == "error" and p.append(f"console: {m.text}"))
                        page.on("pageerror", lambda e, p=problems: p.append(f"exception: {e}"))
                        page.on("requestfailed", lambda r, p=problems: "ERR_ABORTED" not in (r.failure or "") and p.append(f"request failed: {r.url} {r.failure}"))
                        page.on("response", lambda r, p=problems: r.status >= 400 and p.append(f"HTTP {r.status}: {r.url}"))
                        for name, (hash_, action) in pages.items():
                            tag = f"{args.scenario}/{name}/{vp_name}/{scheme}/{lang}"
                            problems.clear()
                            page.evaluate("() => document.querySelectorAll('dialog[open]').forEach(d => d.close())")  # a modal left open by the previous page action would block this page's clicks
                            page.goto(base + "/" + hash_)
                            page.wait_for_load_state("networkidle")
                            page.wait_for_timeout(700)
                            if action:
                                action(page)
                            expected = [re.compile(x) for x in EXPECTED.get((args.scenario, name), [])]
                            for p_ in problems:
                                if any(x.search(p_) for x in expected):
                                    continue
                                findings.append((tag, "network/console", p_))
                            overflow = page.evaluate(
                                "() => { const d = document.documentElement; const bad = [];"
                                "for (const el of document.querySelectorAll('body *')) { const r = el.getBoundingClientRect();"
                                "if (r.width && r.right > d.clientWidth + 1 && getComputedStyle(el).position !== 'fixed') bad.push(el.tagName + '.' + el.className + ' ' + Math.round(r.right));}"
                                "return { page: d.scrollWidth - d.clientWidth, elements: bad.slice(0, 5) }; }")
                            if overflow["page"] > 1 or overflow["elements"]:
                                findings.append((tag, "overflow", json.dumps(overflow)))
                            text = page.evaluate("() => document.body.innerText")
                            for m in sorted(set(KEY_RE.findall(text))):
                                findings.append((tag, "untranslated?", m))
                            for el in page.evaluate("() => [...document.querySelectorAll('[placeholder],[title],[aria-label]')].flatMap(e => ['placeholder','title','aria-label'].map(a => e.getAttribute(a)).filter(Boolean))"):
                                if KEY_RE.fullmatch(el):
                                    findings.append((tag, "untranslated attr", el))
                            if args.axe:
                                page.add_script_tag(path=args.axe)
                                res = page.evaluate("async () => (await axe.run(document, {resultTypes: ['violations']})).violations.map(v => ({id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.slice(0, 3).map(n => n.target.join(' ') + ' :: ' + (n.any[0]?.message || n.failureSummary || ''))}))")
                                for v in res:
                                    findings.append((tag, f"axe {v['impact']} {v['id']}", v["help"] + " | " + " || ".join(v["nodes"])))
                            findings += [(tag, "focus", f) for f in focus_problems(page)]
                            if not parity_done:
                                parity_done = True
                                findings += [(f"{args.scenario}/i18n", "dictionary", f) for f in i18n_parity(page)]
                            if args.shots:
                                shot = out / f"{args.scenario}-{name}-{vp_name}-{scheme}-{lang}.png"
                                page.screenshot(path=str(shot), full_page=True)
                        ctx.close()
            if args.scenario == "upload":
                findings += [("upload/flow", "upload", f) for f in upload_flow(browser, base, Path(tmp))]
            if args.scenario == "record":
                from tools.qa_record import record_flow

                findings += [("record/flow", "record", f) for f in record_flow(browser, base, VIEWPORTS)]
            if args.scenario == "demo":
                from tools.qa_folders import folder_flow

                findings += [("demo/folder-flow", "folders", f) for f in folder_flow(browser, base)]  # before voice_flow adds transcripts
                findings += [("demo/voice-flow", "voices", f) for f in voice_flow(browser, base, Path(tmp) / "root" / "output")]
            if args.scenario == "jobs":
                from tools.qa_jobs import jobs_flow

                findings += [("jobs/flow", "jobs", f) for f in jobs_flow(browser, base, Path(tmp), VIEWPORTS)]
            if args.scenario == "translate":
                from tools.qa_translate import translate_flow

                findings += [("translate/flow", "translate", f) for f in translate_flow(browser, base, Path(tmp), VIEWPORTS)]
            if args.scenario == "empty":
                from tools.qa_stale import stale_flow

                findings += [("empty/stale-flow", "stale-server", f) for f in stale_flow(browser, base)]
                findings += [("empty/voice-flow", "voices", f) for f in empty_voices_flow(browser, base)]
            browser.close()
        server.should_exit = True
    seen = set()
    lines = []
    for tag, kind, detail in findings:
        key = (tag.split("/", 1)[1].split("/")[0], kind, detail)  # same finding on every viewport/theme: report once
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"[{tag}] {kind}: {detail}")
    (out / f"report-{args.scenario}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines) or "no findings")
    return 1 if lines else 0


def make_media(work: Path, name: str = "clip.wav") -> Path:
    """A short audio file: the project's test clip when it can be made, else a second of silence."""
    import wave

    try:
        from tests.make_clip import ensure_clip

        clip = ensure_clip()
        target = work / name
        target.write_bytes(clip.read_bytes())
        return target
    except Exception:
        target = work / name
        with wave.open(str(target), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\0\0" * 16000)
        return target


def stage_files(work: Path):
    """Page action: choose two files with the button's file input, so the staged list is on screen."""
    first, second = make_media(work, "interview-one.wav"), make_media(work, "interview-two.wav")
    return lambda page: (page.set_input_files("#file-input", [str(first), str(second)]), page.wait_for_timeout(300))


def upload_flow(browser, base: str, work: Path) -> list[str]:
    """Drive the upload: keyboard + file chooser, refused files, drag & drop, remove, start, job, audio."""
    out: list[str] = []
    clip = make_media(work, "clip.wav")
    big = work / "too-big.wav"
    big.write_bytes(b"\0" * (3 * 1024 * 1024))  # the scenario's limit is 2 MB
    txt = work / "notes.txt"
    txt.write_text("not audio", encoding="utf-8")
    ctx = browser.new_context(bypass_csp=True, viewport=VIEWPORTS["desktop"], locale="en-GB")
    ctx.add_init_script("try{localStorage.setItem('lang','en')}catch(e){}")
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"exception: {e}"))
    page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))
    page.goto(base + "/#/")
    page.wait_for_selector("#pick-btn")
    page.wait_for_function("document.querySelector('#upload-help').textContent.includes('2 MB')")  # limit from /api/info

    page.focus("#pick-btn")  # the keyboard opens the file chooser
    with page.expect_file_chooser() as chooser:
        page.keyboard.press("Enter")
    chooser.value.set_files([str(clip), str(txt), str(big)])
    page.wait_for_timeout(200)
    names = page.eval_on_selector_all("#upload-list .upload-name", "els => els.map(e => e.textContent)")
    if names != ["clip.wav"]:
        out.append(f"staged files should be ['clip.wav'], got {names}")
    msg = page.inner_text("#form-msg")
    if "notes.txt" not in msg or "too-big.wav" not in msg:
        out.append(f"refused files are not explained: {msg!r}")

    drop = """(name) => { const dt = new DataTransfer(); dt.items.add(new File([new Uint8Array(64)], name, {type: 'audio/wav'}));
      const card = document.querySelector('#new-job'); const fire = (type) => card.dispatchEvent(new DragEvent(type, {dataTransfer: dt, bubbles: true, cancelable: true}));
      fire('dragenter'); fire('dragover'); const shown = card.classList.contains('dragging'); fire('drop');
      return { shown, after: card.classList.contains('dragging') }; }"""
    state = page.evaluate(drop, "dropped.wav")
    if not state["shown"] or state["after"]:
        out.append(f"drop zone highlight wrong: {state}")
    page.wait_for_timeout(200)
    if page.locator("#upload-list li").count() != 2:
        out.append("a dropped file was not added to the list")
    page.click('button[aria-label="Remove dropped.wav"]')
    if page.locator("#upload-list li").count() != 1:
        out.append("the remove button did not remove the file")
    if not page.eval_on_selector("#url", "e => !e.required"):
        out.append("the URL field is still required")
    page.fill("#url", "")

    page.click("#start-btn")
    page.wait_for_function("document.querySelectorAll('#upload-list li').length === 0", timeout=30000)
    try:  # the job runs, then its end-of-job hook keeps the upload as the transcript's audio
        page.wait_for_function("fetch('/api/transcripts').then(r => r.json()).then(l => l.some(t => t.audio))", timeout=30000)
    except Exception:
        out.append("no transcript with audio appeared")
    found = page.evaluate("fetch('/api/transcripts').then(r => r.json())")
    if found and found[0].get("audio"):
        audio = page.evaluate("(id) => fetch('/api/transcripts/' + id).then(r => r.json())", found[0]["id"])
        if audio["metadata"]["source"] != "clip.wav":
            out.append(f"the source should be the file name, got {audio['metadata']['source']!r}")
        status = page.evaluate("(a) => fetch(a).then(r => r.status)", f"/api/transcripts/{found[0]['id']}/files/{found[0]['audio']}")
        if status != 200:
            out.append(f"the player's audio answers {status}")
    if page.evaluate("fetch('/api/jobs/upload', {method: 'POST', body: new FormData()}).then(r => r.status)") != 422:
        out.append("an upload without a file should be 422")
    out += [f"page: {e}" for e in errors if "422" not in e]  # the 422 is the deliberate empty POST above
    ctx.close()
    return out


def empty_voices_flow(browser, base: str) -> list[str]:
    """Empty voice memory: the page explains the workflow and the scan says why nothing was found."""
    out: list[str] = []
    ctx = browser.new_context(bypass_csp=True, viewport=VIEWPORTS["desktop"], locale="en-GB")
    ctx.add_init_script("try{localStorage.setItem('lang','en')}catch(e){}")
    page = ctx.new_page()
    page.goto(base + "/#/voices")
    page.wait_for_selector(".v-empty")
    text = page.inner_text(".v-empty")
    if "Remember voice" not in text or page.locator(".v-empty li").count() != 4:
        out.append(f"the empty Voices page should list the four steps, got {text!r}")
    if "remember a voice first" not in page.inner_text(".v-job"):
        out.append(f"the scan hint for an empty memory is missing: {page.inner_text('.v-job')!r}")
    page.click(".v-scan")
    page.wait_for_timeout(500)
    if "remember a voice first" not in page.inner_text(".v-job"):
        out.append("after scanning an empty memory the page does not say why nothing was found")
    page.goto(base + "/#/")  # the empty library says a sentence, never a raw dictionary key
    page.wait_for_selector("#library .empty")
    if "Paste a link" not in page.inner_text("#library .empty"):
        out.append(f"the empty library shows {page.inner_text('#library .empty')!r}")
    ctx.close()
    return out


def voice_flow(browser, base: str, output: Path) -> list[str]:
    """Remembering a voice from the transcript page: the form for an unnamed speaker (Enter saves, Esc cancels), the
    prompt after a rename (Remember / Not now, not nagging), and one click for a speaker that already has a name."""
    import numpy as np

    from mgtranscriber import voices
    from mgtranscriber.library import Library
    from tests.test_voice_archive import make

    out: list[str] = []
    lib = Library(output)
    rng = np.random.default_rng(7)
    ids = [make(lib, f"qa-voice-{n}", {"SPEAKER_00": 40.0}, date="2026-01-0%d" % n,
                embeddings={"SPEAKER_00": rng.normal(size=voices.DIM)}) for n in (1, 2, 3)]
    ctx = browser.new_context(bypass_csp=True, viewport=VIEWPORTS["desktop"], locale="en-GB")
    ctx.add_init_script("try{localStorage.setItem('lang','en');localStorage.setItem('setup.skip-speakers','0')}catch(e){}")
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"exception: {e}"))
    page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))

    def open_transcript(tid):
        page.goto(f"{base}/#/t/{tid}")
        page.wait_for_selector("[data-speaker='SPEAKER_00'] .voice")

    def people():
        return page.evaluate("fetch('/api/voices').then(r => r.json()).then(d => d.people.map(p => p.name))")

    row = "[data-speaker='SPEAKER_00']"
    # 1. no name: one click opens a focused form; Esc cancels, an empty name is an error, Enter saves
    open_transcript(ids[0])
    page.click(f"{row} .v-enroll")
    if not page.evaluate("document.activeElement?.classList.contains('v-form-name')"):
        out.append("the name field of the voice form is not focused")
    if page.inner_text(f"{row} .voice-form button[type=submit]").strip() != "Save voice":
        out.append("the form's button should be labelled 'Save voice'")
    page.keyboard.press("Escape")
    if page.locator(f"{row} .voice-form").count() or not page.locator(f"{row} .v-enroll").count():
        out.append("Esc should close the form and bring the button back")
    page.click(f"{row} .v-enroll")
    page.keyboard.press("Enter")
    if not page.locator(f"{row} .voice-error").count():
        out.append("saving an empty name shows no error text")
    page.keyboard.type("QA Erste")
    page.keyboard.press("Enter")
    page.wait_for_selector(f"{row} .voice-tag.ok")
    if "QA Erste" not in page.inner_text(f"{row} .voice-tag"):
        out.append(f"the saved state should name the person: {page.inner_text(f'{row} .voice-tag')!r}")
    if "QA Erste" not in page.inner_text("[role=status].sr-only"):
        out.append("the result is not announced in the aria-live region")
    if "QA Erste" not in people():
        out.append("the voice was not stored")

    # 2. rename, then the prompt; Remember saves in one click
    open_transcript(ids[1])
    if page.locator(f"{row} .voice-offer").count():
        out.append("the prompt appears before any rename")
    page.fill(f"{row} .speaker-name input", "QA Zweite")
    page.wait_for_selector(f"{row} .voice-offer", timeout=5000)
    if "QA Zweite" not in page.inner_text(f"{row} .voice-offer"):
        out.append("the prompt does not name the speaker")
    page.click(f"{row} .voice-offer .v-enroll")
    page.wait_for_selector(f"{row} .voice-tag.ok")
    if "QA Zweite" not in people():
        out.append("Remember in the prompt did not store the voice")

    # 3. rename, "Not now" once per transcript+speaker; a plain button later saves in one click with the name
    open_transcript(ids[2])
    page.fill(f"{row} .speaker-name input", "QA Dritte")
    page.wait_for_selector(f"{row} .voice-offer", timeout=5000)
    page.click(f"{row} .v-not-now")
    if page.locator(f"{row} .voice-offer").count() or not page.locator(f"{row} .v-enroll").count():
        out.append("'Not now' should hide the prompt and keep the Remember button")
    open_transcript(ids[2])
    page.fill(f"{row} .speaker-name input", "QA Dritte B")
    page.wait_for_timeout(1500)
    if page.locator(f"{row} .voice-offer").count():
        out.append("the prompt came back after 'Not now'")
    if "QA Dritte B" in people():
        out.append("a voice was stored without an explicit action")
    page.click(f"{row} .v-enroll")
    page.wait_for_selector(f"{row} .voice-tag.ok")
    if "QA Dritte B" not in people() or page.locator(f"{row} .voice-form").count():
        out.append("one click on Remember voice with a typed name should save it")
    out += [f"page: {e}" for e in errors]
    ctx.close()
    return out


def i18n_parity(page) -> list[str]:
    """Keys that exist in one language dictionary but not in the other (plural forms count as one entry)."""
    return page.evaluate("""async () => {
      const { LANGUAGES } = await import('/js/core/i18n.js');
      const plural = new Set(['zero', 'one', 'two', 'few', 'many', 'other']);
      const flat = (o, p = '') => Object.entries(o).flatMap(([k, v]) => {
        if (v && typeof v === 'object' && !Object.keys(v).every((x) => plural.has(x))) return flat(v, p + k + '.');
        return [p + k];
      });
      const en = new Set(flat(LANGUAGES.en)), cs = new Set(flat(LANGUAGES.cs));
      return [...en].filter((k) => !cs.has(k)).map((k) => 'missing in cs: ' + k)
        .concat([...cs].filter((k) => !en.has(k)).map((k) => 'missing in en: ' + k));
    }""")


def focus_problems(page, limit: int = 60) -> list[str]:
    """Tab through the page; every element that gets focus must show an outline or a box-shadow ring."""
    bad: list[str] = []
    page.evaluate("() => { document.activeElement?.blur(); window.scrollTo(0, 0); }")
    for _ in range(limit):
        page.keyboard.press("Tab")
        info = page.evaluate(
            "() => { const e = document.activeElement; if (!e || e === document.body) return null;"
            "const s = getComputedStyle(e); const r = e.getBoundingClientRect();"
            "let p = e.parentElement, ring = false; if (e.matches('input[type=radio],input[type=checkbox]') && e.nextElementSibling) {"
            "const ls = getComputedStyle(e.nextElementSibling); ring = ls.outlineStyle !== 'none' && parseFloat(ls.outlineWidth) > 0;}"
            "const parent = e.closest('.search'); const ps = parent ? getComputedStyle(parent) : null;"
            "return { tag: e.tagName + (e.id ? '#' + e.id : '') + '.' + e.className, outline: s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0,"
            "shadow: s.boxShadow !== 'none', border: s.borderColor, ring, parentShadow: !!ps && ps.boxShadow !== 'none', w: r.width, h: r.height }; }")
        if info is None:
            break
        if not (info["outline"] or info["shadow"] or info["ring"] or info["parentShadow"]):
            bad.append(f"no visible focus indicator on {info['tag']}")
    return bad


if __name__ == "__main__":
    sys.exit(main())
