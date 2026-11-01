"""Screenshots for the README (docs/img/), taken from invented demo data only.

    uv run python -m tools.readme_shots OUT_DIR [--workdir DIR]

Library, transcript with speakers and summary, voices and watched sources, at 1366 px, light and dark, Czech.
Pass ``--workdir`` (an empty folder, e.g. a ``subst`` drive such as X:/) so that paths shown in the UI are neutral.
"""

from __future__ import annotations

import argparse
import contextlib
import tempfile
from pathlib import Path

from tools.qa_ui import start_server


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--workdir")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    workdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True) if not args.workdir else contextlib.nullcontext(args.workdir)
    with workdir as tmp:
        server, base, ids = start_server("demo", Path(tmp))
        tid = ids["transcripts"][0]
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome", headless=True)
            for scheme in ("light", "dark"):
                ctx = browser.new_context(viewport={"width": 1366, "height": 860}, color_scheme=scheme, locale="cs-CZ")
                ctx.add_init_script("try{localStorage.setItem('lang','cs')}catch(e){}")
                page = ctx.new_page()

                def shot(name, hash_, before=None, full=False, height=860):
                    page.set_viewport_size({"width": 1366, "height": height})
                    page.goto(f"{base}/{hash_}")
                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(600)
                    if before:
                        before()
                    page.mouse.move(0, 0)
                    page.screenshot(path=str(out / f"{name}-{scheme}.png"), full_page=full)

                shot("library", "#/", height=720)

                def summary():
                    page.get_by_role("tab", name="Shrnutí").click()
                    page.wait_for_timeout(500)

                shot("summary", f"#/t/{tid}", before=summary, height=1500)
                shot("voices", "#/voices", height=900)
                shot("watch", "#/watch", height=540)
                ctx.close()
            browser.close()
        server.should_exit = True


if __name__ == "__main__":
    main()
