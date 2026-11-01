"""Library folders in the web UI, for tools/qa_ui.py: page actions (screenshots, axe and focus checks see the folder
tree, the selection bar and the "Move to…" dialog) and ``folder_flow`` (drives create / rename / move / delete, drag
& drop, keyboard selection, search in a folder, the form's folder and the detail page) on the demo data.

The demo library (tools/make_demo_data.py) has: Podcasty/Zahrada (2), Spolek/Porady (2), Klient Novák/2026 (empty)
and one unfiled transcript.
"""

from __future__ import annotations

VIEWPORTS = {"desktop": {"width": 1366, "height": 800}, "mobile": {"width": 390, "height": 844}}
DROP = """([ids, target]) => {
  const dt = new DataTransfer();
  dt.setData('application/x-mgtranscriber-ids', JSON.stringify(ids));
  const row = document.querySelector(`.ft-row[data-drop="${CSS.escape(target)}"]`);
  const fire = (type) => row.dispatchEvent(new DragEvent(type, {dataTransfer: dt, bubbles: true, cancelable: true}));
  fire('dragenter'); fire('dragover'); const over = row.classList.contains('drop-over'); fire('drop');
  return over;
}"""


def tree_counts(page) -> dict[str, str]:
    """Visible label of every folder button -> its count (the desktop tree)."""
    return page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('.ft-label')].map(
        (b) => [b.dataset.folder, b.querySelector('.ft-count').textContent.trim()]))""")


def show_folder(path: str):
    """Page action: expand the ancestors and open a folder (the mobile layout uses the <select>)."""
    def action(page):
        if page.viewport_size["width"] < 760:
            page.select_option("#fs-select-input", path)
        else:
            parts = path.split("/")
            for i in range(1, len(parts)):
                toggle = page.locator(f'.ft-toggle[data-toggle="{"/".join(parts[:i])}"]')
                if toggle.count() and toggle.get_attribute("aria-expanded") == "false":
                    toggle.click()
            page.click(f'.ft-label[data-folder="{path}"]')
        page.wait_for_timeout(300)
    return action


def select_first_tile(page) -> None:
    page.locator(".tile-check input").first.check()
    page.wait_for_timeout(200)


def open_move_dialog(page) -> None:
    select_first_tile(page)
    page.click('[data-sel="move"]')
    page.wait_for_selector("dialog[open]")


def page_actions() -> dict[str, tuple[str, object]]:
    def folder_then(action):
        def run(page):
            show_folder("Podcasty/Zahrada")(page)
            action(page)
        return run

    return {
        "home-folder": ("#/", show_folder("Podcasty/Zahrada")),
        "home-selected": ("#/", folder_then(select_first_tile)),
        "home-move-dialog": ("#/", folder_then(open_move_dialog)),
    }


def folder_flow(browser, base: str) -> list[str]:
    out: list[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            out.append(message)

    ctx = browser.new_context(bypass_csp=True, viewport=VIEWPORTS["desktop"], locale="en-GB")
    ctx.add_init_script("try{localStorage.setItem('lang','en');localStorage.removeItem('library.folder')}catch(e){}")
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"exception: {e}"))
    page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))
    page.goto(base + "/#/")
    page.wait_for_selector(".ft-label")
    page.wait_for_selector(".tile")

    # the tree: all, unfiled, top level folders with the totals of their subtrees; an empty folder exists
    counts = tree_counts(page)
    check(counts.get("*") == "5" and counts.get("") == "1", f"All/Unfiled counts wrong: {counts}")
    check(counts.get("Podcasty") == "2" and counts.get("Spolek") == "2", f"top level totals wrong: {counts}")
    check(counts.get("Klient Novák") == "0", f"the empty folder is missing: {counts}")
    check(page.locator('.ft-toggle[data-toggle="Podcasty"]').get_attribute("aria-expanded") == "false", "aria-expanded missing")
    page.click('.ft-toggle[data-toggle="Podcasty"]')
    check(page.locator('.ft-toggle[data-toggle="Podcasty"]').get_attribute("aria-expanded") == "true", "toggle did not expand")
    check(page.locator('.ft-label[data-folder="Podcasty/Zahrada"]').is_visible(), "the subfolder is not shown after expanding")
    page.click('.ft-label[data-folder="Podcasty/Zahrada"]')
    check(page.locator(".tile").count() == 2, f"folder should show 2 tiles, got {page.locator('.tile').count()}")
    crumbs = page.inner_text(".crumbs")
    check("Podcasty" in crumbs and "Zahrada" in crumbs, f"breadcrumb wrong: {crumbs!r}")
    check(page.locator('.ft-label[aria-current="true"]').get_attribute("data-folder") == "Podcasty/Zahrada", "aria-current wrong")
    check(page.input_value("#job-folder") == "Podcasty/Zahrada", "the form's folder should follow the open folder")

    # keyboard: select with Space, "Move to…" with Enter, choose with the keyboard
    first = page.locator(".tile-check input").first
    first.focus()
    page.keyboard.press("Space")
    check("1 transcript selected" in page.inner_text("#selection-bar"), f"selection bar: {page.inner_text('#selection-bar')!r}")
    page.focus('[data-sel="move"]')
    page.keyboard.press("Enter")
    page.wait_for_selector("dialog[open]")
    page.select_option('dialog select[name="folder"]', "Klient Novák/2026")
    page.focus("dialog .fx-ok")
    page.keyboard.press("Enter")
    page.wait_for_selector("dialog[open]", state="detached")
    page.wait_for_function("document.querySelectorAll('.tile').length === 1")
    counts = tree_counts(page)
    check(counts.get("Klient Novák/2026") == "1" or counts.get("Klient Novák") == "1", f"move not reflected in the tree: {counts}")

    # Escape cancels the dialog without moving anything
    before = tree_counts(page)
    page.locator(".tile-check input").first.check()
    page.click('[data-sel="move"]')
    page.wait_for_selector("dialog[open]")
    page.select_option('dialog select[name="folder"]', "Podcasty/Zahrada")  # a target different from the preselected one
    page.keyboard.press("Escape")
    page.wait_for_selector("dialog[open]", state="detached")
    check(page.locator(".tile").count() == 1 and tree_counts(page) == before, "Escape in the dialog moved something")
    page.click('[data-sel="clear"]')

    # a route change closes an open dialog (Back while "Move to…" is open) and nothing is moved
    before = tree_counts(page)
    page.locator(".tile-check input").first.check()
    page.click('[data-sel="move"]')
    page.wait_for_selector("dialog[open]")
    page.select_option('dialog select[name="folder"]', "Podcasty/Zahrada")  # a target different from the preselected one
    page.evaluate("location.hash = '#/system'")
    page.wait_for_selector("dialog[open]", state="detached")
    check(page.locator("dialog").count() == 0, "dialog left in the DOM after a route change")
    page.go_back()
    page.wait_for_selector(".tile")
    check(page.locator(".tile").count() == 1 and tree_counts(page) == before, "route change in the dialog moved something")
    page.click('[data-sel="clear"]') if page.locator('[data-sel="clear"]').is_visible() else None

    # drag & drop onto the Unfiled row
    only = page.locator(".tile").first.get_attribute("data-id")
    over = page.evaluate(DROP, [[only], ""])
    check(over, "the drop target was not highlighted")
    page.wait_for_function("document.querySelector('#library .empty') !== null")
    check(tree_counts(page).get("") == "2", f"Unfiled should have 2 after the drop: {tree_counts(page)}")

    # create, rename, delete
    page.click('[data-action="new"]')
    page.wait_for_selector("dialog[open]")
    page.fill('dialog input[name="name"]', "Nový/Test")
    page.keyboard.press("Enter")
    page.wait_for_selector("dialog[open]", state="detached")
    page.wait_for_selector('.ft-label[aria-current="true"][data-folder="Nový/Test"]')
    page.click('.folder-actions [data-action="rename"]')
    page.wait_for_selector("dialog[open]")
    page.fill('dialog input[name="name"]', "Nový/Přejmenováno")
    page.keyboard.press("Enter")
    page.wait_for_selector('.ft-label[aria-current="true"][data-folder="Nový/Přejmenováno"]')
    page.click('.folder-actions [data-action="delete"]')
    page.wait_for_selector("dialog[open]")
    page.click("dialog .fx-ok")
    page.wait_for_selector("dialog[open]", state="detached")
    page.wait_for_function("!document.querySelector('.ft-label[data-folder=\"Nový/Přejmenováno\"]')")
    check(page.locator("dialog").count() == 0, "dialog left in the DOM")
    page.click('.ft-label[data-folder="*"]')
    check(page.locator(".tile").count() == 5, "deleting a folder must not delete transcripts")

    # search limited to the open folder
    page.click('.ft-toggle[data-toggle="Spolek"]') if page.locator('.ft-toggle[data-toggle="Spolek"][aria-expanded="false"]').count() else None
    page.click('.ft-label[data-folder="Spolek/Porady"]')
    page.fill("#library-filter", "dobrý")
    page.wait_for_selector("#search-results .hit-group")
    everywhere = page.locator("#search-results .hit-group").count()
    page.check("#search-here")
    page.wait_for_function(
        f"document.querySelectorAll('#search-results .hit-group').length < {everywhere}")
    inside = page.locator("#search-results .hit-group").count()
    check(0 < inside < everywhere, f"search in folder: {inside} groups vs {everywhere} everywhere")
    api = page.evaluate("fetch('/api/search?q=' + encodeURIComponent('dobrý') + '&folder=Spolek%2FPorady').then(r => r.json())")
    check(all(h["title"].startswith("Porada") for h in api["results"]), "folder search returned other transcripts")
    page.fill("#library-filter", "")
    page.uncheck("#search-here")

    # the detail page: folder line and "Move to…"
    page.locator(".tile-link").first.click()
    page.wait_for_selector(".d-folder .crumb")
    check("Spolek" in page.inner_text(".d-folder") and "Porady" in page.inner_text(".d-folder"), "detail folder line wrong")
    page.click(".d-move")
    page.wait_for_selector("dialog[open]")
    page.select_option('dialog select[name="folder"]', "")
    page.click("dialog .fx-ok")
    page.wait_for_function("document.querySelector('.d-folder').innerText.includes('Unfiled')")

    # mobile: the drop-down replaces the tree
    mobile = browser.new_context(bypass_csp=True, viewport=VIEWPORTS["mobile"], locale="en-GB")
    mobile.add_init_script("try{localStorage.setItem('lang','en');localStorage.removeItem('library.folder')}catch(e){}")
    m = mobile.new_page()
    m.goto(base + "/#/")
    m.wait_for_selector("#fs-select-input")
    check(m.locator("#fs-select-input").is_visible() and not m.locator(".fs-tree").is_visible(),
          "mobile should show the folder drop-down instead of the tree")
    m.select_option("#fs-select-input", "Spolek/Porady")  # one of its two was moved to Unfiled on the detail page
    m.wait_for_function("document.querySelectorAll('.tile').length === 1")
    mobile.close()

    out += [f"page: {e}" for e in errors]
    ctx.close()
    return out
