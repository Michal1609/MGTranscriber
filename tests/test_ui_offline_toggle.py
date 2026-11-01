"""The Offline mode switch on the Settings page saves at once, without the Save button."""

from .test_ui_watch import page, server  # noqa: F401  (fixtures)


def privacy(page, server):  # noqa: F811
    return page.evaluate("fetch('/api/privacy').then(r => r.json())")["offline"]


def test_offline_switch_saves_immediately_and_back(page, server):  # noqa: F811
    page.goto(f"{server}/#/system")
    page.wait_for_selector("#set-offline_mode")
    assert privacy(page, server) is False
    page.locator("#set-offline_mode").check()  # no click on Save
    page.wait_for_selector(".offline-badge", timeout=10000)  # the page reloaded and shows the badge
    assert privacy(page, server) is True
    page.wait_for_selector("#set-offline_mode")
    assert page.locator("#set-offline_mode").is_checked()
    page.locator("#set-offline_mode").uncheck()
    page.wait_for_function("!document.querySelector('.offline-badge')", timeout=10000)
    assert privacy(page, server) is False
