"""Version-skew journeys: a newer UI served by an older backend.

Static frontend files are re-read from disk on every request while the
Python process keeps running old code — so after an update, users can see
new buttons that hit routes the running server does not have (POST to the
static mount answers 405). These tests cover both sides: the happy path
(no warning on a current server) and the stale path, simulated by
intercepting the backend responses in the browser.
"""

from e2e_helpers import REF_IMAGE
from playwright.sync_api import Page, expect


def test_no_server_warning_on_current_server(page: Page, server_url):
    page.goto(f"{server_url}/#/")
    expect(page.get_by_test_id("run-list")).to_be_visible()
    expect(page.get_by_test_id("server-warning")).to_have_count(0)


def test_stale_backend_shows_restart_banner(page: Page, server_url):
    # Old backends answered /api/health without a feature list
    page.route(
        "**/api/health",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status": "ok", "version": "0.0.9"}',
        ),
    )
    page.goto(f"{server_url}/#/")
    banner = page.get_by_test_id("server-warning")
    expect(banner).to_be_visible()
    expect(banner).to_contain_text("older than this user interface")
    expect(banner).to_contain_text("Restart")
    expect(banner).to_contain_text("browse, upload, recompare, upload-results, batch-accept, lifecycle")


def test_unreachable_backend_shows_warning(page: Page, server_url):
    page.route("**/api/health", lambda route: route.abort())
    page.goto(f"{server_url}/#/")
    expect(page.get_by_test_id("server-warning")).to_contain_text(
        "Cannot reach the dashboard server")


def test_stale_backend_upload_405_explains_restart(page: Page, server_url):
    # POST /api/upload on an old backend falls through to the static mount → 405
    page.route(
        "**/api/upload",
        lambda route: route.fulfill(status=405, body="Method Not Allowed"),
    )
    page.goto(f"{server_url}/#/editor")
    page.set_input_files('[data-testid="upload-input"]', str(REF_IMAGE))
    message = page.get_by_test_id("editor-message")
    expect(message).to_contain_text("does not support uploads yet")
    expect(message).to_contain_text("Restart")
