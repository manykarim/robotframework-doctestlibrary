"""Full user journeys against the served dashboard (Playwright, no mocks).

J1 ingest → browse → diff viewer → accept → file changed on disk + audit
J2 reject with reason → bug-data ZIP with expected contents
J3 diff region → mask editor → recompare PASS → save → robot re-run passes
J4 shorthand import → edit → export round trip
"""

import io
import json
import shutil
import zipfile

from e2e_helpers import REF_IMAGE, SUITE_TEMPLATE, make_image_run
from playwright.sync_api import Page, expect


def ingest_via_ui(page: Page, server_url: str, output_xml) -> None:
    page.goto(f"{server_url}/#/")
    page.fill('[data-testid="ingest-path"]', str(output_xml))
    page.click('[data-testid="ingest-button"]')
    expect(page.get_by_test_id("ingest-message")).to_contain_text("Ingested run")


def open_first_failing_comparison(page: Page) -> None:
    page.locator('[data-testid="run-list"] tr.clickable').first.click()
    page.select_option('[data-testid="status-filter"]', "fail")
    page.locator('[data-testid="test-grid"] tr.clickable').first.click()
    expect(page.get_by_test_id("comparison-status")).to_have_text("FAIL")


def test_j1_ingest_review_accept(page: Page, server_url, workspace, api):
    output_xml, reference, candidate = make_image_run(workspace / "j1")
    before = reference.read_bytes()

    ingest_via_ui(page, server_url, output_xml)
    open_first_failing_comparison(page)

    # viewer modes: overlay default, keyboard switch to side-by-side and swipe
    expect(page.get_by_test_id("viewer-overlay")).to_be_visible()
    page.keyboard.press("1")
    expect(page.get_by_test_id("viewer-side-by-side")).to_be_visible()
    page.keyboard.press("4")
    expect(page.get_by_test_id("viewer-swipe")).to_be_visible()

    # diff-region navigation
    page.click('[data-testid="next-region"]')
    expect(page.get_by_test_id("region-indicator")).to_contain_text("region 1/")

    # accept the page → baseline promoted on disk
    page.fill('[data-testid="decision-reason"]', "intended change")
    page.click('[data-testid="accept-page"]')
    expect(page.get_by_test_id("action-message")).to_contain_text("accepted", ignore_case=True)
    expect(page.get_by_test_id("review-state")).to_have_text("accepted")

    assert reference.read_bytes() == candidate.read_bytes() != before

    # audit row exists with hashes
    comparison_id = int(page.url.rsplit("/", 1)[-1])
    decisions = api.get(f"/api/comparisons/{comparison_id}/decisions").json()
    assert decisions[0]["action"] == "accept"
    assert decisions[0]["reason"] == "intended change"
    assert decisions[0]["prev_sha256"] and decisions[0]["new_sha256"]


def test_j2_reject_with_bug_bundle(page: Page, server_url, workspace, api):
    output_xml, reference, candidate = make_image_run(workspace / "j2")
    ingest_via_ui(page, server_url, output_xml)
    open_first_failing_comparison(page)

    page.fill('[data-testid="decision-reason"]', "real rendering bug")
    page.click('[data-testid="reject-comparison"]')
    expect(page.get_by_test_id("review-state")).to_have_text("rejected")

    comparison_id = int(page.url.rsplit("/", 1)[-1])
    response = api.get(f"/api/comparisons/{comparison_id}/bugdata")
    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = archive.namelist()
    assert "reference/reference.png" in names
    assert "candidate/candidate.png" in names
    assert "comparison_result.json" in names
    assert any("diff" in name for name in names if name.startswith("pages/"))
    metadata = json.loads(archive.read("metadata.json"))
    assert metadata["decisions"][0]["reason"] == "real rendering bug"

    # reference untouched by reject
    assert reference.read_bytes() != candidate.read_bytes()


def test_j3_mask_from_diff_recompare_save_rerun(page: Page, server_url, workspace, api):
    output_xml, reference, candidate = make_image_run(workspace / "j3")
    ingest_via_ui(page, server_url, output_xml)
    open_first_failing_comparison(page)

    # seed the editor from every diff region: step through them, take the first
    page.click('[data-testid="next-region"]')
    with page.expect_navigation():
        page.click('[data-testid="add-mask-from-region"]')
    expect(page.get_by_test_id("editor-dpi")).to_be_visible()
    expect(page.get_by_test_id("mask-row-0")).to_contain_text("coordinates")

    # the seeded region may not cover all diffs — import masks covering all
    # regions like a user drawing the remaining ones
    comparison_id = int(
        dict(p.split("=") for p in page.url.split("?")[1].split("&"))["comparison"])
    detail = api.get(f"/api/comparisons/{comparison_id}").json()
    regions = detail["pages"][0]["regions"]
    extra = [{
        "page": "all", "type": "coordinates",
        "x": max(0, r["x"] - 5), "y": max(0, r["y"] - 5),
        "width": r["width"] + 10, "height": r["height"] + 10, "unit": "px",
    } for r in regions[1:]]
    if extra:
        page.fill('[data-testid="import-text"]', json.dumps(extra))
        page.click('[data-testid="import-button"]')

    # recompare with the masks: FAIL flips to PASS
    page.click('[data-testid="recompare"]')
    expect(page.get_by_test_id("recompare-result")).to_have_text("PASS", timeout=120_000)

    # save to masks.json
    masks_file = workspace / "j3" / "artifacts" / "masks.json"
    page.fill('[data-testid="mask-file"]', str(masks_file))
    page.click('[data-testid="save-masks"]')
    expect(page.get_by_test_id("editor-message")).to_contain_text("Saved")

    # the saved file parses through the library schema...
    from DocTest.IgnoreAreaManager import IgnoreAreaManager
    parsed = IgnoreAreaManager(ignore_area_file=str(masks_file)).read_ignore_areas()
    assert parsed and all(entry["type"] == "coordinates" for entry in parsed)

    # ...and an actual robot re-run with that mask passes
    sys_path_suite = f"""
*** Settings ***
Library    DocTest.VisualTest    take_screenshots=false

*** Test Cases ***
Masked Comparison Passes
    Compare Images    {reference}    {candidate}    placeholder_file={masks_file}
"""
    from e2e_helpers import run_robot_suite
    run_robot_suite(sys_path_suite, workspace / "j3" / "rerun")  # asserts rc == 0


def _browse_to(page: Page, workspace, *names: str) -> None:
    """Navigate the file-browser modal into workspace/<names...>.

    Handles both browser states: a roots list (several roots configured)
    and the auto-entered directory view (single root).
    """
    expect(page.get_by_test_id("file-browser")).to_be_visible()
    page.wait_for_timeout(400)  # let the initial listing (and auto-enter) settle
    if page.get_by_test_id(f"fb-entry-{names[0]}").count() == 0:
        page.locator(".fb-entry", has_text=str(workspace)).first.click()
    for name in names:
        page.click(f'[data-testid="fb-entry-{name}"]')


def test_j5_file_browser_picks_document_and_mask_target(page: Page, server_url, workspace):
    artifacts = workspace / "j5browse" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REF_IMAGE, artifacts / "reference.png")

    page.goto(f"{server_url}/#/editor")

    # pick the document via Browse… instead of typing a path
    page.click('[data-testid="browse-doc"]')
    _browse_to(page, workspace, "j5browse", "artifacts", "reference.png")
    expect(page.get_by_test_id("doc-file")).to_have_value(str(artifacts / "reference.png"))
    expect(page.get_by_test_id("editor-dpi")).to_be_visible()

    # pick the masks.json target via the save-mode browser
    page.click('[data-testid="browse-mask"]')
    _browse_to(page, workspace, "j5browse", "artifacts")
    page.fill('[data-testid="fb-filename"]', "browser_masks.json")
    page.click('[data-testid="fb-select"]')
    expect(page.get_by_test_id("mask-file")).to_have_value(
        str(artifacts / "browser_masks.json"))

    # add a mask and save to the picked location
    page.click('[data-testid="add-coordinates-mask"]')
    page.click('[data-testid="save-masks"]')
    expect(page.get_by_test_id("editor-message")).to_contain_text("Saved")
    saved = json.loads((artifacts / "browser_masks.json").read_text())
    assert saved[0]["type"] == "coordinates"

    # re-opening the mask browser and picking the existing file auto-loads it
    page.reload()
    page.click('[data-testid="browse-mask"]')
    _browse_to(page, workspace, "j5browse", "artifacts", "browser_masks.json")
    page.click('[data-testid="fb-select"]')
    expect(page.get_by_test_id("editor-message")).to_contain_text("Loaded 1 masks")


def test_j6_upload_local_image_and_edit_masks(page: Page, server_url, workspace):
    """A user with zero configured roots uploads an image from their machine
    and immediately edits + saves masks for it."""
    page.goto(f"{server_url}/#/editor")

    page.set_input_files('[data-testid="upload-input"]', str(REF_IMAGE))
    expect(page.get_by_test_id("editor-message")).to_contain_text("Uploaded birthday_1080.png")
    expect(page.get_by_test_id("editor-dpi")).to_be_visible()
    # uploaded path landed in the document field, masks.json suggested next to it
    doc_value = page.get_by_test_id("doc-file").input_value()
    assert "uploads" in doc_value and doc_value.endswith("birthday_1080.png")
    mask_value = page.get_by_test_id("mask-file").input_value()
    assert mask_value == doc_value.rsplit("/", 1)[0] + "/masks.json"

    page.click('[data-testid="add-coordinates-mask"]')
    page.click('[data-testid="save-masks"]')
    expect(page.get_by_test_id("editor-message")).to_contain_text("Saved 1 masks")


def test_j7_typing_pattern_mask_never_errors(page: Page, server_url, workspace):
    """Reproduces the reported journey: upload an image, add a text-pattern
    mask, and type a regex character by character. Mid-typing states are
    invalid regexes — the editor must pause the preview with a hint, produce
    no failed requests, and resume once the pattern compiles."""
    failures = []
    page.on(
        "response",
        lambda response: failures.append(response)
        if "/api/mask-preview" in response.url and response.status >= 500
        else None,
    )

    page.goto(f"{server_url}/#/editor")
    page.set_input_files('[data-testid="upload-input"]', str(REF_IMAGE))
    expect(page.get_by_test_id("editor-dpi")).to_be_visible()

    page.click('[data-testid="add-pattern-mask"]')
    pattern_input = page.locator('[data-testid="prop-pattern"]')

    # simulate keystroke-by-keystroke typing of "[0-9]{2}"
    for prefix in ("[", "[0", "[0-9", "[0-9]", "[0-9]{2", "[0-9]{2}"):
        pattern_input.fill(prefix)
        page.wait_for_timeout(120)

    status = page.get_by_test_id("pattern-status")
    expect(status).to_be_visible()
    # final pattern is valid: hint must clear and preview must run through
    page.wait_for_timeout(1000)
    expect(status).not_to_contain_text("Incomplete")
    expect(status).to_contain_text("highlighted on this page")
    expect(page.get_by_test_id("editor-message")).to_have_count(0)

    # an unterminated set left standing shows the hint, still without errors
    pattern_input.fill("[0-9]{2}[")
    expect(status).to_contain_text("Incomplete or invalid")

    assert not failures, f"mask-preview returned 5xx: {failures}"


def test_j8_upload_results_folder_from_disk(page: Page, server_url, workspace):
    """A user picks their Robot Framework output folder in the browser; the
    whole tree (output.xml + sidecars + images) uploads and ingests, and the
    run is immediately reviewable."""
    output_xml, reference, candidate = make_image_run(workspace / "j8")
    run_dir = output_xml.parent

    page.goto(f"{server_url}/#/")
    # Playwright supports directory upload for webkitdirectory inputs
    page.set_input_files('[data-testid="upload-results-input"]', str(run_dir))
    message = page.get_by_test_id("ingest-message")
    expect(message).to_contain_text("ingested run", timeout=30_000)

    # the uploaded run is reviewable end to end
    page.locator('[data-testid="run-list"] tr.clickable').first.click()
    page.select_option('[data-testid="status-filter"]', "fail")
    page.locator('[data-testid="test-grid"] tr.clickable').first.click()
    expect(page.get_by_test_id("comparison-status")).to_have_text("FAIL")
    expect(page.get_by_test_id("viewer-overlay")).to_be_visible()


def test_j4_shorthand_import_export_roundtrip(page: Page, server_url, workspace):
    page.goto(f"{server_url}/#/editor")
    page.fill('[data-testid="import-text"]', "top:10;bottom:5")
    page.click('[data-testid="import-button"]')
    expect(page.get_by_test_id("mask-row-0")).to_contain_text("area")
    expect(page.get_by_test_id("mask-row-1")).to_contain_text("area")

    # edit: change percent of the first mask via the panel
    page.click('[data-testid="mask-row-0"]')
    page.locator('[data-testid="prop-percent"]').fill("15")

    masks_file = workspace / "j4_masks.json"
    page.fill('[data-testid="mask-file"]', str(masks_file))
    page.click('[data-testid="save-masks"]')
    expect(page.get_by_test_id("editor-message")).to_contain_text("Saved")

    saved = json.loads(masks_file.read_text())
    assert [entry["type"] for entry in saved] == ["area", "area"]
    assert saved[0]["location"] == "top"
    assert int(saved[0]["percent"]) == 15
    assert saved[1]["location"] == "bottom"

    # round trip: load it back, list shows both masks again
    page.click('[data-testid="load-masks"]')
    expect(page.get_by_test_id("editor-message")).to_contain_text("Loaded 2 masks")


def test_j9_batch_accept_and_run_lifecycle(page: Page, server_url, workspace, api):
    """Multi-select two failing comparisons, batch-accept them with the
    confirmation bar, verify baselines changed on disk, then delete the run."""
    import shutil
    from e2e_helpers import run_robot_suite

    base = workspace / "j9"
    artifacts = base / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    pairs = []
    suite_lines = ["*** Settings ***",
                   "Library    DocTest.VisualTest    result_json=true    take_screenshots=false",
                   "", "*** Test Cases ***"]
    for index in (1, 2):
        reference = artifacts / f"ref_{index}.png"
        candidate = artifacts / f"cand_{index}.png"
        shutil.copyfile(REF_IMAGE, reference)
        shutil.copyfile(REF_IMAGE.parent / "birthday_1080_date_id.png", candidate)
        pairs.append((reference, candidate))
        suite_lines += [f"Comparison {index}",
                        "    Run Keyword And Expect Error    The compared images are different.",
                        f"    ...    Compare Images    {reference}    {candidate}", ""]
    output_xml = run_robot_suite("\n".join(suite_lines), base / "run")

    page.goto(f"{server_url}/#/")
    page.fill('[data-testid="ingest-path"]', str(output_xml))
    page.click('[data-testid="ingest-button"]')
    expect(page.get_by_test_id("ingest-message")).to_contain_text("Ingested run")
    # runs are listed newest first
    page.locator('[data-testid="run-list"] tr.clickable').first.click()

    # select all eligible on the page, batch accept through the confirm bar
    page.click('[data-testid="select-page"]')
    accept_button = page.get_by_test_id("accept-selected")
    expect(accept_button).to_contain_text("(2)")
    accept_button.click()
    expect(page.get_by_test_id("confirm-batch")).to_contain_text("2 candidate file(s)")
    page.click('[data-testid="confirm-batch-yes"]')
    expect(page.get_by_test_id("batch-message")).to_contain_text("Accepted 2 comparison(s)")

    for reference, candidate in pairs:
        assert reference.read_bytes() == candidate.read_bytes()

    # zoom is present in the viewer
    page.locator('[data-testid="test-grid"] tr.clickable').first.click()
    expect(page.get_by_test_id("zoom-level")).to_have_text("1.00×")
    expect(page.get_by_test_id("pane-overlay")).to_be_visible()

    # delete the run from the dashboard
    page.goto(f"{server_url}/#/")
    delete_button = page.locator('[data-testid^="delete-run-"]').first
    run_row_id = delete_button.get_attribute("data-testid").rsplit("-", 1)[-1]
    delete_button.click()
    page.click(f'[data-testid="confirm-delete-yes-{run_row_id}"]')
    expect(page.get_by_test_id("ingest-message")).to_contain_text(f"Deleted run {run_row_id}")
    # artifacts on disk survive dashboard deletion
    assert pairs[0][0].exists()


def test_j10_similarity_group_review(page: Page, server_url, workspace, api):
    """Two comparisons failing with the identical difference group together;
    the group is accepted with one confirmed action."""
    import shutil
    from e2e_helpers import run_robot_suite

    base = workspace / "j10"
    artifacts = base / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    pairs = []
    lines = ["*** Settings ***",
             "Library    DocTest.VisualTest    result_json=true    take_screenshots=false",
             "", "*** Test Cases ***"]
    cand_sources = [
        REF_IMAGE.parent / "birthday_1080_date_id.png",
        REF_IMAGE.parent / "birthday_1080_date_id.png",
        REF_IMAGE.parent / "birthday_1080_noise_001.png",
    ]
    for index, cand_src in enumerate(cand_sources, start=1):
        reference = artifacts / f"ref_{index}.png"
        candidate = artifacts / f"cand_{index}.png"
        shutil.copyfile(REF_IMAGE, reference)
        shutil.copyfile(cand_src, candidate)
        pairs.append((reference, candidate))
        lines += [f"Comparison {index}",
                  "    Run Keyword And Expect Error    The compared images are different.",
                  f"    ...    Compare Images    {reference}    {candidate}", ""]
    output_xml = run_robot_suite("\n".join(lines), base / "run")

    page.goto(f"{server_url}/#/")
    page.fill('[data-testid="ingest-path"]', str(output_xml))
    page.click('[data-testid="ingest-button"]')
    expect(page.get_by_test_id("ingest-message")).to_contain_text("Ingested run")
    page.locator('[data-testid="run-list"] tr.clickable').first.click()

    page.click('[data-testid="view-groups"]')
    group_card = page.locator(".group-card").first
    expect(group_card).to_contain_text("2 comparisons with the same difference")

    # expand shows both members
    group_card.locator('[data-testid^="expand-group-"]').click()
    expect(group_card.locator(".group-members li")).to_have_count(2)

    # accept the group through the confirmation bar
    group_card.locator('[data-testid^="accept-group-"]').click()
    expect(page.get_by_test_id("confirm-batch")).to_contain_text("2 candidate file(s)")
    page.click('[data-testid="confirm-batch-yes"]')
    expect(page.get_by_test_id("batch-message")).to_contain_text("Accepted 2 comparison(s)")

    assert pairs[0][0].read_bytes() == pairs[0][1].read_bytes()
    assert pairs[1][0].read_bytes() == pairs[1][1].read_bytes()
    # the unique failure remains unresolved
    assert pairs[2][0].read_bytes() != pairs[2][1].read_bytes()
    expect(page.get_by_test_id("groups-empty")).to_contain_text("1 unresolved failure(s)")


def test_j11_root_cause_explain_region_and_facets(page: Page, server_url, workspace, api):
    """Explain-region shows the changed text of a diff region; PdfTest
    comparisons render their structured facet differences."""
    import shutil
    from e2e_helpers import run_robot_suite
    from helpers import REPO_ROOT

    base = workspace / "j11"
    artifacts = base / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    reference = artifacts / "reference.png"
    candidate = artifacts / "candidate.png"
    shutil.copyfile(REF_IMAGE, reference)
    shutil.copyfile(REF_IMAGE.parent / "birthday_1080_date_id.png", candidate)
    pdf_ref = artifacts / "ref.pdf"
    pdf_cand = artifacts / "cand.pdf"
    shutil.copyfile(REF_IMAGE.parent / "sample_1_page.pdf", pdf_ref)
    shutil.copyfile(REF_IMAGE.parent / "sample_1_page_different_text.pdf", pdf_cand)

    suite = f"""*** Settings ***
Library    DocTest.VisualTest    result_json=true    take_screenshots=false
Library    DocTest.PdfTest    result_json=true

*** Test Cases ***
Image Comparison
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images    {reference}    {candidate}

Pdf Comparison
    Run Keyword And Expect Error    The compared PDF Document Data is different.
    ...    Compare Pdf Documents    {pdf_ref}    {pdf_cand}
"""
    output_xml = run_robot_suite(suite, base / "run")

    page.goto(f"{server_url}/#/")
    page.fill('[data-testid="ingest-path"]', str(output_xml))
    page.click('[data-testid="ingest-button"]')
    expect(page.get_by_test_id("ingest-message")).to_contain_text("Ingested run")
    page.locator('[data-testid="run-list"] tr.clickable').first.click()

    # image comparison: explain the first diff region
    page.select_option('[data-testid="status-filter"]', "fail")
    rows = page.locator('[data-testid="test-grid"] tr.clickable')
    expect(rows).to_have_count(2)
    rows.first.click()
    expect(page.get_by_test_id("comparison-status")).to_have_text("FAIL")
    page.click('[data-testid="next-region"]')
    page.click('[data-testid="explain-region"]')
    panel = page.get_by_test_id("region-text-panel")
    expect(panel).to_be_visible(timeout=60_000)
    expect(page.get_by_test_id("region-text-verdict")).to_have_text("text differs")
    ref_text = page.get_by_test_id("region-text-reference").inner_text()
    cand_text = page.get_by_test_id("region-text-candidate").inner_text()
    assert ref_text != cand_text

    # PdfTest comparison: structured facets rendered
    page.go_back()
    page.select_option('[data-testid="status-filter"]', "fail")
    page.locator('[data-testid="test-grid"] tr.clickable').nth(1).click()
    expect(page.get_by_test_id("comparison-keyword")).to_have_text("Compare Pdf Documents")
    expect(page.get_by_test_id("facets-panel")).to_be_visible()
    facet = page.locator('[data-testid="facet-text"]').first
    expect(facet).to_contain_text("text")


def test_j12_history_across_runs(page: Page, server_url, workspace, api):
    """Re-ingesting the same test identity produces a history timeline in
    the comparison view."""
    for version in ("v1", "v2"):
        output_xml, reference, candidate = make_image_run(workspace / "j12" / version)
        page.goto(f"{server_url}/#/")
        page.fill('[data-testid="ingest-path"]', str(output_xml))
        page.click('[data-testid="ingest-button"]')
        expect(page.get_by_test_id("ingest-message")).to_contain_text("Ingested run")

    page.locator('[data-testid="run-list"] tr.clickable').first.click()
    page.select_option('[data-testid="status-filter"]', "fail")
    page.locator('[data-testid="test-grid"] tr.clickable').first.click()

    history = page.get_by_test_id("history-panel")
    expect(history).to_be_visible()
    # other journeys in this shared session reuse the same test identity, so
    # the timeline holds at least our two runs
    assert history.locator("tbody tr").count() >= 2
    expect(history.locator("tbody tr").first).to_contain_text("FAIL")
