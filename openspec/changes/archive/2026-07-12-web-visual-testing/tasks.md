# Tasks: web-visual-testing

- [x] 1.1 `DocTest/WebCapture.py`: adapter protocol, Browser/Selenium adapters (run_keyword only), detect_adapter + explicit override; unit tests with faked BuiltIn
- [x] 1.2 `DocTest/WebVisualTest.py`: keywords, baseline lifecycle (create/REFERENCE_RUN/compare), name sanitization, retry loop; unit tests with FakeAdapter (create, pass, retry-succeeds, retry-times-out, traversal)
- [x] 1.3 Acceptance: deterministic test page + `atest/web/browser_vrt.robot` + `atest/web/selenium_vrt.robot` (create → match → mutate → fail → mask pass); `web-test` dependency group; local green runs
- [x] 1.4 CI `web` job (rfbrowser init + chrome), README + docs/web-visual-testing.md; full verification
