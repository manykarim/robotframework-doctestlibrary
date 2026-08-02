# Tasks: fix-oversized-log-screenshots

## 1. Fix

- [x] 1.1 `DocTest/VisualTest.py`: change the non-`original_size` style from `width:50%; height: auto;` to `max-width:50%; height: auto;` (leave the `original_size=True` branch untouched)

## 2. Regression test

- [x] 2.1 Add `utest/test_screenshot_log_style.py` patching `DocTest.VisualTest.robot_logger`: assert the default file-link path (jpg, `embed_screenshots=False`) emits `max-width:50%; height: auto;` inside an `<a target="_blank">` wrapper
- [x] 2.2 Same test file: assert the embedded base64 paths (jpg and png) emit `max-width:50%`, and that `original_size=True` still emits `width: auto; height: auto;`
- [x] 2.3 Assert `width:50%` (the fixed-width form) no longer appears in any emitted HTML

## 3. Verification

- [x] 3.1 New tests pass; `uv run pytest utest -q -n auto --timeout=300` green; `uvx ruff check DocTest` green
