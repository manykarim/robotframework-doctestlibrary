# Proposal: packaging-extras-clarity

## Why

Users must be able to install exactly what they need. The dashboard is already a
proper `[dashboard]` extra; web visual testing deliberately adds **zero runtime
dependencies** (it drives whichever web library the user installed) — but nothing
documents that guarantee, and there is no convenient one-liner to install the
library together with a web driver. The README also lacks a single overview of
what each install form brings.

## What Changes

- Convenience extras: `[browser]` → `robotframework-browser`, `[selenium]` →
  `robotframework-seleniumlibrary` (unpinned; the user's driver choice stays
  theirs). `[all]` keeps meaning ai+dashboard — web drivers remain an explicit
  choice; `[dashboard]`/`[ai]` unchanged.
- **Install matrix** in the README: base / `[dashboard]` / `[ai]` / `[browser]` /
  `[selenium]` / `[all]` — what each adds, including the note that
  `DocTest.WebVisualTest` itself ships in the base package with no extra
  dependencies.
- Guarantees enforced: wheel-parity gate baseline extended with the new extras;
  a test asserts the base dependency set contains no dashboard/ai/web packages
  and that the wheel advertises all extras.

## Capabilities

### Modified Capabilities

- `unified-packaging`: convenience web extras and base-dependency purity.

## Impact

`pyproject.toml`, `scripts/wheel_baseline.json`, `utest/test_package_parity.py`
(or comparator), README, docs/web-visual-testing.md setup section.
