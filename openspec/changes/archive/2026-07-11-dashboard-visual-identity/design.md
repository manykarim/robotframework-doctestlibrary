# Design: dashboard-visual-identity

## Decisions

- **D1 Token layer**: semantic CSS custom properties on `:root` — `--bg`, `--surface`, `--surface-2`, `--border`, `--border-strong`, `--text`, `--text-dim`, `--accent`, `--accent-soft`, `--danger`, `--danger-soft`, `--ok`, `--ok-soft`, `--warn`, `--warn-soft`, `--violet`, `--violet-soft`, `--shadow`, `--radius`, `--radius-sm`. Every component rule references tokens; no raw hex outside the two token blocks.
- **D2 Theme switching**: `html[data-theme="dark"]` overrides the tokens. On boot, `App` reads `localStorage["doctest-theme"]`, falling back to `prefers-color-scheme`; a 🌙/☀ toggle in the top bar flips and persists it (`data-testid="theme-toggle"`). Zoom panes and the editor canvas keep a neutral mid-grey backdrop in both themes — document pages are white and must not sit on pure black.
- **D3 Empty states**: `RunList` with zero runs renders an onboarding card (`data-testid="empty-runs"`): title, one-line explanation, the three ingestion paths (path field, **Upload results folder…**, `doctest-dashboard ingest`). `MaskEditor` without a document renders a matching card (`data-testid="empty-editor"`) pointing at **Browse…** and **Upload image…**. The existing `groups-empty` note stays.
- **D4 A11y/polish**: global `:focus-visible` outline using `--accent`; shadows and radii come from tokens so both themes stay coherent.
- **D5 Verification**: e2e J14 on a *function-scoped fresh server* (empty DB): empty-runs card visible; theme toggle flips `data-theme` and persists across reload; editor empty state visible. Screenshot capture re-run (scratchpad `capture.py`) for a manual visual pass in both themes.

## Risks

- [Dark mode illegible badges] → badge colors get explicit dark-theme soft/strong pairs rather than auto-inversion.
- [Theme flash on load] → theme applied in a tiny inline effect before first paint (React state init reads localStorage synchronously).
