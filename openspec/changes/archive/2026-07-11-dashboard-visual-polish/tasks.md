# Tasks: dashboard-visual-polish

- [x] 1.1 Ingest/DAL: thumbnail preference chain; `comparisons.label` from sidecar name (+identity `name::<label>`); `runs.label` + name derivation + `PATCH /api/runs/{id}`; defensive ALTERs; backend tests (thumb preference, label identity/history, rename)
- [x] 1.2 Viewer: fitTo on load + Fit/100% buttons + double-click refit; `highlight` mode (key 5); all-regions outlines toggle; header test-name/label with keyword chip; sticky decision bar; hide viewer chrome for page-less comparisons
- [x] 1.3 Grid/groups state fixes (selection reset, accept-selected only in flat view); run list labels + relative time + inline rename; browser root display; editor default names + disabled-button titles + height-aware canvas fit
- [x] 1.4 e2e J13 (fit zoom on open, highlight mode, outlines visible, header name, rename run) + full dashboard/e2e/core verification + docs touch-up
