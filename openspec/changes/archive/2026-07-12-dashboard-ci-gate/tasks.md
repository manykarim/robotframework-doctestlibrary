# Tasks: dashboard-ci-gate

- [x] 1.1 `gate` CLI command (run-id/latest, exit 0/1/2, unresolved listing); dependency guard scoped to serve/ingest; tests for all exit codes + guard-free operation
- [x] 1.2 `db.comparison_history` + `GET /api/comparisons/{id}/history`; `db.flaky_identities` + `GET /api/flaky`; `history` feature flag; endpoint tests (two-run timeline, flip detection)
- [x] 1.3 UI: history timeline in the comparison view, flaky panel on the runs page; e2e (two runs same identity → history panel shows both)
- [x] 1.4 Docs (CI gating section incl. example workflow snippet) + full verification
