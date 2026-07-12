# Tasks: dashboard-diff-groups

- [x] 1.1 dHash + group-key computation in `ingest.py` (failing sidecar pages only); `group_key` column + index (+ defensive ALTER); unit tests for determinism (identical → same key, different → different, degraded/pass → null)
- [x] 1.2 `GET /api/runs/{id}/groups` endpoint (size ≥ 2 groups + ungrouped count, unresolved failures only) + `diff-groups` feature flag; endpoint tests incl. member-resolution behavior
- [x] 1.3 Grouped view UI: flat/similarity toggle, group cards (count, thumbnail, expandable members), Accept group via existing confirm bar; group accept test through API + e2e journey (two identical + one different failure → group of 2 → accept group → files promoted)
- [x] 1.4 Docs (dashboard.md review section) + full dashboard/e2e verification
