# Design: dashboard-diff-groups

## Context

Sidecar-backed failing comparisons store per-page `diff_regions` (x/y/w/h) and diff-image paths. Percy groups snapshots whose diffs share identical geometry and identical pixels; document renderings are deterministic per engine version, so byte-level-deterministic hashing is viable, but a perceptual hash (dHash) tolerates PNG encoding variance while still being strict about content.

## Goals / Non-Goals

**Goals:** deterministic, strict grouping computed at ingest; group review actions built on `accept_many`; groups never mix degraded or pass records.
**Non-Goals:** tolerance knobs / near-match clustering (future), cross-run grouping (groups are per run view; the key itself is run-independent by construction and can enable it later).

## Decisions

- **D1 Group key**: per failing page → `f"{sorted((w,h) for r in regions)}|{dhash(diff_image)}"`; comparison `group_key = sha256("\n".join(page_keys))[:16]`. Geometry uses sizes only (Percy: same size regardless of coordinates) — a moved-but-identical diff groups; the dHash of the *diff image* (which is position-preserving) keeps this honest: identical size + identical diff pixels.
- **D2 dHash**: 8×8 gradient hash via cv2 (grayscale → resize 9×8 → adjacent-pixel comparison → 64-bit hex). Deterministic, cheap (<1 ms/page), robust to encoder differences.
- **D3 Storage**: `comparisons.group_key TEXT` in the schema plus a defensive `ALTER TABLE ... ADD COLUMN` for pre-existing dev databases (unreleased product — no formal migration). Index on `(group_key)`.
- **D4 API**: `GET /api/runs/{run_id}/groups` → `{groups: [{group_key, count, members: [{comparison_id, name, thumbnail}]}], ungrouped: n}` restricted to `status=FAIL AND review_state='unresolved'`; groups of size 1 fold into `ungrouped` listing (a group of one is noise). Accept/reject group = existing `accept-batch` / per-member reject with the member ids — no new mutation endpoints.
- **D5 UI**: view toggle `flat | by similarity` on the run page; group cards show count, sample thumbnail, expandable member list, *Accept group (n)* via the existing confirmation bar. Feature flag `diff-groups` added to `API_FEATURES`/`REQUIRED_FEATURES`.

## Risks / Trade-offs

- [False grouping → wrong batch accept] → strict key (sizes + diff-image dHash); dHash collisions on *different* diffs of identical size are the residual risk — 64-bit gradient hashes on structured document diffs make this negligible, and the member list in the confirm flow keeps a human in the loop.
- [Old runs lack group keys] → keys compute at ingest; re-ingest recomputes (idempotent upsert). Documented.

## Migration Plan

Additive column + endpoint + UI view; nothing existing changes behavior. Rollback = drop the view toggle.
