# Spec: dashboard-lifecycle

## ADDED Requirements

### Requirement: Paginated, aggregate list queries
List endpoints SHALL accept `limit`/`offset`, return total counts, apply status/review-state filters in SQL, and SHALL NOT issue per-row queries.

#### Scenario: Large run grid page
- **WHEN** a run with many comparisons is listed with `limit=50&offset=50`
- **THEN** the response contains at most 50 rows, the total count, and the server executed a constant number of queries

#### Scenario: Filters in SQL
- **WHEN** the grid is filtered by `status=fail&review_state=unresolved`
- **THEN** only matching rows are returned and counted

### Requirement: Run deletion
`DELETE /api/runs/{id}` SHALL remove the run with its tests, comparisons, pages, and asset registrations, and the UI SHALL offer deletion with confirmation. Decisions history remains queryable (decision rows survive with null references).

#### Scenario: Delete cascades
- **WHEN** a run is deleted
- **THEN** its comparisons and pages are gone, its asset tokens no longer resolve, and other runs are untouched

### Requirement: Storage garbage collection and bounded caches
Engine scratch directories and uploads older than configurable TTLs (defaults: scratch 7 days, uploads 30 days) SHALL be swept at server startup; the engine result cache SHALL be bounded (LRU). Runs SHALL never be auto-deleted.

#### Scenario: Old scratch swept
- **WHEN** the server starts with a scratch directory older than the TTL present
- **THEN** it is removed, while recent scratch directories remain

#### Scenario: Cache stays bounded
- **WHEN** more distinct engine requests than the cache limit are served
- **THEN** memory does not grow beyond the limit and oldest entries are evicted
