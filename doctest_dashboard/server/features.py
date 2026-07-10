"""Capabilities of this backend build, advertised via /api/health.

The UI checks them at load time and tells the user to restart the server
when it is older than the served frontend (static files are re-read from
disk, the Python process is not).
"""

API_FEATURES = [
    "ingest", "review", "masks", "engine", "recompare", "browse", "upload",
    "upload-results", "batch-accept", "lifecycle", "diff-groups", "root-cause", "history",
]
