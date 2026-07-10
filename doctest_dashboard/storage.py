"""Storage garbage collection for the dashboard workspace.

Engine scratch results and browser uploads accumulate over time; both are
reproducible (recompare re-runs, uploads re-upload), so age-based sweeping
is safe. Runs and their ingested artifacts are NEVER auto-deleted — run
deletion is an explicit user action.
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Dict

from doctest_dashboard.config import AppConfig

LOG = logging.getLogger(__name__)


def sweep_directory(root: Path, ttl_days: float, now: float = None) -> int:
    """Remove direct children of ``root`` older than ``ttl_days``.

    Returns the number of entries removed. Missing roots are a no-op.
    """
    if not root.is_dir():
        return 0
    cutoff = (now if now is not None else time.time()) - ttl_days * 86400
    removed = 0
    for child in root.iterdir():
        try:
            if child.stat().st_mtime >= cutoff:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            removed += 1
        except OSError:
            continue
    return removed


def run_gc(config: AppConfig, now: float = None) -> Dict[str, int]:
    """Sweep aged scratch and upload entries. Called at server startup."""
    return {
        "scratch": sweep_directory(
            config.data_dir / "scratch", config.scratch_ttl_days, now=now),
        "uploads": sweep_directory(
            config.data_dir / "uploads", config.uploads_ttl_days, now=now),
    }
