"""Root-confined file browsing for the UI's file picker."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from doctest_dashboard.server.state import AppState, get_state, require_token

router = APIRouter(dependencies=[Depends(require_token)])


@router.get("/api/browse")
def browse(path: Optional[str] = None, state: AppState = Depends(get_state)):
    """List configured roots, or the contents of a directory under them.

    Powers the file picker in the UI; every path is validated against
    the configured roots like any other filesystem access.
    """
    config = state.config
    if not path:
        # Hide the internal recompare scratch area; the uploads workspace
        # stays visible so uploaded files can be browsed again.
        visible = [root for root in config.roots
                   if root != config.data_dir / "scratch"]
        return {"roots": [{"name": root.name or str(root), "path": str(root)}
                          for root in visible]}
    directory = Path(path)  # NOSONAR: paths are confined to configured roots (config.is_within_roots, symlink-safe resolve) and covered by traversal tests
    if not config.is_within_roots(directory):
        raise HTTPException(status_code=403, detail="Path outside configured roots")
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Not a directory")
    entries = []
    try:
        children = sorted(
            directory.iterdir(),
            key=lambda child: (not child.is_dir(), child.name.lower()))
    except OSError as error:
        raise HTTPException(status_code=400, detail=str(error))
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
            entries.append({
                "name": child.name,
                "path": str(child),
                "type": "dir" if is_dir else "file",
                "size": None if is_dir else child.stat().st_size,
            })
        except OSError:
            continue
    parent = directory.parent
    return {
        "path": str(directory),
        "parent": str(parent) if config.is_within_roots(parent) else None,
        "entries": entries,
    }
