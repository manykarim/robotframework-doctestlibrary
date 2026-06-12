"""masks.json I/O: schema-exact load/save with the library's own parser.

Normalization goes through ``DocTest.IgnoreAreaManager`` so the dashboard
accepts exactly what the library accepts (JSON file, inline list/dict,
JSON string, or the ``top:10;bottom:10`` shorthand) — and never invents a
schema of its own. Exports are pretty-printed JSON with stable key order
to keep git diffs clean.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, List

from DocTest.IgnoreAreaManager import IgnoreAreaManager

# Canonical key order for exported mask entries (extras keep insertion order)
KEY_ORDER = [
    "page", "name", "type",
    "x", "y", "width", "height", "unit",
    "location", "percent",
    "pattern", "xoffset", "yoffset",
]


class MaskError(Exception):
    pass


PATTERN_TYPES = {"pattern", "line_pattern", "word_pattern"}


def validate_pattern_masks(masks: List[dict]) -> None:
    """Reject pattern masks whose regex does not compile.

    Called on every input path that feeds masks into the comparison engine
    (preview, recompare, save) — the library itself compiles patterns lazily
    per text token and would raise deep inside a worker process otherwise.
    Intentionally NOT applied when loading existing files, so a file with a
    broken pattern can still be opened and repaired in the editor.
    """
    import re as re_module

    for entry in masks:
        if not isinstance(entry, dict) or entry.get("type") not in PATTERN_TYPES:
            continue
        pattern = entry.get("pattern")
        if not pattern:
            raise MaskError("Pattern mask has an empty 'pattern'")
        try:
            re_module.compile(pattern)
        except re_module.error as error:
            raise MaskError(f"Invalid regular expression {pattern!r}: {error}")


def normalize_masks(raw: Any) -> List[dict]:
    """Normalize any library-accepted mask input into a list of dicts."""
    if raw is None:
        return []
    masks = IgnoreAreaManager(mask=raw).read_ignore_areas()
    if not isinstance(masks, list):
        raise MaskError("Mask input did not normalize to a list")
    for entry in masks:
        if not isinstance(entry, dict):
            raise MaskError(f"Mask entry is not an object: {entry!r}")
        if "type" not in entry:
            raise MaskError(f"Mask entry has no 'type': {entry!r}")
    return masks


def load_mask_file(path: Path) -> List[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path}")
    masks = IgnoreAreaManager(ignore_area_file=str(path)).read_ignore_areas()
    return masks if isinstance(masks, list) else [masks]


def _ordered(entry: dict) -> dict:
    ordered = {key: entry[key] for key in KEY_ORDER if key in entry}
    ordered.update({key: value for key, value in entry.items() if key not in ordered})
    return ordered


def dumps_masks(masks: List[dict]) -> str:
    return json.dumps([_ordered(entry) for entry in masks], indent=4) + "\n"


def save_mask_file(path: Path, masks: List[dict]) -> str:
    """Atomically write masks (temp + rename), keeping a ``.bak`` of the
    previous content. Returns the SHA-256 of the new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = dumps_masks(masks)
    if path.exists():
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
