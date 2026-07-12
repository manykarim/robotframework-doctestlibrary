"""Public keyword-surface baseline for backwards-compatibility gating.

Records keyword names, argument names, kinds and defaults of the shipped Robot
Framework libraries via robot's own LibraryDocumentation — exactly the surface
users see. ``utest/test_keyword_surface.py`` fails the build when anything in
the baseline disappears or changes; additions are always allowed.

Regenerate after an intentional, reviewed surface change:

    uv run python scripts/update_keyword_surface.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

BASELINE_PATH = Path(__file__).parent / "keyword_surface_baseline.json"

LIBRARIES = [
    "DocTest.VisualTest",
    "DocTest.PdfTest",
    "DocTest.PrintJobTests",
    "DocTest.WebVisualTest",
]


def build_surface() -> Dict[str, Dict[str, list]]:
    from robot.libdocpkg import LibraryDocumentation

    surface: Dict[str, Dict[str, list]] = {}
    for library in LIBRARIES:
        doc = LibraryDocumentation(library)
        keywords = {}
        for keyword in doc.keywords:
            keywords[keyword.name] = [
                {
                    "name": arg.name,
                    "kind": str(arg.kind),
                    "required": bool(arg.required),
                    "default": arg.default_repr,
                }
                for arg in keyword.args
            ]
        surface[library] = keywords
    return surface


def compare_surfaces(baseline: dict, current: dict) -> List[str]:
    """Violations of backwards compatibility: anything baselined that is
    missing or changed in the current surface. Additions never appear here."""
    violations = []
    for library, keywords in baseline.items():
        current_keywords = current.get(library)
        if current_keywords is None:
            violations.append(f"{library}: library missing from surface")
            continue
        for keyword_name, args in keywords.items():
            current_args = current_keywords.get(keyword_name)
            if current_args is None:
                violations.append(f"{library}: keyword '{keyword_name}' removed")
                continue
            current_by_name = {a["name"]: a for a in current_args}
            for arg in args:
                current_arg = current_by_name.get(arg["name"])
                if current_arg is None:
                    violations.append(
                        f"{library}: '{keyword_name}' lost argument '{arg['name']}'"
                    )
                    continue
                for field in ("kind", "required", "default"):
                    if current_arg[field] != arg[field]:
                        violations.append(
                            f"{library}: '{keyword_name}' argument '{arg['name']}' "
                            f"changed {field}: {arg[field]!r} → {current_arg[field]!r}"
                        )
    return violations


def main() -> None:
    surface = build_surface()
    BASELINE_PATH.write_text(json.dumps(surface, indent=2, sort_keys=True) + "\n")
    total = sum(len(k) for k in surface.values())
    print(f"Baseline written: {BASELINE_PATH} ({total} keywords, {len(surface)} libraries)")


if __name__ == "__main__":
    main()
