"""Post-execution ingestion of Robot Framework ``output.xml`` files.

Walks the result model with a ``ResultVisitor``, extracting DocTest
comparison keywords at *keyword level* (a failing comparison wrapped in
``Run Keyword And Expect Error`` passes at test level — keyword status is
the truth). Sidecar JSON referenced by a ``DOCTEST_RESULT:`` message is the
preferred source; without it, ``<img>`` references are scraped from HTML
log messages and the comparison is stored as *degraded*.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from robot.api import ExecutionResult, ResultVisitor

from doctest_dashboard.config import AppConfig
from doctest_dashboard.db import Database
from doctest_dashboard.models.sidecar import parse_sidecar

LOG = logging.getLogger(__name__)

DOCTEST_LIBRARIES = {"DocTest.VisualTest", "DocTest.PdfTest"}
RESULT_MESSAGE_RE = re.compile(r"DOCTEST_RESULT:\s*(\S+)")
IMG_SRC_RE = re.compile(r'<img[^>]+src="([^"]+)"')


@dataclass
class IngestSummary:
    run_id: int
    tests: int
    comparisons: int
    sidecar_comparisons: int
    degraded_comparisons: int


def diff_dhash(image_path) -> Optional[str]:
    """64-bit gradient (difference) hash of an image — deterministic and
    robust to PNG encoder differences. Used for similarity grouping."""
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(small[row, col] > small[row, col + 1])
    return f"{bits:016x}"


def comparison_group_key(sidecar_data, base_dir) -> Optional[str]:
    """Deterministic similarity key: per failing page, the sorted diff-region
    sizes plus the diff image's dHash. Identical differences across documents
    yield identical keys (Percy-style strict matching)."""
    page_keys = []
    for page in sidecar_data.pages:
        if page.status != "FAIL":
            continue
        sizes = sorted((region.width, region.height) for region in page.diff_regions)
        diff_rel = page.images.get("diff")
        image_hash = diff_dhash(base_dir / diff_rel) if diff_rel else None
        if image_hash is None:
            return None  # without pixels the key would be too weak to batch on
        page_keys.append(f"{sizes}|{image_hash}")
    if not page_keys:
        return None
    digest = hashlib.sha256("\n".join(page_keys).encode("utf-8")).hexdigest()
    return digest[:16]


def asset_token(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:32]


def _file_sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _iter_messages(item):
    for child in getattr(item, "body", []):
        if child.__class__.__name__ == "Message":
            yield child
        else:
            yield from _iter_messages(child)


def _keyword_owner(keyword) -> Optional[str]:
    owner = getattr(keyword, "owner", None) or getattr(keyword, "libname", None)
    return str(owner) if owner else None


class _ComparisonCollector(ResultVisitor):
    def __init__(self, database: Database, config: AppConfig, run_id: int, base_dir: Path):
        self.database = database
        self.config = config
        self.run_id = run_id
        self.base_dir = base_dir
        self.tests = 0
        self.comparisons = 0
        self.sidecar_comparisons = 0
        self.degraded_comparisons = 0

    def visit_test(self, test):
        self.tests += 1
        test_id = self.database.insert_test(
            run_id=self.run_id,
            suite=test.parent.longname if test.parent else "",
            name=test.name,
            status=test.status,
            message=test.message or "",
        )
        index = 0
        for keyword in self._find_comparison_keywords(test):
            index += 1
            identity = f"{test.longname}::{keyword.name}::{index}"
            self._store_comparison(test_id, keyword, identity)

    def _find_comparison_keywords(self, item) -> List[Any]:
        found = []
        for child in getattr(item, "body", []):
            if child.__class__.__name__ == "Message":
                continue
            is_keyword = child.__class__.__name__ == "Keyword"
            owner = _keyword_owner(child) if is_keyword else None
            if owner in DOCTEST_LIBRARIES and str(
                getattr(child, "name", "") or getattr(child, "kwname", "")
            ).lower().startswith("compare"):
                found.append(child)
            else:
                found.extend(self._find_comparison_keywords(child))
        return found

    def _store_comparison(self, test_id: int, keyword, identity: str) -> None:
        self.comparisons += 1
        name = str(getattr(keyword, "name", "") or getattr(keyword, "kwname", ""))
        owner = _keyword_owner(keyword)
        status = keyword.status

        sidecar_rel = None
        images: List[str] = []
        for message in _iter_messages(keyword):
            match = RESULT_MESSAGE_RE.search(message.message or "")
            if match:
                sidecar_rel = match.group(1)
            for src in IMG_SRC_RE.findall(message.message or ""):
                images.append(src)

        sidecar_data = None
        if sidecar_rel:
            sidecar_path = (self.base_dir / sidecar_rel).resolve()
            try:
                with open(sidecar_path, encoding="utf-8") as file:
                    sidecar_data = parse_sidecar(json.load(file))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                LOG.warning("Cannot load sidecar %s: %s", sidecar_path, error)
                sidecar_data = None

        if sidecar_data is not None:
            group_key = (
                comparison_group_key(sidecar_data, self.base_dir)
                if status == "FAIL" else None
            )
            label = getattr(sidecar_data, "name", None)
            if label:
                # user-assigned comparison labels survive test renames
                identity = f"name::{label}"
            comparison_id = self.database.insert_comparison(
                test_id=test_id,
                keyword=name,
                library=owner,
                status=status,
                degraded=False,
                identity=identity,
                sidecar_path=str((self.base_dir / sidecar_rel).resolve()),
                sidecar_json=sidecar_data.model_dump(),
                reference_path=sidecar_data.reference.path,
                candidate_path=sidecar_data.candidate.path,
                group_key=group_key,
                label=label,
            )
            self.sidecar_comparisons += 1
            for page in sidecar_data.pages:
                page_images: Dict[str, str] = {}
                content_parts = []
                for kind, rel in page.images.items():
                    absolute = (self.base_dir / rel).resolve()
                    token = asset_token(str(absolute))
                    self.database.register_asset(token, str(absolute))
                    page_images[kind] = token
                    if kind == "candidate":
                        content_parts.append(_file_sha256(absolute) or rel)
                content_key = content_parts[0] if content_parts else None
                page_no = page.page if isinstance(page.page, int) else 0
                page_id = self.database.insert_page(
                    comparison_id=comparison_id,
                    page_no=page_no,
                    status=page.status,
                    score=page.score,
                    threshold=page.threshold,
                    regions=[region.model_dump() for region in page.diff_regions],
                    images=page_images,
                    content_key=content_key,
                )
                self._inherit_page_state(page_id, identity, page_no, page.status, content_key)
        else:
            tokens = []
            for rel in images:
                absolute = (self.base_dir / rel).resolve()
                token = asset_token(str(absolute))
                self.database.register_asset(token, str(absolute))
                tokens.append(token)
            self.database.insert_comparison(
                test_id=test_id,
                keyword=name,
                library=owner,
                status=status,
                degraded=True,
                identity=identity,
                images=tokens,
            )
            self.degraded_comparisons += 1

    def _inherit_page_state(self, page_id: int, identity: str, page_no: int,
                            status: str, content_key: Optional[str]) -> None:
        """Keep an accepted/rejected state only while the page content is unchanged."""
        if status != "FAIL" or not content_key:
            return
        previous = self.database.query_one(
            "SELECT p.review_state, p.content_key FROM pages p "
            "JOIN comparisons c ON p.comparison_id = c.id "
            "WHERE c.identity = ? AND p.page_no = ? AND p.id != ? "
            "ORDER BY p.id DESC LIMIT 1",
            (identity, page_no, page_id))
        if (
            previous
            and previous["review_state"] in ("accepted", "rejected")
            and previous["content_key"] == content_key
        ):
            self.database.set_page_state(page_id, previous["review_state"])


def ingest_output_xml(database: Database, config: AppConfig, output_xml) -> IngestSummary:
    output_xml = Path(output_xml).resolve()  # NOSONAR: paths are confined to configured roots (config.is_within_roots, symlink-safe resolve) and covered by traversal tests
    if not output_xml.is_file():
        raise FileNotFoundError(f"output.xml not found: {output_xml}")
    base_dir = output_xml.parent
    config.add_root(base_dir)

    result = ExecutionResult(str(output_xml))
    run_id = database.upsert_run(
        output_xml_path=str(output_xml),
        name=f"{result.suite.name} · {base_dir.name}",
        started=getattr(result.suite, "starttime", None) or None,
        rf_version=getattr(result, "generator", None),
    )
    collector = _ComparisonCollector(database, config, run_id, base_dir)
    result.visit(collector)
    return IngestSummary(
        run_id=run_id,
        tests=collector.tests,
        comparisons=collector.comparisons,
        sidecar_comparisons=collector.sidecar_comparisons,
        degraded_comparisons=collector.degraded_comparisons,
    )
