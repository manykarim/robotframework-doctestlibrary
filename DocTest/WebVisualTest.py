"""Visual regression testing for web apps driven by Browser Library or SeleniumLibrary.

``WebVisualTest`` extends ``VisualTest`` with page/element baseline keywords:
captures go through the web library already running in the suite, baselines are
created automatically on first run, and failing comparisons are recaptured until
``retry_timeout`` expires (stabilization against timing flake).
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from robot.api.deco import keyword
from robot.api import logger as robot_logger
from robot.utils import timestr_to_secs

from DocTest.VisualTest import VisualTest
from DocTest.WebCapture import ElementRect, WebCaptureAdapter, detect_adapter

MASK_PADDING_PX = 2

BASELINE_DIRECTORY_DEFAULT = "visual_baselines"
RETRY_TIMEOUT_DEFAULT = "3s"
RETRY_INTERVAL_DEFAULT = "500ms"
CAPTURE_SUBDIR = "doctest_web"
SPLIT_KEYS_ALLOWED = ("browser", "viewport")

_NAME_SANITIZER = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_baseline_name(name: str) -> str:
    """Reduce a baseline name to a safe filename: no separators, no traversal."""
    cleaned = _NAME_SANITIZER.sub("_", str(name)).strip("._")
    if not cleaned:
        raise ValueError(f"Baseline name '{name}' contains no usable characters.")
    return cleaned


class WebVisualTest(VisualTest):
    """Visual regression testing for web applications with automatic baseline management.

    ``WebVisualTest`` extends `VisualTest` with keywords that capture pages and
    elements through the web automation library your suite already uses —
    **Browser Library** (Playwright) or **SeleniumLibrary** are auto-detected —
    and compare them against named baselines.

    = Requirements =

    The base package is enough: this library adds **no dependencies**. Install
    one web automation library yourself (or use the ``[browser]`` /
    ``[selenium]`` convenience extras) and import it in your suite.

    = Quickstart =

    | *** Settings ***
    | Library    Browser
    | Library    DocTest.WebVisualTest    result_json=true
    |
    | *** Test Cases ***
    | Checkout Page Looks Right
    |     New Page    https://shop.example.com/checkout
    |     Compare Page To Baseline       checkout
    |     Compare Element To Baseline    id=price-table    checkout-prices

    = Baseline lifecycle =

    Baselines are plain PNG files under ``baseline_directory``
    (default ``visual_baselines``): the first run stores the capture as the
    baseline and passes with a warning; later runs compare against it. Update
    baselines with ``--variable REFERENCE_RUN:True`` or by accepting failures in
    the review dashboard (``pip install robotframework-doctestlibrary[dashboard]``).

    = Reliability toolbox =

    - ``ignore_elements=`` masks dynamic elements by locator at comparison time
    - ``retry_timeout`` recaptures failing comparisons until the page settles
    - ``ignore_antialiasing`` / ``max_diff_pixels`` tolerate cross-browser rendering noise
    - ``dom_analysis`` / ``accept_rendering_only`` accept style-only changes while
      recording them for review
    - ``split_baselines_by=browser,viewport`` keeps one baseline per configuration
    - ``llm=True`` (with the ``[ai]`` extra) asks a model to judge the difference

    Every `Compare Images` option (masks, thresholds, move tolerance, ``name=``, …)
    is accepted by all keywords. Full guide with examples:
    https://github.com/manykarim/robotframework-doctestlibrary/blob/main/docs/web-visual-testing.md
    """

    def __init__(
        self,
        baseline_directory: str = BASELINE_DIRECTORY_DEFAULT,
        web_library: Optional[str] = None,
        retry_timeout: str = RETRY_TIMEOUT_DEFAULT,
        retry_interval: str = RETRY_INTERVAL_DEFAULT,
        split_baselines_by: Optional[str] = None,
        dom_analysis: bool = False,
        accept_rendering_only: bool = False,
        **kwargs,
    ):
        """
        | =Arguments= | =Description= |
        | ``baseline_directory`` | Directory holding the named baselines. Relative paths resolve against ``${EXECDIR}``. Default ``visual_baselines``. |
        | ``web_library`` | Force a capture library (``Browser`` or ``SeleniumLibrary``) when both are imported. Auto-detected by default. |
        | ``retry_timeout`` | How long to keep recapturing after a failing comparison before giving up (Robot time string). ``0`` disables retries. Default ``3s``. |
        | ``retry_interval`` | Pause between recapture attempts. Default ``500ms``. |
        | ``split_baselines_by`` | Comma-separated configuration keys (``browser``, ``viewport``) that become baseline path segments — each configuration gets its own baseline, e.g. ``visual_baselines/chromium/1280x720/home.png``. |
        | ``dom_analysis`` | Store a semantic DOM snapshot beside each baseline and record whether the DOM changed (``identical``/``changed``/``missing-baseline``) in the sidecar on every comparison. Overridable per keyword call. |
        | ``accept_rendering_only`` | With ``dom_analysis``: pass failing visual comparisons whose DOM is unchanged (pure rendering differences) with a warning; the failure stays in the sidecar for dashboard review. Overridable per keyword call. |

        All further arguments are passed to `VisualTest` (threshold, ocr_engine,
        result_json, …).
        """
        super().__init__(**kwargs)
        self.baseline_directory = baseline_directory
        self.web_library = web_library
        self.retry_timeout = timestr_to_secs(retry_timeout)
        self.retry_interval = timestr_to_secs(retry_interval)
        self.split_baselines_by = [
            key.strip() for key in (split_baselines_by or "").split(",") if key.strip()
        ]
        self.dom_analysis = dom_analysis
        self.accept_rendering_only = accept_rendering_only
        unknown = [k for k in self.split_baselines_by if k not in SPLIT_KEYS_ALLOWED]
        if unknown:
            raise ValueError(
                f"split_baselines_by supports {', '.join(SPLIT_KEYS_ALLOWED)} — "
                f"got: {', '.join(unknown)}"
            )
        self._adapter: Optional[WebCaptureAdapter] = None

    # -- keywords ----------------------------------------------------------

    @keyword
    def compare_page_to_baseline(self, name: str, full_page: bool = True, **kwargs):
        """Captures the current page and compares it against baseline ``name``.

        With Browser Library the capture is a full-page, CSS-pixel screenshot
        (animation-free); pass ``full_page=False`` for the visible viewport.
        SeleniumLibrary supports viewport captures only (``full_page=False``).

        If the baseline does not exist yet, the capture becomes the baseline and
        the keyword passes with a warning. When ``${REFERENCE_RUN}`` is truthy the
        capture overwrites the baseline. On failure the page is recaptured and
        recompared until ``retry_timeout`` expires.

        | =Arguments= | =Description= |
        | ``name`` | Baseline name → ``{baseline_directory}/{name}.png``. |
        | ``full_page`` | Capture the full scrollable page instead of the viewport. Default ``True`` (Browser Library only). |
        | ``ignore_elements`` | Locator(s) whose live bounding boxes are ignored: a single locator, a ``;``-separated string, or a list. Masks are applied at comparison time — baselines stay clean. |
        | ``**kwargs`` | Any `Compare Images` option (``placeholder_file``, ``mask``, ``threshold``, ``move_tolerance``, …). |

        Examples:
        | `Compare Page To Baseline`    checkout
        | `Compare Page To Baseline`    home    full_page=False    threshold=0.05
        | `Compare Page To Baseline`    news    ignore_elements=id=clock;css=.ad-banner
        """
        adapter = self._get_adapter()
        self._compare_with_baseline(
            name,
            lambda path: adapter.capture_page(path, full_page=full_page),
            kwargs,
        )

    @keyword
    def compare_element_to_baseline(self, locator: str, name: str, /, **kwargs):
        """Captures the element matched by ``locator`` and compares it against baseline ``name``.

        Baseline lifecycle and retry behavior are identical to
        `Compare Page To Baseline`.

        | =Arguments= | =Description= |
        | ``locator`` | Element locator in the syntax of the running web library. |
        | ``name`` | Baseline name → ``{baseline_directory}/{name}.png``. |
        | ``ignore_elements`` | Locator(s) to ignore inside the element (masks are translated into the element's coordinate space). |
        | ``**kwargs`` | Any `Compare Images` option. |

        Example:
        | `Compare Element To Baseline`    id=price-table    prices
        """
        adapter = self._get_adapter()
        self._compare_with_baseline(
            name,
            lambda path: adapter.capture_element(path, locator),
            kwargs,
            origin_locator=locator,
        )

    @keyword
    def set_baseline_directory(self, baseline_directory: str):
        """Changes the baseline directory for subsequent comparisons.

        | =Arguments= | =Description= |
        | ``baseline_directory`` | New baseline directory. Relative paths resolve against ``${EXECDIR}``. |

        Example:
        | `Set Baseline Directory`    ${EXECDIR}/mobile_baselines
        """
        self.baseline_directory = baseline_directory

    # -- internals ---------------------------------------------------------

    def _get_adapter(self) -> WebCaptureAdapter:
        if self._adapter is None:
            self._adapter = detect_adapter(self.web_library)
        return self._adapter

    def _resolve_baseline_dir(self) -> Path:
        directory = Path(self.baseline_directory)
        if not directory.is_absolute():
            directory = Path(self._robot_variable("${EXECDIR}", ".")) / directory
        return directory

    def _capture_dir(self) -> Path:
        base = Path(self._robot_variable("${OUTPUT_DIR}", "."))
        capture_dir = base / CAPTURE_SUBDIR
        capture_dir.mkdir(parents=True, exist_ok=True)
        return capture_dir

    def _robot_variable(self, variable: str, default):
        try:
            from robot.libraries.BuiltIn import BuiltIn

            return BuiltIn().get_variable_value(variable, default)
        except Exception:
            return default

    def _parse_ignore_elements(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if str(item).strip()]
        return [part.strip() for part in str(value).split(";") if part.strip()]

    def _collect_ignore_masks(self, locators, origin_locator):
        """Live element rects → coordinate masks in capture pixels.

        Re-read on every attempt so masks track moving elements between
        recaptures. For element comparisons the compared element's origin is
        subtracted (masks live in the captured element's coordinate space)."""
        adapter = self._get_adapter()
        scale = adapter.capture_scale()
        offset_x = offset_y = 0.0
        if origin_locator is not None:
            origin = adapter.element_rects(origin_locator)
            if origin:
                offset_x, offset_y = origin[0].x, origin[0].y
        masks = []
        for locator in locators:
            rects = adapter.element_rects(locator)
            if not rects:
                robot_logger.info(
                    f"ignore_elements: no elements match '{locator}' — skipped."
                )
                continue
            for rect in rects:
                shifted = ElementRect(
                    rect.x - offset_x, rect.y - offset_y, rect.width, rect.height
                )
                masks.append(shifted.to_mask(scale=scale, padding=MASK_PADDING_PX))
        return masks

    @staticmethod
    def _truthy(value) -> bool:
        return str(value).lower() in ("true", "1", "yes")

    @staticmethod
    def _canonical_json(text: str) -> str:
        return json.dumps(json.loads(text), sort_keys=True)

    def _dom_verdict(self, dom_path: Path, current: str) -> dict:
        if not dom_path.exists():
            return {"verdict": "missing-baseline"}
        try:
            baseline_snapshot = self._canonical_json(dom_path.read_text(encoding="utf-8"))
            current_snapshot = self._canonical_json(current)
        except ValueError:
            return {"verdict": "missing-baseline"}
        if baseline_snapshot == current_snapshot:
            return {"verdict": "identical"}
        changes: list = []
        try:
            from deepdiff import DeepDiff

            diff = DeepDiff(json.loads(baseline_snapshot), json.loads(current_snapshot))
            for change_type, entries in diff.items():
                for entry in list(entries)[:5]:
                    if len(changes) >= 5:
                        break
                    changes.append(f"{change_type}: {entry}")
        except Exception:  # defensive: the summary is optional metadata
            pass
        return {"verdict": "changed", "changes": changes}

    @staticmethod
    def _merge_masks(user_mask, ignore_masks):
        if user_mask is None:
            return ignore_masks
        from DocTest.IgnoreAreaManager import IgnoreAreaManager

        parsed = IgnoreAreaManager(mask=user_mask).read_ignore_areas() or []
        if isinstance(parsed, dict):
            parsed = [parsed]
        return list(parsed) + ignore_masks

    def _stable_capture(self, capture, capture_dir: Path, safe_name: str, attempt) -> Path:
        """Capture until two consecutive captures are byte-identical (≤3 rounds).

        Full-page captures can reflow the page (scrollbar toggling changes the
        document height between the very first and later captures) — a baseline
        taken from that half-settled state would never match again."""
        paths = [
            capture_dir / f"{safe_name}-{attempt}a.png",
            capture_dir / f"{safe_name}-{attempt}b.png",
        ]
        previous = capture(paths[0])
        for round_index in range(1, 3):
            current = capture(paths[round_index % 2])
            if Path(previous).read_bytes() == Path(current).read_bytes():
                return Path(current)
            previous = current
        robot_logger.info("Capture did not stabilize after 3 rounds — using the latest.")
        return Path(previous)

    def _compare_with_baseline(
        self, name: str, capture, compare_kwargs: dict, origin_locator: Optional[str] = None
    ):
        safe_name = sanitize_baseline_name(name)
        context = self._get_adapter().describe()
        segments = [
            sanitize_baseline_name(str(context.get(key) or f"unknown-{key}"))
            for key in self.split_baselines_by
        ]
        baseline = self._resolve_baseline_dir().joinpath(*segments, f"{safe_name}.png")
        label = "/".join(segments + [safe_name])
        capture_dir = self._capture_dir()
        ignore_locators = self._parse_ignore_elements(
            compare_kwargs.pop("ignore_elements", None)
        )
        dom_analysis = self._truthy(compare_kwargs.pop("dom_analysis", self.dom_analysis))
        accept_rendering_only = self._truthy(
            compare_kwargs.pop("accept_rendering_only", self.accept_rendering_only)
        )
        dom_path = baseline.with_suffix(".dom.json")
        dom_current: Optional[str] = None
        dom_verdict: Optional[dict] = None
        if dom_analysis:
            try:
                dom_current = self._get_adapter().dom_snapshot(origin_locator)
            except Exception as exc:  # snapshot failure must not break comparison
                robot_logger.warn(f"dom_analysis: snapshot failed — {exc}")

        candidate = self._stable_capture(capture, capture_dir, safe_name, 1)

        if self.reference_run and baseline.exists():
            baseline.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, baseline)
            if dom_current is not None:
                dom_path.write_text(dom_current, encoding="utf-8")
            robot_logger.info(f"Reference run: baseline updated: {baseline}")
            return

        if not baseline.exists():
            baseline.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, baseline)
            if dom_current is not None:
                dom_path.write_text(dom_current, encoding="utf-8")
            robot_logger.warn(f"Baseline created: {baseline}")
            return

        if dom_current is not None:
            dom_verdict = self._dom_verdict(dom_path, dom_current)
            context["dom_analysis"] = dom_verdict
            robot_logger.info(f"dom_analysis: {dom_verdict['verdict']}")

        compare_kwargs.setdefault("name", label)
        compare_kwargs.setdefault("context", context)
        deadline = time.monotonic() + self.retry_timeout
        attempt = 1
        while True:
            attempt_kwargs = dict(compare_kwargs)
            if ignore_locators:
                masks = self._collect_ignore_masks(ignore_locators, origin_locator)
                if masks:
                    attempt_kwargs["mask"] = self._merge_masks(
                        attempt_kwargs.get("mask"), masks
                    )
            try:
                self.compare_images(str(baseline), str(candidate), **attempt_kwargs)
                if dom_current is not None:
                    # keep the DOM baseline aligned with the visually accepted state
                    dom_path.write_text(dom_current, encoding="utf-8")
                return
            except AssertionError:
                if time.monotonic() >= deadline:
                    if (
                        accept_rendering_only
                        and dom_verdict is not None
                        and dom_verdict.get("verdict") == "identical"
                    ):
                        robot_logger.warn(
                            "Rendering-only difference accepted (DOM unchanged) — "
                            "the visual failure is recorded for dashboard review."
                        )
                        return
                    raise
                time.sleep(self.retry_interval)
                attempt += 1
                robot_logger.info(
                    f"Comparison failed — recapturing (attempt {attempt})."
                )
                candidate = self._stable_capture(capture, capture_dir, safe_name, attempt)
