"""Capture adapters bridging web automation libraries to the comparison engine.

Adapters talk to Browser Library / SeleniumLibrary exclusively through
``BuiltIn.run_keyword`` so this module never imports either package — the
user's own installation is the only dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ElementRect:
    """Element bounding box in CSS pixels (fractional, as reported by browsers)."""

    x: float
    y: float
    width: float
    height: float

    def to_mask(self, scale: float = 1.0, padding: int = 0) -> Dict[str, Any]:
        """Convert to a coordinate ignore mask, rounding outward so the
        fractional rect is always fully covered. ``scale`` maps CSS pixels to
        screenshot pixels (device pixel ratio for device-scaled captures)."""
        left = max(0, math.floor(self.x * scale) - padding)
        top = max(0, math.floor(self.y * scale) - padding)
        right = math.ceil((self.x + self.width) * scale) + padding
        bottom = math.ceil((self.y + self.height) * scale) + padding
        return {
            "page": "all",
            "type": "coordinates",
            "unit": "px",
            "x": left,
            "y": top,
            "width": max(1, right - left),
            "height": max(1, bottom - top),
        }


# Semantic DOM walker: visible elements only, semantic attributes, normalized
# text. Verified byte-identical across chromium and firefox for the same page.
DOM_WALKER_JS = """(root) => {
  const skip = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE']);
  const walk = (node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent.replace(/\\s+/g, ' ').trim();
      return text ? text : null;
    }
    if (node.nodeType !== Node.ELEMENT_NODE || skip.has(node.tagName)) return null;
    const style = getComputedStyle(node);
    if (style.display === 'none' || style.visibility === 'hidden') return null;
    const entry = { tag: node.tagName.toLowerCase() };
    const role = node.getAttribute('role'); if (role) entry.role = role;
    const label = node.getAttribute('aria-label'); if (label) entry.label = label;
    if (['INPUT','TEXTAREA','SELECT'].includes(node.tagName)) entry.value = String(node.value);
    const href = node.getAttribute('href'); if (href) entry.href = href;
    const src = node.getAttribute('src'); if (src) entry.src = src;
    const alt = node.getAttribute('alt'); if (alt) entry.alt = alt;
    const children = [];
    for (const child of node.childNodes) {
      const result = walk(child);
      if (result !== null) children.push(result);
    }
    if (children.length) entry.children = children;
    return entry;
  };
  return JSON.stringify(walk(root || document.body));
}"""


class WebCaptureAdapter:
    """Protocol for capturing screenshots through a running web library."""

    library_name: str = ""

    def capture_page(self, path: Path, full_page: bool = True) -> Path:
        raise NotImplementedError

    def capture_element(self, path: Path, locator: str) -> Path:
        raise NotImplementedError

    def element_rects(self, locator: str) -> List[ElementRect]:
        raise NotImplementedError

    def device_pixel_ratio(self) -> float:
        raise NotImplementedError

    def capture_scale(self) -> float:
        """CSS-pixel → screenshot-pixel factor for page captures."""
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        """Best-effort capture context (never raises)."""
        return {"library": self.library_name}

    def dom_snapshot(self, locator: Optional[str] = None) -> str:
        """JSON string of the semantic DOM tree (page body or one element)."""
        raise NotImplementedError


class BrowserLibraryAdapter(WebCaptureAdapter):
    """Browser Library (Playwright). Page captures use ``scale=css`` so
    baselines and element rects are DPR-independent (verified by experiment:
    fullPage at deviceScaleFactor=2 renders identical pixels with scale=css)."""

    library_name = "Browser"

    def __init__(self, builtin):
        self._builtin = builtin

    def _run(self, keyword: str, *args):
        return self._builtin.run_keyword(f"Browser.{keyword}", *args)

    def capture_page(self, path: Path, full_page: bool = True) -> Path:
        # Take Screenshot appends the extension itself → pass without suffix
        result = self._run(
            "Take Screenshot",
            f"filename={path.with_suffix('')}",
            f"fullPage={full_page}",
            "scale=css",
            "disableAnimations=True",
            "log_screenshot=False",
        )
        return Path(str(result))

    def capture_element(self, path: Path, locator: str) -> Path:
        result = self._run(
            "Take Screenshot",
            f"filename={path.with_suffix('')}",
            f"selector={locator}",
            "scale=css",
            "disableAnimations=True",
            "log_screenshot=False",
        )
        return Path(str(result))

    def element_rects(self, locator: str) -> List[ElementRect]:
        elements = self._run("Get Elements", f"selector={locator}")
        rects = []
        for element in elements:
            box = self._run("Get BoundingBox", f"selector={element}")
            rects.append(ElementRect(box["x"], box["y"], box["width"], box["height"]))
        return rects

    def device_pixel_ratio(self) -> float:
        return float(self._run("Evaluate JavaScript", None, "() => window.devicePixelRatio"))

    def capture_scale(self) -> float:
        return 1.0  # scale=css captures are already in CSS pixels

    def dom_snapshot(self, locator: Optional[str] = None) -> str:
        # with a selector, Evaluate JavaScript hands the element to the function
        return str(self._run("Evaluate JavaScript", locator, DOM_WALKER_JS))

    def describe(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {"library": self.library_name}
        try:
            context["url"] = str(self._run("Get Url"))
            viewport = self._run("Get Viewport Size")
            context["viewport"] = f"{viewport['width']}x{viewport['height']}"
            context["device_pixel_ratio"] = self.device_pixel_ratio()
            catalog = self._run("Get Browser Catalog")
            active = [b for b in catalog if b.get("activeBrowser")]
            if active:
                context["browser"] = active[0].get("type")
        except Exception:  # defensive: context is metadata, never a failure source
            pass
        return context


class SeleniumLibraryAdapter(WebCaptureAdapter):
    """SeleniumLibrary. Captures are device pixels; rects are CSS pixels →
    masks must scale by the session's devicePixelRatio."""

    library_name = "SeleniumLibrary"

    def __init__(self, builtin):
        self._builtin = builtin

    def _run(self, keyword: str, *args):
        return self._builtin.run_keyword(f"SeleniumLibrary.{keyword}", *args)

    def capture_page(self, path: Path, full_page: bool = True) -> Path:
        if full_page:
            raise RuntimeError(
                "SeleniumLibrary cannot capture full-page screenshots; "
                "call with full_page=False to compare the visible viewport, "
                "or use Browser Library for full-page comparisons."
            )
        result = self._run("Capture Page Screenshot", f"filename={path}")
        return Path(str(result))

    def capture_element(self, path: Path, locator: str) -> Path:
        result = self._run(
            "Capture Element Screenshot", f"locator={locator}", f"filename={path}"
        )
        return Path(str(result))

    def element_rects(self, locator: str) -> List[ElementRect]:
        elements = self._run("Get WebElements", f"locator={locator}")
        return [
            ElementRect(rect["x"], rect["y"], rect["width"], rect["height"])
            for rect in (element.rect for element in elements)
        ]

    def device_pixel_ratio(self) -> float:
        return float(self._run("Execute Javascript", "return window.devicePixelRatio"))

    def capture_scale(self) -> float:
        return self.device_pixel_ratio()

    def dom_snapshot(self, locator: Optional[str] = None) -> str:
        script = f"return ({DOM_WALKER_JS})(arguments[0]);"
        if locator is None:
            return str(self._run("Execute Javascript", script))
        element = self._run("Get WebElement", f"locator={locator}")
        return str(self._run("Execute Javascript", script, "ARGUMENTS", element))

    def describe(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {"library": self.library_name}
        try:
            context["url"] = str(self._run("Get Location"))
            size = self._run(
                "Execute Javascript",
                "return [window.innerWidth, window.innerHeight]",
            )
            context["viewport"] = f"{size[0]}x{size[1]}"
            context["device_pixel_ratio"] = self.device_pixel_ratio()
            instance = self._builtin.get_library_instance("SeleniumLibrary")
            context["browser"] = instance.driver.capabilities.get("browserName")
        except Exception:  # defensive: context is metadata, never a failure source
            pass
        return context


_ADAPTERS = {
    "Browser": BrowserLibraryAdapter,
    "SeleniumLibrary": SeleniumLibraryAdapter,
}


def detect_adapter(web_library: Optional[str] = None, builtin=None) -> WebCaptureAdapter:
    """Return the adapter for the web library running in this suite.

    ``web_library`` forces a specific one (when both are imported); otherwise
    Browser is probed first, then SeleniumLibrary.
    """
    if builtin is None:
        from robot.libraries.BuiltIn import BuiltIn

        builtin = BuiltIn()
    if web_library is not None and web_library not in _ADAPTERS:
        raise RuntimeError(
            f"Unsupported web_library '{web_library}'. "
            f"Supported: {', '.join(_ADAPTERS)}."
        )
    candidates = [web_library] if web_library else list(_ADAPTERS)
    for name in candidates:
        try:
            builtin.get_library_instance(name)
        except Exception:
            continue
        return _ADAPTERS[name](builtin)
    raise RuntimeError(
        "No supported web automation library is loaded. Import Browser "
        "(robotframework-browser) or SeleniumLibrary in your suite before "
        "using web comparison keywords."
    )
