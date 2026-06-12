"""Contract: the backend advertises every feature the UI build requires.

Reads ``REQUIRED_FEATURES`` straight out of the frontend source, so adding
a requirement in the UI without advertising it from the backend fails here
instead of as a 405 in a user's browser.
"""

import re
from pathlib import Path

from doctest_dashboard.server.app import API_FEATURES

APP_TSX = Path(__file__).resolve().parents[2] / "frontend" / "src" / "App.tsx"


def _ui_required_features() -> list:
    source = APP_TSX.read_text(encoding="utf-8")
    match = re.search(r"REQUIRED_FEATURES\s*=\s*\[([^\]]*)\]", source)
    assert match, "REQUIRED_FEATURES not found in App.tsx"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_backend_advertises_all_ui_required_features():
    required = _ui_required_features()
    assert required, "UI must declare its required features"
    missing = [feature for feature in required if feature not in API_FEATURES]
    assert not missing, f"Backend does not advertise UI-required features: {missing}"


def test_health_endpoint_reports_features(client):
    body = client.get("/api/health").json()
    assert body["features"] == API_FEATURES
