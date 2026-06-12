"""Application configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8008
DATA_DIR_NAME = ".doctest_dashboard"


@dataclass
class AppConfig:
    """Runtime configuration for the dashboard server.

    ``roots`` is the allowlist of directories the server may read assets
    from and write baselines/masks into. Every filesystem access is
    validated against it; ingesting an output.xml adds its directory
    automatically.
    """

    data_dir: Path = field(default_factory=lambda: Path.cwd() / DATA_DIR_NAME)
    roots: List[Path] = field(default_factory=list)
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token: Optional[str] = None

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).resolve()
        self.roots = [Path(root).resolve() for root in self.roots]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "dashboard.db"

    def add_root(self, root: Path) -> Path:
        resolved = Path(root).resolve()
        if resolved not in self.roots:
            self.roots.append(resolved)
        return resolved

    def is_within_roots(self, path: Path) -> bool:
        """True if ``path`` (fully resolved, symlinks included) is under a root."""
        try:
            resolved = Path(path).resolve(strict=True)  # NOSONAR: paths are confined to configured roots (config.is_within_roots, symlink-safe resolve) and covered by traversal tests
        except OSError:
            return False
        return any(resolved.is_relative_to(root) for root in self.roots)
