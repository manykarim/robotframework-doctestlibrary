import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
RUST_MANIFEST = ROOT / "rust" / "ocrs_py" / "Cargo.toml"
WHEELS_DIR = ROOT / "rust" / "ocrs_py" / "target" / "wheels"
TARGET_PACKAGE_DIR = ROOT / "DocTest" / "_ocrs"


def build(setup_kwargs):
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--release",
            "--manifest-path",
            str(RUST_MANIFEST),
            "--interpreter",
            sys.executable,
        ],
        cwd=ROOT,
    )

    wheel = max(WHEELS_DIR.glob("ocrs_py-*.whl"), key=lambda path: path.stat().st_mtime)

    if TARGET_PACKAGE_DIR.exists():
        shutil.rmtree(TARGET_PACKAGE_DIR)
    TARGET_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    extensions = {".so", ".pyd", ".dylib"}
    copied = False
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            suffix = Path(member).suffix
            if suffix not in extensions:
                continue
            name = Path(member).name
            with archive.open(member) as src, open(TARGET_PACKAGE_DIR / name, "wb") as dst:
                dst.write(src.read())
            copied = True

    if not copied:
        raise RuntimeError("maturin wheel did not contain a shared library")

    init_code = (
        '"""Load the compiled OCRS extension bundled with DocTest."""\n'
        'from importlib import import_module as _import_module\n'
        'import importlib.util as _importlib_util\n'
        '# Prefer the bundled shared library, fallback to globally installed module.\n'
        '_spec = _importlib_util.find_spec("DocTest._ocrs._ocrs")\n'
        'if _spec is not None:\n'
        '    module = _import_module("DocTest._ocrs._ocrs")\n'
        'else:\n'
        '    module = _import_module("_ocrs")\n'
        'globals().update(module.__dict__)\n'
    )
    (TARGET_PACKAGE_DIR / "__init__.py").write_text(init_code)

    setup_kwargs.setdefault("packages", []).append("DocTest._ocrs")
