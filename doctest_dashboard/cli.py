"""Command line interface: ``doctest-dashboard serve|ingest``."""

import argparse
import sys
from pathlib import Path

from doctest_dashboard import __version__
from doctest_dashboard.config import DEFAULT_HOST, DEFAULT_PORT, AppConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doctest-dashboard",
        description="Visual review dashboard for robotframework-doctestlibrary",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory for the dashboard database (default: ./.doctest_dashboard)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the dashboard server")
    serve.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--token", default=None, help="Require this bearer token on all API calls")
    serve.add_argument(
        "--root",
        action="append",
        type=Path,
        default=[],
        dest="roots",
        help="Allow asset/baseline/mask access under this directory (repeatable)",
    )
    serve.add_argument("--reload", action="store_true", help="Auto-reload (development)")

    ingest = subparsers.add_parser("ingest", help="Ingest a Robot Framework output.xml")
    ingest.add_argument("output_xml", type=Path)

    gate = subparsers.add_parser(
        "gate",
        help="CI gate: exit 0 when a run has no unresolved failed comparisons")
    gate.add_argument(
        "run", help="Run id, or 'latest' for the most recently imported run")

    return parser


def _require_dashboard_dependencies() -> bool:
    """The console script is installed with every variant of the package;
    the server/ingest dependencies only come with the [dashboard] extra."""
    try:
        import fastapi  # noqa: F401
        import pydantic  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        print(
            "The dashboard dependencies are not installed.\n"
            "Install them with:\n\n"
            "    pip install robotframework-doctestlibrary[dashboard]\n",
            file=sys.stderr,
        )
        return False
    return True


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir or Path.cwd() / ".doctest_dashboard"

    if args.command == "gate":
        # database-only: works on a base install without the [dashboard] extra
        return _gate(data_dir, args.run)

    if not _require_dashboard_dependencies():
        return 3

    if args.command == "serve":
        import uvicorn

        from doctest_dashboard.server.app import create_app

        config = AppConfig(
            data_dir=data_dir,
            roots=args.roots,
            host=args.host,
            port=args.port,
            token=args.token,
        )
        app = create_app(config)
        uvicorn.run(app, host=config.host, port=config.port)
        return 0

    if args.command == "ingest":
        from doctest_dashboard.db import Database
        from doctest_dashboard.ingest import ingest_output_xml

        config = AppConfig(data_dir=data_dir)
        config.data_dir.mkdir(parents=True, exist_ok=True)
        database = Database(config.db_path)
        try:
            summary = ingest_output_xml(database, config, args.output_xml)
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(
            f"Ingested run {summary.run_id}: {summary.tests} tests, "
            f"{summary.comparisons} comparisons "
            f"({summary.sidecar_comparisons} with sidecar, {summary.degraded_comparisons} degraded)"
        )
        return 0

    return 1


def _gate(data_dir: Path, run_spec: str) -> int:
    """Exit 0 when the run has no unresolved failed comparisons, 1 when it
    does, 2 when the run cannot be found. Reads the database directly."""
    from doctest_dashboard.db import Database

    db_path = data_dir / "dashboard.db"
    if not db_path.exists():
        print(f"error: no dashboard database at {db_path}", file=sys.stderr)
        return 2
    database = Database(db_path)
    try:
        if run_spec == "latest":
            row = database.query_one(
                "SELECT id, name FROM runs ORDER BY imported_at DESC, id DESC LIMIT 1")
        else:
            try:
                row = database.query_one(
                    "SELECT id, name FROM runs WHERE id = ?", (int(run_spec),))
            except ValueError:
                row = None
        if not row:
            print(f"error: run '{run_spec}' not found", file=sys.stderr)
            return 2
        unresolved = database.query(
            "SELECT c.id, t.name FROM comparisons c JOIN tests t ON c.test_id = t.id "
            "WHERE t.run_id = ? AND c.status = 'FAIL' AND c.review_state = 'unresolved' "
            "ORDER BY c.id", (row["id"],))
        if unresolved:
            print(f"GATE FAILED: run {row['id']} ({row['name']}) has "
                  f"{len(unresolved)} unreviewed failed comparison(s):")
            for item in unresolved:
                print(f"  - #{item['id']} {item['name']}")
            print("Review them in the dashboard (accept or reject), then re-run the gate.")
            return 1
        print(f"GATE PASSED: run {row['id']} ({row['name']}) has no "
              "unresolved failed comparisons.")
        return 0
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
