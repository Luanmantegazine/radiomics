"""Schema migrations (Alembic), applied programmatically at start-up.

    python -m backend.app.migrations upgrade          # apply pending migrations
    python -m backend.app.migrations revision -m msg  # autogenerate a new one
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, inspect

SCRIPT_LOCATION = Path(__file__).resolve().parents[1] / "migrations"


class UnversionedDatabase(RuntimeError):
    """Tables exist but were not created by these migrations."""


def _config(connection) -> Config:
    config = Config()
    config.set_main_option("script_location", str(SCRIPT_LOCATION))
    config.attributes["connection"] = connection
    return config


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def upgrade(engine: Engine, revision: str = "head") -> None:
    """Bring the database schema to ``revision``.

    Refuses to touch a database that has tables but no migration history (for
    example a development database created before Stage 7 with ``create_all``),
    because its schema cannot be trusted to match any revision.
    """
    tables = set(inspect(engine).get_table_names())
    if tables and "alembic_version" not in tables:
        raise UnversionedDatabase(
            "the database has tables but no migration history; recreate it (development "
            "databases) or stamp it after checking its schema: "
            "python -m backend.app.migrations stamp head"
        )
    with engine.begin() as connection:
        command.upgrade(_config(connection), revision)


def main(argv: list[str] | None = None) -> int:
    from backend.app.db import make_engine
    from backend.app.settings import get_settings

    parser = argparse.ArgumentParser(prog="python -m backend.app.migrations")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("upgrade")
    stamp = sub.add_parser("stamp")
    stamp.add_argument("revision")
    revision = sub.add_parser("revision")
    revision.add_argument("-m", "--message", required=True)
    args = parser.parse_args(argv)
    engine = make_engine(get_settings().database_url)
    if args.command == "upgrade":
        upgrade(engine)
    else:
        with engine.begin() as connection:
            if args.command == "stamp":
                command.stamp(_config(connection), args.revision)
            else:
                command.revision(_config(connection), message=args.message, autogenerate=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
