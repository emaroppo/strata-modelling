"""Where this package's migration chain lives, and the command that runs it; the plumbing is
:mod:`strata.common.migrations`.
"""

from pathlib import Path

from strata.common.migrations import migrate_main

MIGRATIONS = Path(__file__).resolve().parent / "migrations"


def migrate(argv: list[str] | None = None) -> None:
    """``strata-modelling-migrate``: alembic's commands, over this package's chain."""
    migrate_main("strata-modelling-migrate", MIGRATIONS, argv)


__all__ = ["MIGRATIONS", "migrate"]
