"""Where this package's migration chain lives, and the command that runs it.

The plumbing is :mod:`strata.common.migrations`; what is this package's own
is the directory of scripts beside its tables, and the name of the command
that runs them from an installed wheel. Which database that command reaches
is ``migrations/env.py``'s business.
"""

from pathlib import Path

from strata.common.migrations import migrate_main

MIGRATIONS = Path(__file__).resolve().parent / "migrations"


def migrate(argv: list[str] | None = None) -> None:
    """``strata-modelling-migrate``: alembic's commands, over this package's chain."""
    migrate_main("strata-modelling-migrate", MIGRATIONS, argv)


__all__ = ["MIGRATIONS", "migrate"]
