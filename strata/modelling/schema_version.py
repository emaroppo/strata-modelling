"""Marking a freshly created database as already current.

``create_all`` builds the whole schema in one step, which is what keeps a
checkout runnable and a test suite fast — replaying every migration to make
a scratch catalog would cost more than the tests do. But a database built
that way has no revision recorded, so the first ``alembic upgrade head``
against it would try to create tables that already exist.

Stamping closes that: a database this process just created is, by
definition, at head. A database that already carried a revision is left
alone, because it is alembic's to move.

The script directory is found from this package rather than from
``alembic.ini``, which lives at the root of a checkout and is not part of
an installed wheel.
"""

from pathlib import Path

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Engine

MIGRATIONS = Path(__file__).resolve().parent / "migrations"


def script_directory() -> ScriptDirectory:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    return ScriptDirectory.from_config(config)


def stamp_if_new(engine: Engine) -> str | None:
    """Record head against a database that has no revision yet.

    **Only correct for a database this process just created.** A database
    that already held tables and no revision predates migrations, and is at
    the baseline rather than at head — stamping it head would have it claim
    columns it does not have, and a later ``upgrade`` would find nothing to
    do. :func:`require_current` is the guard for that case.

    Returns the revision stamped, or None where one was already recorded.
    """
    with engine.begin() as conn:
        context = MigrationContext.configure(conn)
        if context.get_current_revision() is not None:
            return None
        scripts = script_directory()
        context.stamp(scripts, "head")
        return scripts.get_current_head()


class SchemaOutOfDate(Exception):
    """A database whose schema is not the one this code was written against."""


def require_current(engine: Engine, name: str) -> None:
    """Refuse a database the code would misread, and say how to fix it.

    Refused rather than upgraded in passing: a migration rewrites somebody's
    data, and doing that as a side effect of opening a connection is not a
    decision this should be making on their behalf. Refused rather than
    ignored, because the alternative is a missing column surfacing as a
    query error somewhere far from the cause.
    """
    scripts = script_directory()
    head = scripts.get_current_head()
    with engine.connect() as conn:
        current = MigrationContext.configure(conn).get_current_revision()
    if current == head:
        return
    if current is None:
        raise SchemaOutOfDate(
            f"This {name} predates migrations. Its schema is the baseline, so "
            f"record that and then bring it up to date:\n"
            f"  uv run alembic --name {name} stamp {scripts.get_base()}\n"
            f"  uv run alembic --name {name} upgrade head"
        )
    raise SchemaOutOfDate(
        f"This {name} is at revision {current}, and the code expects {head}:\n"
        f"  uv run alembic --name {name} upgrade head"
    )


__all__ = [
    "MIGRATIONS",
    "SchemaOutOfDate",
    "require_current",
    "script_directory",
    "stamp_if_new",
]
