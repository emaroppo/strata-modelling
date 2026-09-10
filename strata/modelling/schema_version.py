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

    Returns the revision stamped, or None where one was already recorded —
    which is the case worth not touching, since a database mid-history is
    alembic's to move rather than ours to relabel.
    """
    with engine.begin() as conn:
        context = MigrationContext.configure(conn)
        if context.get_current_revision() is not None:
            return None
        scripts = script_directory()
        context.stamp(scripts, "head")
        return scripts.get_current_head()


__all__ = ["MIGRATIONS", "script_directory", "stamp_if_new"]
