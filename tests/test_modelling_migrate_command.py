"""``strata-modelling-migrate``, which is all an installed wheel has to migrate with."""

from alembic.migration import MigrationContext
from sqlalchemy import create_engine

from strata.common.migrations import script_directory
from strata.modelling.schema_version import MIGRATIONS, migrate


def test_the_migrate_command_needs_no_alembic_ini(tmp_path, monkeypatch):
    monkeypatch.delenv("STRATA_RUNS_URL", raising=False)
    monkeypatch.setenv("STRATA_RUNS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    migrate(["upgrade", "head"])

    [store] = tmp_path.glob("*.db")
    with create_engine(f"sqlite:///{store}").connect() as conn:
        current = MigrationContext.configure(conn).get_current_revision()
    assert current == script_directory(MIGRATIONS).get_current_head()
