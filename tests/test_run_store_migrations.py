"""The run store's migration chain builds the schema the code expects.

Shorter than the catalog's because there is less to guard: a run store is
always SQLite and always local to a project. What is the same is the way it
rots — ``create_all`` keeps the suite green while the chain drifts, and the
drift shows up on the machine that already has runs recorded.
"""

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine

from strata.common.migrations import script_directory
from strata.modelling import RunStore
from strata.modelling import tables as t
from strata.modelling.schema_version import MIGRATIONS


@pytest.fixture
def url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'runs.db'}"


def test_the_chain_produces_the_schema_the_code_expects(url, monkeypatch):
    monkeypatch.setenv("STRATA_RUNS_URL", url)
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")

    with create_engine(url).connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        difference = compare_metadata(context, t.metadata)

    assert not difference, f"the chain and tables.py disagree — add a migration: {difference}"


def test_a_created_run_store_is_stamped_at_head(tmp_path):
    store = RunStore.local(tmp_path / "runs")

    with store.engine.connect() as conn:
        current = MigrationContext.configure(conn).get_current_revision()

    assert current == script_directory(MIGRATIONS).get_current_head()
