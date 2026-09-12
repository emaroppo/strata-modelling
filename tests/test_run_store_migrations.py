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


def test_cached_spans_written_with_one_label_are_rewritten(url, monkeypatch):
    import json

    from sqlalchemy import text

    from strata.labels import SpansPrediction

    monkeypatch.setenv("STRATA_RUNS_URL", url)
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "882788cac3cf")

    old = {
        "kind": "spans",
        "confidences": [0.9],
        "values": [{"label": "PER", "start": 0, "end": 3, "text": "Ada"}],
    }
    choices = {"kind": "choices", "values": ["cat"], "confidences": [0.5]}
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO run (id, dataset, label_set, model, model_version, classes) "
                "VALUES ('r', 'd', 's', 'm', '1', '[]')"
            )
        )
        for checksum, value in (("a", old), ("b", choices)):
            conn.execute(
                text(
                    "INSERT INTO prediction (run_id, checksum, feature_digest, value) "
                    "VALUES ('r', :c, '', :v)"
                ),
                {"c": checksum, "v": json.dumps(value)},
            )

    command.upgrade(config, "head")

    with engine.connect() as conn:
        rows = dict(conn.execute(text("SELECT checksum, value FROM prediction")).all())
    assert SpansPrediction.model_validate_json(rows["a"]).values[0].labels == ["PER"]
    assert json.loads(rows["b"]) == choices
