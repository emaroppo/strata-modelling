"""Alembic environment for the run store and the prediction cache.

They share one metadata and one SQLite file, so they share one history.

``$STRATA_RUNS_URL`` is required and there is no fallback: a run store
belongs to one project or one modelling host, so unlike the catalog there
is no host-wide setting that could name the right one. Guessing would
migrate somebody else's runs.
"""

from __future__ import annotations

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

from strata.modelling.tables import metadata

alembic_config = context.config

if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

target_metadata = metadata


def database_url() -> str:
    url = os.environ.get("STRATA_RUNS_URL")
    if url:
        return url
    root = os.environ.get("STRATA_RUNS_ROOT")
    if root:
        return f"sqlite:///{Path(root) / 'runs.db'}"
    raise SystemExit(
        "No run store to migrate. Set STRATA_RUNS_URL to it, or "
        "STRATA_RUNS_ROOT to a project's runs/ directory."
    )


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = alembic_config.get_section(alembic_config.config_ini_section, {})
    section["sqlalchemy.url"] = database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # A run store is always SQLite, which cannot ALTER a column.
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
