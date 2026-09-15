"""Alembic environment for the run store and the prediction cache.

They share one metadata and one SQLite file, so they share one history.

``$STRATA_RUNS_URL`` names the store, or ``$STRATA_RUNS_ROOT`` a project's
``runs/`` directory; there is no host-wide default. See ``docs/adr/0019``.
"""

from __future__ import annotations

import os
from pathlib import Path

from alembic import context

from strata.common.migrations import run_alembic
from strata.modelling.store.tables import metadata


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


run_alembic(context, metadata, database_url())
