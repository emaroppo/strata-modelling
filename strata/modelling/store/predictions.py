"""Predictions already made, keyed on run, checkpoint, checksum and feature digest, beside the runs.

See ``docs/adr/0006`` and ``docs/adr/0054``.
"""

from collections.abc import Mapping
from pathlib import Path

from pydantic import TypeAdapter
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from strata.contracts import AnyPrediction, Prediction

from . import tables as t
from .runs import RunStore

_PREDICTION = TypeAdapter(AnyPrediction)


def _chunks(items: list, size: int):
    """SQLite caps parameters per statement."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


class PredictionCache:
    """What a run already said about a sample."""

    def __init__(self, engine):
        self.engine = engine

    @classmethod
    def local(cls, root: Path) -> "PredictionCache":
        """The cache in the run store under ``root``, made if there is none (``docs/adr/0006``)."""
        return cls.beside(RunStore.local(root))

    @classmethod
    def beside(cls, store) -> "PredictionCache":
        """The cache belonging to a run store."""
        return cls(store.engine)

    def last_checkpoint(self, run_id: str) -> str:
        """The name of a run's last checkpoint; empty for a run that kept none or is not held here.
        See ``docs/adr/0054``.
        """
        with self.engine.connect() as conn:
            name = conn.execute(
                select(t.checkpoint.c.name)
                .where(t.checkpoint.c.run_id == run_id)
                .order_by(t.checkpoint.c.saved_at.desc(), t.checkpoint.c.id.desc())
                .limit(1)
            ).scalar_one_or_none()
        return name or ""

    def get(
        self,
        run_id: str,
        checksums: list[str],
        digests: dict[str, str] | None = None,
        checkpoint: str | None = None,
    ) -> dict[str, AnyPrediction]:
        """Whatever of ``checksums`` this run's checkpoint has already answered.

        ``digests`` is each sample's :func:`strata.contracts.feature_digest`,
        empty when omitted; ``checkpoint`` None is the run's last. See ``docs/adr/0006`` and
        ``docs/adr/0054``.
        """
        found: dict[str, AnyPrediction] = {}
        if not checksums:
            return found
        digests = digests or {}
        if checkpoint is None:
            checkpoint = self.last_checkpoint(run_id)
        with self.engine.connect() as conn:
            for chunk in _chunks(list(checksums), 500):
                rows = conn.execute(
                    select(
                        t.prediction.c.checksum,
                        t.prediction.c.feature_digest,
                        t.prediction.c.value,
                    ).where(
                        t.prediction.c.run_id == run_id,
                        t.prediction.c.checkpoint == checkpoint,
                        t.prediction.c.checksum.in_(chunk),
                    )
                )
                for row in rows:
                    if (row.feature_digest or "") != digests.get(row.checksum, ""):
                        continue
                    found[row.checksum] = _PREDICTION.validate_json(row.value)
        return found

    def put(
        self,
        run_id: str,
        made: Mapping[str, Prediction],
        digests: dict[str, str] | None = None,
        checkpoint: str | None = None,
    ) -> int:
        """Record what a run's checkpoint said. Rewriting an entry is a no-op by construction.

        ``checkpoint`` is a name; None is the run's last, as for :meth:`get`.
        """
        if not made:
            return 0
        wrong = {
            checksum: type(value).__name__
            for checksum, value in made.items()
            if not isinstance(value, Prediction)
        }
        if wrong:
            kinds = ", ".join(sorted(set(wrong.values())))
            raise TypeError(
                f"A prediction cache holds model output, not {kinds}. "
                f"Anything else — a wrapper around one, or a plain annotation "
                f"value — serialises happily and reads back with its "
                f"confidences gone rather than failing."
            )
        if checkpoint is None:
            checkpoint = self.last_checkpoint(run_id)
        rows = [
            {
                "run_id": run_id,
                "checkpoint": checkpoint,
                "checksum": checksum,
                "feature_digest": (digests or {}).get(checksum, ""),
                "value": value.model_dump_json(),
            }
            for checksum, value in made.items()
        ]
        with self.engine.begin() as conn:
            for chunk in _chunks(rows, 500):
                # docs/adr/0006
                conn.execute(
                    sqlite_insert(t.prediction).on_conflict_do_nothing(
                        index_elements=["run_id", "checkpoint", "checksum", "feature_digest"]
                    ),
                    chunk,
                )
        return len(rows)

    def forget(self, run_id: str) -> None:
        """Drop a run's predictions, for when disk matters more than time."""
        with self.engine.begin() as conn:
            conn.execute(delete(t.prediction).where(t.prediction.c.run_id == run_id))

    def counts(self) -> dict[str, int]:
        """How many predictions are held per run."""
        with self.engine.connect() as conn:
            return {
                row[0]: row[1]
                for row in conn.execute(
                    select(t.prediction.c.run_id, func.count()).group_by(t.prediction.c.run_id)
                )
            }
