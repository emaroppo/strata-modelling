"""Predictions already made, so nobody pays for them twice.

Ranking a review queue needs a score for every unlabelled sample, not just
the ones about to be shown — least-confident-first cannot pick a top 200
without having looked at all of them. That is a full inference pass per
push, and asking again from the same checkpoint pays for an identical
answer.

**Nothing here is ever invalidated, and that is a property rather than an
omission.** A prediction is a function of a checkpoint and some bytes.
Checkpoints are immutable — a run writes one and never rewrites it — and
blobs are addressed by content. So an entry keyed on both cannot go stale;
the only reason to drop one is disk.

**It lives beside the runs, which means beside whoever trained.** A run id
means something only within one store, so a cache keyed on one belongs in
the same database. It also puts the cache on the machine that does the
work: an answer is the same for every caller, and a client-side cache would
help only the machine that happened to ask first, leaving a second machine
to buy the same minutes of GPU again.

Keyed on the checksum rather than a sample id, for the same reason task
URLs are: an id belongs to one catalog's numbering, while the bytes are
what the model actually saw.
"""

from pathlib import Path

from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from strata.labels import ChoicesPrediction

from . import tables as t


def _chunks(items: list, size: int):
    """SQLite caps parameters per statement; a review pool is well past it."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


class PredictionCache:
    """What a run already said about a sample."""

    def __init__(self, engine):
        self.engine = engine

    @classmethod
    def local(cls, root: Path) -> "PredictionCache":
        """The same database the runs are in, since it is keyed on them."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{root / 'runs.db'}")
        t.metadata.create_all(engine)
        return cls(engine)

    @classmethod
    def beside(cls, store) -> "PredictionCache":
        """The cache belonging to a run store."""
        return cls(store.engine)

    def get(self, run_id: int, checksums: list[str]) -> dict[str, ChoicesPrediction]:
        """Whatever of ``checksums`` this run has already answered."""
        found: dict[str, ChoicesPrediction] = {}
        if not checksums:
            return found
        with self.engine.connect() as conn:
            for chunk in _chunks(list(checksums), 500):
                rows = conn.execute(
                    select(t.prediction.c.checksum, t.prediction.c.value).where(
                        t.prediction.c.run_id == run_id,
                        t.prediction.c.checksum.in_(chunk),
                    )
                )
                for row in rows:
                    found[row.checksum] = ChoicesPrediction.model_validate_json(row.value)
        return found

    def put(self, run_id: int, made: dict[str, ChoicesPrediction]) -> int:
        """Record what a run said. Rewriting an entry is a no-op by construction."""
        if not made:
            return 0
        wrong = {
            checksum: type(value).__name__
            for checksum, value in made.items()
            if not isinstance(value, ChoicesPrediction)
        }
        if wrong:
            # A wrapper around a prediction serialises happily and reads
            # back as an empty one — pydantic ignores the keys it does not
            # know — so a cache full of nothing looks exactly like a cache
            # full of answers until a ranking sorts on them.
            kinds = ", ".join(sorted(set(wrong.values())))
            raise TypeError(
                f"A prediction cache holds ChoicesPrediction, not {kinds}. "
                f"Stored as-is these read back empty rather than failing."
            )
        rows = [
            {"run_id": run_id, "checksum": checksum, "value": value.model_dump_json()}
            for checksum, value in made.items()
        ]
        with self.engine.begin() as conn:
            for chunk in _chunks(rows, 500):
                # A checkpoint and some bytes give one answer, so a second
                # write of the same key carries the same value. Ignoring the
                # conflict keeps a re-run from failing on work it repeated.
                conn.execute(
                    sqlite_insert(t.prediction).on_conflict_do_nothing(
                        index_elements=["run_id", "checksum"]
                    ),
                    chunk,
                )
        return len(rows)

    def forget(self, run_id: int) -> None:
        """Drop a run's predictions, for when disk matters more than time."""
        with self.engine.begin() as conn:
            conn.execute(delete(t.prediction).where(t.prediction.c.run_id == run_id))

    def counts(self) -> dict[int, int]:
        """How many predictions are held per run."""
        with self.engine.connect() as conn:
            return {
                row[0]: row[1]
                for row in conn.execute(
                    select(t.prediction.c.run_id, func.count()).group_by(
                        t.prediction.c.run_id
                    )
                )
            }
