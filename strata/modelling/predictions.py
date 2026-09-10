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

It holds whatever a model produced — choices, spans, boxes — and reads it
back as what it was, through the discriminator. Pinning it to one of them
would make a cache that quietly refuses, or worse mangles, every task type
but the first.
"""

from pathlib import Path

from pydantic import TypeAdapter
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from strata.labels import AnyPrediction, Prediction

from . import tables as t

_PREDICTION = TypeAdapter(AnyPrediction)


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

    def get(
        self,
        run_id: str,
        checksums: list[str],
        digests: dict[str, str] | None = None,
    ) -> dict[str, Prediction]:
        """Whatever of ``checksums`` this run has already answered.

        ``digests`` says what the model was told about each sample, by
        :func:`strata.labels.feature_digest`. A checksum whose
        features have changed since it was scored simply misses — the
        stored row is still the right answer for the inputs it was computed
        from, and those inputs are no longer the ones being asked about.

        Omitted, every lookup uses the empty digest, which is what a
        project declaring no features has always written.
        """
        found: dict[str, Prediction] = {}
        if not checksums:
            return found
        digests = digests or {}
        with self.engine.connect() as conn:
            for chunk in _chunks(list(checksums), 500):
                rows = conn.execute(
                    select(
                        t.prediction.c.checksum,
                        t.prediction.c.feature_digest,
                        t.prediction.c.value,
                    ).where(
                        t.prediction.c.run_id == run_id,
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
        made: dict[str, Prediction],
        digests: dict[str, str] | None = None,
    ) -> int:
        """Record what a run said. Rewriting an entry is a no-op by construction."""
        if not made:
            return 0
        wrong = {
            checksum: type(value).__name__
            for checksum, value in made.items()
            if not isinstance(value, Prediction)
        }
        if wrong:
            # A wrapper around a prediction serialises happily and reads
            # back as an empty one — pydantic ignores the keys it does not
            # know — so a cache full of nothing looks exactly like a cache
            # full of answers until a ranking sorts on them.
            kinds = ", ".join(sorted(set(wrong.values())))
            raise TypeError(
                f"A prediction cache holds model output, not {kinds}. "
                f"Anything else — a wrapper around one, or a plain annotation "
                f"value — serialises happily and reads back with its "
                f"confidences gone rather than failing."
            )
        rows = [
            {
                "run_id": run_id,
                "checksum": checksum,
                "feature_digest": (digests or {}).get(checksum, ""),
                "value": value.model_dump_json(),
            }
            for checksum, value in made.items()
        ]
        with self.engine.begin() as conn:
            for chunk in _chunks(rows, 500):
                # A checkpoint and some bytes give one answer, so a second
                # write of the same key carries the same value. Ignoring the
                # conflict keeps a re-run from failing on work it repeated.
                conn.execute(
                    sqlite_insert(t.prediction).on_conflict_do_nothing(
                        index_elements=["run_id", "checksum", "feature_digest"]
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
                    select(t.prediction.c.run_id, func.count()).group_by(
                        t.prediction.c.run_id
                    )
                )
            }
