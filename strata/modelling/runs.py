"""The model catalog: recording runs, and reading their history back."""

from pathlib import Path

from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Engine

from . import tables as t
from .requests import Run


class RunStore:
    """Where runs, their metrics and their checkpoints live."""

    def __init__(self, engine: Engine, checkpoints: Path):
        self.engine = engine
        self.checkpoints = Path(checkpoints)
        self.checkpoints.mkdir(parents=True, exist_ok=True)

    @classmethod
    def local(cls, root: Path) -> "RunStore":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{root / 'runs.db'}")
        store = cls(engine, root / "checkpoints")
        t.metadata.create_all(engine)
        return store

    def checkpoint_path(self, run_id: int) -> Path:
        return self.checkpoints / f"run_{run_id:04d}.pt"

    # ------------------------------------------------------------------

    def record(self, run: Run, metrics: dict[str, float]) -> Run:
        """Write a completed run and its final metrics."""
        with self.engine.begin() as conn:
            run_id = conn.execute(
                insert(t.run).values(
                    parent_run_id=run.parent_run_id,
                    dataset=run.dataset,
                    dataset_version=run.dataset_version,
                    label_set=run.label_set,
                    model=run.model,
                    model_version=run.model_version,
                    params=run.params,
                    classes=run.classes,
                    checkpoint=str(run.checkpoint) if run.checkpoint else None,
                )
            ).inserted_primary_key[0]
            self._write_metrics(conn, run_id, metrics)
        return run.model_copy(update={"id": run_id, "metrics": metrics})

    def _write_metrics(self, conn, run_id: int, metrics: dict[str, float], epoch=None) -> None:
        for name, value in metrics.items():
            conn.execute(
                insert(t.metric).values(
                    run_id=run_id, name=name, value=float(value), epoch=epoch
                )
            )

    def get(self, run_id: int) -> Run | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(t.run).where(t.run.c.id == run_id)).first()
            if row is None:
                return None
            metrics = dict(
                conn.execute(
                    select(t.metric.c.name, t.metric.c.value).where(
                        (t.metric.c.run_id == run_id) & (t.metric.c.epoch.is_(None))
                    )
                ).all()
            )
        return Run(
            id=row.id,
            parent_run_id=row.parent_run_id,
            dataset=row.dataset,
            dataset_version=row.dataset_version,
            label_set=row.label_set,
            model=row.model,
            model_version=row.model_version,
            params=row.params or {},
            classes=row.classes or [],
            checkpoint=Path(row.checkpoint) if row.checkpoint else None,
            metrics=metrics,
        )

    def latest(self, dataset: str) -> Run | None:
        """The newest run over a dataset — what a warm start continues from."""
        with self.engine.connect() as conn:
            run_id = conn.execute(
                select(t.run.c.id)
                .where(t.run.c.dataset == dataset)
                .order_by(t.run.c.id.desc())
                .limit(1)
            ).scalar_one_or_none()
        return self.get(run_id) if run_id is not None else None

    def history(self, dataset: str, metric: str) -> list[tuple[int, int, float]]:
        """``(run id, dataset version, value)`` for one metric, oldest first.

        The question a training history exists to answer, and the reason
        this is a database: metric progression across rounds is one query
        rather than a glob over directories.
        """
        with self.engine.connect() as conn:
            return [
                (r.id, r.dataset_version, r.value)
                for r in conn.execute(
                    select(t.run.c.id, t.run.c.dataset_version, t.metric.c.value)
                    .join(t.metric, t.metric.c.run_id == t.run.c.id)
                    .where(
                        (t.run.c.dataset == dataset)
                        & (t.metric.c.name == metric)
                        & (t.metric.c.epoch.is_(None))
                    )
                    .order_by(t.run.c.id)
                ).all()
            ]

    def chain(self, run_id: int) -> list[Run]:
        """A run and everything it continued from, oldest first.

        Warm-started metrics only mean something against the run before
        them, so the edge has to be walkable.
        """
        walked: list[Run] = []
        seen: set[int] = set()
        current = self.get(run_id)
        while current is not None and current.id not in seen:
            seen.add(current.id)
            walked.append(current)
            current = self.get(current.parent_run_id) if current.parent_run_id else None
        return list(reversed(walked))

    def delete(self, run_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(t.metric).where(t.metric.c.run_id == run_id))
            conn.execute(delete(t.run).where(t.run.c.id == run_id))
