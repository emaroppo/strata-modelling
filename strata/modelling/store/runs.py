"""The model catalog: recording runs, and reading their history back."""

import socket
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, insert, select
from sqlalchemy.engine import Engine

from strata.common import database
from strata.common.migrations import require_current, stamp_if_new

from ..requests import Run
from . import tables as t
from .schema_version import MIGRATIONS


def _over(dataset: str, catalog_id: str | None):
    """The runs over a dataset, within a catalog when one is named.

    A run with no catalog recorded matches, since null is unknown rather
    than "some other": excluding it would cold-start every project whose
    history predates catalog identities. See ``docs/adr/0008``.
    """
    where = t.run.c.dataset == dataset
    if catalog_id is not None:
        where = where & ((t.run.c.catalog_id == catalog_id) | t.run.c.catalog_id.is_(None))
    return where


def host_token(name: str | None = None) -> str:
    """A hostname reduced to something safe to put in an identifier."""
    raw = (name or socket.gethostname()).split(".")[0].lower()
    cleaned = "".join(c if c.isalnum() else "-" for c in raw).strip("-")
    return cleaned[:16] or "unknown"


def new_run_id(origin: str | None = None) -> str:
    """A run id: when it happened, and where.

    Unique without coordination and legible on purpose. See
    ``docs/adr/0005``.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"{stamp}-{host_token(origin)}"


class RunStoreError(Exception):
    """A run store that cannot be used as it stands."""


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
        path = root / "runs.db"
        engine = database.engine(f"sqlite:///{path}")
        from sqlalchemy import inspect

        empty = not inspect(engine).has_table("run")
        t.metadata.create_all(engine)
        if empty:
            stamp_if_new(engine, MIGRATIONS)
        else:
            require_current(engine, MIGRATIONS, "modelling")
        return cls(engine, root / "checkpoints")

    def checkpoint_path(self, run_id: str) -> Path:
        # Ids are timestamps and hex, so they are already filename-safe
        return self.checkpoints / f"run_{run_id}.pt"

    # ------------------------------------------------------------------

    def record(
        self,
        run: Run,
        metrics: dict[str, float],
        curve: list[tuple[int, dict[str, float]]] | None = None,
        saw: list[tuple[str, str]] | None = None,
    ) -> Run:
        """Write a completed run, its final metrics, its curve, and what it saw.

        The id is minted here. ``curve`` is ``(epoch, metrics)`` oldest
        first, and ``saw`` is ``(checksum, side)`` for every sample of the
        manifest the run trained from. Both are written with the run: a run
        appears only once it finished, and a round that died halfway leaves
        nothing. See ``docs/adr/0005``.
        """
        origin = run.origin or socket.gethostname()
        run_id = run.id or new_run_id(origin)
        with self.engine.begin() as conn:
            conn.execute(
                insert(t.run).values(
                    id=run_id,
                    origin=origin,
                    catalog_id=run.catalog_id,
                    parent_run_id=run.parent_run_id,
                    dataset=run.dataset,
                    dataset_version=run.dataset_version,
                    label_set=run.label_set,
                    model=run.model,
                    model_version=run.model_version,
                    params=run.params,
                    classes=run.classes,
                    checkpoint=str(run.checkpoint) if run.checkpoint else None,
                    experiment_id=run.experiment_id,
                )
            )
            self._write_metrics(conn, run_id, metrics)
            for epoch, reported in curve or ():
                self._write_metrics(conn, run_id, reported, epoch=epoch)
            if saw:
                conn.execute(
                    insert(t.run_sample),
                    [{"run_id": run_id, "checksum": c, "side": s} for c, s in saw],
                )
        return run.model_copy(update={"id": run_id, "origin": origin, "metrics": metrics})

    def curve(self, run_id: str) -> list[tuple[int, dict[str, float]]]:
        """What a run reported as it trained, oldest epoch first.

        Separate from :meth:`get` because only a chart wants epochs times
        metrics.
        """
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(t.metric.c.epoch, t.metric.c.name, t.metric.c.value)
                .where((t.metric.c.run_id == run_id) & (t.metric.c.epoch.is_not(None)))
                .order_by(t.metric.c.epoch)
            ).all()
        by_epoch: dict[int, dict[str, float]] = {}
        for epoch, name, value in rows:
            by_epoch.setdefault(epoch, {})[name] = value
        return sorted(by_epoch.items())

    def _write_metrics(self, conn, run_id: str, metrics: dict[str, float], epoch=None) -> None:
        for name, value in metrics.items():
            conn.execute(
                insert(t.metric).values(run_id=run_id, name=name, value=float(value), epoch=epoch)
            )

    def get(self, run_id: str) -> Run | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(t.run).where(t.run.c.id == run_id)).first()
            if row is None:
                return None
            # A comprehension, not dict(rows): a result has keys(), so dict()
            # would read it as a mapping and fail
            metrics: dict[str, float] = {  # noqa: C416
                name: value
                for name, value in conn.execute(
                    select(t.metric.c.name, t.metric.c.value).where(
                        (t.metric.c.run_id == run_id) & (t.metric.c.epoch.is_(None))
                    )
                )
            }
        return Run(
            id=row.id,
            parent_run_id=row.parent_run_id,
            origin=row.origin,
            catalog_id=row.catalog_id,
            experiment_id=row.experiment_id,
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

    def saw(self, run_id: str) -> dict[str, list[str]]:
        """Which side each sample was on for this run: checksums by side.

        Every side is present, empty or not, so a run with no holdout reads
        as one rather than as a missing key.
        """
        sides: dict[str, list[str]] = {"train": [], "val": [], "holdout": []}
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(t.run_sample.c.side, t.run_sample.c.checksum)
                .where(t.run_sample.c.run_id == run_id)
                .order_by(t.run_sample.c.checksum)
            ).all()
        for side, checksum in rows:
            sides.setdefault(side, []).append(checksum)
        return sides

    def for_experiment(self, experiment_id: str) -> list[Run]:
        """Every run an experiment file asked for, oldest first."""
        with self.engine.connect() as conn:
            ids = list(
                conn.execute(
                    select(t.run.c.id)
                    .where(t.run.c.experiment_id == experiment_id)
                    .order_by(t.run.c.created_at, t.run.c.id)
                ).scalars()
            )
        return [run for run in (self.get(str(i)) for i in ids) if run is not None]

    def latest(
        self, dataset: str, catalog_id: str | None = None, since_version: int | None = None
    ) -> Run | None:
        """The newest run over a dataset — what a warm start continues from.

        Scoped to a catalog when one is given; a run with no catalog
        recorded matches, since null is unknown. See ``docs/adr/0008``.

        ``since_version`` is the version the dataset's sides descend from:
        a run over an earlier version trained before a re-split and may have
        seen what is now held out, so it is not continued. A run that
        recorded no version is left out too, since it cannot say.
        """
        where = _over(dataset, catalog_id)
        if since_version is not None:
            where = where & (t.run.c.dataset_version >= since_version)
        with self.engine.connect() as conn:
            run_id = conn.execute(
                select(t.run.c.id)
                .where(where)
                # By time, which is what "latest" asks; the id breaks a tie
                .order_by(t.run.c.created_at.desc(), t.run.c.id.desc())
                .limit(1)
            ).scalar_one_or_none()
        return self.get(run_id) if run_id is not None else None

    def history(
        self, dataset: str, metric: str, catalog_id: str | None = None
    ) -> list[tuple[str, int, float]]:
        """``(run id, dataset version, value)`` for one metric, oldest first.

        Scoped to a catalog when one is given, on the rule ``latest`` uses:
        a dataset name means something within one catalog, and a history
        across two is two histories under one name.
        """
        with self.engine.connect() as conn:
            return [
                (r.id, r.dataset_version, r.value)
                for r in conn.execute(
                    select(t.run.c.id, t.run.c.dataset_version, t.metric.c.value)
                    .join(t.metric, t.metric.c.run_id == t.run.c.id)
                    .where(
                        _over(dataset, catalog_id)
                        & (t.metric.c.name == metric)
                        & (t.metric.c.epoch.is_(None))
                    )
                    # Oldest first, by when rather than by id — the same
                    # reason latest does: a store can hold ids minted
                    # elsewhere, and their order is their timestamps'
                    .order_by(t.run.c.created_at, t.run.c.id)
                ).all()
            ]

    def metric_names(self, dataset: str, catalog_id: str | None = None) -> list[str]:
        """Which metrics this dataset's runs actually recorded.

        Models report what they like — accuracy for a classifier, span F1
        for a tagger — so a caller asking for one that is absent is better
        told what is there than guessed at.
        """
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(t.metric.c.name)
                .join(t.run, t.metric.c.run_id == t.run.c.id)
                .where(_over(dataset, catalog_id) & (t.metric.c.epoch.is_(None)))
                .distinct()
            ).all()
        return sorted(row.name for row in rows)

    def chain(self, run_id: str) -> list[Run]:
        """A run and everything it continued from, oldest first.

        Warm-started metrics only mean something against the run before
        them, so the edge has to be walkable.
        """
        walked: list[Run] = []
        seen: set[str] = set()
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
