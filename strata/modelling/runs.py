"""The model catalog: recording runs, and reading their history back."""

import socket
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, insert, select
from sqlalchemy.engine import Engine

from strata.common import database
from strata.common.migrations import require_current, stamp_if_new

from . import tables as t
from .requests import Run
from .schema_version import MIGRATIONS


def host_token(name: str | None = None) -> str:
    """A hostname reduced to something safe to put in an identifier."""
    raw = (name or socket.gethostname()).split(".")[0].lower()
    cleaned = "".join(c if c.isalnum() else "-" for c in raw).strip("-")
    return cleaned[:16] or "unknown"


def new_run_id(origin: str | None = None) -> str:
    """A run id: when it happened, and where.

    Unique without coordination, which is what lets a laptop train offline
    and fold its history into the main store afterwards. Two machines cannot
    collide because the host differs; one machine cannot collide with itself
    because the timestamp carries microseconds and training takes minutes.

    Legible on purpose. A random suffix would be unique too, and would say
    nothing — while the two facts worth knowing about a run you are looking
    at months later are when it happened and which machine did it.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"{stamp}-{host_token(origin)}"


def _refuse_a_store_from_before_string_ids(engine, path: Path) -> None:
    """Say what is wrong, rather than failing on a missing column later.

    Run ids used to autoincrement, which meant something only inside one
    store — and there were two, both numbering from one. There is no
    migration: an id minted here and an integer from before it sort against
    each other by their first digit, so mixing them is worse than starting
    a store whose history is all of one kind.
    """
    from sqlalchemy import inspect

    if "run" not in inspect(engine).get_table_names():
        return
    columns = {c["name"] for c in inspect(engine).get_columns("run")}
    if "origin" in columns:
        return
    raise RunStoreError(
        f"{path} predates run ids carrying when and where they were made. "
        f"Move it aside and a new one will be created:\n"
        f"  mv {path} {path}.archived"
    )


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
        _refuse_a_store_from_before_string_ids(engine, path)
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
    ) -> Run:
        """Write a completed run, its final metrics, and its training curve.

        The id is minted here, where the run happened, and is unique without
        asking anyone — which is what lets a laptop train offline and fold
        its history into the main store later.

        ``curve`` is what the model reported as it went, ``(epoch,
        metrics)`` oldest first. It is written here rather than as it
        arrived because the run row does not exist until now and the metric
        rows point at it — and because a run should appear in this store
        only once it finished. A round that died halfway leaves no curve for
        the same reason it leaves no run.
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
                )
            )
            self._write_metrics(conn, run_id, metrics)
            for epoch, reported in curve or ():
                self._write_metrics(conn, run_id, reported, epoch=epoch)
        return run.model_copy(
            update={"id": run_id, "origin": origin, "metrics": metrics}
        )

    def curve(self, run_id: str) -> list[tuple[int, dict[str, float]]]:
        """What a run reported as it trained, oldest epoch first.

        Separate from :meth:`get` rather than a field on :class:`Run`,
        because every caller that reads a run wants its final numbers and
        only a chart wants the rest — and there are as many curve rows as
        epochs times metrics.
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
                insert(t.metric).values(
                    run_id=run_id, name=name, value=float(value), epoch=epoch
                )
            )

    def get(self, run_id: str) -> Run | None:
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
            # Coerced, because a store written before ids were strings holds
            # integers and SQLite hands them back as it stored them
            id=str(row.id),
            parent_run_id=None if row.parent_run_id is None else str(row.parent_run_id),
            origin=row.origin,
            catalog_id=row.catalog_id,
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

    def latest(self, dataset: str, catalog_id: str | None = None) -> Run | None:
        """The newest run over a dataset — what a warm start continues from.

        Scoped to a catalog when one is given, because a dataset name means
        something within one and a host can serve several.

        A run recorded before catalogs had identities carries none, and that
        null is *unknown* rather than *some other catalog* — matching it
        keeps every existing history findable, where excluding it would cold
        start a project whose runs are all pre-identity. New runs all carry
        one, so the scoping tightens on its own.
        """
        where = t.run.c.dataset == dataset
        if catalog_id is not None:
            where = where & (
                (t.run.c.catalog_id == catalog_id) | t.run.c.catalog_id.is_(None)
            )
        with self.engine.connect() as conn:
            run_id = conn.execute(
                select(t.run.c.id)
                .where(where)
                # By time, not by id. Ids are minted where a run happens and
                # sort by their timestamp, but a store holding both those and
                # older numeric ones would order them by their first digit.
                .order_by(t.run.c.created_at.desc(), t.run.c.id.desc())
                .limit(1)
            ).scalar_one_or_none()
        return self.get(run_id) if run_id is not None else None

    def history(self, dataset: str, metric: str) -> list[tuple[str, int, float]]:
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
                    # Oldest first, by when rather than by id — the same
                    # reason latest does: a store can hold ids minted
                    # elsewhere, and their order is their timestamps'
                    .order_by(t.run.c.created_at, t.run.c.id)
                ).all()
            ]

    def metric_names(self, dataset: str) -> list[str]:
        """Which metrics this dataset's runs actually recorded.

        Models report what they like — accuracy for a classifier, span F1
        for a tagger — so a caller asking for one that is absent is better
        told what is there than guessed at.
        """
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(t.metric.c.name)
                .join(t.run, t.metric.c.run_id == t.run.c.id)
                .where((t.run.c.dataset == dataset) & (t.metric.c.epoch.is_(None)))
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
