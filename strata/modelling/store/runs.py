"""The model catalog: recording runs, and reading their history back."""

import hashlib
import re
import shutil
import socket
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from sqlalchemy import case, delete, func, insert, inspect, select
from sqlalchemy.engine import Engine

from strata.common import database
from strata.common.canonical import hash_file
from strata.common.migrations import require_current, stamp_if_new

from ..requests import Checkpoint, Run
from . import tables as t
from .schema_version import MIGRATIONS


def _over(dataset: str, catalog_id: str | None):
    """The runs over a dataset, within a catalog when one is named; a null catalog matches. See
    ``docs/adr/0008``.
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
    """A run id: when it happened, and where. See ``docs/adr/0005``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"{stamp}-{host_token(origin)}"


#: What a run's one checkpoint is called when its model named none (``docs/adr/0054``).
FINAL = "final"

#: A checkpoint's name becomes a file name, so it is kept to what is safe as one everywhere.
CHECKPOINT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def now() -> datetime:
    """Naive UTC, as everything stored here is."""
    return datetime.now(UTC).replace(tzinfo=None)


def checkpoint_digest(path: Path) -> str | None:
    """SHA-256 over what a model's ``save`` wrote, or None where it wrote nothing.

    A directory hashes each file by relative path: path, NUL, its SHA-256, newline. See
    ``docs/adr/0054``.
    """
    path = Path(path)
    if path.is_file():
        return hash_file(path)[0]
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for name in sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()):
        digest.update(f"{name}\0{hash_file(path / name)[0]}\n".encode())
    return digest.hexdigest()


class Seen(NamedTuple):
    """One sample a run saw: its side, and what the manifest said about its label."""

    checksum: str
    side: str
    #: Which import the label arrived in; null for a label nothing imported.
    batch: str | None = None
    #: Whether a person vouched for it; null where the manifest did not say.
    reviewed: bool | None = None


class Unchecked(NamedTuple):
    """Of the samples a run saw on one side from one batch, how many nobody checked."""

    side: str
    batch: str | None
    samples: int
    unreviewed: int


def _exists(engine: Engine) -> bool:
    inspector = inspect(engine)
    return inspector.has_table("run") or inspector.has_table("alembic_version")


class RunStoreError(Exception):
    """A run store that cannot be used as it stands."""


class RunStoreMissing(RunStoreError):
    """No run store where one was asked for: nothing has trained here."""


class RunStore:
    """Where runs, their metrics and their checkpoints live."""

    def __init__(self, engine: Engine, checkpoints: Path):
        """Over ``engine``, keeping checkpoints under ``checkpoints``, which is made when one is
        first written; constructing touches nothing.
        """
        self.engine = engine
        self.checkpoints = Path(checkpoints)

    @classmethod
    def open(cls, root: Path) -> "RunStore":
        """The store under ``root``, which must exist; :class:`RunStoreMissing` otherwise. Nothing
        is created, not even an empty ``checkpoints/``. See ``docs/adr/0018``.
        """
        root = Path(root)
        path = root / "runs.db"
        if not path.exists():
            raise RunStoreMissing(f"No runs recorded at {root}. Run 'train' first.")
        engine = database.engine(f"sqlite:///{path}", foreign_keys=False)
        if not _exists(engine):
            raise RunStoreMissing(f"No runs recorded at {root}. Run 'train' first.")
        require_current(engine, MIGRATIONS, "modelling")
        return cls(engine, root / "checkpoints")

    @classmethod
    def local(cls, root: Path) -> "RunStore":
        """The store under ``root``, made if there is none; a caller that only consults uses
        :meth:`open`.
        """
        root = Path(root)
        checkpoints = root / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        # Foreign keys off: a parent may be in another store. docs/adr/0005
        engine = database.engine(f"sqlite:///{root / 'runs.db'}", foreign_keys=False)
        if _exists(engine):
            require_current(engine, MIGRATIONS, "modelling")
            return cls(engine, checkpoints)
        t.metadata.create_all(engine)
        stamp_if_new(engine, MIGRATIONS)
        return cls(engine, checkpoints)

    @classmethod
    def scratch(cls, root: Path) -> "RunStore":
        """An empty store in memory, with ``root`` as where its checkpoints would go: what a report
        of a write that is not made is computed against. Nothing under ``root`` is touched.
        """
        engine = database.engine("sqlite://", foreign_keys=False)
        t.metadata.create_all(engine)
        return cls(engine, Path(root) / "checkpoints")

    def checkpoint_path(self, run_id: str, name: str = FINAL) -> Path:
        """Where a run's checkpoint lives: ``final`` as a file, named ones in a directory. See
        ``docs/adr/0054``.
        """
        if name == FINAL:
            return self.checkpoints / f"run_{run_id}.pt"
        return self.checkpoints / f"run_{run_id}" / name

    def staging(self) -> Path:
        """A fresh directory for checkpoints saved before their run has an id; :meth:`record` moves
        them, the caller removes it.

        See ``docs/adr/0054``.
        """
        path = self.checkpoints / ".staging" / uuid.uuid4().hex
        path.mkdir(parents=True)
        return path

    def checkpoint(self, run_id: str, name: str | None = None) -> Checkpoint | None:
        """One of a run's checkpoints by name, or its last; None if it has no such one."""
        query = select(t.checkpoint).where(t.checkpoint.c.run_id == run_id)
        if name is not None:
            query = query.where(t.checkpoint.c.name == name)
        query = query.order_by(t.checkpoint.c.saved_at.desc(), t.checkpoint.c.id.desc()).limit(1)
        with self.engine.connect() as conn:
            row = conn.execute(query).first()
        return _checkpoint(row) if row is not None else None

    # ------------------------------------------------------------------

    def record(
        self,
        run: Run,
        metrics: dict[str, float],
        curve: list[tuple[int, dict[str, float]]] | None = None,
        saw: Sequence[Seen | tuple[str, str]] | None = None,
        checkpoints: Sequence[Checkpoint] | None = None,
    ) -> Run:
        """Write a completed run, its final metrics, its curve, what it saw and what it kept, in one
        transaction.

        The id is minted here. ``checkpoints`` are moved under the run;
        omitted, the run's ``checkpoint`` is recorded as ``final``. See ``docs/adr/0005`` and
        ``docs/adr/0054``.
        """
        origin = run.origin or socket.gethostname()
        run_id = run.id or new_run_id(origin)
        with self.engine.begin() as conn:
            if checkpoints is None:
                kept = (
                    [
                        Checkpoint(
                            name=FINAL,
                            path=run.checkpoint,
                            digest=checkpoint_digest(run.checkpoint),
                            saved_at=now(),
                        )
                    ]
                    if run.checkpoint is not None
                    else []
                )
            else:
                # Inside the transaction, so a failed move leaves no run.
                kept = self._keep(run_id, checkpoints)
            last = kept[-1].path if kept else None
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
                    checkpoint=str(last) if last else None,
                    experiment_id=run.experiment_id,
                )
            )
            if kept:
                conn.execute(
                    insert(t.checkpoint),
                    [
                        {
                            "run_id": run_id,
                            **c.model_dump(),
                            "path": str(c.path) if c.path else None,
                        }
                        for c in kept
                    ],
                )
            self._write_metrics(conn, run_id, metrics)
            for epoch, reported in curve or ():
                self._write_metrics(conn, run_id, reported, epoch=epoch)
            if saw:
                seen = [entry if isinstance(entry, Seen) else Seen(*entry) for entry in saw]
                conn.execute(
                    insert(t.run_sample),
                    [{"run_id": run_id, **entry._asdict()} for entry in seen],
                )
        return run.model_copy(
            update={
                "id": run_id,
                "origin": origin,
                "metrics": metrics,
                "checkpoint": last,
                "checkpoints": kept,
            }
        )

    def _keep(self, run_id: str, saved: Sequence[Checkpoint]) -> list[Checkpoint]:
        """Each checkpoint moved to where its run keeps it; one already there, or with no file, is
        recorded as is.
        """
        names = [c.name for c in saved]
        twice = sorted({n for n in names if names.count(n) > 1})
        if twice:
            raise RunStoreError(f"Checkpoint(s) named more than once: {', '.join(twice)}.")
        kept = []
        for c in saved:
            path = c.path
            if path is not None and Path(path).exists():
                destination = self.checkpoint_path(run_id, c.name)
                if Path(path).resolve() != destination.resolve():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), destination)
                    path = destination
            kept.append(c.model_copy(update={"path": path, "saved_at": c.saved_at or now()}))
        return kept

    def curve(self, run_id: str) -> list[tuple[int, dict[str, float]]]:
        """What a run reported as it trained, oldest epoch first. See ``docs/adr/0005``."""
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
            kept = [
                _checkpoint(c)
                for c in conn.execute(
                    select(t.checkpoint)
                    .where(t.checkpoint.c.run_id == run_id)
                    .order_by(t.checkpoint.c.saved_at, t.checkpoint.c.id)
                )
            ]
            # Not dict(rows): a result has keys(), so dict() would read it as a mapping.
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
            # An older writer set the column with no row beside it. docs/adr/0054
            checkpoints=kept
            or (
                [Checkpoint(name=FINAL, path=Path(row.checkpoint), saved_at=row.created_at)]
                if row.checkpoint
                else []
            ),
            metrics=metrics,
        )

    def saw(self, run_id: str) -> dict[str, list[str]]:
        """Which side each sample was on for this run: checksums by side, every side present."""
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

    def unchecked(self, run_id: str) -> list[Unchecked]:
        """How much of what this run saw nobody checked, per side and per batch.

        A sample whose manifest said nothing about review counts as neither. See ``docs/adr/0005``.
        """
        unreviewed = func.sum(case((t.run_sample.c.reviewed.is_(False), 1), else_=0))
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(
                    t.run_sample.c.side,
                    t.run_sample.c.batch,
                    func.count(),
                    unreviewed,
                )
                .where(t.run_sample.c.run_id == run_id)
                .group_by(t.run_sample.c.side, t.run_sample.c.batch)
                .order_by(t.run_sample.c.side, t.run_sample.c.batch)
            ).all()
        return [
            Unchecked(side=side, batch=batch, samples=int(samples), unreviewed=int(missed or 0))
            for side, batch, samples, missed in rows
        ]

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
        """The newest run over a dataset, what a warm start continues from.

        Scoped to a catalog when given (``docs/adr/0008``); ``since_version``
        leaves out runs over earlier versions (``docs/adr/0024``).
        """
        where = _over(dataset, catalog_id)
        if since_version is not None:
            where = where & (t.run.c.dataset_version >= since_version)
        with self.engine.connect() as conn:
            run_id = conn.execute(
                select(t.run.c.id)
                .where(where)
                # docs/adr/0005
                .order_by(t.run.c.created_at.desc(), t.run.c.id.desc())
                .limit(1)
            ).scalar_one_or_none()
        return self.get(run_id) if run_id is not None else None

    def history(
        self, dataset: str, metric: str, catalog_id: str | None = None
    ) -> list[tuple[str, int, float]]:
        """``(run id, dataset version, value)`` for one metric, oldest first, scoped as ``latest``
        is. See ``docs/adr/0008``.
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
                    # docs/adr/0005
                    .order_by(t.run.c.created_at, t.run.c.id)
                ).all()
            ]

    def metric_names(self, dataset: str, catalog_id: str | None = None) -> list[str]:
        """Which metrics this dataset's runs recorded."""
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(t.metric.c.name)
                .join(t.run, t.metric.c.run_id == t.run.c.id)
                .where(_over(dataset, catalog_id) & (t.metric.c.epoch.is_(None)))
                .distinct()
            ).all()
        return sorted(row.name for row in rows)

    def chain(self, run_id: str) -> list[Run]:
        """A run and everything it continued from, oldest first. See ``docs/adr/0005``."""
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
            conn.execute(delete(t.checkpoint).where(t.checkpoint.c.run_id == run_id))
            conn.execute(delete(t.run).where(t.run.c.id == run_id))


def _checkpoint(row) -> Checkpoint:
    return Checkpoint(
        name=row.name,
        path=Path(row.path) if row.path else None,
        digest=row.digest,
        saved_at=row.saved_at,
    )
