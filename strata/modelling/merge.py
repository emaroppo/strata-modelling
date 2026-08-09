"""Folding one run store into another.

There are two stores by design: a project keeps runs beside its own
checkpoints, and a modelling host keeps its own. Training moved to the GPU
host part-way through, so a project's history is now split across both —
the early rounds on the desktop, the later ones where the GPU is.

This is what puts them back together, and it is only possible because run
ids stopped being integers. Two stores numbering from one produced the same
id for different models with nothing to say so; an id that is a timestamp
and a host token is unique without anyone coordinating, which is exactly
the property a merge needs.

Checkpoints are the awkward part and are left behind by default. They are
the large half by orders of magnitude, and a merge is usually done to read
a history rather than to train from it. A row whose checkpoint did not come
along has its ``checkpoint`` column cleared rather than left pointing at a
file on another machine — everything that warm-starts or predicts checks
that column, and a path that looks real and is not would fail at the far
end of a long round.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import insert, select

from . import tables as t
from .runs import RunStore, RunStoreError


class StoreMergeError(RunStoreError):
    """The two stores cannot be merged as they stand.

    Named for the store rather than called ``MergeError``, because the
    catalog has one of those too and the two are merged in the same
    breath — a caller importing both should not have to alias one.
    """


@dataclass
class StoreMergeReport:
    """What a run store merge did."""

    #: Runs copied in.
    runs: int = 0
    #: Runs the target already had, by id. A merge run twice.
    already_present: int = 0
    #: Metric rows copied, final figures and training curves alike.
    metrics: int = 0
    #: Cached predictions copied.
    predictions: int = 0
    #: Checkpoint files copied.
    checkpoints: int = 0
    #: Runs whose parent is in neither store. Copied, with the link dropped.
    orphaned: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"{self.runs} run(s)",
            f"{self.metrics} metric row(s)",
        ]
        if self.predictions:
            out.append(f"{self.predictions} cached prediction(s)")
        if self.checkpoints:
            out.append(f"{self.checkpoints} checkpoint(s)")
        if self.already_present:
            out.append(f"{self.already_present} already there")
        if self.orphaned:
            out.append(
                f"{len(self.orphaned)} whose parent is in neither store, "
                f"copied as cold"
            )
        return out


def merge_stores(
    source: RunStore,
    target: RunStore,
    checkpoints: bool = False,
    predictions: bool = True,
    dry_run: bool = False,
) -> StoreMergeReport:
    """Copy every run ``target`` does not have out of ``source``.

    Ids are globally unique, so an id the target already holds is the same
    run rather than a collision — this is re-runnable, and a merge
    interrupted halfway can simply be repeated.

    Runs are copied oldest first so that a parent is in place before the
    run that continues it. A parent in neither store cannot be honoured:
    the link is dropped and the run is reported, because a dangling
    ``parent_run_id`` would make the history claim a lineage it cannot
    show.
    """
    if source.engine.url == target.engine.url:
        raise StoreMergeError(
            "Source and target are the same store. Nothing would move, and "
            "the ids would all collide with themselves."
        )

    rows = _runs(source)
    have = _ids(target)
    report = StoreMergeReport()

    # Both sides, because a run whose parent is already in the target is
    # not an orphan — only one whose parent exists nowhere is
    known = have | {row.id for row in rows}

    for row in rows:
        if row.id in have:
            report.already_present += 1
            continue

        values = {c.name: getattr(row, c.name) for c in t.run.columns}
        parent = values.get("parent_run_id")
        if parent is not None and parent not in known:
            report.orphaned.append(row.id)
            values["parent_run_id"] = None

        if checkpoints and row.checkpoint:
            source_file = Path(row.checkpoint)
            if source_file.exists():
                destination = target.checkpoint_path(row.id)
                report.checkpoints += 1
                if not dry_run:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, destination)
                values["checkpoint"] = str(destination)
            else:
                # Asked for and not there: the row must not keep a path to
                # a file that exists on neither machine
                values["checkpoint"] = None
        else:
            values["checkpoint"] = None

        report.runs += 1
        if not dry_run:
            with target.engine.begin() as conn:
                conn.execute(insert(t.run).values(**values))

        report.metrics += _copy(source, target, t.metric, t.metric.c.run_id, row.id,
                                dry_run, drop_id=True)
        if predictions:
            report.predictions += _copy(
                source, target, t.prediction, t.prediction.c.run_id, row.id, dry_run
            )

    return report


def _runs(store: RunStore):
    with store.engine.connect() as conn:
        # Oldest first: a child inserted before its parent violates the
        # foreign key on any database that enforces one
        return list(conn.execute(select(t.run).order_by(t.run.c.created_at, t.run.c.id)))


def _ids(store: RunStore) -> set[str]:
    with store.engine.connect() as conn:
        return set(conn.execute(select(t.run.c.id)).scalars())


def _copy(source, target, table, run_column, run_id: str, dry_run: bool,
          drop_id: bool = False) -> int:
    """Copy one run's rows out of a side table."""
    with source.engine.connect() as conn:
        rows = list(conn.execute(select(table).where(run_column == run_id)))
    if not rows:
        return 0
    payload = []
    for row in rows:
        values = {c.name: getattr(row, c.name) for c in table.columns}
        if drop_id:
            # A metric's own id is that store's numbering and means nothing
            # here; letting the target mint one avoids a collision that has
            # no meaning to resolve
            values.pop("id", None)
        payload.append(values)
    if not dry_run:
        with target.engine.begin() as conn:
            conn.execute(insert(table), payload)
    return len(payload)


__all__ = ["StoreMergeError", "StoreMergeReport", "merge_stores"]
