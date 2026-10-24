"""Folding one run store into another.

A project keeps runs beside its checkpoints and a modelling host keeps its
own; this puts a history split across both back together. Checkpoints are
left behind unless asked for, and a run whose checkpoint did not come has
its column cleared. Every checkpoint a run kept is recorded either way, its
path cleared where its file did not come. See ``docs/adr/0005`` and
``docs/adr/0054``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field

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
    #: Rows of what each run saw — a sample and its side — copied.
    samples: int = 0
    #: Runs whose parent is in neither store. Copied, with the link dropped.
    orphaned: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"{self.runs} run(s)",
            f"{self.metrics} metric row(s)",
        ]
        if self.samples:
            out.append(f"{self.samples} sample side(s)")
        if self.predictions:
            out.append(f"{self.predictions} cached prediction(s)")
        if self.checkpoints:
            out.append(f"{self.checkpoints} checkpoint(s)")
        if self.already_present:
            out.append(f"{self.already_present} already there")
        if self.orphaned:
            out.append(f"{len(self.orphaned)} whose parent is in neither store, copied as cold")
        return out


def merge_stores(
    source: RunStore,
    target: RunStore,
    checkpoints: bool = False,
    predictions: bool = True,
    dry_run: bool = False,
) -> StoreMergeReport:
    """Copy every run ``target`` does not have out of ``source``.

    Re-runnable: an id the target holds is the same run. Oldest first; a
    parent in neither store is dropped and reported. See ``docs/adr/0005``.
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

        # Every checkpoint the run kept, its record copied whether or not its
        # file comes too. docs/adr/0054
        kept = []
        held = source.get(row.id)
        for saved in held.checkpoints if held is not None else []:
            entry = {
                "name": saved.name,
                "digest": saved.digest,
                # Never null in the table; a run read without one is from
                # before it, and is dated by the run
                "saved_at": saved.saved_at or row.created_at,
            }
            source_file = saved.path
            if checkpoints and source_file is not None and source_file.exists():
                destination = target.checkpoint_path(row.id, saved.name)
                report.checkpoints += 1
                if not dry_run:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if source_file.is_dir():
                        shutil.copytree(source_file, destination)
                    else:
                        shutil.copy2(source_file, destination)
                entry["path"] = str(destination)
            else:
                # Not asked for, or asked for and not there: the path is
                # cleared. docs/adr/0005
                entry["path"] = None
            kept.append(entry)
        # The run's column names its last, as it did where it came from
        values["checkpoint"] = kept[-1]["path"] if kept else None

        report.runs += 1
        if not dry_run:
            with target.engine.begin() as conn:
                conn.execute(insert(t.run).values(**values))
                if kept:
                    conn.execute(insert(t.checkpoint), [{**c, "run_id": row.id} for c in kept])

        report.metrics += _copy(
            source, target, t.metric, t.metric.c.run_id, row.id, dry_run, drop_id=True
        )
        report.samples += _copy(
            source, target, t.run_sample, t.run_sample.c.run_id, row.id, dry_run
        )
        if predictions:
            report.predictions += _copy(
                source, target, t.prediction, t.prediction.c.run_id, row.id, dry_run
            )

    return report


def _runs(store: RunStore):
    with store.engine.connect() as conn:
        # Oldest first, a parent before its child. docs/adr/0005
        return list(conn.execute(select(t.run).order_by(t.run.c.created_at, t.run.c.id)))


def _ids(store: RunStore) -> set[str]:
    with store.engine.connect() as conn:
        return set(conn.execute(select(t.run.c.id)).scalars())


def _copy(
    source, target, table, run_column, run_id: str, dry_run: bool, drop_id: bool = False
) -> int:
    """Copy one run's rows out of a side table."""
    with source.engine.connect() as conn:
        rows = list(conn.execute(select(table).where(run_column == run_id)))
    if not rows:
        return 0
    payload = []
    for row in rows:
        values = {c.name: getattr(row, c.name) for c in table.columns}
        if drop_id:
            # A metric's own id is that store's numbering; the target mints
            # its own. docs/adr/0005
            values.pop("id", None)
        payload.append(values)
    if not dry_run:
        with target.engine.begin() as conn:
            conn.execute(insert(table), payload)
    return len(payload)


__all__ = ["StoreMergeError", "StoreMergeReport", "merge_stores"]
