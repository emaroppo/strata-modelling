"""``strata-runs``: looking after a run store from the command line.

One command for now, ``merge``, which folds one store into another.
Standard library only. See ``docs/adr/0005`` and ``docs/adr/0030``.
"""

import argparse
from pathlib import Path

from strata.common.cli import emit, run

from .store.merge import merge_stores
from .store.runs import RunStore, RunStoreError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="strata-runs", description="Look after a run store.")
    parser.add_argument("--json", action="store_true", help="Print the record as JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    merge = commands.add_parser("merge", help="Fold another run store into this one")
    merge.add_argument(
        "--from",
        dest="source",
        type=Path,
        required=True,
        metavar="DIR",
        help="A runs directory: runs.db and its checkpoints",
    )
    merge.add_argument(
        "--into",
        dest="target",
        type=Path,
        required=True,
        metavar="DIR",
        help="The runs directory to fold it into",
    )
    merge.add_argument(
        "--checkpoints", action="store_true", help="Copy the checkpoint files too. They are large."
    )
    merge.add_argument("--apply", action="store_true", help="Write; otherwise only report")
    merge.set_defaults(run=_merge)

    args = parser.parse_args(argv)
    return run(lambda: args.run(args), RunStoreError)


def _merge(args) -> int:
    report = merge_stores(
        RunStore.open(args.source),
        RunStore.local(args.target),
        checkpoints=args.checkpoints,
        dry_run=not args.apply,
    )
    payload = {
        "runs": report.runs,
        "metrics": report.metrics,
        "samples": report.samples,
        "predictions": report.predictions,
        "checkpoints": report.checkpoints,
        "already_present": report.already_present,
        "orphaned": report.orphaned,
        "applied": args.apply,
    }

    def render(_):
        if not report.runs and not report.already_present:
            yield f"No runs in {args.source}."
            return
        for line in report.lines():
            yield f"  {line}"
        if report.orphaned:
            yield (
                "Some runs continued from a run in neither store. They were copied "
                "without the link, so they read as cold."
            )
        if not args.apply:
            yield "Nothing was written. Re-run with --apply."
            return
        yield f"{report.runs} run(s) merged"
        if not args.checkpoints and report.runs:
            yield (
                "Checkpoints were left behind, so the merged runs cannot be trained or "
                "predicted from here. Pass --checkpoints if you need them."
            )

    emit(payload, render, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
