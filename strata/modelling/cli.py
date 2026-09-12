"""``strata-runs``: looking after a run store from the command line.

One command for now, ``merge``, which folds one store into another. A
project keeps runs beside its own checkpoints and a modelling host keeps
its own, so a project trained on both has its history split in two.
Standard library only, for the same reason as ``strata-catalog``.
"""

import argparse
import json
import sys
from pathlib import Path

from .merge import StoreMergeError, merge_stores
from .runs import RunStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="strata-runs", description="Look after a run store.")
    parser.add_argument("--json", action="store_true", help="Print the record as JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    merge = commands.add_parser("merge", help="Fold another run store into this one")
    merge.add_argument(
        "--from", dest="source", type=Path, required=True, metavar="DIR",
        help="A runs directory: runs.db and its checkpoints",
    )
    merge.add_argument(
        "--into", dest="target", type=Path, required=True, metavar="DIR",
        help="The runs directory to fold it into",
    )
    merge.add_argument(
        "--checkpoints", action="store_true", help="Copy the checkpoint files too. They are large."
    )
    merge.add_argument("--apply", action="store_true", help="Write; otherwise only report")
    merge.set_defaults(run=_merge)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except StoreMergeError as e:
        print(str(e), file=sys.stderr)
        return 1


def _merge(args) -> int:
    if not (args.source / "runs.db").exists():
        print(f"No runs.db under {args.source}.", file=sys.stderr)
        return 1
    report = merge_stores(
        RunStore.local(args.source),
        RunStore.local(args.target),
        checkpoints=args.checkpoints,
        dry_run=not args.apply,
    )
    if args.json:
        payload = {
            "runs": report.runs,
            "metrics": report.metrics,
            "predictions": report.predictions,
            "checkpoints": report.checkpoints,
            "already_present": report.already_present,
            "orphaned": report.orphaned,
            "applied": args.apply,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not report.runs and not report.already_present:
        print(f"No runs in {args.source}.")
        return 0
    for line in report.lines():
        print(f"  {line}")
    if report.orphaned:
        print(
            "Some runs continued from a run in neither store. They were copied "
            "without the link, so they read as cold."
        )
    if not args.apply:
        print("Nothing was written. Re-run with --apply.")
        return 0
    print(f"{report.runs} run(s) merged")
    if not args.checkpoints and report.runs:
        print(
            "Checkpoints were left behind, so the merged runs cannot be trained or "
            "predicted from here. Pass --checkpoints if you need them."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
