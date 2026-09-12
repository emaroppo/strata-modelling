"""``strata-runs merge``: two stores become one, reported first and written on request."""

import json

from run_factory import recorded

from strata.modelling import RunStore
from strata.modelling.cli import main


def test_merge_reports_then_writes(tmp_path, capsys):
    source, target = RunStore.local(tmp_path / "gpu"), RunStore.local(tmp_path / "laptop")
    recorded(source, origin="gpu")
    recorded(source, origin="gpu")

    assert main(["merge", "--from", str(tmp_path / "gpu"), "--into", str(tmp_path / "laptop")]) == 0
    out = capsys.readouterr().out
    assert "2 run(s)" in out and "Nothing was written" in out
    assert target.history("demo", "val_accuracy") == []

    args = ["--json", "merge", "--from", str(tmp_path / "gpu"), "--into", str(tmp_path / "laptop")]
    assert main([*args, "--apply"]) == 0
    assert json.loads(capsys.readouterr().out)["runs"] == 2
    assert len(target.history("demo", "val_accuracy")) == 2


def test_a_directory_without_a_store_is_refused(tmp_path, capsys):
    assert main(["merge", "--from", str(tmp_path / "empty"), "--into", str(tmp_path / "x")]) == 1
    assert "No runs.db" in capsys.readouterr().err
