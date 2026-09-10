"""Shared fixtures.

The model under test carries no ML framework at all. That is not only for
speed: it is the standing proof that the interface has no torch shaped into
it, and that a model bringing its own dependencies — or none — is a
first-class citizen rather than a special case.
"""

import json
from pathlib import Path

import pytest
from counting_model import COUNTER, COUNTING_MODEL

from strata.labels import ChoicesPrediction
from strata.modelling import RunStore


@pytest.fixture
def store(tmp_path) -> RunStore:
    return RunStore.local(tmp_path / "runs")


@pytest.fixture
def dataset_dir(tmp_path):
    """A materialised dataset, written by hand.

    Hand-written rather than produced by the catalog, so this suite tests
    the manifest *contract* rather than agreeing with one implementation of
    it — and so modelling keeps no catalog dependency.
    """

    def _make(
        n_train: int = 8,
        n_val: int = 2,
        classes: tuple[str, ...] = ("cat", "dog"),
        n_skipped: int = 0,
        name: str = "d",
        version: int = 1,
    ) -> Path:
        root = tmp_path / f"{name}-v{version}"
        (root / "files").mkdir(parents=True, exist_ok=True)
        (root / COUNTER.split(":")[0]).write_text(COUNTING_MODEL)

        samples = []
        for i in range(n_train + n_val + n_skipped):
            relative = f"files/img{i:03d}.jpg"
            (root / relative).write_bytes(f"image {i}".encode())
            skipped = i >= n_train + n_val
            samples.append(
                {
                    "id": i + 1,
                    "checksum": f"{i:064d}",
                    "path": relative,
                    "group_id": None,
                    "split": "val" if n_train <= i < n_train + n_val else "train",
                    "value": None if skipped else {"kind": "choices", "values": [classes[0]]},
                }
            )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset": name,
                    "version": version,
                    "label_set": "presence",
                    "label_schema": {
                        "task": "classification",
                        "classes": list(classes),
                        "multiple": True,
                    },
                    "val_ratio": 0.2,
                    "val_ratio_achieved": n_val / max(n_train + n_val, 1),
                    "samples": samples,
                }
            )
        )
        return root

    return _make


@pytest.fixture
def prediction() -> ChoicesPrediction:
    return ChoicesPrediction(values=["cat"], confidences=[0.8])
