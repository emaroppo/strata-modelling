"""A media model with no ML framework behind it.

In its own module for the reason ``counting_model`` is. It learns nothing:
it remembers what it was shown, keeps a checkpoint per epoch when it is
handed the means to, and "generates" by writing down what it was asked
with, so a test can read back which input, setting, seed and checkpoint
made each output.
"""

ECHO = "echo.py:EchoModel"

ECHO_MODEL = """
import json
from pathlib import Path

from strata.contracts import SFTTextExample
from strata.modelling import Generated, MediaModel


class EchoModel(MediaModel):
    trains_on = SFTTextExample
    version = "1"

    def __init__(self, epochs: int = 3):
        self.epochs = epochs
        self.epoch = 0
        self.seen = {"targets": [], "features": [], "val": 0}

    def finetune(self, train, val=None, on_epoch=None, save_checkpoint=None):
        self.seen = {
            "targets": [e.target for e in train],
            "features": [e.features for e in train],
            "val": len(val or []),
        }
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            if on_epoch is not None:
                on_epoch(epoch, self.epochs, {"loss": 1.0 / epoch})
            if save_checkpoint is not None:
                save_checkpoint(f"epoch-{epoch}")
        return {"n_train": float(len(train)), "n_val": float(len(val or []))}

    def generate(self, inputs, settings, seeds, on_batch=None):
        return [
            Generated(
                data=json.dumps(
                    {
                        "values": i.values,
                        "samples": {
                            role: [Path(p).read_text() for p in paths]
                            for role, paths in i.samples.items()
                        },
                        "settings": s,
                        "seed": seed,
                        "epoch": self.epoch,
                    },
                    sort_keys=True,
                ).encode(),
                ended=seed % 2 == 0,
            )
            for i, s, seed in zip(inputs, settings, seeds, strict=True)
        ]

    def save(self, path):
        Path(path).write_text(json.dumps({"epoch": self.epoch, "seen": self.seen}))

    def load(self, path):
        self.epoch = json.loads(Path(path).read_text())["epoch"]
"""
