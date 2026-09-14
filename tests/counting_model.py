"""A model with no ML framework behind it.

Lives in its own module rather than in ``conftest`` because every
package's tests directory would otherwise claim the name ``tests.conftest``
and the first one imported wins.
"""

COUNTER = "counter.py:CountingModel"

#: A model with no dependency on anything. It records what it was given, so
#: a test can assert on the split it received, and predicts the class it saw
#: most often — enough to be deterministic without being a real learner.
COUNTING_MODEL = """
import json
from pathlib import Path

from strata.labels import ChoicesPrediction
from strata.modelling import Model


class CountingModel(Model):
    task = "classification"
    version = "1"

    def __init__(self, bias: float = 0.5):
        self.bias = bias
        self.classes = []
        self.seen = {"train": 0, "val": 0, "warm_started": False}

    def finetune(self, train, classes, val=None, on_epoch=None):
        self.classes = list(classes)
        self.seen["train"] = len(train)
        self.seen["val"] = len(val or [])
        return {
            "accuracy": self.bias,
            "val_accuracy": self.bias / 2,
            "n_train": float(len(train)),
            "n_val": float(len(val or [])),
        }

    def predict(self, paths, on_batch=None, *, features=None):
        top = self.classes[0] if self.classes else "unknown"
        return [ChoicesPrediction(values=[top], confidences=[self.bias]) for _ in paths]

    def save(self, path):
        Path(path).write_text(json.dumps({"classes": self.classes, "bias": self.bias}))

    def load(self, path):
        payload = json.loads(Path(path).read_text())
        self.classes = payload["classes"]
        self.bias = payload["bias"]
        self.seen["warm_started"] = True
"""
