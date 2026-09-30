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

from strata.contracts import ChoicesPrediction
from strata.modelling import Model


class CountingModel(Model):
    label_type = "classification"
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


SPAN_COUNTER = "span_counter.py:SpanCountingModel"

#: A span model with nothing behind it, which splits the name in every
#: document it is shown into its two words: the fragmentation the mask task
#: exists to count.
SPAN_MODEL = """
import json
from pathlib import Path

from strata.contracts import Span, SpansPrediction
from strata.modelling import Model


class SpanCountingModel(Model):
    label_type = "span"
    version = "1"

    def finetune(self, train, classes, val=None, on_epoch=None):
        return {"n_train": float(len(train))}

    def predict(self, paths, on_batch=None, *, features=None):
        return [
            SpansPrediction(
                values=[
                    Span(labels=["PER"], start=5, end=9, text="John"),
                    Span(labels=["PER"], start=10, end=15, text="Smith"),
                ],
                confidences=[0.9, 0.8],
            )
            for _ in paths
        ]

    def save(self, path):
        Path(path).write_text(json.dumps({}))

    def load(self, path):
        pass
"""
