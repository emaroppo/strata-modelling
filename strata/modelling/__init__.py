"""Training and prediction, and the record of what was trained.

The model catalog: plugins, runs, metrics and checkpoints. A run names the
dataset version and model version behind it, so a checkpoint resolves back
to the exact samples and annotations that produced it.

**May import:** ``labels``, an ML framework behind the optional extras, and
``catalog`` *only* from the service layer.

**May not import:** ``strata.labeller``, or Label Studio.

The training core takes a directory and a manifest, nothing else
(``docs/adr/0004``); the service layer materialises before invoking it.
Runs are a chain and carry the class list as trained (``docs/adr/0005``).
Predictions are returned by the core, not written; ``PredictionCache`` is
that persistence (``docs/adr/0006``).
"""

from .handlers import TrainingError, examples, predict, train
from .merge import StoreMergeError, StoreMergeReport, merge_stores
from .model import Example, Model
from .predictions import PredictionCache
from .registry import ENTRY_POINT_GROUP, ModelError, available, resolve
from .requests import PredictRequest, Run, ScoredPath, TrainRequest
from .runs import RunStore

__all__ = [
    "ENTRY_POINT_GROUP",
    "Example",
    "Model",
    "ModelError",
    "PredictRequest",
    "ScoredPath",
    "Run",
    "PredictionCache",
    "RunStore",
    "StoreMergeError",
    "StoreMergeReport",
    "merge_stores",
    "TrainRequest",
    "TrainingError",
    "available",
    "examples",
    "predict",
    "resolve",
    "train",
]
