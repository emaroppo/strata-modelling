"""Training and prediction, and the record of what was trained.

The model catalog: plugins, runs, metrics and checkpoints. A run names the
dataset version and model version behind it, so a checkpoint resolves back
to the exact samples and annotations that produced it.

**May import:** ``labels``, an ML framework behind the optional extras, and
``catalog`` *only* from the service layer.

**May not import:** ``strata.labeller``, or Label Studio.

The training core takes a directory and a manifest — nothing else. The
service layer materialises a dataset before invoking it, which keeps the
catalog dependency in a thin outer shell rather than running through the
training code, and keeps the core testable against a fixture directory.

Two properties this has to record that are easy to leave out:

- **Runs are a chain, not a set.** Rounds warm-start from the previous
  checkpoint, so a run has a parent and its metrics only mean something
  relative to it.
- **The class list belongs on the run.** Checkpoints map output neurons to
  classes by position, so a warm start from a checkpoint whose class list
  has since been reordered corrupts silently rather than failing.

Predictions are returned by the core, not written: persisting them is the
caller's business, and doing it here would put the catalog back into it.
``PredictionCache`` is that persistence, and it lives beside the runs
because a prediction is a function of a checkpoint and some bytes.
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
