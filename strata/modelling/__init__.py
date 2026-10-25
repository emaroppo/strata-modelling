"""Training, prediction and generation, and the record of what was trained.

Plugins, runs, metrics and checkpoints; imports ``contracts`` and a
framework behind an extra, never the labeller. See ``docs/adr/0004``,
``docs/adr/0005``, ``docs/adr/0006`` and ``docs/adr/0054``.
"""

from .handlers import TrainingError, examples, generate, predict, train
from .model import Example, Generated, GenerationInput, LabelModel, MediaModel, Model
from .plugins.registry import ENTRY_POINT_GROUP, ModelError, absolute, available, resolve
from .requests import (
    Checkpoint,
    GeneratedOutput,
    GenerateRequest,
    PredictRequest,
    Run,
    ScoredPath,
    TrainRequest,
)
from .store.merge import StoreMergeError, StoreMergeReport, merge_stores
from .store.predictions import PredictionCache
from .store.runs import RunStore, RunStoreError, RunStoreMissing, Seen, Unchecked

#: Modules another package may import by path (``docs/adr/0015``).
PUBLIC_MODULES = frozenset({"stages", "remote.client", "remote.wire", "plugins.registry"})

__all__ = [
    "ENTRY_POINT_GROUP",
    "Checkpoint",
    "Example",
    "GenerateRequest",
    "Generated",
    "GeneratedOutput",
    "GenerationInput",
    "LabelModel",
    "MediaModel",
    "Model",
    "ModelError",
    "PredictRequest",
    "PredictionCache",
    "Run",
    "RunStore",
    "RunStoreError",
    "RunStoreMissing",
    "ScoredPath",
    "Seen",
    "StoreMergeError",
    "StoreMergeReport",
    "TrainRequest",
    "TrainingError",
    "Unchecked",
    "absolute",
    "available",
    "examples",
    "generate",
    "merge_stores",
    "predict",
    "resolve",
    "train",
]
