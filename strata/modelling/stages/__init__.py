"""Modelling's stages: ``train`` records a run, here or on a host; ``evaluate`` scores one with the
tasks asked.

See ``docs/adr/0035``, ``docs/adr/0037`` and ``docs/adr/0054``.
"""

from strata.common.stages import Stage

from ._context import (
    DATASET_DIR,
    METRICS,
    RUN,
    Context,
    DatasetIdentity,
    Host,
    Keyed,
    StageError,
)
from .evaluate import (
    DEFAULT_GENERATION_TASKS,
    DEFAULT_TASKS,
    CheckpointScores,
    ClassScore,
    EvaluateRecord,
    EvaluateRequest,
    GeneratedText,
    TaskRef,
    evaluate,
    makes,
    resolved,
    sample_seed,
)
from .train import TrainRecord, TrainStageRequest, train

STAGES = (
    Stage("train", "1", (DATASET_DIR,), RUN, train),
    # 3: tasks with identities; a request naming no later field keys as before. docs/adr/0042,
    # docs/adr/0054
    Stage("evaluate", "3", (DATASET_DIR, RUN), METRICS, evaluate),
)


__all__ = [
    "DATASET_DIR",
    "DEFAULT_GENERATION_TASKS",
    "DEFAULT_TASKS",
    "METRICS",
    "RUN",
    "STAGES",
    "CheckpointScores",
    "ClassScore",
    "Context",
    "DatasetIdentity",
    "EvaluateRecord",
    "EvaluateRequest",
    "GeneratedText",
    "Host",
    "Keyed",
    "StageError",
    "TaskRef",
    "TrainRecord",
    "TrainStageRequest",
    "evaluate",
    "makes",
    "resolved",
    "sample_seed",
    "train",
]
