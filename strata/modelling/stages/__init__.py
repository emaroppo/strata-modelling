"""Modelling's stages: training and scoring as an experiment asks for them.

``train`` takes a materialised directory and returns the run it recorded,
here or on the modelling host — the request says which by whether the
context names a host, and the record is the same shape either way.
``evaluate`` scores one side of a directory with a recorded run, by one
implementation (``docs/adr/0035``). Requests and records are plain models;
nothing here names a catalog.
"""

from strata.common.stages import Stage

from ._context import DATASET_DIR, METRICS, RUN, Context, DatasetIdentity, Host, StageError
from .evaluate import (
    DEFAULT_TASKS,
    ClassScore,
    EvaluateRecord,
    EvaluateRequest,
    TaskRef,
    evaluate,
    resolved,
)
from .train import TrainRecord, TrainStageRequest, train

STAGES = (
    Stage("train", "1", (DATASET_DIR,), RUN, train),
    # 2: exact_match rather than accuracy, and per-class scores. docs/adr/0035
    # 3: the tasks asked, by strata-evaluation, each with its identity. docs/adr/0042
    Stage("evaluate", "3", (DATASET_DIR, RUN), METRICS, evaluate),
)


__all__ = [
    "DATASET_DIR",
    "DEFAULT_TASKS",
    "METRICS",
    "RUN",
    "STAGES",
    "ClassScore",
    "Context",
    "DatasetIdentity",
    "EvaluateRecord",
    "EvaluateRequest",
    "Host",
    "StageError",
    "TaskRef",
    "TrainRecord",
    "TrainStageRequest",
    "evaluate",
    "resolved",
    "train",
]
