"""Modelling's stages: training and scoring as an experiment asks for them.

``train`` takes a materialised directory and returns the run it recorded,
here or on the modelling host — the request says which by whether the
context names a host, and the record is the same shape either way.
``evaluate`` scores one side of a directory with a recorded run and
computes the number by one implementation, which is what makes two runs'
numbers comparable. Requests and records are plain models; nothing here
names a catalog.
"""

from strata.common.stages import Stage

from ._context import DATASET_DIR, METRICS, RUN, Context, DatasetIdentity, Host, StageError
from .evaluate import ClassScore, EvaluateRecord, EvaluateRequest, evaluate
from .train import TrainRecord, TrainStageRequest, train

STAGES = (
    Stage("train", "1", (DATASET_DIR,), RUN, train),
    # 2: exact_match rather than accuracy, and per-class scores. A record
    # written by 1 has the old shape, so it is not handed back as this one's.
    Stage("evaluate", "2", (DATASET_DIR, RUN), METRICS, evaluate),
)


__all__ = [
    "DATASET_DIR",
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
    "TrainRecord",
    "TrainStageRequest",
    "evaluate",
    "train",
]
