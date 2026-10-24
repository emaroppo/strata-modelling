"""Modelling's stages: training and scoring as an experiment asks for them.

``train`` takes a materialised directory and returns the run it recorded,
here or on the modelling host — the request says which by whether the
context names a host, and the record is the same shape either way.
``evaluate`` scores one side of a directory with a recorded run, by one
implementation (``docs/adr/0035``), at every checkpoint the run kept, and a
media model's by what it generates (``docs/adr/0054``). Requests and records
are plain models; nothing here names a catalog.
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
    # 2: exact_match rather than accuracy, and per-class scores. docs/adr/0035
    # 3: the tasks asked, by strata-evaluation, each with its identity. docs/adr/0042
    # Still 3 with every checkpoint scored and a media model generated from:
    # a run with one checkpoint scores as it did, and a request that names
    # none of the later fields is keyed as it was. docs/adr/0054
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
