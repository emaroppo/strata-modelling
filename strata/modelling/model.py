"""The contract a model implements.

Four methods, and no knowledge of where its data came from. A model is
handed file paths and values; it never sees a catalog, a database or Label
Studio. See ``docs/adr/0004``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from strata.labels import AnyPrediction, AnyValue

#: What a model reports as it trains: epochs done, epochs in total, and
#: whatever it knows so far. The metrics are a snapshot rather than a
#: result — the returned value is the result.
#:
#: The callback may answer ``True`` to ask the model to stop early. See
#: :meth:`Model.finetune`: honouring it is optional.
EpochReport = Callable[[int, int, dict[str, float]], bool | None]

#: What a model reports as it scores: samples done, samples in total.
#: Nothing more, because a prediction in progress says nothing useful and
#: the answer is the returned value.
BatchReport = Callable[[int, int], None]


@dataclass(frozen=True)
class Example:
    """One labelled sample as a model sees it.

    ``target`` is whatever the label set stores — choices for
    classification, spans for a tagger, boxes for a detector. See
    ``docs/adr/0004``.
    """

    path: Path
    target: AnyValue
    #: What is already known about this sample and may be told to the
    #: model — a species, a coordinate. Empty for a project that declares
    #: none. Plain values rather than :mod:`strata.labels` ones. See
    #: ``docs/adr/0011``.
    features: dict = field(default_factory=dict)


class Model(ABC):
    """Subclass this and implement the four methods below.

    Targets and outputs are :mod:`strata.labels` values rather than any
    storage format, so a model neither knows nor cares that annotations came
    out of Label Studio.
    """

    #: Which task this model handles. Checked before training. See
    #: ``docs/adr/0014``.
    task: ClassVar[str] = "classification"

    #: Classes this model emits that a caller would not otherwise declare.
    #: The label set has to declare them, and training refuses if it does
    #: not. Read from the built model, not the class. See ``docs/adr/0014``.
    requires_classes: tuple[str, ...] = ()

    #: Features this model cannot predict without. Declared the way
    #: :attr:`requires_classes` is, and checked the same way, before the
    #: round. "Cannot predict without", not "cannot train without". See
    #: ``docs/adr/0011``.
    requires_features: tuple[str, ...] = ()

    #: Bumped when a change makes existing checkpoints unreadable. A run
    #: records it, and warm-starting from a checkpoint written by a
    #: different version is refused. See ``docs/adr/0005``.
    version: ClassVar[str] = "1"

    def requires_schema(self, schema) -> None:
        """Refuse a label set this model cannot learn from. Raise, or return.

        For a label set of the right task whose *shape* the model cannot
        represent. Raise ``ValueError`` naming what would be needed;
        training refuses before the round. The default accepts anything.
        See ``docs/adr/0014``.
        """
        return None

    @abstractmethod
    def finetune(
        self,
        train: list[Example],
        classes: list[str],
        val: list[Example] | None = None,
        on_epoch: "EpochReport | None" = None,
    ) -> dict[str, float]:
        """Train, and return metrics.

        ``val`` is held out: evaluate on it afterwards and include the
        results, conventionally prefixed ``val_``.

        ``on_epoch(done, total, metrics)`` is called as training proceeds.
        Calling it is optional and a model that ignores it still conforms;
        the caller must treat silence as "no news", never as "stalled".

        The callback may return ``True`` to ask the model to stop early. A
        model that honours it stops after the epoch it just reported and
        returns its metrics as they stand; a model that ignores the answer
        still conforms. See ``docs/adr/0031``.
        """
        ...

    @abstractmethod
    def predict(
        self,
        paths: list[Path],
        on_batch: "BatchReport | None" = None,
        *,
        features: list[dict] | None = None,
    ) -> list[AnyPrediction]:
        """One prediction per path, in order.

        ``features`` is positional against ``paths``: the nth entry belongs
        to the nth path. None where the project declares none. Keyword-only
        because it arrived after ``on_batch`` (``docs/adr/0011``).

        ``on_batch(done, total)`` is called as scoring proceeds, like
        :meth:`finetune`'s ``on_epoch``. A model may ignore it but has to
        accept it. See ``docs/adr/0031``.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write everything needed to rebuild this model."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore from :meth:`save`."""
        ...
