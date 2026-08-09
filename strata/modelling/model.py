"""The contract a model implements.

Four methods, and no knowledge of where its data came from. A model is
handed file paths and values; it never sees a catalog, a database or Label
Studio, which is what lets it be tested against a directory and run on a
machine that has neither.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from strata.labels import Choices, ChoicesPrediction

#: What a model reports as it trains: epochs done, epochs in total, and
#: whatever it knows so far. The metrics are a snapshot rather than a
#: result — the returned value is the result.
EpochReport = Callable[[int, int, dict[str, float]], None]

#: What a model reports as it scores: samples done, samples in total.
#: Nothing more, because a prediction in progress says nothing useful and
#: the answer is the returned value.
BatchReport = Callable[[int, int], None]


@dataclass(frozen=True)
class Example:
    """One labelled sample as a model sees it."""

    path: Path
    target: Choices


class Model(ABC):
    """Subclass this and implement the four methods below.

    Targets and outputs are :mod:`strata.labels` values rather than any
    storage format, so a model neither knows nor cares that annotations came
    out of Label Studio.
    """

    #: Which task this model handles. Checked before training, so a
    #: classifier pointed at a span label set fails immediately rather than
    #: after a queue wait.
    task: ClassVar[str] = "classification"

    #: Classes this model emits that a caller would not otherwise declare.
    #: A model with an implicit negative class predicts a token nothing else
    #: knows about, and an annotation tool given one silently drops the
    #: prediction — so the label set has to declare it, and training refuses
    #: if it does not.
    requires_classes: ClassVar[tuple[str, ...]] = ()

    #: Bumped when a change makes existing checkpoints unreadable. A run
    #: records it, and warm-starting from a checkpoint written by a
    #: different version is refused — output neurons map to the class list by
    #: position, so a silent mismatch corrupts rather than fails.
    version: ClassVar[str] = "1"

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

        ``on_epoch(done, total, metrics)`` is called as training proceeds,
        and exists because training is the long part and the only machine
        that can see it is the one doing it. A model served over HTTP prints
        to a journal nobody is reading; a caller watching a round otherwise
        cannot tell minute one from minute nine. Calling it is optional and
        a model that ignores it still conforms — the caller must treat
        silence as "no news", never as "stalled".
        """
        ...

    @abstractmethod
    def predict(
        self, paths: list[Path], on_batch: "BatchReport | None" = None
    ) -> list[ChoicesPrediction]:
        """One prediction per path, in order.

        ``on_batch(done, total)`` is called as scoring proceeds, for the same
        reason :meth:`finetune` takes ``on_epoch``: ranking a review queue
        means scoring every unlabelled sample, and a caller on another
        machine cannot see it happen. Calling it is optional and a model
        that ignores it still conforms — but it has to accept one, because
        refusing fails a scoring pass minutes in rather than failing the
        contract.
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
