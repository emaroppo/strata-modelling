"""The contract a model implements.

A :class:`LabelModel` predicts, a :class:`MediaModel` generates; either is
handed paths and values, never a catalog. See ``docs/adr/0004`` and ``docs/adr/0054``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from strata.contracts import AnyPrediction, AnyValue, SampleType

#: Reported as a model trains: epochs done, epochs in total, metrics so far. May answer ``True`` to
#: ask it to stop.
EpochReport = Callable[[int, int, dict[str, float]], bool | None]

#: Reported as a model scores: samples done, samples in total.
BatchReport = Callable[[int, int], None]

#: Handed to ``finetune``: called with a name, keeps the model as it stands as a checkpoint
#: (``docs/adr/0054``).
SaveCheckpoint = Callable[[str], None]


@dataclass(frozen=True)
class Example:
    """One labelled sample as a model sees it: a path, the label set's value, its features. See
    ``docs/adr/0004``.
    """

    path: Path
    #: None for a :class:`MediaModel`, whose target is the file at ``path``.
    target: AnyValue | None
    #: What the model may be told about this sample, as plain values; empty when none are declared
    #: (``docs/adr/0011``).
    features: dict = field(default_factory=dict)


class Model(ABC):
    """What every trainable model shares. Subclass :class:`LabelModel` or :class:`MediaModel`.

    A direct subclass is refused. See ``docs/adr/0054``.
    """

    #: Features this model cannot predict or generate without; checked before the round
    #: (``docs/adr/0011``).
    requires_features: tuple[str, ...] = ()

    #: Bumped when a change makes existing checkpoints unreadable; a warm start across versions is
    #: refused (``docs/adr/0005``).
    version: ClassVar[str] = "1"

    def resolved_params(self) -> dict | None:
        """The parameters as used, defaults filled in, or None to record those asked for.

        Asked after ``finetune`` and recorded on the run; each must be a
        constructor argument, since the model is rebuilt from them. See ``docs/adr/0054``.
        """
        return None

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write everything needed to rebuild this model."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore from :meth:`save`."""
        ...


class LabelModel(Model):
    """A model that learns a label set and predicts on samples, in ``strata.contracts`` values."""

    #: The label type it learns and emits: classification, span or bbox (``docs/adr/0014``,
    #: ``docs/adr/0041``).
    label_type: ClassVar[str] = "classification"

    #: Classes this model emits that the label set must declare; read from the built model
    #: (``docs/adr/0014``).
    requires_classes: tuple[str, ...] = ()

    def requires_schema(self, schema) -> None:
        """Raise ``ValueError`` for a label set of the right type whose shape this model cannot
        represent.

        The default accepts anything. See ``docs/adr/0014``.
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
        """Train, and return metrics, those on the held-out ``val`` prefixed ``val_``.

        ``on_epoch(done, total, metrics)`` may be ignored, and may answer
        ``True`` to ask for an early stop (``docs/adr/0031``). A ``finetune``
        may also take ``save_checkpoint``, a :data:`SaveCheckpoint`, as a
        keyword; one that never calls it is kept once, at the end (``docs/adr/0054``).
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

        ``features`` is positional against ``paths``, None where none are
        declared (``docs/adr/0011``); ``on_batch`` may be ignored (``docs/adr/0031``).
        """
        ...


@dataclass(frozen=True)
class GenerationInput:
    """What one output is generated from: ``values`` by input field, ``samples`` by role.

    See ``docs/adr/0053`` and ``docs/adr/0054``.
    """

    values: dict = field(default_factory=dict)
    samples: dict[str, list[Path]] = field(default_factory=dict)


@dataclass(frozen=True)
class Generated:
    """One output, as the model made it; the caller commits it as a sample.

    ``ended`` is whether it stopped on its own, None for a media with no such notion. See
    ``docs/adr/0054``.
    """

    data: bytes
    ended: bool | None = None


class MediaModel(Model):
    """A model that learns from samples of the type it ``trains_on`` and generates new ones.

    See ``docs/adr/0053`` and ``docs/adr/0054``.
    """

    #: The sample type it learns from, subtypes included; checked before training.
    trains_on: ClassVar[type[SampleType] | None] = None

    @abstractmethod
    def finetune(
        self,
        train: list[Example],
        val: list[Example] | None = None,
        on_epoch: "EpochReport | None" = None,
        save_checkpoint: "SaveCheckpoint | None" = None,
    ) -> dict[str, float]:
        """Train, and return metrics. As :meth:`LabelModel.finetune`, without classes."""
        ...

    @abstractmethod
    def generate(
        self,
        inputs: list[GenerationInput],
        settings: list[dict],
        seeds: list[int],
        on_batch: "BatchReport | None" = None,
    ) -> list[Generated]:
        """One output per input, in order; ``settings`` and ``seeds`` are positional against
        ``inputs``. See ``docs/adr/0054``.
        """
        ...
