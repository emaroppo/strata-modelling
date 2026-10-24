"""The contract a model implements.

A model makes labels or media, and the two are siblings: a
:class:`LabelModel` learns a label set and predicts on samples that exist,
a :class:`MediaModel` learns from samples and generates new ones. Neither
knows where its data came from. A model is handed file paths and values;
it never sees a catalog, a database or Label Studio. See ``docs/adr/0004``
and ``docs/adr/0054``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from strata.contracts import AnyPrediction, AnyValue, SampleType

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

#: Handed to ``finetune``: call it with a name, ``epoch-2``, to keep the
#: model as it stands as one of the run's checkpoints. Optional, like the
#: reports: a model that never calls it is kept once, at the end. See
#: ``docs/adr/0054``.
SaveCheckpoint = Callable[[str], None]


@dataclass(frozen=True)
class Example:
    """One labelled sample as a model sees it.

    ``target`` is whatever the label set stores — choices for
    classification, spans for a tagger, boxes for a detector. See
    ``docs/adr/0004``.
    """

    path: Path
    #: None for a :class:`MediaModel`, whose target is the file at ``path``.
    target: AnyValue | None
    #: What is already known about this sample and may be told to the
    #: model — a species, a coordinate. Empty for a project that declares
    #: none. Plain values rather than :mod:`strata.contracts` ones. See
    #: ``docs/adr/0011``.
    features: dict = field(default_factory=dict)


class Model(ABC):
    """What every trainable model shares. Subclass :class:`LabelModel` or :class:`MediaModel`.

    A model subclassing this directly is refused: it would not say whether
    it makes labels or media. See ``docs/adr/0054``.
    """

    #: Features this model cannot predict or generate without. Declared the
    #: way :attr:`LabelModel.requires_classes` is, and checked the same way,
    #: before the round. "Cannot predict without", not "cannot train
    #: without". See ``docs/adr/0011``.
    requires_features: tuple[str, ...] = ()

    #: Bumped when a change makes existing checkpoints unreadable. A run
    #: records it, and warm-starting from a checkpoint written by a
    #: different version is refused. See ``docs/adr/0005``.
    version: ClassVar[str] = "1"

    def resolved_params(self) -> dict | None:
        """The parameters as this model used them, its defaults filled in, or None.

        Asked after ``finetune``; the run records what comes back as its
        params, so a changed default never changes what a recorded run
        means. They are what the model is rebuilt from to predict or
        generate, so each must be a constructor argument. None, the
        default, records the parameters asked for. See ``docs/adr/0054``.
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
    """A model that learns a label set and predicts on samples.

    Targets and outputs are :mod:`strata.contracts` values rather than any
    storage format, so a model neither knows nor cares that annotations came
    out of Label Studio.
    """

    #: Which label type this model learns from and emits: classification,
    #: span or bbox. Checked before training. See ``docs/adr/0014`` and
    #: ``docs/adr/0041``.
    label_type: ClassVar[str] = "classification"

    #: Classes this model emits that a caller would not otherwise declare.
    #: The label set has to declare them, and training refuses if it does
    #: not. Read from the built model, not the class. See ``docs/adr/0014``.
    requires_classes: tuple[str, ...] = ()

    def requires_schema(self, schema) -> None:
        """Refuse a label set this model cannot learn from. Raise, or return.

        For a label set of the right label type whose *shape* the model cannot
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

        A ``finetune`` may also take ``save_checkpoint``, a
        :data:`SaveCheckpoint`, as a keyword: calling it with a name keeps
        the model as it stands as one of the run's checkpoints. Not in this
        signature, so a model written before it still overrides this one
        cleanly; the handler passes it only to a ``finetune`` that accepts
        it. A model that never calls it is kept once, at the end. See
        ``docs/adr/0054``.
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


@dataclass(frozen=True)
class GenerationInput:
    """What one output is generated from.

    ``values`` fill the prompt template, by the input fields' names.
    ``samples`` are the samples the model is given, by the role each fills
    and in order within a role: the text to summarise, the images to
    combine. Empty for a model that writes from values alone. See
    ``docs/adr/0053`` and ``docs/adr/0054``.
    """

    values: dict = field(default_factory=dict)
    samples: dict[str, list[Path]] = field(default_factory=dict)


@dataclass(frozen=True)
class Generated:
    """One output, as the model made it. Not yet a sample: the caller commits it.

    ``ended`` is whether it stopped on its own rather than at a limit, or
    None for a media with no such notion. See ``docs/adr/0054``.
    """

    data: bytes
    ended: bool | None = None


class MediaModel(Model):
    """A model that learns from samples and generates new ones.

    Its training examples are samples of the type it ``trains_on``: the
    file is the target, the features its input. What it generates becomes
    generated media in the catalog, never predictions. See
    ``docs/adr/0053`` and ``docs/adr/0054``.
    """

    #: The sample type it learns from; its subtypes are accepted too.
    #: Checked against every manifest sample's type before training.
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
        """One output per input, in order.

        ``settings`` and ``seeds`` are positional against ``inputs``: one
        run can sweep a setting, and each output says which made it. See
        ``docs/adr/0054``.
        """
        ...
