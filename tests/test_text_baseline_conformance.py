"""The text baselines, held to the same contract as any plugin.

Nothing ran these before. ``ModelContract`` covered the image classifiers,
so the text ones' training and evaluation path was never exercised by the
suite — which was demonstrated once by breaking it and watching everything
pass.

Two stubs, for two different reasons.

The *encoder* is stubbed because the real one fetches pretrained weights,
exactly as ``_TinyBackbone`` stands in for ConvNeXt. Nothing here is about
whether a transformer learns.

The *tokenizer* is not stubbed. It is built in-process from a tiny
vocabulary, because a real fast tokenizer's ``offset_mapping`` is the thing
the span path depends on: whether a span decoded out of a window lands on
the right character is a fact about the tokenizer, and asserting our own
arithmetic against a fake would prove nothing. Built rather than downloaded
so this runs offline.
"""

from typing import ClassVar

import pytest

torch = pytest.importorskip("torch", reason="needs the text extra")
pytest.importorskip("transformers", reason="needs the text extra")

from strata.labels import Choices, ClassificationSchema, Span, Spans  # noqa: E402
from strata.modelling import Example  # noqa: E402
from strata.modelling.baselines.text import (  # noqa: E402
    TextClassifier,
    TextMulticlassClassifier,
    TextSpanTagger,
)
from strata.modelling.plugins.conformance import ModelContract  # noqa: E402

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


@pytest.fixture(scope="module")
def tokenizer():
    """A real fast tokenizer over a six-word vocabulary.

    Whitespace pre-tokenisation, which is what makes the offsets real: they
    are computed from where each word actually sits in the string.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1}
    for word in WORDS:
        vocab[word] = len(vocab)
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


class _TinyEncoder(torch.nn.Module):
    """Stands in for the transformer, and returns what the heads expect."""

    def __init__(self, num_labels: int, per_token: bool):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, num_labels)
        self.per_token = per_token
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden = self.embed(input_ids.clamp(max=15))
        logits = self.head(hidden if self.per_token else hidden.mean(dim=1))
        loss = None
        if labels is not None:
            if self.per_token:
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, self.num_labels),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            elif labels.dtype == torch.long:
                # A class index per document: the single-label head. Branched
                # on the target rather than on a flag, which is what the real
                # encoder does with `problem_type`.
                loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return type("Output", (), {"loss": loss, "logits": logits})()


class _TextContract(ModelContract):
    MODEL: type = TextClassifier
    PER_TOKEN: bool = False
    #: Constructor settings, so the model a checkpoint is restored into is
    #: configured like the one that wrote it — as a real restore is, from the
    #: params recorded on the run. Set on ``model`` alone, the windowed tests
    #: compared a windowed model with an unwindowed one: padded to different
    #: lengths, the stub's logits differed, and a near-even pair of classes
    #: swapped places about one run in six.
    WINDOW: ClassVar[dict] = {}

    def _build(self, tokenizer, monkeypatch):
        instance = self.MODEL(num_epochs=1, batch_size=2, device="cpu", **self.WINDOW)
        instance._tokenizer = tokenizer
        monkeypatch.setattr(
            type(instance),
            "_build_model",
            lambda self, n: _TinyEncoder(n, per_token=self.PER_TOKEN_FLAG),
            raising=False,
        )
        type(instance).PER_TOKEN_FLAG = self.PER_TOKEN
        return instance

    @pytest.fixture
    def model(self, tokenizer, monkeypatch):
        return self._build(tokenizer, monkeypatch)

    @pytest.fixture
    def fresh(self, tokenizer, monkeypatch):
        # Pinned to CPU and given the same tokenizer: a default-constructed
        # one would try to fetch the real encoder to tokenize with.
        return self._build(tokenizer, monkeypatch)

    @pytest.fixture
    def documents(self, tmp_path):
        paths = []
        for i in range(4):
            path = tmp_path / f"doc{i}.txt"
            path.write_text("alpha beta gamma delta")
            paths.append(path)
        return paths


class TestTextClassifier(_TextContract):
    MODEL = TextClassifier
    PER_TOKEN = False

    @pytest.fixture
    def examples(self, documents):
        return [
            Example(path=path, target=Choices(values=["alpha" if i % 2 else "beta"]))
            for i, path in enumerate(documents)
        ]


class TestTextMulticlassClassifier(_TextContract):
    """The single-label head, held to the same contract.

    Its ``schema`` fixture is overridden because the plainest classification
    label set is multi-choice, and this model exists precisely to refuse
    that one.
    """

    MODEL = TextMulticlassClassifier
    PER_TOKEN = False

    @pytest.fixture
    def schema(self, classes):
        return ClassificationSchema(classes=list(classes), multiple=False)

    @pytest.fixture
    def examples(self, documents):
        return [
            Example(path=path, target=Choices(values=["alpha" if i % 2 else "beta"]))
            for i, path in enumerate(documents)
        ]


class TestWindowedMulticlassClassifier(TestTextMulticlassClassifier):
    """The same contract with windowing on.

    One class for the document out of one class per window, which is the
    part of this head windowing can break.
    """

    WINDOW: ClassVar[dict] = {"window": 8, "window_overlap": 2, "window_aggregation": "mean"}


class TestTextSpanTagger(_TextContract):
    MODEL = TextSpanTagger
    PER_TOKEN = True

    @pytest.fixture
    def examples(self, documents):
        # "alpha" at 0..5 and "gamma" at 11..16 of "alpha beta gamma delta"
        return [
            Example(
                path=path,
                target=Spans(
                    values=[
                        Span(labels=["WORD"], start=0, end=5, text="alpha"),
                        Span(labels=["OTHER"], start=11, end=16, text="gamma"),
                    ]
                ),
            )
            for path in documents
        ]


class TestWindowedSpanTagger(TestTextSpanTagger):
    """The same contract with windowing on.

    A model whose documents become several training items each still has to
    return one prediction per path, and still has to keep to its class
    list. That is the property windowing is most able to break.
    """

    WINDOW: ClassVar[dict] = {"window": 8, "window_overlap": 2}
