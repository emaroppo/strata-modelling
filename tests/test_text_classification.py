"""The two document-classification heads, and what separates them.

A label set says whether its classes are mutually exclusive, and until now
nothing read that: the only text head scored every class independently, so
pointing a single-choice project at it trained a model that could assert two
classes where the label set permits one — and that lands as a pre-annotation
on a control that will not display it.

Needs the text extra. The tokenizer is built in-process from a tiny
vocabulary; nothing here is about whether a transformer learns.
"""

import pytest

torch = pytest.importorskip("torch", reason="needs the text extra")
pytest.importorskip("transformers", reason="needs the text extra")

from strata.labels import (  # noqa: E402
    Choices,
    ChoicesPrediction,
    ClassificationSchema,
)
from strata.modelling import Example  # noqa: E402
from strata.modelling.baselines.text import (  # noqa: E402
    TextClassifier,
    TextMulticlassClassifier,
)

DOCUMENT = "alpha beta gamma"


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3, "gamma": 4}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


class _Tiny(torch.nn.Module):
    """Stands in for the encoder, and takes a class index as its target."""

    def __init__(self, num_labels):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        logits = self.head(self.embed(input_ids.clamp(max=15)).mean(dim=1))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
        return type("Output", (), {"loss": loss, "logits": logits})()


@pytest.fixture
def single(tokenizer, monkeypatch):
    model = TextMulticlassClassifier(device="cpu", num_epochs=1, batch_size=1)
    model._tokenizer = tokenizer
    monkeypatch.setattr(type(model), "_build_model", lambda self, n: _Tiny(n))
    return model


# ----------------------------------------------------------------------
# Which head is for which label set
# ----------------------------------------------------------------------


def test_the_sigmoid_head_refuses_a_single_choice_label_set(tokenizer):
    # It scores every class independently, so it can assert two at once
    with pytest.raises(ValueError, match="single-choice"):
        TextClassifier(device="cpu").requires_schema(
            ClassificationSchema(classes=["a", "b"], multiple=False)
        )


def test_the_softmax_head_refuses_a_multi_choice_label_set(single):
    with pytest.raises(ValueError, match="multi-choice"):
        single.requires_schema(ClassificationSchema(classes=["a", "b"], multiple=True))


def test_each_refusal_names_the_other_head(single):
    # A refusal that does not say what to use instead reads as a dead end
    with pytest.raises(ValueError, match="text-multiclass"):
        TextClassifier(device="cpu").requires_schema(
            ClassificationSchema(classes=["a"], multiple=False)
        )
    with pytest.raises(ValueError, match="'text'"):
        single.requires_schema(ClassificationSchema(classes=["a"], multiple=True))


def test_each_accepts_the_label_set_it_is_for(single):
    assert single.requires_schema(ClassificationSchema(classes=["a"], multiple=False)) is None
    assert (
        TextClassifier(device="cpu").requires_schema(
            ClassificationSchema(classes=["a"], multiple=True)
        )
        is None
    )


# ----------------------------------------------------------------------
# What it predicts
# ----------------------------------------------------------------------


def test_it_names_exactly_one_class(single, tmp_path):
    document = tmp_path / "doc.txt"
    document.write_text(DOCUMENT)
    single.finetune([Example(path=document, target=Choices(values=["alpha"]))], ["alpha", "beta"])

    [prediction] = single.predict([document])
    assert len(prediction.values) == 1
    assert prediction.values[0] in ("alpha", "beta")
    assert len(prediction.confidences) == 1


def test_its_confidence_is_a_share_of_one(single, tmp_path):
    document = tmp_path / "doc.txt"
    document.write_text(DOCUMENT)
    single.finetune([Example(path=document, target=Choices(values=["alpha"]))], ["alpha", "beta"])

    [prediction] = single.predict([document])
    # Softmax over the classes, so the winner's score is what is left after
    # the others — unlike a sigmoid, where every class can score 0.9
    assert 0.5 <= prediction.confidences[0] <= 1.0


# ----------------------------------------------------------------------
# "None of these", which this head cannot say
# ----------------------------------------------------------------------


def test_a_document_with_no_class_is_not_trained_on(single, tmp_path):
    """Encoding it as class zero would teach the first class every time.

    An empty value is a real answer — a reviewer looked and found none of
    the classes present — and a softmax head has no way to give it back.
    """
    document = tmp_path / "doc.txt"
    document.write_text(DOCUMENT)
    windows, noticed = single._scan(DOCUMENT, Choices(values=[]))
    assert windows == 0
    assert noticed == {"empty_targets": 1}


def test_the_dropped_ones_are_counted_on_the_run(single, tmp_path):
    answered = tmp_path / "answered.txt"
    answered.write_text(DOCUMENT)
    empty = tmp_path / "empty.txt"
    empty.write_text("beta gamma alpha")

    metrics = single.finetune(
        [
            Example(path=answered, target=Choices(values=["alpha"])),
            Example(path=empty, target=Choices(values=[])),
        ],
        ["alpha", "beta"],
    )
    # A corpus quietly training on less than it holds is the fault this
    # whole family of counts exists for
    assert metrics["train_empty_targets"] == 1


def test_a_document_with_a_class_still_trains(single, tmp_path):
    document = tmp_path / "doc.txt"
    document.write_text(DOCUMENT)
    windows, noticed = single._scan(DOCUMENT, Choices(values=["alpha"]))
    assert windows == 1
    assert noticed.get("empty_targets", 0) == 0


# ----------------------------------------------------------------------
# Combining windows
# ----------------------------------------------------------------------


def test_it_will_not_take_the_union_of_its_windows():
    # "any" means the union of what the windows asserted, which is not an
    # answer a single-label head is allowed to give
    with pytest.raises(ValueError, match="cannot combine windows by 'any'"):
        TextMulticlassClassifier(device="cpu", window=64, window_aggregation="any")


def test_the_sigmoid_head_still_takes_the_union():
    assert (
        TextClassifier(device="cpu", window=64, window_aggregation="any").window_aggregation
        == "any"
    )


def test_an_unknown_strategy_is_refused_at_construction():
    """Not at merge time, which is a whole training run later.

    `_merge` runs when a document needs combining, so a typo here used to
    survive training and fail on the first prediction.
    """
    with pytest.raises(ValueError, match="cannot combine windows by 'meen'"):
        TextClassifier(device="cpu", window=64, window_aggregation="meen")


def test_one_class_comes_out_of_several_windows(single):
    single.classes = ["alpha", "beta"]
    single.window_aggregation = "max"
    merged = single._merge(
        DOCUMENT,
        [
            ChoicesPrediction(values=["alpha"], confidences=[0.6]),
            ChoicesPrediction(values=["beta"], confidences=[0.9]),
        ],
    )
    assert merged.values == ["beta"]
    assert merged.confidences == [0.9]


def test_mean_counts_the_windows_that_said_nothing(single):
    """A class that won once in three does not outrank one that won twice."""
    single.classes = ["alpha", "beta"]
    single.window_aggregation = "mean"
    merged = single._merge(
        DOCUMENT,
        [
            ChoicesPrediction(values=["alpha"], confidences=[0.6]),
            ChoicesPrediction(values=["alpha"], confidences=[0.6]),
            ChoicesPrediction(values=["beta"], confidences=[0.95]),
        ],
    )
    # alpha: (0.6 + 0.6) / 3 = 0.4; beta: 0.95 / 3 ≈ 0.317
    assert merged.values == ["alpha"]


# ----------------------------------------------------------------------
# The checkpoint
# ----------------------------------------------------------------------


def test_a_checkpoint_round_trips_under_the_restricted_loader(single, tokenizer, tmp_path):
    # A checkpoint is tensors, the class list and the encoder name, so the
    # loader that refuses arbitrary pickles is enough (docs/adr/0005)
    document = tmp_path / "doc.txt"
    document.write_text(DOCUMENT)
    single.finetune([Example(path=document, target=Choices(values=["alpha"]))], ["alpha", "beta"])
    path = tmp_path / "run.pt"
    single.save(path)

    again = TextMulticlassClassifier(device="cpu", num_epochs=1, batch_size=1)
    again._tokenizer = tokenizer
    again._build_model = lambda n: _Tiny(n)
    again.load(path)

    assert again.classes == ["alpha", "beta"]
    assert again.encoder == single.encoder
    before, after = single.predict([document]), again.predict([document])
    assert after[0].values == before[0].values
    assert after[0].confidences == pytest.approx(before[0].confidences)
