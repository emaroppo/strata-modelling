"""Aligning a target span to tokens, and what that loses.

Both failures here were silent, and they look identical afterwards to a
model that simply did not learn a class.

A span whose characters do not sit on token boundaries used to supervise
*nothing*: the rule asked for tokens contained by the span, and a span
cutting a token contains none of it. It now labels the tokens it overlaps,
which is the standard rule — and the count of spans that reach no token at
all, and of spans the tokens do not fit exactly, is reported on the run.

Needs the text extra. The tokenizer is built in-process from a tiny
vocabulary rather than stubbed, because offsets are the thing under test:
asserting our own arithmetic against a fake would prove nothing, and
building it rather than downloading it keeps this runnable offline.
"""

import pytest

torch = pytest.importorskip("torch", reason="needs the text extra")
pytest.importorskip("transformers", reason="needs the text extra")

from strata.labels import Span, Spans, SpanSchema  # noqa: E402
from strata.modelling import Example  # noqa: E402
from strata.modelling.baselines.text import TextSpanTagger  # noqa: E402

#: "alpha beta gamma" tokenises as three words at 0..5, 6..10 and 11..16.
DOCUMENT = "alpha beta gamma"


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3, "gamma": 4}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


@pytest.fixture
def tagger(tokenizer):
    model = TextSpanTagger(device="cpu")
    model._tokenizer = tokenizer
    model.classes = ["WORD"]
    return model


def _tagged(tagger, target) -> list[int]:
    """The tag ids this target produces, minus the padding."""
    [item] = tagger._encode(DOCUMENT, target)
    return [tag for tag in item["labels"].tolist() if tag > 0]


# ----------------------------------------------------------------------
# The alignment rule
# ----------------------------------------------------------------------


def test_a_span_on_token_boundaries_is_tagged(tagger):
    target = Spans(values=[Span(labels=["WORD"], start=0, end=5, text="alpha")])
    # B-WORD is tag 1
    assert _tagged(tagger, target) == [1]


def test_a_span_cutting_a_token_still_supervises_it(tagger):
    # "lph", inside "alpha". The old rule asked for tokens the span
    # contains, so this taught the model nothing at all.
    target = Spans(values=[Span(labels=["WORD"], start=1, end=4, text="lph")])
    assert _tagged(tagger, target) == [1]


def test_a_span_across_two_tokens_continues(tagger):
    target = Spans(values=[Span(labels=["WORD"], start=0, end=10, text="alpha beta")])
    # B-WORD then I-WORD
    assert _tagged(tagger, target) == [1, 2]


def test_a_span_over_no_token_tags_nothing(tagger):
    # The space between two words is not part of either
    target = Spans(values=[Span(labels=["WORD"], start=5, end=6, text=" ")])
    assert _tagged(tagger, target) == []


def test_a_class_the_model_does_not_have_is_not_tagged(tagger):
    target = Spans(values=[Span(labels=["ELSEWHERE"], start=0, end=5)])
    assert _tagged(tagger, target) == []


# ----------------------------------------------------------------------
# What it reports
# ----------------------------------------------------------------------


def _counts(tagger, target) -> dict:
    windows, noticed = tagger._scan(DOCUMENT, target)
    assert windows == 1
    return noticed


def test_a_span_that_fits_its_tokens_counts_as_neither(tagger):
    target = Spans(values=[Span(labels=["WORD"], start=0, end=5)])
    assert _counts(tagger, target) == {"spans_unaligned": 0, "spans_inexact": 0}


def test_a_span_reaching_no_token_is_counted(tagger):
    target = Spans(values=[Span(labels=["WORD"], start=5, end=6)])
    assert _counts(tagger, target)["spans_unaligned"] == 1


def test_a_span_the_tokens_do_not_fit_is_counted_separately(tagger):
    # It trains; what it cannot do is come back out at these offsets, which
    # is a gap between exact and partial F1 rather than a bug
    target = Spans(values=[Span(labels=["WORD"], start=1, end=4)])
    counts = _counts(tagger, target)
    assert (counts["spans_unaligned"], counts["spans_inexact"]) == (0, 1)


def test_a_document_with_no_spans_reports_zero_rather_than_nothing(tagger):
    # A run reporting 0 is evidence; a run reporting no number at all is
    # indistinguishable from a version that never counted
    assert _counts(tagger, Spans()) == {"spans_unaligned": 0, "spans_inexact": 0}


def test_the_counts_reach_the_run(tagger, tmp_path, monkeypatch):
    """They are metrics, so `report` prints them beside the F1 they explain."""

    class _Tiny(torch.nn.Module):
        def __init__(self, num_labels):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, num_labels)
            self.num_labels = num_labels

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            logits = self.head(self.embed(input_ids.clamp(max=15)))
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, self.num_labels),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            return type("Output", (), {"loss": loss, "logits": logits})()

    monkeypatch.setattr(type(tagger), "_build_model", lambda self, n: _Tiny(n))
    tagger.num_epochs = 1
    tagger.batch_size = 1

    document = tmp_path / "doc.txt"
    document.write_text(DOCUMENT)
    metrics = tagger.finetune(
        [
            Example(
                path=document,
                target=Spans(
                    values=[
                        Span(labels=["WORD"], start=1, end=4),
                        Span(labels=["WORD"], start=5, end=6),
                    ]
                ),
            )
        ],
        ["WORD"],
    )
    assert metrics["train_spans_inexact"] == 1
    assert metrics["train_spans_unaligned"] == 1


# ----------------------------------------------------------------------
# What it will not train on at all
# ----------------------------------------------------------------------


def test_it_refuses_a_label_set_with_overlapping_regions(tagger):
    with pytest.raises(ValueError, match="overlapping"):
        tagger.requires_schema(SpanSchema(classes=["WORD"], overlapping=True))


def test_it_refuses_a_label_set_with_multi_label_regions(tagger):
    with pytest.raises(ValueError, match="more than one label"):
        tagger.requires_schema(SpanSchema(classes=["WORD"], multi_label=True))


def test_it_says_what_would_be_needed_instead(tagger):
    # A refusal that does not name the alternative reads as a limitation
    # nobody thought about
    with pytest.raises(ValueError, match="head per class"):
        tagger.requires_schema(SpanSchema(classes=["WORD"], overlapping=True))


def test_the_ordinary_label_set_is_accepted(tagger):
    assert tagger.requires_schema(SpanSchema(classes=["WORD"])) is None
