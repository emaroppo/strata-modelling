"""Windowing, and the two ways it fails quietly.

Needs the text extra and a tokenizer, because the tokenizer's offset
mapping is the thing under test: whether a span decoded out of the third
window lands at the right character in the original document is a fact
about the tokenizer, and a stub asserting our own arithmetic would prove
nothing. Skipped rather than failed where the tokenizer cannot be loaded.

What is checked here is not that windowing produces good spans — that is
the model's business — but that it produces *the same* spans a single
window would, at the same offsets, without doubling the ones that sit in
the overlap.
"""

import pytest

pytest.importorskip("transformers", reason="needs the text extra")

import torch

from strata.labels import Span, Spans, SpansPrediction
from strata.modelling.baselines.text import (
    TextClassifier,
    TextSpanTagger,
    bio,
)

CLASSES = ["PER", "ORG"]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")
    except Exception as e:  # pragma: no cover - depends on cache and network
        pytest.skip(f"no tokenizer available ({type(e).__name__})")


@pytest.fixture
def tagger(tokenizer):
    model = TextSpanTagger(window=64, window_overlap=16)
    model.classes = list(CLASSES)
    model._tokenizer = tokenizer
    return model


def _long_text(marker: str = "Alice Smith") -> tuple[str, int, int]:
    """A document several windows long, with one entity late in it."""
    filler = " ".join(f"word{i}" for i in range(400))
    text = f"{filler} {marker} {filler}"
    start = text.index(marker)
    return text, start, start + len(marker)


# -- the refusal ---------------------------------------------------------


def test_a_windowed_classifier_must_declare_how_windows_combine():
    # Several windows give several answers about one document, and max,
    # mean and any disagree. Defaulting would pick one silently.
    with pytest.raises(ValueError, match="window_aggregation"):
        TextClassifier(window=64)


def test_a_windowed_tagger_needs_no_aggregation():
    # Spans from adjacent windows are spans in the same document, so there
    # is nothing to decide and nothing to declare
    model = TextSpanTagger(window=64)
    assert (model.window, model.window_overlap) == (64, 16)


def test_the_overlap_defaults_to_a_quarter_of_the_window():
    # A fixed default is larger than a short window, and windows that do
    # not advance is the failure that produces
    assert TextSpanTagger(window=512).window_overlap == 128
    assert TextSpanTagger(window=128).window_overlap == 32


def test_windows_must_advance():
    with pytest.raises(ValueError, match="must be smaller"):
        TextSpanTagger(window=64, window_overlap=64)


# -- offsets stay absolute ----------------------------------------------


def test_a_long_document_becomes_several_windows(tagger):
    text, _, _ = _long_text()
    items = tagger._encode(text, None)
    assert len(items) > 2


def test_every_window_reports_offsets_into_the_original_document(tagger):
    text, _, _ = _long_text()
    items = tagger._encode(text, None)

    # The window after the first must begin partway through the document.
    # If offsets were window-relative every window would start near zero,
    # and every span decoded out of one would land in the wrong place.
    starts = []
    for item in items:
        real = [o for o in item["offsets"].tolist() if o != [0, 0]]
        starts.append(real[0][0])
    assert starts[0] == 0
    assert all(later > 0 for later in starts[1:])
    assert starts == sorted(starts)


def test_a_span_late_in_a_document_is_still_supervised(tagger):
    """The point of the whole exercise.

    Truncation drops this span: it sits past token 512 equivalent, so no
    window under the old scheme contained it and nothing was trained on it.
    """
    text, start, end = _long_text()
    target = Spans(values=[Span(labels=["PER"], start=start, end=end, text=text[start:end])])
    items = tagger._encode(text, target)

    begin, inside = bio.tag_ids(tagger.classes, "PER")
    tagged = [i for i, item in enumerate(items) if bool((item["labels"] == begin).any())]
    assert tagged, "the entity was in no window's supervision"

    # And it is tagged where the characters actually are
    for i in tagged:
        offsets = items[i]["offsets"].tolist()
        marked = [
            offsets[t]
            for t, label in enumerate(items[i]["labels"].tolist())
            if label in (begin, inside)
        ]
        assert min(o[0] for o in marked) >= start
        assert max(o[1] for o in marked) <= end


# -- the seam ------------------------------------------------------------


def test_an_entity_found_by_two_windows_is_reported_once(tagger):
    """The overlap means adjacent windows both see the same entity."""
    text, start, end = _long_text()
    span = Span(labels=["PER"], start=start, end=end, text=text[start:end])
    merged = tagger._merge(
        text,
        [
            SpansPrediction(values=[span], confidences=[0.7]),
            SpansPrediction(values=[span], confidences=[0.9]),
        ],
    )
    assert len(merged.values) == 1
    assert len(merged.confidences) == 1
    # The window that saw it with more context around it is the one believed
    assert merged.confidences[0] == pytest.approx(0.9)


def test_merging_keeps_confidences_paired_with_their_spans(tagger):
    """Spans sort themselves into reading order on construction.

    A merge that built the values first and attached confidences afterwards
    would have them reassigned by that sort — lengths still matching, so no
    guard could see it. This is the bug that has already shipped once.
    """
    text = "x" * 200
    late = Span(labels=["PER"], start=100, end=110, text=text[100:110])
    early = Span(labels=["ORG"], start=10, end=20, text=text[10:20])
    merged = tagger._merge(
        text,
        [
            SpansPrediction(values=[late], confidences=[0.11]),
            SpansPrediction(values=[early], confidences=[0.99]),
        ],
    )
    paired = {(s.label, s.start): c for s, c in zip(merged.values, merged.confidences, strict=True)}
    assert paired == {("ORG", 10): pytest.approx(0.99), ("PER", 100): pytest.approx(0.11)}


def test_distinct_entities_survive_the_merge(tagger):
    text = "x" * 200
    a = Span(labels=["PER"], start=10, end=20, text=text[10:20])
    b = Span(labels=["ORG"], start=30, end=40, text=text[30:40])
    merged = tagger._merge(
        text,
        [
            SpansPrediction(values=[a], confidences=[0.8]),
            SpansPrediction(values=[b], confidences=[0.6]),
        ],
    )
    assert [(s.label, s.start, s.end) for s in merged.values] == [
        ("PER", 10, 20),
        ("ORG", 30, 40),
    ]


# -- windowing off is the old behaviour ---------------------------------


def test_windowing_off_gives_exactly_one_window(tokenizer):
    model = TextSpanTagger()
    model.classes = list(CLASSES)
    model._tokenizer = tokenizer
    text, _, _ = _long_text()
    items = model._encode(text, None)
    assert len(items) == 1
    assert len(items[0]["input_ids"]) == model.MAX_LENGTH


def test_the_classifier_aggregates_across_windows(tokenizer):
    model = TextClassifier(window=64, window_aggregation="max")
    model.classes = ["a", "b"]
    model._tokenizer = tokenizer
    from strata.labels import ChoicesPrediction

    merged = model._merge(
        "irrelevant",
        [
            ChoicesPrediction(values=["a"], confidences=[0.9]),
            ChoicesPrediction(values=["a", "b"], confidences=[0.2, 0.8]),
        ],
    )
    assert merged.values[0] == "a"
    assert dict(zip(merged.values, merged.confidences, strict=True))["a"] == pytest.approx(0.9)


def test_collate_leaves_offsets_out_of_the_model(tagger):
    text, _, _ = _long_text()
    items = tagger._encode(text, None)
    batch = tagger._collate(items[:2])
    # offsets are for decoding; the forward pass would reject the keyword
    assert "offsets" not in batch
    assert isinstance(batch["input_ids"], torch.Tensor)
