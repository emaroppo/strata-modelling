"""Entity-level scoring.

Pure functions over ``strata.labels`` values, so this needs no framework
and no tokenizer: the arithmetic is the thing under test, and the cases
that matter are the ones where a plausible-looking implementation is
wrong.
"""

import pytest

from strata.labels import Span

# span_scores is pure, but it lives in the text baselines' module, which
# needs the text extra to import at all
pytest.importorskip("torch", reason="needs the text extra")
pytest.importorskip("transformers", reason="needs the text extra")

from strata.modelling.baselines.text import span_scores


def s(label, start, end):
    return Span(labels=[label], start=start, end=end, text="x" * (end - start))


def test_a_perfect_prediction_scores_one():
    truth = [[s("PER", 0, 5), s("ORG", 10, 15)]]
    assert span_scores(truth, truth)["val_span_f1"] == 1.0


def test_finding_nothing_scores_zero_rather_than_raising():
    # "The model found nothing" is a real outcome and a common one early,
    # not a division to fall over
    scores = span_scores([[s("PER", 0, 5)]], [[]])
    assert scores["val_span_recall"] == 0.0
    assert scores["val_span_f1"] == 0.0


def test_an_empty_document_on_both_sides_is_not_a_failure():
    scores = span_scores([[]], [[]])
    assert scores["val_span_precision"] == 0.0
    assert scores["val_span_f1"] == 0.0


def test_a_wrong_label_at_the_right_place_is_not_a_match():
    scores = span_scores([[s("PER", 0, 5)]], [[s("ORG", 0, 5)]])
    assert scores["val_span_f1"] == 0.0
    # and not by overlap either: the span is right, the answer is not
    assert scores["val_span_partial_f1"] == 0.0


def test_exact_and_partial_disagree_on_a_boundary():
    """The case the two definitions exist for.

    A test where they agree proves nothing about either. Here the model
    found the entity and took one character too many — a correction for a
    reviewer, a total miss for exact scoring.
    """
    scores = span_scores([[s("PER", 0, 5)]], [[s("PER", 0, 6)]])
    assert scores["val_span_f1"] == 0.0
    assert scores["val_span_partial_f1"] == 1.0


def test_one_greedy_span_cannot_match_every_entity():
    """Otherwise saying almost nothing scores almost perfectly."""
    truth = [[s("PER", 0, 5), s("PER", 10, 15), s("PER", 20, 25)]]
    predicted = [[s("PER", 0, 25)]]
    scores = span_scores(truth, predicted)
    # One of the three paired off; the other two are misses
    assert scores["val_span_partial_recall"] < 0.4


def test_per_class_scores_are_reported_separately():
    truth = [[s("PER", 0, 5), s("ORG", 10, 15)]]
    predicted = [[s("PER", 0, 5)]]
    scores = span_scores(truth, predicted)
    assert scores["val_span_f1_PER"] == 1.0
    assert scores["val_span_f1_ORG"] == 0.0
    # The aggregate hides exactly this, which is why both are reported
    assert 0.0 < scores["val_span_f1"] < 1.0


def test_scores_run_over_many_documents():
    truth = [[s("PER", 0, 5)], [s("ORG", 0, 4)], []]
    predicted = [[s("PER", 0, 5)], [], [s("PER", 0, 3)]]
    scores = span_scores(truth, predicted)
    # one hit, one miss, one invention
    assert scores["val_span_precision"] == 0.5
    assert scores["val_span_recall"] == 0.5


def test_a_region_with_two_labels_is_two_entities():
    """Scoring regions rather than entities would flatter a half-right answer.

    The truth marks one phrase as both a name and an organisation; the
    model found only the name. That is one of two, not one of one.
    """
    truth = [[Span(labels=["PER", "ORG"], start=0, end=4)]]
    predicted = [[Span(labels=["PER"], start=0, end=4)]]
    scores = span_scores(truth, predicted)
    assert scores["val_span_recall"] == 0.5
    # And what it did say, it said correctly
    assert scores["val_span_precision"] == 1.0


def test_per_class_scores_split_a_shared_region():
    truth = [[Span(labels=["PER", "ORG"], start=0, end=4)]]
    predicted = [[Span(labels=["PER"], start=0, end=4)]]
    scores = span_scores(truth, predicted)
    assert scores["val_span_f1_PER"] == 1.0
    assert scores["val_span_f1_ORG"] == 0.0


def test_a_region_with_no_labels_asserts_nothing():
    # Not an entity, and counting it as one would invent a class
    scores = span_scores([[]], [[Span(labels=[], start=0, end=4)]])
    assert scores["val_span_precision"] == 0.0
