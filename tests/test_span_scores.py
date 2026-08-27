"""Entity-level scoring.

Pure functions over ``strata.labels`` values, so this needs no framework
and no tokenizer: the arithmetic is the thing under test, and the cases
that matter are the ones where a plausible-looking implementation is
wrong.
"""

from strata.labels import Span
from strata.modelling.baselines.text_classifier import span_scores


def s(label, start, end):
    return Span(label=label, start=start, end=end, text="x" * (end - start))


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
