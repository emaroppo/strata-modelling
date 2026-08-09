"""Not paying twice for the same inference pass.

Ranking a queue needs a score for every unlabelled sample, so a push over a
pool of tens of thousands is minutes of GPU whether it shows 200 tasks or
20. Asking again from the same checkpoint should not repeat it — for any
caller, which is why this lives beside the runs rather than beside whoever
asked first.
"""

import pytest

from strata.labels import ChoicesPrediction
from strata.modelling.predictions import PredictionCache


@pytest.fixture
def cache(tmp_path):
    return PredictionCache.local(tmp_path / "runs")


def guess(*values, confidences=None):
    return ChoicesPrediction(
        values=list(values), confidences=confidences or [0.9] * len(values)
    )


def test_what_went_in_comes_back(cache):
    cache.put(7, {"a" * 64: guess("cat"), "b" * 64: guess("dog", "cat")})
    got = cache.get(7, ["a" * 64, "b" * 64])

    assert got["a" * 64].values == ["cat"]
    assert got["b" * 64].values == ["dog", "cat"]
    # Confidences are what the ranking sorts on, so losing them would leave
    # a cache that silently reorders the queue
    assert got["b" * 64].confidences == [0.9, 0.9]


def test_a_miss_is_simply_absent(cache):
    cache.put(7, {"a" * 64: guess("cat")})
    got = cache.get(7, ["a" * 64, "c" * 64])
    # The caller predicts the difference, so a miss must not be an error or
    # a placeholder that ranks like a real score
    assert set(got) == {"a" * 64}


def test_runs_do_not_share_predictions(cache):
    cache.put(7, {"a" * 64: guess("cat")})
    assert cache.get(8, ["a" * 64]) == {}


def test_writing_the_same_answer_again_is_harmless(cache):
    cache.put(7, {"a" * 64: guess("cat")})
    cache.put(7, {"a" * 64: guess("cat")})
    # A checkpoint and some bytes give one answer, so a re-run repeating
    # work must not fail on it
    assert cache.get(7, ["a" * 64])["a" * 64].values == ["cat"]
    assert cache.counts() == {7: 1}


def test_a_pool_larger_than_sqlite_will_bind_still_works(cache):
    # A review pool is tens of thousands; SQLite caps parameters per
    # statement well below that, so this is the ordinary case rather than
    # an edge one
    many = {f"{i:064x}": guess("cat") for i in range(1500)}
    cache.put(7, many)
    got = cache.get(7, list(many))
    assert len(got) == 1500


def test_forgetting_a_run_leaves_the_others(cache):
    cache.put(7, {"a" * 64: guess("cat")})
    cache.put(8, {"a" * 64: guess("dog")})
    cache.forget(7)
    assert cache.counts() == {8: 1}


def test_a_cache_survives_being_reopened(tmp_path):
    root = tmp_path / "runs"
    PredictionCache.local(root).put(7, {"a" * 64: guess("cat")})
    # The whole point is the next push, which is a different process
    assert PredictionCache.local(root).get(7, ["a" * 64])["a" * 64].values == ["cat"]


def test_what_comes_back_is_what_went_in(cache):
    """A round trip, because a near-miss here is silent.

    Storing a wrapper around a prediction and reading it back as a bare one
    parses without complaint and yields empty values — so a cache holding
    nothing looks exactly like a cache holding answers, until a ranking
    sorts on them.
    """
    original = guess("cat", "dog", confidences=[0.7, 0.2])
    cache.put(9, {"a" * 64: original})
    assert cache.get(9, ["a" * 64])["a" * 64] == original


def test_a_wrapped_prediction_is_refused(cache):
    """The bug this guard exists for.

    A Prediction wraps a ChoicesPrediction with the path it came from. Stored
    as-is it serialises without complaint and reads back with empty values,
    because pydantic drops the keys it does not recognise — so the cache
    holds nothing while looking exactly like a cache holding answers, and
    the failure surfaces as a ranking sorted on empty confidences.
    """
    from pathlib import Path

    from strata.modelling.requests import ScoredPath

    wrapped = ScoredPath(path=Path("/x.jpg"), value=guess("cat"))
    with pytest.raises(TypeError, match="model output"):
        cache.put(1, {"a" * 64: wrapped})


def test_it_holds_whatever_a_model_produced(cache):
    """Choices, spans and boxes alike, read back as what they were.

    Being indifferent to the label format is the point of the project. A
    cache pinned to one task type refuses every other, and a prediction
    parsed as a plain value keeps the answer and loses how sure the model
    was of it.
    """
    from strata.labels import Box, BoxesPrediction, Span, SpansPrediction

    boxes = BoxesPrediction(
        values=[Box(label="cat", x=0.1, y=0.2, width=0.3, height=0.4)],
        confidences=[0.6],
    )
    spans = SpansPrediction(
        values=[Span(label="name", start=0, end=4)], confidences=[0.7]
    )
    cache.put(3, {"a" * 64: boxes, "b" * 64: spans})

    back = cache.get(3, ["a" * 64, "b" * 64])
    assert back["a" * 64] == boxes
    assert back["b" * 64] == spans
    # The confidences are what a ranking sorts on; losing them silently
    # reorders the queue rather than failing
    assert back["a" * 64].confidences == [0.6]


def test_a_plain_value_is_refused_too(cache):
    """Not model output, and storing it loses the same thing silently.

    A Choices has no confidences, so it reads back as a prediction that was
    sure of nothing — which a ranking then sorts to the front.
    """
    from strata.labels import Choices

    with pytest.raises(TypeError, match="model output"):
        cache.put(1, {"a" * 64: Choices(values=["cat"])})
