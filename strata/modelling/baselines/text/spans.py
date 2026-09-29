"""Entity-level scores for span predictions against a validation set.

The arithmetic is strata-evaluation's ``entities`` task, the same the
``evaluate`` stage runs; this only names the numbers the way a run records
its own. See ``docs/adr/0035``.
"""

from strata.evaluation.tasks import entities


def span_scores(truth: list, predicted: list) -> dict[str, float]:
    """Entity-level precision, recall and F1 over a validation set.

    *Exact* requires the label and both offsets to agree; *partial* accepts
    an overlap of the same label. Per class as well as overall.

    ``truth`` and ``predicted`` are parallel lists, one entry per document,
    each a list of spans.
    """
    scores = entities.score(truth, predicted)
    exact, partial = scores.exact, scores.partial
    named = {
        "val_span_precision": round(exact.precision, 4),
        "val_span_recall": round(exact.recall, 4),
        "val_span_f1": round(exact.f1, 4),
        "val_span_partial_precision": round(partial.precision, 4),
        "val_span_partial_recall": round(partial.recall, 4),
        "val_span_partial_f1": round(partial.f1, 4),
    }
    for label, tally in scores.per_class.items():
        named[f"val_span_f1_{label}"] = round(tally.f1, 4)
    return named
