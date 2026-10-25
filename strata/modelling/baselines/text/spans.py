"""Entity-level scores for span predictions, by strata-evaluation's ``entities`` task. See
``docs/adr/0035``.
"""

from strata.evaluation.tasks import entities


def span_scores(truth: list, predicted: list) -> dict[str, float]:
    """Exact and partial precision, recall and F1 over parallel lists of documents' spans, overall
    and per class.
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
    for label, tally in scores.per_class_partial.items():
        named[f"val_span_partial_f1_{label}"] = round(tally.f1, 4)
    return named
