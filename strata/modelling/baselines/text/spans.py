"""Entity-level scores for span predictions against a validation set."""


def _prf(true_positive: int, false_positive: int, false_negative: int) -> tuple:
    precision = true_positive / (true_positive + false_positive or 1)
    recall = true_positive / (true_positive + false_negative or 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _entities(spans: list) -> list[tuple[str, int, int]]:
    """One entity per label a region carries. See ``docs/adr/0035``."""
    return [(label, s.start, s.end) for s in spans for label in s.labels]


def _overlap_matches(predicted: list, truth: list) -> int:
    """Predicted entities overlapping a true one of the same label, paired off.

    One-to-one on purpose. See ``docs/adr/0035``.
    """
    unmatched = list(truth)
    matched = 0
    for label, start, end in predicted:
        for i, (other_label, other_start, other_end) in enumerate(unmatched):
            if label == other_label and start < other_end and other_start < end:
                del unmatched[i]
                matched += 1
                break
    return matched


def span_scores(truth: list, predicted: list) -> dict[str, float]:
    """Entity-level precision, recall and F1 over a validation set.

    *Exact* requires the label and both offsets to agree; *partial* accepts
    an overlap of the same label. Per class as well as overall. See
    ``docs/adr/0035``.

    ``truth`` and ``predicted`` are parallel lists, one entry per document,
    each a list of spans.
    """
    exact_tp = exact_fp = exact_fn = 0
    partial_tp = 0
    per_class: dict[str, list[int]] = {}

    for wanted, got in zip(truth, predicted, strict=True):
        wanted_keys = set(_entities(wanted))
        got_keys = set(_entities(got))
        exact_tp += len(wanted_keys & got_keys)
        exact_fp += len(got_keys - wanted_keys)
        exact_fn += len(wanted_keys - got_keys)
        partial_tp += _overlap_matches(_entities(got), _entities(wanted))

        for label in {k[0] for k in wanted_keys | got_keys}:
            w = {k for k in wanted_keys if k[0] == label}
            g = {k for k in got_keys if k[0] == label}
            counts = per_class.setdefault(label, [0, 0, 0])
            counts[0] += len(w & g)
            counts[1] += len(g - w)
            counts[2] += len(w - g)

    precision, recall, f1 = _prf(exact_tp, exact_fp, exact_fn)
    # Partial shares the exact counts' denominators: the same predictions
    # and the same truth, scored by a looser notion of a match.
    p_precision, p_recall, p_f1 = _prf(
        partial_tp, (exact_tp + exact_fp) - partial_tp, (exact_tp + exact_fn) - partial_tp
    )
    scores = {
        "val_span_precision": round(precision, 4),
        "val_span_recall": round(recall, 4),
        "val_span_f1": round(f1, 4),
        "val_span_partial_precision": round(p_precision, 4),
        "val_span_partial_recall": round(p_recall, 4),
        "val_span_partial_f1": round(p_f1, 4),
    }
    for label, (tp, fp, fn) in sorted(per_class.items()):
        scores[f"val_span_f1_{label}"] = round(_prf(tp, fp, fn)[2], 4)
    return scores
