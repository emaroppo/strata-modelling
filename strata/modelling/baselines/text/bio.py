"""BIO tags: from a label set's classes to tag ids, and from a window's tags back to spans."""

import torch

from strata.labels import Span, SpansPrediction


def tag_ids(classes: list[str], class_name: str) -> tuple[int, int]:
    """The B and I tag for a class. Tag 0 is O."""
    i = classes.index(class_name)
    return 1 + 2 * i, 2 + 2 * i


def label_tokens(classes: list[str], offsets, spans) -> torch.Tensor:
    """One tag per token for a window, ``-100`` where a token carries no supervision."""
    labels = torch.zeros(len(offsets), dtype=torch.long)
    # Padding and special tokens carry no supervision
    labels[(offsets[:, 0] == 0) & (offsets[:, 1] == 0)] = -100
    for span in spans:
        if span.label not in classes:
            continue
        begin, inside = tag_ids(classes, span.label)
        first = True
        for t, (start, end) in enumerate(offsets.tolist()):
            if start == end == 0:
                continue
            # Overlap, not containment: a span that cuts a token still
            # supervises it. docs/adr/0036
            if start < span.end and end > span.start:
                labels[t] = begin if first else inside
                first = False
    return labels


def decode_window(classes: list[str], text: str, logits: torch.Tensor, offsets) -> SpansPrediction:
    """The spans one window's tags spell out, each scored by its tokens' mean confidence."""
    probs = torch.softmax(logits.float(), dim=-1)
    tags = probs.argmax(dim=-1).tolist()
    confidences = probs.max(dim=-1).values.tolist()

    spans: list[dict] = []
    current: dict | None = None
    for tag, conf, (start, end) in zip(tags, confidences, offsets.tolist(), strict=True):
        if start == end == 0:
            continue
        if tag == 0:
            current = None
            continue
        class_name = classes[(tag - 1) // 2]
        is_begin = (tag - 1) % 2 == 0
        if current is not None and not is_begin and current["labels"] == [class_name]:
            current["end"] = end
            current["scores"].append(conf)
        else:
            current = {"labels": [class_name], "start": start, "end": end, "scores": [conf]}
            spans.append(current)

    # Sorted as Spans sorts on the way in, so confidences are ordered to
    # match. docs/adr/0004
    found = sorted(spans, key=lambda s: (s["start"], s["end"]))
    return SpansPrediction(
        values=[
            Span(
                labels=s["labels"], start=s["start"], end=s["end"], text=text[s["start"] : s["end"]]
            )
            for s in found
        ],
        confidences=[round(sum(s["scores"]) / len(s["scores"]), 4) for s in found],
    )


def merge_windows(text: str, outputs: list) -> SpansPrediction:
    """Concatenate the windows' spans, minus what the overlap said twice.

    An entity in the shared region is the same entity when the label and
    both offsets agree, and the higher confidence wins. See
    ``docs/adr/0014``.
    """
    if len(outputs) == 1:
        return outputs[0]
    best: dict[tuple[str, int, int], float] = {}
    for output in outputs:
        for span, confidence in zip(output.values, output.confidences, strict=True):
            key = (span.label, span.start, span.end)
            best[key] = max(best.get(key, 0.0), confidence)
    order = sorted(best, key=lambda k: (k[1], k[2]))
    # One construction, values and confidences together. docs/adr/0004
    return SpansPrediction(
        values=[
            Span(labels=[label], start=start, end=end, text=text[start:end])
            for label, start, end in order
        ],
        confidences=[round(best[key], 4) for key in order],
    )
