"""Multi-label document classification with a sigmoid head."""

from typing import ClassVar

import torch
from transformers import AutoModelForSequenceClassification

from strata.labels import ChoicesPrediction

from .._shared import threshold_choices
from .base import TransformerBase


class TextClassifier(TransformerBase):
    """Multi-label document classification with a sigmoid head."""

    task = "classification"
    version = "1"

    #: How the head is trained: several labels may be on at once.
    problem_type: ClassVar[str] = "multi_label_classification"
    aggregates_windows: ClassVar[bool] = True

    def requires_schema(self, schema) -> None:
        """A sigmoid head cannot promise to name only one class. See ``docs/adr/0014``."""
        if getattr(schema, "multiple", True) is False:
            raise ValueError(
                "it is single-choice, and this head scores every class "
                "independently, so it can assert two at once. Use "
                "'text-multiclass', whose softmax head names exactly one."
            )

    def _build_model(self, num_labels: int):
        return AutoModelForSequenceClassification.from_pretrained(
            self.encoder,
            num_labels=num_labels,
            problem_type=self.problem_type,
            # The encoder may already carry a head for someone else's classes;
            # ours is sized for this project's, so re-initialise it
            ignore_mismatched_sizes=True,
        ).to(self.device)

    def _item(self, fields: dict, text: str, offsets, target) -> dict:
        # Every window of a document carries the document's labels.
        # docs/adr/0014
        labels = torch.zeros(len(self.classes))
        for name in target.values if target is not None else []:
            if name in self.classes:
                labels[self.classes.index(name)] = 1.0
        return {**fields, "labels": labels, "offsets": offsets}

    def _decode(self, text: str, logits: torch.Tensor, offsets) -> ChoicesPrediction:
        return threshold_choices(torch.sigmoid(logits.float()), self.classes)

    def _merge(self, text: str, outputs: list) -> ChoicesPrediction:
        """Combine per-window answers under the declared strategy. See ``docs/adr/0014``."""
        if len(outputs) == 1:
            return outputs[0]
        scores = _by_class(outputs)
        how = self.window_aggregation
        if how == "mean":
            combined = {name: sum(values) / len(outputs) for name, values in scores.items()}
        elif how in ("max", "any"):
            combined = {name: max(values) for name, values in scores.items()}
        else:
            raise ValueError(f"Unknown window_aggregation {how!r}")

        keep = combined if how == "any" else {n: s for n, s in combined.items() if s > 0.5}
        if not keep:
            keep = {max(combined, key=lambda n: combined[n]): max(combined.values())}
        order = sorted(keep, key=lambda n: keep[n], reverse=True)
        # Built in one construction: confidences are positional against
        # values. docs/adr/0004
        return ChoicesPrediction(values=order, confidences=[round(keep[n], 4) for n in order])


def _by_class(outputs: list) -> dict[str, list[float]]:
    """Each class's confidence in every window that named it."""
    scores: dict[str, list[float]] = {}
    for output in outputs:
        for name, confidence in zip(output.values, output.confidences, strict=True):
            scores.setdefault(name, []).append(confidence)
    return scores
