"""Single-label document classification: the classes are exclusive."""

from typing import ClassVar

import torch

from strata.contracts import ChoicesPrediction

from .classifier import TextClassifier, _by_class


class TextMulticlassClassifier(TextClassifier):
    """Single-label document classification: the classes are exclusive."""

    version: ClassVar[str] = "1"
    problem_type: ClassVar[str] = "single_label_classification"

    #: No "any": the union is not an answer a single-label head may give (docs/adr/0014).
    AGGREGATIONS: ClassVar[tuple[str, ...]] = ("max", "mean")

    def requires_schema(self, schema) -> None:
        """The mirror of the sigmoid head's refusal. See ``docs/adr/0014``."""
        if getattr(schema, "multiple", False) is True:
            raise ValueError(
                "it is multi-choice, and this head names exactly one class "
                "per document, so every other answer would be dropped. Use "
                "'text', whose sigmoid head scores each class separately."
            )

    def _trainable(self, target) -> bool:
        # An empty target is counted, not encoded as class zero. docs/adr/0036
        return bool(getattr(target, "values", None))

    def _item(self, fields: dict, text: str, offsets, target) -> dict:
        named = [n for n in (target.values if target is not None else []) if n in self.classes]
        index = self.classes.index(named[0]) if named else -100
        return {
            **fields,
            # -100 is cross-entropy's ignore index, so an empty target gets no supervision.
            "labels": torch.tensor(index, dtype=torch.long),
            "offsets": offsets,
        }

    def _decode(self, text: str, logits: torch.Tensor, offsets) -> ChoicesPrediction:
        probs = torch.softmax(logits.float(), dim=-1)
        index = int(probs.argmax().item())
        return ChoicesPrediction(
            values=[self.classes[index]], confidences=[round(probs[index].item(), 4)]
        )

    def _merge(self, text: str, outputs: list) -> ChoicesPrediction:
        """One class for the document, from one class per window. See ``docs/adr/0014``."""
        if len(outputs) == 1:
            return outputs[0]
        scores = _by_class(outputs)
        if self.window_aggregation == "mean":
            # Divided by every window, so a class that won once does not outrank one that won
            # steadily.
            combined = {name: sum(values) / len(outputs) for name, values in scores.items()}
        else:
            combined = {name: max(values) for name, values in scores.items()}
        best = max(combined, key=lambda n: combined[n])
        return ChoicesPrediction(values=[best], confidences=[round(combined[best], 4)])
