"""Single-label document classification: the classes are exclusive."""

from typing import ClassVar

import torch

from strata.labels import ChoicesPrediction

from .classifier import TextClassifier, _by_class


class TextMulticlassClassifier(TextClassifier):
    """Single-label document classification: the classes are exclusive.

    The same encoder, training loop and windowing as :class:`TextClassifier`.
    Only the loss, the target encoding and the decoding differ.
    """

    version: ClassVar[str] = "1"
    problem_type: ClassVar[str] = "single_label_classification"

    #: "any" means the union of what the windows asserted, which is not an
    #: answer a single-label head is allowed to give.
    AGGREGATIONS: ClassVar[tuple[str, ...]] = ("max", "mean")

    def requires_schema(self, schema) -> None:
        """The mirror of the sigmoid head's refusal.

        A softmax head names one class, so a label set expecting several
        would be trained on the first of them and scored as though the rest
        had been asked for.
        """
        if getattr(schema, "multiple", False) is True:
            raise ValueError(
                "it is multi-choice, and this head names exactly one class "
                "per document, so every other answer would be dropped. Use "
                "'text', whose sigmoid head scores each class separately."
            )

    def _trainable(self, target) -> bool:
        # "None of these" is a real answer and a common one, but it is not
        # one this head can represent — the argmax always names something.
        # Counted rather than encoded as class zero, which would teach the
        # first class every time a reviewer found nothing.
        return bool(getattr(target, "values", None))

    def _item(self, fields: dict, text: str, offsets, target) -> dict:
        # A class index rather than a multi-hot row, which is what
        # cross-entropy takes. Every window carries the document's class.
        named = [n for n in (target.values if target is not None else []) if n in self.classes]
        index = self.classes.index(named[0]) if named else -100
        return {
            **fields,
            # -100 is cross-entropy's ignore index. Unreachable in practice —
            # a target with nothing in it contributes no windows — and here
            # so that items assembled by hand get no supervision rather than
            # the wrong supervision.
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
        """One class for the document, from one class per window.

        Each window has already been decoded to its own winner, so a class
        that came second everywhere never reaches here. Merging the
        probability vectors would keep it, at the cost of holding every
        window's logits for every document; the sigmoid head has the same
        limit, so the two stay consistent.
        """
        if len(outputs) == 1:
            return outputs[0]
        scores = _by_class(outputs)
        if self.window_aggregation == "mean":
            # Divided by every window, not just the ones that named it, so
            # a class that won once in twenty does not outrank one that won
            # steadily
            combined = {name: sum(values) / len(outputs) for name, values in scores.items()}
        else:
            combined = {name: max(values) for name, values in scores.items()}
        best = max(combined, key=lambda n: combined[n])
        return ChoicesPrediction(values=[best], confidences=[round(combined[best], 4)])
