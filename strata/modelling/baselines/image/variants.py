"""The single-label and presence variants: the same backbone and loops, different task hooks."""

import torch
import torch.nn as nn

from strata.labels import ChoicesPrediction

from .._shared import threshold_choices
from .classifier import MultiLabelClassifier


class MulticlassClassifier(MultiLabelClassifier):
    """Single-label variant: classes are mutually exclusive.

    Only the loss (cross-entropy vs BCE), target encoding, and prediction
    decoding differ. Predictions carry exactly one label with its softmax
    confidence.
    """

    def _make_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    @staticmethod
    def _encode_target(labels: list[str], class_to_idx: dict[str, int]) -> torch.Tensor:
        # Exactly one class per image; first label wins if data has extras
        idx = class_to_idx.get(labels[0], 0) if labels else 0
        return torch.tensor(idx, dtype=torch.long)

    @staticmethod
    def _count_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
        return int((logits.argmax(dim=1) == targets).sum().item())

    @staticmethod
    def _activation(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.float(), dim=1)

    def _to_output(self, probs: torch.Tensor) -> ChoicesPrediction:
        idx = int(probs.argmax().item())
        return ChoicesPrediction(
            values=[self.classes[idx]], confidences=[round(probs[idx].item(), 4)]
        )


class PresenceClassifier(MultiLabelClassifier):
    """Independent presence detectors with an implicit negative class.

    One sigmoid per positive class ("is X present in the picture?"), so any
    combination of classes can co-occur. NEGATIVE_LABEL is a dataset marker
    meaning "reviewed, nothing present": it gets no output neuron, trains as
    an all-zeros target, and is emitted as the prediction when no class
    clears the threshold, so the model cannot contradict itself.
    """

    NEGATIVE_LABEL = "none"
    requires_classes = (NEGATIVE_LABEL,)

    def _effective_classes(self, classes: list[str]) -> list[str]:
        return [c for c in classes if c != self.NEGATIVE_LABEL]

    def _to_output(self, probs: torch.Tensor) -> ChoicesPrediction:
        if not bool((probs > 0.5).any()):
            return ChoicesPrediction(
                values=[self.NEGATIVE_LABEL], confidences=[round(1.0 - probs.max().item(), 4)]
            )
        return threshold_choices(probs, self.classes)
