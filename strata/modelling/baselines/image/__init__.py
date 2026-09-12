"""ConvNeXt V2 baselines for images: multi-label, single-label, and presence."""

from .classifier import MultiLabelClassifier
from .variants import MulticlassClassifier, PresenceClassifier

__all__ = ["MultiLabelClassifier", "MulticlassClassifier", "PresenceClassifier"]
