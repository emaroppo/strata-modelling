"""Transformer baselines for text: document classification and spans.

Each fine-tunes a pretrained encoder from Hugging Face over the same
plumbing, tokenisation into windows, a training loop and checkpointing,
and differs in what a target is: a set of classes for the whole document,
one class, or labelled character ranges inside it.
"""

from .classifier import TextClassifier
from .multiclass import TextMulticlassClassifier
from .spans import span_scores
from .tagger import TextSpanTagger

__all__ = ["TextClassifier", "TextMulticlassClassifier", "TextSpanTagger", "span_scores"]
