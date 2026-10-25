"""Span labelling as token classification, in BIO tagging."""

import torch
from transformers import AutoModelForTokenClassification

from strata.contracts import SpansPrediction

from . import bio
from .base import TransformerBase
from .spans import span_scores
from .windows import tokens_touching


class TextSpanTagger(TransformerBase):
    """Span labelling as token classification, in BIO tagging, aligned through the tokenizer's
    offset mapping.
    """

    label_type = "span"
    version = "1"

    def _label_count(self) -> int:
        # O, plus B and I for each class.
        return 1 + 2 * len(self.classes)

    def _build_model(self, num_labels: int):
        return AutoModelForTokenClassification.from_pretrained(
            self.encoder, num_labels=num_labels, ignore_mismatched_sizes=True
        ).to(self.device)

    def requires_schema(self, schema) -> None:
        """Refuse overlapping or multi-label regions, which one tag per token cannot say. See
        ``docs/adr/0014``.
        """
        cannot = []
        if getattr(schema, "overlapping", False):
            cannot.append("overlapping regions")
        if getattr(schema, "multi_label", False):
            cannot.append("regions carrying more than one label")
        if cannot:
            raise ValueError(
                f"it declares {' and '.join(cannot)}, and BIO tagging gives each "
                f"token exactly one tag. Learning these needs a tagger with one "
                f"binary B/I/O head per class, which is not what this is."
            )

    def _alignment(self, target, windows) -> dict[str, int]:
        """Spans that reached no token, and spans off token boundaries. See ``docs/adr/0036``."""
        spans = [
            span
            for span in (target.values if target is not None else [])
            if span.label in self.classes
        ]
        unaligned = inexact = 0
        for span in spans:
            touched = tokens_touching(span, windows)
            if not touched:
                unaligned += 1
            elif (
                min(start for start, _ in touched) != span.start
                or max(end for _, end in touched) != span.end
            ):
                inexact += 1
        return {"spans_unaligned": unaligned, "spans_inexact": inexact}

    def _item(self, fields: dict, text: str, offsets, target) -> dict:
        spans = target.values if target is not None else []
        return {
            **fields,
            "labels": bio.label_tokens(self.classes, offsets, spans),
            "offsets": offsets,
        }

    def _evaluate(self, samples: list) -> dict:
        """Loss, and entity-level scores from :meth:`predict`. See ``docs/adr/0035``."""
        metrics = super()._evaluate(samples)
        try:
            predicted = self.predict([sample.path for sample in samples])
            truth = [
                list(sample.target.values) if sample.target is not None else []
                for sample in samples
            ]
            metrics.update(span_scores(truth, [list(p.values) for p in predicted]))
        finally:
            # predict() leaves the model in eval mode, and finetune carries on.
            self._net().train()
        return metrics

    def _merge(self, text: str, outputs: list) -> SpansPrediction:
        return bio.merge_windows(text, outputs)

    def _decode(self, text: str, logits: torch.Tensor, offsets) -> SpansPrediction:
        return bio.decode_window(self.classes, text, logits, offsets)
