"""Span labelling as token classification, in BIO tagging."""

import torch
from transformers import AutoModelForTokenClassification

from strata.labels import SpansPrediction

from . import bio
from .base import TransformerBase
from .spans import span_scores
from .windows import tokens_touching


class TextSpanTagger(TransformerBase):
    """Span labelling as token classification, in BIO tagging.

    Character offsets are what the schema stores, so targets are aligned to
    tokens on the way in and decoded back to character ranges on the way
    out; the tokenizer's offset mapping is what makes both directions
    exact.
    """

    task = "span"
    version = "1"

    def _label_count(self) -> int:
        # O, plus B- and I- for each class
        return 1 + 2 * len(self.classes)

    def _build_model(self, num_labels: int):
        return AutoModelForTokenClassification.from_pretrained(
            self.encoder, num_labels=num_labels, ignore_mismatched_sizes=True
        ).to(self.device)

    def requires_schema(self, schema) -> None:
        """What BIO cannot say, refused before a round rather than during it.

        A tag per token means one thing per token. A region carrying two
        labels loses the second; the later of two overlapping regions
        overwrites the earlier. See ``docs/adr/0014``.
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
        """What this document's spans lose against the tokenizer.

        ``spans_unaligned`` is supervision that never happened: the span
        reached no token, so nothing taught the model that class.
        ``spans_inexact`` is a span whose characters do not sit on token
        boundaries: it trains, but decoding snaps back to whole tokens, which
        shows as a gap between exact and partial F1 rather than as a bug.
        """
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
        """Loss, and the entity-level numbers a person can read.

        Scored from :meth:`predict` rather than from the validation loop's
        logits: ``predict`` is what reaches the cache, Label Studio and the
        reviewer, so scoring it measures what the system actually does. The
        ``evaluate`` stage scores classification today and not spans, so
        this stays until it does.
        """
        metrics = super()._evaluate(samples)
        try:
            predicted = self.predict([sample.path for sample in samples])
            truth = [
                list(sample.target.values) if sample.target is not None else []
                for sample in samples
            ]
            metrics.update(span_scores(truth, [list(p.values) for p in predicted]))
        finally:
            # predict() leaves the model in eval mode, and finetune carries
            # on training after this returns.
            self._model.train()
        return metrics

    def _merge(self, text: str, outputs: list) -> SpansPrediction:
        return bio.merge_windows(text, outputs)

    def _decode(self, text: str, logits: torch.Tensor, offsets) -> SpansPrediction:
        return bio.decode_window(self.classes, text, logits, offsets)
