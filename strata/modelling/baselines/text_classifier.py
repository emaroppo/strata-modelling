"""Transformer baselines for text: document classification and spans.

Both fine-tune a pretrained encoder from Hugging Face and share the same
plumbing — tokenisation, training loop, checkpointing. They differ in what
a target is: a set of classes for the whole document, or labelled
character ranges inside it.

Documents are read from disk, so a model sees the same absolute paths an
image model does.
"""

import sys
from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from strata.labels import ChoicesPrediction, Span, SpansPrediction

from ..model import Example, Model
from ._shared import console, epoch_progress, pick_device, predict_progress, threshold_choices

DEFAULT_ENCODER = "distilbert-base-uncased"


def _read_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"warning: unreadable document {path} ({e}), using empty text", file=sys.stderr)
        return ""


def _prf(true_positive: int, false_positive: int, false_negative: int) -> tuple:
    precision = true_positive / (true_positive + false_positive or 1)
    recall = true_positive / (true_positive + false_negative or 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _tokens_touching(span, windows) -> list[tuple[int, int]]:
    """Every token a span overlaps, across each window the document became.

    Bisected rather than scanned: a dense document runs to hundreds of
    spans over five hundred tokens, and this is asked once per span on
    every dataset built.
    """
    touched: list[tuple[int, int]] = []
    for window in windows:
        tokens = [(int(s), int(e)) for s, e in window if not (s == 0 and e == 0)]
        if not tokens:
            continue
        starts = [s for s, _ in tokens]
        ends = [e for _, e in tokens]
        # First token reaching past the span's start, up to the first one
        # starting at or after its end
        touched.extend(tokens[bisect_right(ends, span.start) : bisect_left(starts, span.end)])
    return touched


def _entities(spans: list) -> list[tuple[str, int, int]]:
    """One entity per label a region carries.

    A region marked both PER and ORG asserts two things, and counting it
    once would be scoring regions rather than entities — which is not what
    anyone reads an F1 as.
    """
    return [(label, s.start, s.end) for s in spans for label in s.labels]


def _overlap_matches(predicted: list, truth: list) -> int:
    """Predicted entities overlapping a true one of the same label, paired off.

    One-to-one on purpose: without it, a model emitting one span across a
    whole sentence would "match" every entity in it and score perfectly for
    saying almost nothing.
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

    Two definitions, because neither alone is honest about this kind of
    data. *Exact* requires the label and both offsets to agree, which is
    the standard and what a downstream consumer actually gets. *Partial*
    accepts an overlap of the same label, which is what a reviewer sees:
    a boundary off by a trailing bracket is a correction, not a miss.
    Reporting only the first understates the model; only the second
    flatters it.

    Per class as well as overall, because the classes here are wildly
    uneven — a corpus with forty times more URLs than organisations has an
    aggregate that says almost nothing about organisations.

    ``truth`` and ``predicted`` are parallel lists, one entry per document,
    each a list of spans.
    """
    exact_tp = exact_fp = exact_fn = 0
    partial_tp = 0
    per_class: dict[str, list[int]] = {}

    for wanted, got in zip(truth, predicted):
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
        partial_tp,
        (exact_tp + exact_fp) - partial_tp,
        (exact_tp + exact_fn) - partial_tp,
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


class _TextDataset(Dataset):
    """One item per *window*, not per document.

    A document longer than the encoder's input becomes several training
    items rather than a truncated one. The index is built once by counting
    each document's windows; ``__getitem__`` re-encodes its document and
    returns the window asked for, which costs a tokenisation per access and
    saves holding every padded window in memory at once.

    With windowing off, a document is one window and this is the shape it
    always had.
    """

    def __init__(self, samples: list[Example], encode, scan):
        self.samples = samples
        self.encode = encode
        self.index: list[tuple[int, int]] = []
        #: What was noticed while counting windows — supervision a target
        #: loses on the way in. Collected here because this is the one pass
        #: that already tokenises every document, and a second one over a
        #: corpus this size is not free.
        self.diagnostics: Counter = Counter()
        for i, sample in enumerate(samples):
            windows, noticed = scan(_read_text(sample.path), sample.target)
            self.index.extend((i, w) for w in range(windows))
            self.diagnostics.update(noticed)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sample_idx, window = self.index[idx]
        sample = self.samples[sample_idx]
        return self.encode(_read_text(sample.path), sample.target)[window]


class _TransformerBase(Model):
    """Shared training loop for the two text heads."""

    MAX_LENGTH = 512

    #: Whether this head has to combine several windows into one answer for
    #: the document. A tagger does not — spans from adjacent windows are
    #: spans in the same document, and concatenating them is the whole
    #: operation. A classifier does, and there is no obvious right way, so
    #: it has to be told rather than defaulted.
    aggregates_windows: ClassVar[bool] = False

    #: Which combinations this head can actually perform. Narrower for a
    #: single-label head, where "the union of what the windows asserted" is
    #: not an answer it is allowed to give.
    AGGREGATIONS: ClassVar[tuple[str, ...]] = ("max", "mean", "any")

    def __init__(
        self,
        encoder: str = DEFAULT_ENCODER,
        num_epochs: int = 3,
        batch_size: int = 8,
        lr: float = 5e-5,
        device: str | None = None,
        window: int | None = None,
        window_overlap: int | None = None,
        window_aggregation: str | None = None,
    ):
        """``window`` is off by default, and off means today's behaviour.

        Truncation at ``MAX_LENGTH`` stays the default because the right way
        to handle a document past the encoder's limit is not a property of
        text — it depends on the corpus. Windowing suits email, where the
        tail of a long message carries as many entities as the head;
        truncation suits a corpus whose documents lead with what matters; a
        long-context encoder suits others still. So the choice is declared
        in ``[model.params]`` and recorded on the run, rather than inherited
        from a default nobody chose.

        ``window_overlap`` is how many tokens consecutive windows share, so
        the step between them is ``window - window_overlap``. It has to
        exceed the longest thing being labelled or that thing can be split
        across a boundary and found twice, in halves. It defaults to a
        quarter of the window rather than to a fixed number of tokens: a
        fixed default is either too small for a long window or larger than
        a short one, and the second silently means the windows never
        advance.
        """
        if window is not None:
            if window_overlap is None:
                window_overlap = window // 4
            if window_overlap >= window:
                raise ValueError(
                    f"window_overlap ({window_overlap}) must be smaller than "
                    f"window ({window}), or the windows do not advance."
                )
            choices = ", ".join(repr(a) for a in self.AGGREGATIONS)
            if self.aggregates_windows and window_aggregation is None:
                raise ValueError(
                    f"{type(self).__name__} needs window_aggregation when "
                    f"window is set: several windows produce several answers "
                    f"for one document and they have to be combined. Choose "
                    f"{choices} — there is no right default."
                )
            if (
                window_aggregation is not None
                and window_aggregation not in self.AGGREGATIONS
            ):
                # Here rather than in `_merge`, which does not run until a
                # document needs combining — a typo would otherwise survive
                # a whole training run and fail on the first prediction.
                raise ValueError(
                    f"{type(self).__name__} cannot combine windows by "
                    f"{window_aggregation!r}. It takes {choices}."
                )
        self.encoder = encoder
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.window = window
        self.window_overlap = window_overlap if window_overlap is not None else 128
        self.window_aggregation = window_aggregation
        self.device = pick_device(device)
        self.classes: list[str] = []
        self._tokenizer = None
        self._model = None

    # -- pieces the two heads differ on ---------------------------------

    def _build_model(self, num_labels: int):
        raise NotImplementedError

    def _item(self, fields: dict, text: str, offsets, target) -> dict:
        """Add this head's supervision to one already-tokenised window."""
        raise NotImplementedError

    def _merge(self, text: str, outputs: list):
        """Combine one document's windows into one answer.

        Only ever called with more than one window when windowing is on.
        """
        return outputs[0]

    def _trainable(self, target) -> bool:
        """Whether this head can learn anything from a target.

        A head that cannot represent an answer should not be handed it: a
        single-label classifier given "none of these" has no way to encode
        that, and encoding it as class zero teaches the wrong thing. Such a
        sample contributes no windows and is counted instead.
        """
        return True

    def _decode(self, text: str, logits: torch.Tensor):
        raise NotImplementedError

    def _label_count(self) -> int:
        return len(self.classes)

    # -- shared ---------------------------------------------------------

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.encoder)
        return self._tokenizer

    def _encode(self, text: str, target) -> list[dict]:
        """One encoded item per window, and a list even when there is one.

        The tokenizer reports ``offset_mapping`` as character offsets into
        the *original* string for every window, not relative to the window,
        so nothing has to be rebased afterwards — which is the step this
        would otherwise most easily get wrong, and silently: every span in a
        document shifted by a constant still validates and still trains.
        """
        kwargs = {
            "truncation": True,
            "padding": "max_length",
            "return_offsets_mapping": True,
            "return_tensors": "pt",
        }
        if self.window is None:
            kwargs["max_length"] = self.MAX_LENGTH
        else:
            kwargs["max_length"] = self.window
            # The tokenizer calls the overlap "stride"; the step between
            # windows is window - overlap.
            kwargs["stride"] = self.window_overlap
            kwargs["return_overflowing_tokens"] = True

        encoded = self.tokenizer(text, **kwargs)
        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)
        return [
            self._item({k: v[i] for k, v in encoded.items()}, text, offsets[i], target)
            for i in range(len(offsets))
        ]

    def _scan(self, text: str, target) -> tuple[int, dict[str, int]]:
        """How many windows this document needs, and what it loses on the way in.

        One tokenisation answering both, because the dataset already pays
        for one per document and the alignment question needs exactly what
        it produces.
        """
        if self.window is None:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.MAX_LENGTH,
                return_offsets_mapping=True,
            )
            windows = [encoded["offset_mapping"]]
        else:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.window,
                stride=self.window_overlap,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )
            windows = encoded["offset_mapping"]
        if target is not None and not self._trainable(target):
            # No windows, so it contributes no training items — and a count
            # rather than a silence, because a corpus quietly training on
            # less than it holds is the whole family of faults this guards.
            return 0, {"empty_targets": 1}
        return len(windows), self._alignment(target, windows)

    def _alignment(self, target, windows) -> dict[str, int]:
        """What this head loses aligning a target to tokens. Nothing, by default."""
        return {}

    def _prepare(self, classes: list[str]) -> None:
        """Keep the fine-tuned weights when the class list only grew."""
        if self._model is None:
            self.classes = list(classes)
            self._model = self._build_model(self._label_count())
            return
        if list(self.classes) == list(classes):
            return
        if classes[: len(self.classes)] == list(self.classes):
            console.print(
                f"Class list grew {len(self.classes)} -> {len(classes)}; "
                "rebuilding the head, encoder weights kept."
            )
            body = self._model.base_model
            self.classes = list(classes)
            self._model = self._build_model(self._label_count())
            self._model.base_model.load_state_dict(body.state_dict())
            return
        console.print(
            "[yellow]Class list changed incompatibly; starting from the "
            "pretrained encoder.[/yellow]"
        )
        self.classes = list(classes)
        self._model = self._build_model(self._label_count())

    def finetune(
        self,
        train: list[Example],
        classes: list[str],
        val: list[Example] | None = None,
        on_epoch=None,
    ) -> dict:
        self._prepare(classes)
        dataset = _TextDataset(train, self._encode, self._scan)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
        )
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = max(1, self.num_epochs * len(loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=0.1
        )

        self._model.train()
        total_loss = 0.0
        seen = 0
        with epoch_progress("loss={task.fields[loss]:.4f}") as progress:
            task = progress.add_task(
                "training", total=total_steps, epoch=1,
                total_epochs=self.num_epochs, loss=0.0,
            )
            for epoch in range(1, self.num_epochs + 1):
                progress.update(task, epoch=epoch)
                for batch in loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    outputs = self._model(**batch)
                    outputs.loss.backward()
                    nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    total_loss += outputs.loss.item() * batch["input_ids"].size(0)
                    seen += batch["input_ids"].size(0)
                    progress.update(task, advance=1, loss=total_loss / max(seen, 1))

        metrics = {"loss": total_loss / max(seen, 1)}
        # What the training set lost on the way in, recorded on the run.
        # A model that will not learn a class is otherwise indistinguishable
        # from a class it was never actually taught.
        metrics.update({f"train_{k}": v for k, v in dataset.diagnostics.items()})
        if val:
            metrics.update(self._evaluate(val))
        return metrics

    def _evaluate(self, samples: list[dict]) -> dict:
        loader = DataLoader(
            _TextDataset(samples, self._encode, self._scan),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
        )
        self._model.eval()
        total_loss = 0.0
        seen = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self._model(**batch)
                total_loss += outputs.loss.item() * batch["input_ids"].size(0)
                seen += batch["input_ids"].size(0)
        self._model.train()
        return {"val_loss": total_loss / max(seen, 1)}

    def _collate(self, items: list[dict]) -> dict:
        # ``offsets`` rides along for decoding and is not model input; the
        # forward pass would reject it as an unexpected keyword.
        keys = [k for k in items[0] if k != "offsets"]
        return {k: torch.stack([item[k] for item in items]) for k in keys}

    def predict(self, paths: list[Path], on_batch=None, *, features=None) -> list:
        if self._model is None or not self.classes:
            raise RuntimeError("Model has no weights. Call finetune() or load() first.")
        self._model.eval()
        outputs = []
        with torch.no_grad(), predict_progress() as progress:
            task = progress.add_task("predict", total=len(paths))
            for done, path in enumerate(paths, start=1):
                text = _read_text(path)
                # One item per window, so a long document is answered whole
                # rather than up to the encoder's limit and no further.
                items = self._encode(text, None)
                per_window = []
                for item in items:
                    offsets = item["offsets"]
                    fields = {
                        k: v.unsqueeze(0).to(self.device)
                        for k, v in item.items()
                        if k in ("input_ids", "attention_mask", "token_type_ids")
                    }
                    logits = self._model(**fields).logits[0]
                    per_window.append(self._decode(text, logits, offsets))
                outputs.append(self._merge(text, per_window))
                progress.advance(task)
                if on_batch is not None:
                    on_batch(done, len(paths))
        return outputs

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save.")
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "classes": self.classes,
                "encoder": self.encoder,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        self.classes = checkpoint["classes"]
        self.encoder = checkpoint.get("encoder", self.encoder)
        self._model = self._build_model(self._label_count())
        self._model.load_state_dict(checkpoint["state_dict"])


class TextClassifier(_TransformerBase):
    """Multi-label document classification with a sigmoid head."""

    task = "classification"
    version = "1"

    def requires_schema(self, schema) -> None:
        """A sigmoid head cannot promise to name only one class.

        Every class is scored independently, so this can assert two where
        the label set permits one — and that lands as a pre-annotation a
        reviewer has to undo, on a control that will not even display it.
        The label set already says which it is; nothing read it until now.
        """
        if getattr(schema, "multiple", True) is False:
            raise ValueError(
                "it is single-choice, and this head scores every class "
                "independently, so it can assert two at once. Use "
                "'text-multiclass', whose softmax head names exactly one."
            )

    #: How the head is trained: several labels may be on at once.
    problem_type: ClassVar[str] = "multi_label_classification"

    def _build_model(self, num_labels: int):
        return AutoModelForSequenceClassification.from_pretrained(
            self.encoder,
            num_labels=num_labels,
            problem_type=self.problem_type,
            # The encoder may already carry a head for someone else's classes;
            # ours is sized for this project's, so re-initialise it
            ignore_mismatched_sizes=True,
        ).to(self.device)

    aggregates_windows: ClassVar[bool] = True

    def _item(self, fields: dict, text: str, offsets, target) -> dict:
        # Every window of a document carries the document's labels: the
        # target is a statement about the whole thing, and there is no way
        # to know which window earned it.
        labels = torch.zeros(len(self.classes))
        for name in (target.values if target is not None else []):
            if name in self.classes:
                labels[self.classes.index(name)] = 1.0
        return {**fields, "labels": labels, "offsets": offsets}

    def _merge(self, text: str, outputs: list) -> ChoicesPrediction:
        """Combine per-window answers under the declared strategy.

        There is no right default here, which is why the constructor
        refuses to guess: 'max' says a class is present if any window was
        confident, 'mean' averages the evidence, 'any' takes the union of
        what each window asserted. They disagree most on exactly the long
        documents windowing exists for.
        """
        if len(outputs) == 1:
            return outputs[0]

        scores: dict[str, list[float]] = {}
        for output in outputs:
            for name, confidence in zip(output.values, output.confidences):
                scores.setdefault(name, []).append(confidence)

        how = self.window_aggregation
        if how == "any":
            combined = {name: max(values) for name, values in scores.items()}
        elif how == "mean":
            combined = {
                name: sum(values) / len(outputs) for name, values in scores.items()
            }
        elif how == "max":
            combined = {name: max(values) for name, values in scores.items()}
        else:
            raise ValueError(f"Unknown window_aggregation {how!r}")

        keep = combined if how == "any" else {
            name: score for name, score in combined.items() if score > 0.5
        }
        if not keep:
            keep = {max(combined, key=combined.get): max(combined.values())}
        order = sorted(keep, key=lambda n: keep[n], reverse=True)
        # Built in one construction: confidences are positional against
        # values, so appending to a value already made reassigns them.
        return ChoicesPrediction(
            values=order, confidences=[round(keep[n], 4) for n in order]
        )

    def _decode(self, text: str, logits: torch.Tensor, offsets) -> ChoicesPrediction:
        return threshold_choices(torch.sigmoid(logits.float()), self.classes)


class TextMulticlassClassifier(TextClassifier):
    """Single-label document classification: the classes are exclusive.

    The same encoder, training loop and windowing as
    :class:`TextClassifier`. Only the loss, the target encoding and the
    decoding differ, which is exactly the split the image baselines already
    make between ``MultiLabelClassifier`` and ``MulticlassClassifier``.
    """

    version: ClassVar[str] = "1"

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

    problem_type: ClassVar[str] = "single_label_classification"

    def _item(self, fields: dict, text: str, offsets, target) -> dict:
        # A class index rather than a multi-hot row, which is what
        # cross-entropy takes. Every window of a document carries the
        # document's class, for the same reason the sigmoid head does.
        named = [
            name
            for name in (target.values if target is not None else [])
            if name in self.classes
        ]
        index = self.classes.index(named[0]) if named else -100
        return {
            **fields,
            # -100 is cross-entropy's ignore index. It is unreachable in
            # practice — a target with nothing in it contributes no windows
            # — and is here so that a caller assembling items by hand gets
            # no supervision rather than the wrong supervision.
            "labels": torch.tensor(index, dtype=torch.long),
            "offsets": offsets,
        }

    def _decode(self, text: str, logits: torch.Tensor, offsets) -> ChoicesPrediction:
        probs = torch.softmax(logits.float(), dim=-1)
        index = int(probs.argmax().item())
        return ChoicesPrediction(
            values=[self.classes[index]],
            confidences=[round(probs[index].item(), 4)],
        )

    def _merge(self, text: str, outputs: list) -> ChoicesPrediction:
        """One class for the document, from one class per window.

        Note what this can and cannot see: each window has already been
        decoded to its own winner, so a class that came second everywhere
        never reaches here. Merging the probability vectors instead would
        keep it, at the cost of holding every window's logits for every
        document — and the same limitation applies to the sigmoid head, so
        the two stay consistent rather than one being quietly cleverer.
        """
        if len(outputs) == 1:
            return outputs[0]

        scores: dict[str, list[float]] = {}
        for output in outputs:
            for name, confidence in zip(output.values, output.confidences):
                scores.setdefault(name, []).append(confidence)

        if self.window_aggregation == "mean":
            # Divided by every window, not just the ones that named it, so
            # a class that won once in twenty does not outrank one that won
            # steadily
            combined = {
                name: sum(values) / len(outputs) for name, values in scores.items()
            }
        else:
            combined = {name: max(values) for name, values in scores.items()}

        best = max(combined, key=combined.get)
        return ChoicesPrediction(values=[best], confidences=[round(combined[best], 4)])


class TextSpanTagger(_TransformerBase):
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
        overwrites the earlier. Either way the model trains on a projection
        of the label set and is scored as though it had learned the whole
        thing — the metric is the last place that failure would show.
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

        Two counts, both otherwise silent, and they read differently.

        ``spans_unaligned`` is supervision that never happened: the span
        reached no token at all, so nothing in the document taught the model
        that class. It looks afterwards exactly like a class the model
        cannot learn.

        ``spans_inexact`` is a span whose characters do not sit on token
        boundaries. It trains, but decoding snaps back to whole tokens, so
        the model cannot reproduce the human's range — which is a gap
        between exact and partial F1 rather than a bug, and is much easier
        to read as one when the count is on the run beside it.
        """
        spans = [
            span
            for span in (target.values if target is not None else [])
            if span.label in self.classes
        ]
        unaligned = inexact = 0
        for span in spans:
            touched = _tokens_touching(span, windows)
            if not touched:
                unaligned += 1
            elif (
                min(start for start, _ in touched) != span.start
                or max(end for _, end in touched) != span.end
            ):
                inexact += 1
        return {"spans_unaligned": unaligned, "spans_inexact": inexact}

    def _tag_ids(self, class_name: str) -> tuple[int, int]:
        i = self.classes.index(class_name)
        return 1 + 2 * i, 2 + 2 * i  # B, I

    def _item(self, fields: dict, text: str, offsets, target) -> dict:
        item = dict(fields)

        labels = torch.zeros(len(offsets), dtype=torch.long)
        # Padding and special tokens carry no supervision
        labels[(offsets[:, 0] == 0) & (offsets[:, 1] == 0)] = -100
        for span in (target.values if target is not None else []):
            if span.label not in self.classes:
                continue
            begin, inside = self._tag_ids(span.label)
            first = True
            for t, (start, end) in enumerate(offsets.tolist()):
                if start == end == 0:
                    continue
                # Overlap, not containment. A span whose characters cut a
                # token — "USA" inside "USA-based" — used to land on no
                # token at all and supervise nothing, which reads later as
                # a class the model simply will not learn.
                if start < span.end and end > span.start:
                    labels[t] = begin if first else inside
                    first = False
        item["labels"] = labels
        item["offsets"] = offsets
        return item

    def _evaluate(self, samples: list) -> dict:
        """Loss, and the numbers a person can read.

        Loss on a token-classification head is not interpretable: it does
        not say whether entities are being found. So this adds entity-level
        precision, recall and F1, which is what ``val_accuracy`` is for the
        image baseline — one number per run that means something.

        Scored from :meth:`predict` rather than from the validation loop's
        logits, deliberately. ``predict`` is what reaches the prediction
        cache, Label Studio and the reviewer, so scoring it measures what
        the system actually does. The image baseline does the other thing —
        ``_count_correct`` thresholds at 0.5 and scores the empty set, while
        ``_to_output`` falls back to argmax — so its accuracy describes a
        rule that exists nowhere else. Not a mistake worth repeating.

        Where this metric eventually belongs is a framework-side evaluator,
        computing it identically for every model.
        With one span model the bias is constant and this is enough.
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
        """Concatenate the windows' spans, minus what the overlap said twice.

        Windows share ``window_overlap`` tokens, so an entity sitting in the
        shared region is found by both and must not be reported twice. It is
        the same entity when the label and both offsets agree, and the
        higher confidence wins — a window that saw the entity with more
        context around it is the one to believe.

        What this cannot repair is an entity longer than the overlap, which
        no single window contains: it is found in halves and reported as
        two. Real entities here are short — ``PER`` tops out at 17
        characters and ``ORG`` at 37 — but one class reaches 11,694,
        so the halves are a known and visible failure rather than a
        surprising one.
        """
        if len(outputs) == 1:
            return outputs[0]

        best: dict[tuple[str, int, int], float] = {}
        for output in outputs:
            for span, confidence in zip(output.values, output.confidences):
                key = (span.label, span.start, span.end)
                best[key] = max(best.get(key, 0.0), confidence)

        order = sorted(best, key=lambda k: (k[1], k[2]))
        # One construction, values and confidences together: Spans sorts
        # itself into reading order and moves confidences with it, so a
        # value list built separately silently pairs the wrong numbers.
        return SpansPrediction(
            values=[
                Span(labels=[label], start=start, end=end, text=text[start:end])
                for label, start, end in order
            ],
            confidences=[round(best[key], 4) for key in order],
        )

    def _decode(self, text: str, logits: torch.Tensor, offsets) -> SpansPrediction:
        probs = torch.softmax(logits.float(), dim=-1)
        tags = probs.argmax(dim=-1).tolist()
        confidences = probs.max(dim=-1).values.tolist()

        spans: list[Span] = []
        current: dict | None = None
        for tag, conf, (start, end) in zip(tags, confidences, offsets.tolist()):
            if start == end == 0:
                continue
            if tag == 0:
                current = None
                continue
            class_name = self.classes[(tag - 1) // 2]
            is_begin = (tag - 1) % 2 == 0
            if current is not None and not is_begin and current["labels"] == [class_name]:
                current["end"] = end
                current["scores"].append(conf)
            else:
                current = {"labels": [class_name], "start": start, "end": end, "scores": [conf]}
                spans.append(current)

        # Sorted by Spans on the way in, so confidences are ordered to match
        # rather than left to line up by luck
        found = sorted(spans, key=lambda s: (s["start"], s["end"]))
        return SpansPrediction(
            values=[
                Span(
                    labels=s["labels"],
                    start=s["start"],
                    end=s["end"],
                    text=text[s["start"] : s["end"]],
                )
                for s in found
            ],
            confidences=[
                round(sum(s["scores"]) / len(s["scores"]), 4) for s in found
            ],
        )
