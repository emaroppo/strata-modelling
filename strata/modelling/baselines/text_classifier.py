"""Transformer baselines for text: document classification and spans.

Both fine-tune a pretrained encoder from Hugging Face and share the same
plumbing — tokenisation, training loop, checkpointing. They differ in what
a target is: a set of classes for the whole document, or labelled
character ranges inside it.

Documents are read from disk, so a model sees the same absolute paths an
image model does.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from strata.labels import ChoicesPrediction, Span, SpansPrediction

from ..model import Example, Model

console = Console()

DEFAULT_ENCODER = "distilbert-base-uncased"


def _read_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"warning: unreadable document {path} ({e}), using empty text", file=sys.stderr)
        return ""


class _TextDataset(Dataset):
    def __init__(self, samples: list[Example], encode):
        self.samples = samples
        self.encode = encode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return self.encode(_read_text(sample.path), sample.target)


class _TransformerBase(Model):
    """Shared training loop for the two text heads."""

    MAX_LENGTH = 512

    def __init__(
        self,
        encoder: str = DEFAULT_ENCODER,
        num_epochs: int = 3,
        batch_size: int = 8,
        lr: float = 5e-5,
        device: str | None = None,
    ):
        self.encoder = encoder
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
        self.classes: list[str] = []
        self._tokenizer = None
        self._model = None

    # -- pieces the two heads differ on ---------------------------------

    def _build_model(self, num_labels: int):
        raise NotImplementedError

    def _encode(self, text: str, target):
        raise NotImplementedError

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
    ) -> dict:
        samples, val_samples = train, val
        self._prepare(classes)
        loader = DataLoader(
            _TextDataset(samples, self._encode),
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
        with Progress(
            TextColumn("[bold cyan]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
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
        if val_samples:
            metrics.update(self._evaluate(val_samples))
        return metrics

    def _evaluate(self, samples: list[dict]) -> dict:
        loader = DataLoader(
            _TextDataset(samples, self._encode),
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
        keys = items[0].keys()
        return {k: torch.stack([item[k] for item in items]) for k in keys}

    def predict(self, paths: list[Path]) -> list:
        if self._model is None or not self.classes:
            raise RuntimeError("Model has no weights. Call finetune() or load() first.")
        self._model.eval()
        outputs = []
        with torch.no_grad(), Progress(
            TextColumn("[bold cyan]Predicting"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("predict", total=len(paths))
            for path in paths:
                text = _read_text(path)
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.MAX_LENGTH,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                )
                offsets = encoded.pop("offset_mapping")[0]
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self._model(**encoded).logits[0]
                outputs.append(self._decode(text, logits, offsets))
                progress.advance(task)
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

    def _build_model(self, num_labels: int):
        return AutoModelForSequenceClassification.from_pretrained(
            self.encoder,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            # The encoder may already carry a head for someone else's classes;
            # ours is sized for this project's, so re-initialise it
            ignore_mismatched_sizes=True,
        ).to(self.device)

    def _encode(self, text: str, target) -> dict:
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.MAX_LENGTH,
            padding="max_length", return_tensors="pt",
        )
        item = {k: v[0] for k, v in encoded.items()}
        labels = torch.zeros(len(self.classes))
        for name in (target.values if target is not None else []):
            if name in self.classes:
                labels[self.classes.index(name)] = 1.0
        item["labels"] = labels
        return item

    def _decode(self, text: str, logits: torch.Tensor, offsets) -> ChoicesPrediction:
        probs = torch.sigmoid(logits.float())
        indices = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
        if not indices:
            indices = [int(probs.argmax().item())]
        indices.sort(key=lambda i: probs[i].item(), reverse=True)
        return ChoicesPrediction(
            values=[self.classes[i] for i in indices],
            confidences=[round(probs[i].item(), 4) for i in indices],
        )


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

    def _tag_ids(self, class_name: str) -> tuple[int, int]:
        i = self.classes.index(class_name)
        return 1 + 2 * i, 2 + 2 * i  # B, I

    def _encode(self, text: str, target) -> dict:
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.MAX_LENGTH,
            padding="max_length", return_tensors="pt", return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0]
        item = {k: v[0] for k, v in encoded.items()}

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
                if start >= span.start and end <= span.end:
                    labels[t] = begin if first else inside
                    first = False
        item["labels"] = labels
        return item

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
            if current is not None and not is_begin and current["label"] == class_name:
                current["end"] = end
                current["scores"].append(conf)
            else:
                current = {"label": class_name, "start": start, "end": end, "scores": [conf]}
                spans.append(current)

        # Sorted by Spans on the way in, so confidences are ordered to match
        # rather than left to line up by luck
        found = sorted(spans, key=lambda s: (s["start"], s["end"]))
        return SpansPrediction(
            values=[
                Span(
                    label=s["label"],
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
