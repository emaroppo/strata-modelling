"""Span labelling with GLiNER: whole spans scored against a prompt per class, zero-shot at
``num_epochs=0``.

Long documents are cut into overlapping word windows and merged back; a
training span not on word boundaries is dropped and counted (``docs/adr/0036``).
"""

import re
import tempfile
from pathlib import Path
from typing import cast

import torch

from strata.contracts import Span, Spans, SpansPrediction

from ...model import Example, LabelModel
from .._shared import pick_device, predict_progress, seed_everything
from .bio import merge_windows
from .spans import span_scores
from .windows import read_text

DEFAULT_CHECKPOINT = "knowledgator/gliner-pii-base-v1.0"

#: What GLiNER is asked to find for each class; a class with no entry is asked for by name.
DEFAULT_PROMPTS = {
    "PER": "person",
    "ORG": "organization",
    "LOC": "location",
    "DATE": "date",
    "EMAIL": "email address",
    "PHONE": "phone number",
    "URL": "url",
    "ATTACHMENT": "attachment file name",
}

#: GLiNER's own word splitter (``WhitespaceTokenSplitter``), which spans and windows must agree
#: with.
WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")

#: The same, but a URL or an email address is one word, since a span is at most ``max_width`` words.
WHOLE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"'()\[\]]*[^\s<>\"'()\[\].,;:!?]"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|\w+(?:[-_]\w+)*|\S"
)

#: The checkpoint file's layout, so a later one can still read this one.
FORMAT = 1


def words(text: str, pattern: re.Pattern = WORD) -> list[tuple[int, int]]:
    """Each word's character range, in order."""
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


class Splitter:
    """A word splitter in the shape GLiNER's processor calls: (word, start, end)."""

    def __init__(self, pattern: re.Pattern):
        self.pattern = pattern

    def __call__(self, text: str):
        for m in self.pattern.finditer(text):
            yield m.group(), m.start(), m.end()


def windows(count: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Word ranges ``[start, end)`` of at most ``size`` words covering ``count``."""
    if count <= size:
        return [(0, count)]
    step = max(1, size - overlap)
    ranges = []
    start = 0
    while True:
        end = min(start + size, count)
        ranges.append((start, end))
        if end == count:
            return ranges
        start += step


def to_training(
    text: str, spans: list[Span], prompts: dict[str, str], size: int, pattern: re.Pattern = WORD
) -> tuple[list[dict], int]:
    """GLiNER's training records for one document, and how many spans did not fit.

    Windows do not overlap, or an entity in the overlap would train twice;
    a span off word boundaries or across an edge is dropped and counted.
    """
    bounds = words(text, pattern)
    first = {start: i for i, (start, _) in enumerate(bounds)}
    last = {end: i for i, (_, end) in enumerate(bounds)}
    placed: list[tuple[int, int, str]] = []
    dropped = 0
    for span in spans:
        i, j = first.get(span.start), last.get(span.end)
        if i is None or j is None or j < i:
            dropped += 1
            continue
        placed.append((i, j, prompts.get(span.label, span.label)))

    records = []
    kept = 0
    for start, end in windows(len(bounds), size, 0):
        inside = [(i, j, p) for i, j, p in placed if start <= i and j < end]
        kept += len(inside)
        records.append(
            {
                "tokenized_text": [text[a:b] for a, b in bounds[start:end]],
                "ner": [[i - start, j - start, p] for i, j, p in inside],
            }
        )
    return records, dropped + len(placed) - kept


class GlinerSpanTagger(LabelModel):
    """GLiNER over a document's text, zero-shot or fine-tuned."""

    label_type = "span"
    version = "1"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        revision: str | None = None,
        num_epochs: int = 0,
        threshold: float = 0.5,
        prompts: dict[str, str] | None = None,
        max_width: int | None = None,
        whole_urls: bool = False,
        chunk_words: int | None = None,
        chunk_overlap: int = 32,
        lr: float = 5e-6,
        others_lr: float = 1e-5,
        batch_size: int = 8,
        seed: int | None = None,
        device: str | None = None,
    ):
        """``max_width`` and ``chunk_words`` default to the checkpoint's own; ``revision`` left out
        is resolved and saved.

        ``whole_urls`` makes each URL and email address one word (``WHOLE``).
        """
        if chunk_words is not None and chunk_overlap >= chunk_words:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_words "
                f"({chunk_words}), or the windows do not advance."
            )
        self.checkpoint = checkpoint
        self.revision = revision
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.prompts = {**DEFAULT_PROMPTS, **(prompts or {})}
        self.max_width = max_width
        self.whole_urls = whole_urls
        self.chunk_words = chunk_words
        self.chunk_overlap = chunk_overlap
        self.lr = lr
        self.others_lr = others_lr
        self.batch_size = batch_size
        self.seed = seed
        self.device = pick_device(device)
        self.classes: list[str] = []
        self.trained = False
        self._model = None

    # -- the checkpoint ------------------------------------------------------

    def _load(self, state_dict: dict | None = None) -> None:
        from gliner import GLiNER  # the gliner extra, imported where it is used

        if self.revision is None:
            self.revision = _resolved(self.checkpoint)
        model = GLiNER.from_pretrained(
            self.checkpoint, revision=self.revision, max_width=self.max_width
        )
        if state_dict is not None:
            model.load_state_dict(state_dict)
        if self.whole_urls:
            model.data_processor.words_splitter = Splitter(WHOLE)
        self._model = model.to(self.device)

    def _net(self):
        if self._model is None or not self.classes:
            raise RuntimeError("Model has no weights. Call finetune() or load() first.")
        return self._model

    def _pattern(self) -> re.Pattern:
        return WHOLE if self.whole_urls else WORD

    def _window(self) -> int:
        size = self.chunk_words or int(getattr(self._net().config, "max_len", 384))
        return max(size, self.chunk_overlap + 1)

    def requires_schema(self, schema) -> None:
        """One label per region and no overlaps, as the merge of windows assumes."""
        cannot = []
        if getattr(schema, "overlapping", False):
            cannot.append("overlapping regions")
        if getattr(schema, "multi_label", False):
            cannot.append("regions carrying more than one label")
        if cannot:
            raise ValueError(
                f"it declares {' and '.join(cannot)}, and this model decodes one "
                f"label per region with no overlaps (flat NER)."
            )

    # -- training ------------------------------------------------------------

    def finetune(
        self,
        train: list[Example],
        classes: list[str],
        val: list[Example] | None = None,
        on_epoch=None,
    ) -> dict:
        seed_everything(self.seed)
        self.classes = list(classes)
        if self._model is None:
            self._load()
        metrics: dict[str, float] = {}
        if self.num_epochs > 0:
            metrics.update(self._train(train))
        if val:
            predicted = self.predict([example.path for example in val])
            truth = [
                list(example.target.values) if isinstance(example.target, Spans) else []
                for example in val
            ]
            metrics.update(span_scores(truth, [list(p.values) for p in predicted]))
        return metrics

    def _train(self, train: list[Example]) -> dict:
        size = self._window()
        records: list[dict] = []
        dropped = 0
        for example in train:
            spans = list(example.target.values) if isinstance(example.target, Spans) else []
            found, lost = to_training(
                read_text(example.path), spans, self.prompts, size, self._pattern()
            )
            records += found
            dropped += lost
        steps = max(1, self.num_epochs * -(-len(records) // self.batch_size))
        model = self._net()
        with tempfile.TemporaryDirectory() as scratch:
            args = model.create_training_args(
                output_dir=scratch,
                learning_rate=self.lr,
                others_lr=self.others_lr,
                per_device_train_batch_size=self.batch_size,
                max_steps=steps,
                save_strategy="no",
                logging_steps=max(1, steps // 10),
                use_cpu=self.device.type == "cpu",
                # With fp16 GLiNER's summed loss overflows and the scaler skips every step.
                fp16=False,
                dataloader_num_workers=0,
                seed=self.seed if self.seed is not None else 42,
            )
            trainer = model.train_model(records, None, training_args=args)
        self._model = model.to(self.device)
        self.trained = True
        loss = next(
            (e["train_loss"] for e in reversed(trainer.state.log_history) if "train_loss" in e),
            0.0,
        )
        # docs/adr/0036
        return {"loss": float(loss), "train_spans_dropped": float(dropped)}

    # -- prediction ----------------------------------------------------------

    def predict(self, paths: list[Path], on_batch=None, *, features=None) -> list:
        if not paths:
            return []
        model = self._net()
        model.eval()
        by_prompt = {self.prompts.get(c, c): c for c in self.classes}
        labels = list(by_prompt)
        size, overlap = self._window(), self.chunk_overlap
        outputs = []
        with torch.no_grad(), predict_progress() as progress:
            task = progress.add_task("predict", total=len(paths))
            for done, path in enumerate(paths, start=1):
                text = read_text(path)
                bounds = words(text, self._pattern())
                per_window = []
                ranges = windows(len(bounds), size, overlap) if bounds else []
                pieces = [(bounds[a][0], text[bounds[a][0] : bounds[b - 1][1]]) for a, b in ranges]
                found = cast(
                    list[list[dict]],
                    model.inference(
                        [piece for _, piece in pieces],
                        labels,
                        flat_ner=True,
                        threshold=self.threshold,
                        batch_size=self.batch_size,
                    )
                    if pieces
                    else [],
                )
                for (offset, _), entities in zip(pieces, found, strict=True):
                    per_window.append(_prediction(text, entities, offset, by_prompt))
                outputs.append(merge_windows(text, per_window) if per_window else SpansPrediction())
                progress.advance(task)
                if on_batch is not None:
                    on_batch(done, len(paths))
        return outputs

    # -- checkpoints ---------------------------------------------------------

    def save(self, path: Path) -> None:
        """One file; zero-shot keeps no weights, only the pinned revision, fetched again on load."""
        net = self._net()
        torch.save(
            {
                "format": FORMAT,
                "checkpoint": self.checkpoint,
                "revision": self.revision,
                "classes": self.classes,
                "prompts": self.prompts,
                "max_width": self.max_width,
                "whole_urls": self.whole_urls,
                "state_dict": net.state_dict() if self.trained else None,
            },
            path,
        )

    def load(self, path: Path) -> None:
        saved = torch.load(path, weights_only=True, map_location="cpu")
        self.checkpoint = saved["checkpoint"]
        self.revision = saved["revision"]
        self.classes = list(saved["classes"])
        self.prompts = dict(saved["prompts"])
        self.max_width = saved["max_width"]
        self.whole_urls = saved.get("whole_urls", False)
        self.trained = saved["state_dict"] is not None
        self._load(saved["state_dict"])


def _prediction(text: str, entities: list[dict], offset: int, by_prompt: dict) -> SpansPrediction:
    """One window's entities, at their place in the whole document."""
    found = []
    for entity in entities:
        label = by_prompt.get(entity["label"])
        if label is None:
            continue
        start, end = entity["start"] + offset, entity["end"] + offset
        found.append((start, end, label, float(entity["score"])))
    found.sort(key=lambda f: (f[0], f[1], f[2]))
    return SpansPrediction(
        values=[Span(labels=[label], start=s, end=e, text=text[s:e]) for s, e, label, _ in found],
        confidences=[score for *_, score in found],
    )


def _resolved(checkpoint: str) -> str | None:
    """The commit a checkpoint name points at now, or None when it cannot be asked."""
    try:
        from huggingface_hub import model_info  # comes with the gliner extra's transformers

        return model_info(checkpoint).sha
    # Not pinned, never a failed run.
    except Exception:
        return None
