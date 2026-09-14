"""The training loop the text heads share, over a windowed encoder."""

from pathlib import Path
from typing import ClassVar, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...model import Example, Model
from .._shared import console, epoch_progress, pick_device, predict_progress
from .windows import DEFAULT_ENCODER, WindowDataset, Windowed, read_text


class TransformerBase(Windowed, Model):
    """Shared training loop for the text heads."""

    #: Whether this head has to combine several windows into one answer for
    #: the document. A tagger does not — spans from adjacent windows are
    #: spans in the same document. A classifier does, and there is no
    #: obvious right way, so it has to be told rather than defaulted.
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
        """``window`` is off by default; off means truncation at ``MAX_LENGTH``.

        Whether to window is a property of the corpus, so it is declared
        in ``[model.params]`` and recorded on the run (``docs/adr/0014``).
        ``window_overlap`` is how many tokens consecutive windows share; it
        must exceed the longest thing being labelled, and defaults to a
        quarter of the window.
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
            if window_aggregation is not None and window_aggregation not in self.AGGREGATIONS:
                # Here rather than in `_merge`, which does not run until a
                # document needs combining — a typo would otherwise survive
                # a whole training run and fail on the first prediction.
                raise ValueError(
                    f"{type(self).__name__} cannot combine windows by "
                    f"{window_aggregation!r}. It takes {choices}."
                )
        super().__init__(encoder, window, window_overlap if window_overlap is not None else 128)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.window_aggregation = window_aggregation
        self.device = pick_device(device)
        self.classes: list[str] = []
        self._model: nn.Module | None = None

    # -- pieces the heads differ on ------------------------------------------

    def _build_model(self, num_labels: int):
        raise NotImplementedError

    def _merge(self, text: str, outputs: list):
        """Combine one document's windows into one answer.

        Called with more than one only when windowing is on.
        """
        return outputs[0]

    def _decode(self, text: str, logits: torch.Tensor, offsets):
        raise NotImplementedError

    def _label_count(self) -> int:
        return len(self.classes)

    def _net(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model has no weights. Call finetune() or load() first.")
        return self._model

    # -- shared --------------------------------------------------------------

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
            # A transformers model: base_model is its encoder, an nn.Module
            body = cast(nn.Module, self._net().base_model)
            self.classes = list(classes)
            self._model = self._build_model(self._label_count())
            cast(nn.Module, self._net().base_model).load_state_dict(body.state_dict())
            return
        console.print(
            "[yellow]Class list changed incompatibly; starting from the "
            "pretrained encoder.[/yellow]"
        )
        self.classes = list(classes)
        self._model = self._build_model(self._label_count())

    def _loader(self, samples: list[Example], shuffle: bool) -> tuple[WindowDataset, DataLoader]:
        dataset = WindowDataset(samples, self._encode, self._scan)
        return dataset, DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self._collate
        )

    @staticmethod
    def _collate(items: list[dict]) -> dict:
        # ``offsets`` rides along for decoding and is not model input; the
        # forward pass would reject it as an unexpected keyword.
        keys = [k for k in items[0] if k != "offsets"]
        return {k: torch.stack([item[k] for item in items]) for k in keys}

    def finetune(
        self,
        train: list[Example],
        classes: list[str],
        val: list[Example] | None = None,
        on_epoch=None,
    ) -> dict:
        self._prepare(classes)
        dataset, loader = self._loader(train, shuffle=True)
        optimizer = torch.optim.AdamW(self._net().parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = max(1, self.num_epochs * len(loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=0.1
        )

        self._net().train()
        total_loss = 0.0
        seen = 0
        with epoch_progress("loss={task.fields[loss]:.4f}") as progress:
            task = progress.add_task(
                "training", total=total_steps, epoch=1, total_epochs=self.num_epochs, loss=0.0
            )
            for epoch in range(1, self.num_epochs + 1):
                progress.update(task, epoch=epoch)
                for batch in loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    outputs = self._net()(**batch)
                    outputs.loss.backward()
                    nn.utils.clip_grad_norm_(self._net().parameters(), 1.0)
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

    def _evaluate(self, samples: list[Example]) -> dict:
        _, loader = self._loader(samples, shuffle=False)
        self._net().eval()
        total_loss = 0.0
        seen = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self._net()(**batch)
                total_loss += outputs.loss.item() * batch["input_ids"].size(0)
                seen += batch["input_ids"].size(0)
        self._net().train()
        return {"val_loss": total_loss / max(seen, 1)}

    def predict(self, paths: list[Path], on_batch=None, *, features=None) -> list:
        if self._model is None or not self.classes:
            raise RuntimeError("Model has no weights. Call finetune() or load() first.")
        self._net().eval()
        outputs = []
        with torch.no_grad(), predict_progress() as progress:
            task = progress.add_task("predict", total=len(paths))
            for done, path in enumerate(paths, start=1):
                text = read_text(path)
                # One item per window, so a long document is answered whole
                # rather than up to the encoder's limit and no further.
                per_window = []
                for item in self._encode(text, None):
                    fields = {
                        k: v.unsqueeze(0).to(self.device)
                        for k, v in item.items()
                        if k in ("input_ids", "attention_mask", "token_type_ids")
                    }
                    logits = self._net()(**fields).logits[0]
                    per_window.append(self._decode(text, logits, item["offsets"]))
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
                "state_dict": self._net().state_dict(),
                "classes": self.classes,
                "encoder": self.encoder,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, weights_only=True, map_location=self.device)
        self.classes = checkpoint["classes"]
        self.encoder = checkpoint.get("encoder", self.encoder)
        self._model = self._build_model(self._label_count())
        self._net().load_state_dict(checkpoint["state_dict"])
