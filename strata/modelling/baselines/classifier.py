import os
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from rich import get_console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from strata.labels import ChoicesPrediction

from ..model import Example, Model

# Video-extracted frames are occasionally cut short; decode what's there
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_rgb(path: str | Path, draft_size: int | None = None) -> Image.Image:
    try:
        img = Image.open(path)
        if draft_size is not None:
            # JPEG-only fast path: decode at reduced scale straight from the
            # DCT domain; PIL picks the smallest scale still >= draft_size
            img.draft("RGB", (draft_size, draft_size))
        return img.convert("RGB")
    except OSError as e:
        print(
            f"warning: unreadable image {path} ({e}), using black placeholder",
            file=sys.stderr,
        )
        return Image.new("RGB", (256, 256))


#: The console rich itself hands out, not one of our own. Two Console
#: objects writing to one terminal cannot coordinate: a live display owned
#: by one knows nothing about text printed through the other, and the two
#: fight over the same lines — which is what made a progress bar flicker
#: against a model's own output.
console = get_console()


class _ImageDataset(Dataset):
    def __init__(
        self,
        samples: list[Example],
        classes: list[str],
        transform,
        draft_size: int | None = None,
        target_fn=None,
    ):
        self.samples = samples
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.draft_size = draft_size
        self.target_fn = target_fn

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = self.transform(_load_rgb(sample.path, self.draft_size))
        return image, self.target_fn(sample.target.values, self.class_to_idx)


class _LetterboxSquash:
    """Resize to a square, splitting the aspect gap between distortion and padding.

    The image is squashed by at most ``max_distortion``; whatever aspect
    difference remains is letterboxed with black bands. For 16:9 input and
    max_distortion=1.4 the content fills ~79% of the square.
    """

    def __init__(self, size: int, max_distortion: float = 1.4):
        self.size = size
        self.max_distortion = max_distortion

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        aspect = w / h
        residual = max(aspect, 1 / aspect) / self.max_distortion
        if residual <= 1:
            content_w = content_h = self.size
        elif aspect > 1:
            content_w, content_h = self.size, round(self.size / residual)
        else:
            content_w, content_h = round(self.size / residual), self.size
        img = img.resize((content_w, content_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size))
        canvas.paste(img, ((self.size - content_w) // 2, (self.size - content_h) // 2))
        return canvas


class _InferenceDataset(Dataset):
    def __init__(self, paths: list[Path], transform, draft_size: int | None = None):
        self.paths = paths
        self.transform = transform
        self.draft_size = draft_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return self.transform(_load_rgb(self.paths[idx], self.draft_size))


class MultiLabelClassifier(Model):
    """ConvNeXt V2 Base fine-tuned multi-label classifier.

    Uses timm's ``convnextv2_base`` with ImageNet-22k pre-trained weights.
    Supports multi-label outputs via ``BCEWithLogitsLoss``.
    """

    task = "classification"
    version = "1"

    IMG_SIZE = 288
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        num_epochs: int = 4,
        batch_size: int = 16,
        lr: float = 5e-5,
        device: str | None = None,
    ):
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
        self._backbone: nn.Module | None = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_backbone(self, num_classes: int) -> nn.Module:
        model = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k",
            pretrained=True,
            num_classes=num_classes,
        )
        return model.to(self.device)

    def _expand_head(self, num_classes: int) -> None:
        """Grow the classifier head, keeping the weights of existing classes.

        Adding a class should not cost the rounds already trained: the new
        neurons start from a fresh init while every existing class keeps the
        row it learned.
        """
        old = self._backbone.get_classifier()
        old_weight = old.weight.data.clone()
        old_bias = old.bias.data.clone() if old.bias is not None else None

        self._backbone.reset_classifier(num_classes)
        self._backbone.to(self.device)

        new = self._backbone.get_classifier()
        with torch.no_grad():
            new.weight[: old_weight.shape[0]] = old_weight
            if old_bias is not None and new.bias is not None:
                new.bias[: old_bias.shape[0]] = old_bias

    def _prepare_backbone(self, classes: list[str]) -> None:
        """Continue from the loaded weights when the class list allows it.

        Training resumes from whatever ``load`` put in place, so each round
        builds on the last instead of restarting from ImageNet. Appending
        classes only grows the head; any other change to the list would
        shift the index each neuron stands for, so the model is rebuilt.
        """
        if self._backbone is None:
            self._backbone = self._build_backbone(len(classes))
            return

        current = list(self.classes)
        if current == classes:
            return

        if classes[: len(current)] == current:
            added = classes[len(current) :]
            console.print(
                f"Expanding head {len(current)} -> {len(classes)} for "
                f"{', '.join(added)}; existing weights kept."
            )
            self._expand_head(len(classes))
            return

        console.print(
            f"[yellow]Class list changed incompatibly "
            f"({', '.join(current)} -> {', '.join(classes)}); "
            f"training from pretrained weights.[/yellow]"
        )
        self._backbone = self._build_backbone(len(classes))

    # ------------------------------------------------------------------
    # Task hooks — override these to change the classification regime
    # (see MulticlassClassifier); the training/eval/predict loops are shared.
    # ------------------------------------------------------------------

    def _effective_classes(self, classes: list[str]) -> list[str]:
        """Which dataset classes get an output neuron. Override to drop e.g.
        an implicit negative class that is only a dataset marker."""
        return list(classes)

    def _make_criterion(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def _encode_target(labels: list[str], class_to_idx: dict[str, int]) -> torch.Tensor:
        vec = torch.zeros(len(class_to_idx))
        for lbl in labels:
            if lbl in class_to_idx:
                vec[class_to_idx[lbl]] = 1.0
        return vec

    @staticmethod
    def _count_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
        preds = (torch.sigmoid(logits.float()) > 0.5).float()
        return int((preds == targets).all(dim=1).sum().item())

    @staticmethod
    def _activation(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits.float())

    def _to_output(self, probs: torch.Tensor) -> ChoicesPrediction:
        indices = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
        if not indices:
            # Fall back to argmax when nothing clears the threshold
            indices = [int(probs.argmax().item())]
        indices.sort(key=lambda i: probs[i].item(), reverse=True)
        return ChoicesPrediction(
            values=[self.classes[i] for i in indices],
            confidences=[round(probs[i].item(), 4) for i in indices],
        )

    @property
    def _train_transform(self):
        return transforms.Compose(
            [
                _LetterboxSquash(self.IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ]
        )

    @property
    def _eval_transform(self):
        return transforms.Compose(
            [
                _LetterboxSquash(self.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ]
        )

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def finetune(
        self,
        train: list[Example],
        classes: list[str],
        val: list[Example] | None = None,
        on_epoch=None,
    ) -> dict:
        samples, val_samples = train, val
        classes = self._effective_classes(classes)
        self._prepare_backbone(classes)
        self.classes = classes

        dataset = _ImageDataset(
            samples,
            classes,
            self._train_transform,
            draft_size=self.IMG_SIZE,
            target_fn=self._encode_target,
        )
        # num_workers > 0 hangs on macOS MPS; pin_memory is unsupported there too
        on_mps = self.device.type == "mps"
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0 if on_mps else 4,
            pin_memory=not on_mps,
        )

        criterion = self._make_criterion()
        optimizer = torch.optim.AdamW(
            self._backbone.parameters(), lr=self.lr, weight_decay=1e-2
        )
        # fp16 autocast roughly halves activation memory; no-op off CUDA
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        # Per-step warmup then cosine decay: the first high-LR steps on a
        # fresh head are where fine-tuning occasionally diverged
        total_steps = self.num_epochs * len(loader)
        warmup_steps = max(1, min(100, total_steps // 10))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_steps - warmup_steps
                ),
            ],
            milestones=[warmup_steps],
        )

        self._backbone.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with Progress(
            TextColumn(
                "[bold cyan]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]:.4f} acc={task.fields[acc]:.3f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            epoch_task = progress.add_task(
                "training",
                total=self.num_epochs * len(loader),
                epoch=1,
                total_epochs=self.num_epochs,
                loss=0.0,
                acc=0.0,
            )

            for epoch in range(1, self.num_epochs + 1):
                progress.update(epoch_task, epoch=epoch)
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_samples = 0

                for images, labels in loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        logits = self._backbone(images)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self._backbone.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    epoch_loss += loss.item() * images.size(0)
                    epoch_correct += self._count_correct(logits, labels)
                    epoch_samples += images.size(0)

                    progress.update(
                        epoch_task,
                        advance=1,
                        loss=epoch_loss / max(epoch_samples, 1),
                        acc=epoch_correct / max(epoch_samples, 1),
                    )

                total_loss += epoch_loss
                total_correct += epoch_correct
                total_samples += epoch_samples

                if on_epoch is not None:
                    # At the epoch boundary rather than per batch: a caller
                    # is on the other end of a network, and a report per
                    # step would be thousands of updates to say the same
                    # thing more often.
                    on_epoch(
                        epoch,
                        self.num_epochs,
                        {
                            "loss": epoch_loss / max(epoch_samples, 1),
                            "accuracy": epoch_correct / max(epoch_samples, 1),
                        },
                    )

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        metrics = {"loss": avg_loss, "accuracy": accuracy}
        if val_samples:
            metrics.update(self._evaluate(val_samples, criterion))
        return metrics

    def _evaluate(self, samples: list[dict], criterion: nn.Module) -> dict:
        on_mps = self.device.type == "mps"
        use_amp = self.device.type == "cuda"
        loader = DataLoader(
            _ImageDataset(
                samples,
                self.classes,
                self._eval_transform,
                draft_size=self.IMG_SIZE,
                target_fn=self._encode_target,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0 if on_mps else 4,
            pin_memory=not on_mps,
        )
        self._backbone.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = self._backbone(images)
                loss = criterion(logits.float(), labels)
                total_loss += loss.item() * images.size(0)
                total_correct += self._count_correct(logits, labels)
                total_samples += images.size(0)
        self._backbone.train()
        return {
            "val_loss": total_loss / max(total_samples, 1),
            "val_accuracy": total_correct / max(total_samples, 1),
        }

    def predict(self, image_paths: list[Path], on_batch=None) -> list[ChoicesPrediction]:
        if self._backbone is None or not self.classes:
            raise RuntimeError("Model has no weights. Call finetune() or load() first.")
        if not image_paths:
            return []

        self._backbone.eval()
        on_mps = self.device.type == "mps"
        use_amp = self.device.type == "cuda"
        loader = DataLoader(
            _InferenceDataset(
                image_paths, self._eval_transform, draft_size=self.IMG_SIZE
            ),
            # No gradients/optimizer state at inference: much larger batches fit,
            # and JPEG decode needs more workers to keep the GPU fed
            batch_size=self.batch_size * 4,
            shuffle=False,
            num_workers=0 if on_mps else min(12, os.cpu_count() or 4),
            pin_memory=not on_mps,
        )

        backbone = self._backbone
        if use_amp and len(image_paths) >= 2000:
            # ~25s one-time compile, ~2x steady-state — only worth it on big jobs
            backbone = torch.compile(self._backbone)

        batch_probs: list[torch.Tensor] = []
        with (
            torch.no_grad(),
            Progress(
                TextColumn("[bold cyan]Predicting"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress,
        ):
            predict_task = progress.add_task("predict", total=len(image_paths))
            done = 0
            for batch in loader:
                batch = batch.to(self.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = backbone(batch)
                batch_probs.append(self._activation(logits).cpu())
                progress.advance(predict_task, batch.size(0))
                done += batch.size(0)
                if on_batch is not None:
                    # Per batch rather than per sample: a caller is on the
                    # other end of a network, and thousands of updates say
                    # the same thing more often.
                    on_batch(done, len(image_paths))

        return [self._to_output(probs) for probs in torch.cat(batch_probs)]

    def save(self, path: Path) -> None:
        if self._backbone is None:
            raise RuntimeError("No model to save.")
        torch.save(
            {
                "state_dict": self._backbone.state_dict(),
                "classes": self.classes,
                "config": {
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                },
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, weights_only=True, map_location=self.device)
        self.classes = checkpoint["classes"]
        # The checkpoint's config is provenance only: hyperparameters belong
        # to [model.params] in project.toml, and letting a checkpoint
        # override them made editing them look like it did nothing
        self._backbone = self._build_backbone(len(self.classes))
        self._backbone.load_state_dict(checkpoint["state_dict"])


class MulticlassClassifier(MultiLabelClassifier):
    """Single-label variant: classes are mutually exclusive.

    Same backbone, training loop, and data pipeline as MultiLabelClassifier — only the
    loss (cross-entropy vs BCE), target encoding, and prediction decoding
    differ. Predictions carry exactly one label with its softmax confidence.
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
            values=[self.classes[idx]],
            confidences=[round(probs[idx].item(), 4)],
        )


class PresenceClassifier(MultiLabelClassifier):
    """Independent presence detectors with an implicit negative class.

    One sigmoid per positive class ("is X present in the picture?"), so any
    combination of classes can co-occur. NEGATIVE_LABEL is a dataset/LS
    marker meaning "reviewed, nothing present": it gets no output neuron,
    trains as an all-zeros target (labels outside the head are ignored by
    ``_encode_target``), and is emitted as the prediction when no class
    clears the threshold — the model cannot contradict itself.
    """

    NEGATIVE_LABEL = "none"
    requires_classes = (NEGATIVE_LABEL,)

    def _effective_classes(self, classes: list[str]) -> list[str]:
        return [c for c in classes if c != self.NEGATIVE_LABEL]

    def _to_output(self, probs: torch.Tensor) -> ChoicesPrediction:
        indices = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
        if not indices:
            return ChoicesPrediction(
                values=[self.NEGATIVE_LABEL],
                confidences=[round(1.0 - probs.max().item(), 4)],
            )
        indices.sort(key=lambda i: probs[i].item(), reverse=True)
        return ChoicesPrediction(
            values=[self.classes[i] for i in indices],
            confidences=[round(probs[i].item(), 4) for i in indices],
        )
