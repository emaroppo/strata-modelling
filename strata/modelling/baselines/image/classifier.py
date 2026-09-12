"""ConvNeXt V2 Base fine-tuned as a multi-label classifier."""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from strata.labels import ChoicesPrediction

from ...model import Example, Model
from .._shared import pick_device, threshold_choices
from . import head, loop
from .data import ImageDataset, InferenceDataset, LetterboxSquash


class MultiLabelClassifier(Model):
    """ConvNeXt V2 Base fine-tuned multi-label classifier.

    Uses timm's ``convnextv2_base`` with ImageNet-22k pre-trained weights.
    Supports multi-label outputs via ``BCEWithLogitsLoss``. The single-label
    and presence variants override the task hooks below and share the rest.
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
        self.device = pick_device(device)
        self.classes: list[str] = []
        self._backbone: nn.Module | None = None

    # -- the backbone, as methods so a test or a subclass can stand in a small one --

    def _build_backbone(self, num_classes: int) -> nn.Module:
        return head.build_backbone(num_classes, self.device)

    def _expand_head(self, num_classes: int) -> None:
        head.expand_head(self._backbone, num_classes, self.device)

    def _prepare_backbone(self, classes: list[str]) -> None:
        self._backbone = head.prepare_backbone(
            self._backbone,
            list(self.classes),
            classes,
            build=self._build_backbone,
            expand=self._expand_head,
        )

    # -- task hooks ---------------------------------------------------------

    def _effective_classes(self, classes: list[str]) -> list[str]:
        """Which dataset classes get an output neuron."""
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
        return threshold_choices(probs, self.classes)

    # -- data ---------------------------------------------------------------

    def _transform(self, train: bool):
        steps = [LetterboxSquash(self.IMG_SIZE)]
        if train:
            steps += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        steps += [transforms.ToTensor(), transforms.Normalize(self.MEAN, self.STD)]
        return transforms.Compose(steps)

    def _loader(self, dataset, *, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
        # num_workers > 0 hangs on macOS MPS; pin_memory is unsupported there too
        on_mps = self.device.type == "mps"
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0 if on_mps else workers,
            pin_memory=not on_mps,
        )

    def _labelled(self, samples: list[Example], train: bool) -> DataLoader:
        dataset = ImageDataset(
            samples,
            self.classes,
            self._transform(train),
            draft_size=self.IMG_SIZE,
            target_fn=self._encode_target,
        )
        return self._loader(dataset, batch_size=self.batch_size, shuffle=train, workers=4)

    # -- the model contract -------------------------------------------------

    def finetune(
        self,
        train: list[Example],
        classes: list[str],
        val: list[Example] | None = None,
        on_epoch=None,
    ) -> dict:
        classes = self._effective_classes(classes)
        self._prepare_backbone(classes)
        self.classes = classes
        criterion = self._make_criterion()
        loss, accuracy = loop.fit(
            self._backbone,
            self._labelled(train, train=True),
            criterion,
            lr=self.lr,
            num_epochs=self.num_epochs,
            device=self.device,
            count_correct=self._count_correct,
            on_epoch=on_epoch,
        )
        metrics = {"loss": loss, "accuracy": accuracy}
        if val:
            metrics.update(
                loop.evaluate(
                    self._backbone,
                    self._labelled(val, train=False),
                    criterion,
                    device=self.device,
                    count_correct=self._count_correct,
                )
            )
        return metrics

    def predict(self, image_paths: list[Path], on_batch=None, *, features=None) -> list:
        if self._backbone is None or not self.classes:
            raise RuntimeError("Model has no weights. Call finetune() or load() first.")
        if not image_paths:
            return []
        loader = self._loader(
            InferenceDataset(image_paths, self._transform(train=False), draft_size=self.IMG_SIZE),
            # No gradients or optimizer state at inference: much larger batches
            # fit, and JPEG decode needs more workers to keep the GPU fed
            batch_size=self.batch_size * 4,
            shuffle=False,
            workers=min(12, os.cpu_count() or 4),
        )
        probs = loop.predict_probs(
            self._backbone,
            loader,
            total=len(image_paths),
            device=self.device,
            activation=self._activation,
            on_batch=on_batch,
        )
        return [self._to_output(p) for p in probs]

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
        # to [model.params], and letting a checkpoint override them made
        # editing them look like it did nothing
        self._backbone = self._build_backbone(len(self.classes))
        self._backbone.load_state_dict(checkpoint["state_dict"])
