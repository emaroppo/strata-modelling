"""The backbone and its classifier head: building one, and growing it as classes are added."""

from typing import Protocol, cast

import timm
import torch
import torch.nn as nn

from .._shared import console

ARCHITECTURE = "convnextv2_base.fcmae_ft_in22k_in1k"


def build_backbone(num_classes: int, device) -> nn.Module:
    model = timm.create_model(ARCHITECTURE, pretrained=True, num_classes=num_classes)
    return model.to(device)


class Backbone(Protocol):
    """What this needs of a timm model beyond ``nn.Module``: its head, by name."""

    def get_classifier(self) -> nn.Linear: ...

    def reset_classifier(self, num_classes: int) -> None: ...

    def to(self, device) -> object: ...


def expand_head(backbone: nn.Module, num_classes: int, device) -> None:
    """Grow the classifier head, keeping the weights of existing classes.

    Adding a class should not cost the rounds already trained: the new
    neurons start from a fresh init while every existing class keeps the
    row it learned.
    """
    # A timm model; the head methods are its API, not nn.Module's
    model = cast(Backbone, backbone)
    old = model.get_classifier()
    old_weight = old.weight.data.clone()
    old_bias = old.bias.data.clone() if old.bias is not None else None

    model.reset_classifier(num_classes)
    model.to(device)

    new = model.get_classifier()
    with torch.no_grad():
        new.weight[: old_weight.shape[0]] = old_weight
        if old_bias is not None and new.bias is not None:
            new.bias[: old_bias.shape[0]] = old_bias


def prepare_backbone(
    backbone: nn.Module | None, current: list[str], classes: list[str], *, build, expand
):
    """The backbone to train ``classes`` with, continuing from ``backbone`` when the list allows.

    Training resumes from whatever ``load`` put in place, so each round
    builds on the last instead of restarting from ImageNet. Appending
    classes only grows the head; any other change to the list would shift
    the index each neuron stands for, so the model is rebuilt. ``build``
    makes a backbone for a class count and ``expand`` grows the current one.
    """
    if backbone is None:
        return build(len(classes))
    if current == classes:
        return backbone
    if classes[: len(current)] == current:
        added = classes[len(current) :]
        console.print(
            f"Expanding head {len(current)} -> {len(classes)} for "
            f"{', '.join(added)}; existing weights kept."
        )
        expand(len(classes))
        return backbone
    console.print(
        f"[yellow]Class list changed incompatibly "
        f"({', '.join(current)} -> {', '.join(classes)}); "
        f"training from pretrained weights.[/yellow]"
    )
    return build(len(classes))
