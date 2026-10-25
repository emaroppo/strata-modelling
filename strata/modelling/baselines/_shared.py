"""What the baseline modules share: console, progress displays, device, seeding, thresholding.

Kept out of ``baselines/__init__.py``, which a base install imports. See ``docs/adr/0034``.
"""

import random

import numpy
import torch
from rich import get_console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from strata.contracts import ChoicesPrediction

#: The console rich itself hands out, not one of our own. See ``docs/adr/0030``.
console = get_console()


def epoch_progress(fields: str) -> Progress:
    """The training display: epoch of total, a bar, and whatever ``fields`` shows."""
    return Progress(
        TextColumn("[bold cyan]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn(fields),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def predict_progress() -> Progress:
    """The scoring display."""
    return Progress(
        TextColumn("[bold cyan]Predicting"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def seed_everything(seed: int | None) -> None:
    """Seed Python's, numpy's and torch's generators, CUDA's included; a no-op without a seed."""
    if seed is None:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str | None) -> torch.device:
    """The device asked for, else the best one present."""
    return torch.device(
        name
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )


def threshold_choices(
    probs: torch.Tensor, classes: list[str], threshold: float = 0.5
) -> ChoicesPrediction:
    """Every class over the threshold, most confident first; the best guess if none."""
    indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
    if not indices:
        indices = [int(probs.argmax().item())]
    indices.sort(key=lambda i: probs[i].item(), reverse=True)
    return ChoicesPrediction(
        values=[classes[i] for i in indices],
        confidences=[round(probs[i].item(), 4) for i in indices],
    )
