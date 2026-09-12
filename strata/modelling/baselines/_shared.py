"""What the two baseline modules share.

They were written in parallel and grew the same furniture: a console, the
two progress displays, the device rule, and the threshold that turns
sigmoid probabilities into a prediction. Here once, so a change to one is a
change to both — and out of ``baselines/__init__.py``, which the registry
imports on a base install that has none of these frameworks.
"""

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

from strata.labels import ChoicesPrediction

#: The console rich itself hands out, not one of our own. Two Console
#: objects writing to one terminal cannot coordinate: a live display owned
#: by one knows nothing about text printed through the other, and the two
#: fight over the same lines — which is what made a progress bar flicker
#: against a model's own output.
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


def pick_device(name: str | None) -> torch.device:
    """The device asked for, else the best one present."""
    return torch.device(
        name
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
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
