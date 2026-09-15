"""The training, validation and prediction loops, over any backbone and loader."""

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler

from .._shared import epoch_progress, predict_progress


def fit(backbone, loader, criterion, *, lr, num_epochs, device, count_correct, on_epoch=None):
    """Train for ``num_epochs`` and return the mean loss and accuracy over all of them."""
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=lr, weight_decay=1e-2)
    # fp16 autocast roughly halves activation memory; no-op off CUDA
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    # Per-step warmup then cosine decay
    total_steps = num_epochs * len(loader)
    warmup_steps = max(1, min(100, total_steps // 10))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps],
    )

    backbone.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with epoch_progress("loss={task.fields[loss]:.4f} acc={task.fields[acc]:.3f}") as progress:
        task = progress.add_task(
            "training", total=total_steps, epoch=1, total_epochs=num_epochs, loss=0.0, acc=0.0
        )
        for epoch in range(1, num_epochs + 1):
            progress.update(task, epoch=epoch)
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = backbone(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item() * images.size(0)
                epoch_correct += count_correct(logits, labels)
                epoch_samples += images.size(0)
                progress.update(
                    task,
                    advance=1,
                    loss=epoch_loss / max(epoch_samples, 1),
                    acc=epoch_correct / max(epoch_samples, 1),
                )

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
            if on_epoch is not None:
                # At the epoch boundary rather than per batch. docs/adr/0031
                on_epoch(
                    epoch,
                    num_epochs,
                    {
                        "loss": epoch_loss / max(epoch_samples, 1),
                        "accuracy": epoch_correct / max(epoch_samples, 1),
                    },
                )
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def evaluate(backbone, loader, criterion, *, device, count_correct) -> dict:
    """Validation loss and accuracy, leaving the backbone in training mode."""
    use_amp = device.type == "cuda"
    backbone.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = backbone(images)
            loss = criterion(logits.float(), labels)
            total_loss += loss.item() * images.size(0)
            total_correct += count_correct(logits, labels)
            total_samples += images.size(0)
    backbone.train()
    return {
        "val_loss": total_loss / max(total_samples, 1),
        "val_accuracy": total_correct / max(total_samples, 1),
    }


def predict_probs(backbone, loader, *, total: int, device, activation, on_batch=None):
    """Every image's class probabilities, in loader order, reporting per batch."""
    use_amp = device.type == "cuda"
    backbone.eval()
    if use_amp and total >= 2000:
        # ~25s one-time compile, ~2x steady-state — only worth it on big jobs
        backbone = torch.compile(backbone)
    batches: list[torch.Tensor] = []
    with torch.no_grad(), predict_progress() as progress:
        task = progress.add_task("predict", total=total)
        done = 0
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = backbone(batch)
            batches.append(activation(logits).cpu())
            progress.advance(task, batch.size(0))
            done += batch.size(0)
            if on_batch is not None:
                # Per batch rather than per sample. docs/adr/0031
                on_batch(done, total)
    return torch.cat(batches)
