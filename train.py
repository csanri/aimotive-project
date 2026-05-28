import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from logging import Logger


def lossfunc(
    preds:   torch.Tensor,
    targets: torch.Tensor,
    masks:   torch.Tensor
) -> torch.Tensor:
    """
    kiszámoljuk a maszkolt l1 veszteséget a ritka mélységtérképen.
    preds: háló tippje [Batch, 1, H, W]
    targets: valódi mélység (depth) [Batch, 1, H, W]
    masks: bináris maszk (gt_mask) [Batch, 1, H, W]
    """
    absolute_error = torch.abs(preds - targets)
    masked_error = absolute_error * masks

    total_loss = torch.sum(masked_error)
    valid_pixels = torch.sum(masks)

    if valid_pixels == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    return total_loss / valid_pixels


def train_epoch(
    model,
    train_loader:     DataLoader,
    optimizer:        torch.optim.Optimizer,
    device:           str,
    scaler:           GradScaler,
    grad_accum_steps: int,
    logger:           Logger
) -> float:

    model.train()
    running_loss = .0
    optimizer.zero_grad()
    total_batches = len(train_loader)
    total_losses = []

    for i, data in enumerate(train_loader):
        images = data["image"].to(device)
        depths = data["depth"].to(device)
        masks = data["gt_mask"].to(device)

        with autocast("cuda"):
            preds = model(images)
            loss = lossfunc(preds, depths, masks)
            loss /= grad_accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_loss = loss.item() * grad_accum_steps
        running_loss += loss.item() * grad_accum_steps
        total_losses.append(running_loss / (i+1))

        logger.info(
            f"Batch [{i+1:4d}/{total_batches}] | "
            f"Loss: {batch_loss:8.4f} m | "
            f"Avg: {running_loss / (i+1):8.4f} m"
        )

    if (i + 1) % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_losses, running_loss / len(train_loader)
