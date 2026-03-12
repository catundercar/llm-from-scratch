"""
Phase 4 - Pretraining Utilities
================================

Utility functions for the GPT pretraining loop:
- Cosine learning rate schedule with linear warmup
- Loss estimation over multiple batches
- Checkpoint saving and loading

These utilities are used by train.py to manage the training process.
"""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Compute the learning rate for a given training step using a cosine
    schedule with linear warmup.

    The schedule has two phases:
    1. **Linear warmup** (steps 0 to warmup_steps): LR increases linearly
       from 0 to max_lr.
    2. **Cosine decay** (steps warmup_steps to max_steps): LR decreases
       following a cosine curve from max_lr down to min_lr.

    For steps beyond max_steps, the learning rate should be min_lr.

    TODO: Implement this function

    Requirements:
    1. During warmup (step < warmup_steps): lr = max_lr * (step / warmup_steps)
    2. During cosine decay (warmup_steps <= step <= max_steps):
       lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
       where progress = (step - warmup_steps) / (max_steps - warmup_steps)
    3. After max_steps: lr = min_lr
    4. Handle edge case: if warmup_steps == 0, skip warmup phase

    HINT: Use math.cos and math.pi for the cosine calculation.

    HINT: The 'progress' variable represents how far through the decay phase
    we are, ranging from 0.0 (just after warmup) to 1.0 (at max_steps).

    Args:
        step: Current training step (0-indexed).
        warmup_steps: Number of warmup steps.
        max_steps: Total number of training steps.
        max_lr: Peak learning rate (reached at end of warmup).
        min_lr: Minimum learning rate (reached at end of training).

    Returns:
        The learning rate for the given step.
    """
    # TODO: Implement cosine learning rate schedule with linear warmup
    # Step 1: Handle the case where step >= max_steps (return min_lr)
    # Step 2: Handle the warmup phase (step < warmup_steps)
    # Step 3: Handle the cosine decay phase
    #   - Compute progress as fraction through decay phase
    #   - Apply cosine formula: min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    raise NotImplementedError("TODO: Implement get_lr")


# ---------------------------------------------------------------------------
# Loss Estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    eval_steps: int,
    device: str = "cpu",
) -> float:
    """Estimate the average loss over a fixed number of evaluation batches.

    This function sets the model to eval mode, computes loss on `eval_steps`
    batches from the dataloader, then restores the model to training mode.

    TODO: Implement this function

    Requirements:
    1. Set model to eval mode before evaluation
    2. Iterate over at most `eval_steps` batches from the dataloader
    3. For each batch, move input_ids and targets to the correct device
    4. Call model(input_ids, targets) which returns (logits, loss)
    5. Accumulate losses and compute the mean
    6. Restore model to training mode before returning
    7. Handle case where dataloader has fewer batches than eval_steps

    HINT: The dataloader yields (input_ids, targets) tuples where both are
    tensors of shape (batch_size, seq_len). The model's forward method
    returns (logits, loss) when targets are provided.

    HINT: Use itertools.islice or a manual counter to limit the number of
    batches evaluated.

    Args:
        model: The GPT model to evaluate.
        dataloader: DataLoader providing (input_ids, targets) batches.
        eval_steps: Maximum number of batches to evaluate.
        device: Device to run evaluation on.

    Returns:
        The average loss as a Python float.
    """
    # TODO: Implement loss estimation
    # Step 1: Set model to eval mode
    # Step 2: Accumulate losses over eval_steps batches
    # Step 3: Compute mean loss
    # Step 4: Restore model to train mode
    # Step 5: Return mean loss as float
    raise NotImplementedError("TODO: Implement estimate_loss")


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    path: str | Path,
) -> None:
    """Save a training checkpoint to disk.

    The checkpoint dictionary should contain:
    - 'model_state_dict': model.state_dict()
    - 'optimizer_state_dict': optimizer.state_dict()
    - 'step': current training step
    - 'loss': current loss value

    TODO: Implement this function

    Requirements:
    1. Create parent directories if they don't exist
    2. Build a checkpoint dict with the four keys listed above
    3. Save using torch.save()

    HINT: Use Path(path).parent.mkdir(parents=True, exist_ok=True) to ensure
    the directory exists.

    Args:
        model: The model whose weights to save.
        optimizer: The optimizer whose state to save.
        step: Current training step number.
        loss: Current loss value.
        path: File path to save the checkpoint.
    """
    # TODO: Implement checkpoint saving
    # Step 1: Ensure parent directory exists
    # Step 2: Build checkpoint dict with model_state_dict, optimizer_state_dict, step, loss
    # Step 3: Save with torch.save()
    raise NotImplementedError("TODO: Implement save_checkpoint")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load a training checkpoint from disk.

    Restores model weights and optionally optimizer state. Returns the
    full checkpoint dict so the caller can read 'step' and 'loss'.

    TODO: Implement this function

    Requirements:
    1. Load the checkpoint using torch.load() with weights_only=True
    2. Load model state dict from checkpoint
    3. If optimizer is provided, load optimizer state dict
    4. Return the full checkpoint dictionary

    HINT: Use torch.load(path, map_location="cpu", weights_only=True) for
    safe loading that works across devices.

    Args:
        path: File path to load the checkpoint from.
        model: The model to load weights into.
        optimizer: Optional optimizer to restore state into.

    Returns:
        The checkpoint dictionary containing 'step', 'loss', etc.
    """
    # TODO: Implement checkpoint loading
    # Step 1: Load checkpoint dict from disk
    # Step 2: Load model state dict
    # Step 3: Optionally load optimizer state dict
    # Step 4: Return checkpoint dict
    raise NotImplementedError("TODO: Implement load_checkpoint")
