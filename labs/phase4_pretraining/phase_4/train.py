"""
Phase 4 - GPT Pretraining Loop
================================

This module implements the main training loop for pretraining a GPT model
on a text dataset. It ties together the model (from Phase 3), the data
pipeline (from Phase 1), and the training utilities (utils.py).

Students implement:
- TrainConfig: a dataclass holding all training hyperparameters
- train(): the main training loop with LR scheduling, evaluation, and checkpointing
- main(): entry point that wires everything together

Dependencies:
- Phase 1: Tokenizer and DataLoader
- Phase 3: GPT model and GPTConfig
- Phase 4: utils.py (get_lr, estimate_loss, save_checkpoint)
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Cross-phase imports (adjust paths as needed for your setup)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase4_pretraining.phase_4.utils import (
    estimate_loss,
    get_lr,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Configuration dataclass for the pretraining loop.

    TODO: Define the following fields with sensible defaults

    Requirements:
    1. max_steps (int): Total number of training steps. Default: 1000
    2. batch_size (int): Batch size for training. Default: 4
    3. learning_rate (float): Peak learning rate. Default: 3e-4
    4. min_lr (float): Minimum learning rate after decay. Default: 3e-5
    5. warmup_steps (int): Number of linear warmup steps. Default: 100
    6. eval_interval (int): Evaluate every N steps. Default: 100
    7. eval_steps (int): Number of batches for each evaluation. Default: 10
    8. checkpoint_dir (str): Directory to save checkpoints. Default: "checkpoints"
    9. checkpoint_interval (int): Save checkpoint every N steps. Default: 500
    10. max_grad_norm (float): Maximum gradient norm for clipping. Default: 1.0
    11. device (str): Device for training. Default: "cpu"

    HINT: Use the @dataclass decorator. Each field should have a type
    annotation and a default value.
    """
    # TODO: Implement TrainConfig fields
    # Example: max_steps: int = 1000
    pass


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: TrainConfig,
) -> dict:
    """Main pretraining loop for the GPT model.

    This function runs the full training procedure:
    1. Set up the AdamW optimizer
    2. For each step, fetch a batch, compute loss, and update weights
    3. Apply learning rate scheduling via get_lr()
    4. Clip gradients to prevent explosion
    5. Periodically evaluate on the validation set
    6. Save checkpoints at regular intervals
    7. Print training progress

    TODO: Implement this function

    Requirements:
    1. Create an AdamW optimizer with config.learning_rate
    2. Move model to config.device
    3. For each training step (0 to config.max_steps - 1):
       a. Get the next batch from train_loader (cycle if exhausted)
       b. Move input_ids and targets to config.device
       c. Forward pass: logits, loss = model(input_ids, targets)
       d. Backward pass: loss.backward()
       e. Clip gradients with torch.nn.utils.clip_grad_norm_(max_norm=config.max_grad_norm)
       f. Optimizer step + zero_grad
       g. Update learning rate using get_lr() for each param group
       h. Every config.eval_interval steps: estimate val loss and print progress
       i. Every config.checkpoint_interval steps: save checkpoint
    4. Return a dict with 'final_train_loss' and 'final_val_loss'

    HINT: To cycle through a DataLoader that runs out of batches, wrap it
    with itertools.cycle() or catch StopIteration and re-create the iterator.

    HINT: Update the learning rate by setting optimizer.param_groups[0]['lr']
    to the value returned by get_lr().

    HINT: The forward pass order matters: zero_grad -> forward -> backward ->
    clip_grad -> step. But it's also valid to do: forward -> backward ->
    clip_grad -> step -> zero_grad(set_to_none=True).

    Args:
        model: The GPT model to train.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation data.
        config: Training configuration.

    Returns:
        A dictionary with 'final_train_loss' and 'final_val_loss' keys.
    """
    # TODO: Implement the training loop
    # Step 1: Move model to device, create optimizer
    # Step 2: Create an iterator from train_loader (use itertools.cycle)
    # Step 3: For each step in range(config.max_steps):
    #   Step 3a: Get learning rate from get_lr() and update optimizer
    #   Step 3b: Get next batch, move to device
    #   Step 3c: Forward pass -> loss
    #   Step 3d: Backward pass
    #   Step 3e: Clip gradients
    #   Step 3f: Optimizer step + zero_grad
    #   Step 3g: Print/log progress at eval_interval
    #   Step 3h: Save checkpoint at checkpoint_interval
    # Step 4: Return final losses
    raise NotImplementedError("TODO: Implement train")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: create model, data, config, and run training.

    TODO: Implement this function

    Requirements:
    1. Create a GPTConfig with small parameters suitable for testing
    2. Create a GPT model from the config
    3. Create or load training and validation DataLoaders
    4. Create a TrainConfig
    5. Call train(model, train_loader, val_loader, config)

    HINT: For quick testing, use small model dimensions:
    GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, block_size=32)

    HINT: You can create a simple synthetic DataLoader for testing:
    random_data = torch.randint(0, vocab_size, (100, block_size + 1))
    Then split each sample into input_ids = data[:, :-1] and targets = data[:, 1:]
    """
    # TODO: Implement main
    # Step 1: Define GPTConfig with small dimensions
    # Step 2: Create GPT model
    # Step 3: Create train/val DataLoaders
    # Step 4: Create TrainConfig
    # Step 5: Call train()
    raise NotImplementedError("TODO: Implement main")


if __name__ == "__main__":
    main()
