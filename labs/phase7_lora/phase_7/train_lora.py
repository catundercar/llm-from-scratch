"""
Phase 7, Lab 2: LoRA Training Pipeline

This module implements the training loop for LoRA fine-tuning.
It demonstrates how to:
    - Apply LoRA to a pretrained model
    - Train only the low-rank parameters
    - Merge weights after training for efficient inference
    - Compare parameter counts before and after LoRA

Students implement:
    - train_with_lora: The training loop that only updates LoRA parameters
    - compare_param_counts: Utility to show parameter efficiency
    - main: End-to-end pipeline
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

from phase_7.types import LoRATrainConfig, GPTConfig
from phase_7.lora import (
    apply_lora_to_model,
    merge_lora_weights,
    count_trainable_params,
    LoRALinear,
)
from phase_7.sample_model import SampleGPT


def create_dummy_data(
    n_samples: int = 100,
    seq_length: int = 32,
    vocab_size: int = 50257,
) -> tuple[DataLoader, DataLoader]:
    """
    Create dummy training and validation data loaders.

    This is PROVIDED for testing purposes. In a real scenario, you would
    load actual text data and tokenize it.
    """
    input_ids = torch.randint(0, vocab_size, (n_samples, seq_length))
    labels = torch.randint(0, vocab_size, (n_samples, seq_length))

    dataset = TensorDataset(input_ids, labels)
    train_size = int(0.8 * n_samples)
    val_size = n_samples - train_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    return train_loader, val_loader


def train_with_lora(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: LoRATrainConfig,
) -> list[float]:
    """
    Train only LoRA parameters of a model.

    TODO: Implement this function

    Requirements:
    1. Verify that only LoRA parameters (lora_A, lora_B) have requires_grad=True.
       Print a warning if any non-LoRA parameter is trainable.
    2. Create an optimizer with ONLY the trainable (LoRA) parameters.
    3. Run the training loop for config.epochs epochs:
       a. For each batch, forward pass through the model to get (logits, loss).
       b. Backward pass and optimizer step.
       c. Every config.log_interval steps, print the training loss.
    4. After each epoch, compute and print the validation loss.
    5. Return a list of average training losses per epoch.

    Args:
        model: Model with LoRA layers already applied.
        train_loader: DataLoader yielding (input_ids, labels) batches.
        val_loader: DataLoader for validation.
        config: Training hyperparameters.

    Returns:
        List of average training losses, one per epoch.

    HINT: Filter parameters with: [p for p in model.parameters() if p.requires_grad]

    HINT: The model's forward() returns (logits, loss) when targets are provided.
          Call it as: logits, loss = model(input_ids, targets=labels)

    HINT: Use torch.no_grad() for the validation loop.
    """
    # TODO: Implement
    # Step 1: Verify only LoRA params are trainable
    # Step 2: Create optimizer (e.g., AdamW) with trainable params only
    # Step 3: Training loop
    #   for epoch in range(config.epochs):
    #       for batch_idx, (input_ids, labels) in enumerate(train_loader):
    #           optimizer.zero_grad()
    #           logits, loss = model(input_ids, targets=labels)
    #           loss.backward()
    #           optimizer.step()
    # Step 4: Validation after each epoch
    # Step 5: Return list of epoch losses
    raise NotImplementedError("TODO: Implement train_with_lora")


def compare_param_counts(
    original_model: nn.Module, lora_model: nn.Module
) -> dict[str, int]:
    """
    Print a comparison table of trainable parameters.

    TODO: Implement this function

    Requirements:
    1. Count total and trainable params in original_model.
    2. Count total and trainable params in lora_model.
    3. Print a formatted comparison table showing:
       - Original: total params, trainable params
       - LoRA: total params, trainable params (lora_A + lora_B only)
       - Reduction: percentage of params that are now trainable
    4. Return a dict with keys: "original_total", "original_trainable",
       "lora_total", "lora_trainable".

    HINT: Use count_trainable_params() from lora.py for the counts.
    """
    # TODO: Implement
    # Step 1: Count params for both models
    # Step 2: Print comparison table
    # Step 3: Return dict
    raise NotImplementedError("TODO: Implement compare_param_counts")


def main() -> None:
    """
    End-to-end LoRA fine-tuning pipeline.

    TODO: Implement this function

    Requirements:
    1. Create a SampleGPT model with a small GPTConfig.
    2. Print original parameter counts.
    3. Apply LoRA to the model (target q_proj and v_proj, rank=4).
    4. Print LoRA parameter counts and compare.
    5. Create dummy data loaders.
    6. Train the LoRA model for a few epochs.
    7. Merge LoRA weights back into the base model.
    8. Print final parameter counts (should be same as original).
    9. Optionally save the merged model.

    HINT: Use the helper functions defined in this file and lora.py.
    """
    # TODO: Implement
    # Step 1: config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=1000)
    # Step 2: model = SampleGPT(config)
    # Step 3: compare_param_counts(model_copy, model)  -- before/after
    # Step 4: apply_lora_to_model(model, rank=4, alpha=1.0)
    # Step 5: train_with_lora(model, train_loader, val_loader, train_config)
    # Step 6: merge_lora_weights(model)
    # Step 7: Save or evaluate
    raise NotImplementedError("TODO: Implement main")


if __name__ == "__main__":
    main()
