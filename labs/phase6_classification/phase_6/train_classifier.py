"""
Phase 6 - Classification Training Loop
========================================

This module implements the training loop for fine-tuning a GPT-based
classifier on a downstream task (e.g., spam detection).

Key components:
- train_epoch(): One pass through the training data
- evaluate(): Compute accuracy and loss on an evaluation set
- train_classifier(): Full training loop with early stopping

Dependencies:
- Phase 6: classifier.py (GPTClassifier)
- Phase 6: dataset.py (SpamDataset, DataLoaders)
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Training Epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    """Run one epoch of classification training.

    TODO: Implement this function

    Requirements:
    1. Set model to training mode.
    2. For each batch in the dataloader:
       a. Move input_ids, attention_mask, labels to device.
       b. Forward pass: logits = model(input_ids, attention_mask)
       c. Compute cross-entropy loss: F.cross_entropy(logits, labels)
       d. Backward pass and optimizer step.
       e. Zero gradients.
    3. Track total loss and return average loss over all batches.

    HINT: Each batch from SpamDataset is a dict with keys
    'input_ids', 'attention_mask', 'label'.

    HINT: The standard training step order is:
    optimizer.zero_grad() -> forward -> loss.backward() -> optimizer.step()

    Args:
        model: The GPTClassifier model.
        dataloader: Training DataLoader (yields dicts).
        optimizer: The optimizer.
        device: Device to train on.

    Returns:
        Average training loss for the epoch (float).
    """
    # TODO: Implement one training epoch
    # Step 1: Set model to train mode
    # Step 2: Initialize total_loss and batch_count
    # Step 3: For each batch:
    #   a. Extract input_ids, attention_mask, labels from batch dict
    #   b. Move tensors to device
    #   c. optimizer.zero_grad()
    #   d. Forward pass: logits = model(input_ids, attention_mask)
    #   e. Compute loss: F.cross_entropy(logits, labels)
    #   f. loss.backward()
    #   g. optimizer.step()
    #   h. Accumulate loss
    # Step 4: Return average loss
    raise NotImplementedError("TODO: Implement train_epoch")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> dict:
    """Evaluate the classifier on a dataset.

    TODO: Implement this function

    Requirements:
    1. Set model to eval mode.
    2. For each batch, compute logits and loss.
    3. Compute accuracy: fraction of correct predictions.
       Prediction = argmax of logits along the class dimension.
    4. Return a dict with 'loss' (average) and 'accuracy' (fraction correct).

    HINT: For accuracy, compare torch.argmax(logits, dim=-1) with labels.
    Count correct predictions and divide by total samples.

    Args:
        model: The GPTClassifier model.
        dataloader: Evaluation DataLoader.
        device: Device to run on.

    Returns:
        Dict with 'loss' (float) and 'accuracy' (float, 0-1 range).
    """
    # TODO: Implement evaluation
    # Step 1: Set model to eval mode
    # Step 2: Initialize counters: total_loss, correct, total
    # Step 3: For each batch:
    #   a. Extract and move tensors to device
    #   b. Forward pass -> logits
    #   c. Compute loss
    #   d. Compute predictions = argmax(logits, dim=-1)
    #   e. Count correct predictions: (predictions == labels).sum()
    #   f. Accumulate totals
    # Step 4: Return {'loss': avg_loss, 'accuracy': correct / total}
    raise NotImplementedError("TODO: Implement evaluate")


# ---------------------------------------------------------------------------
# Full Training Loop
# ---------------------------------------------------------------------------

def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cpu",
    patience: int = 3,
) -> dict:
    """Full training loop with early stopping based on validation accuracy.

    TODO: Implement this function

    Requirements:
    1. Create an AdamW optimizer with the given learning rate.
       Only optimize parameters that have requires_grad=True.
    2. Move model to device.
    3. For each epoch:
       a. Run train_epoch() to train for one epoch.
       b. Run evaluate() on the validation set.
       c. Print epoch number, train loss, val loss, val accuracy.
       d. Track the best validation accuracy. If val accuracy improves,
          reset the patience counter. Otherwise, increment it.
       e. If patience counter reaches `patience`, stop early.
    4. After training, if test_loader is provided, evaluate on the test set.
    5. Return a dict with 'best_val_accuracy', 'final_test_accuracy' (or None),
       and 'epochs_trained'.

    HINT: Early stopping logic:
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            break

    HINT: To only optimize unfrozen parameters:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    Args:
        model: The GPTClassifier model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Optional test DataLoader.
        epochs: Maximum number of epochs.
        lr: Learning rate.
        device: Device to train on.
        patience: Number of epochs without improvement before stopping.

    Returns:
        Dict with 'best_val_accuracy', 'final_test_accuracy', 'epochs_trained'.
    """
    # TODO: Implement full training loop with early stopping
    # Step 1: Create optimizer (only for requires_grad=True params)
    # Step 2: Move model to device
    # Step 3: Initialize best_val_acc, no_improve_count, epochs_trained
    # Step 4: For each epoch:
    #   a. train_epoch()
    #   b. evaluate() on val set
    #   c. Print progress
    #   d. Update best_val_acc and check early stopping
    # Step 5: Evaluate on test set if provided
    # Step 6: Return results dict
    raise NotImplementedError("TODO: Implement train_classifier")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load pretrained GPT, create classifier, train on spam dataset.

    TODO: Implement this function

    Requirements:
    1. Create or load a GPT model (from Phase 3/4).
    2. Create a tokenizer (from Phase 1).
    3. Create DataLoaders using create_classification_dataloaders().
    4. Wrap the GPT model in GPTClassifier with freeze_backbone=True.
    5. Call train_classifier() and print results.

    HINT: For testing, use the small MockGPT from conftest.py or a tiny
    GPTConfig. The spam.csv sample data is in sample_data/spam.csv.
    """
    # TODO: Implement main
    raise NotImplementedError("TODO: Implement main")


if __name__ == "__main__":
    main()
