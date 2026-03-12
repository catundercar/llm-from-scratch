"""
Phase 6 - Classification Dataset
==================================

This module handles loading and preparing text data for classification
fine-tuning. It implements a PyTorch Dataset that:
- Reads a CSV file with 'text' and 'label' columns
- Tokenizes each text using the provided tokenizer
- Pads or truncates sequences to a fixed length
- Returns input_ids, attention_mask, and label tensors

It also provides a utility to split data into train/val/test DataLoaders.

Dependencies:
- Phase 1: Tokenizer (for encoding text to token IDs)
"""

import csv
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# SpamDataset
# ---------------------------------------------------------------------------

class SpamDataset(Dataset):
    """A PyTorch Dataset for the spam/ham classification task.

    Reads a CSV file with 'text' and 'label' columns, tokenizes each text,
    and pads/truncates to a fixed max_length.

    TODO: Implement __init__ and __getitem__

    Attributes:
        texts: List of raw text strings.
        labels: List of integer labels (0 = ham, 1 = spam).
        tokenizer: Tokenizer with an encode() method.
        max_length: Maximum sequence length (pad or truncate to this).
        encoded: List of encoded (token ID) sequences.
    """

    def __init__(self, csv_path: str | Path, tokenizer, max_length: int = 128):
        """Load the CSV file and tokenize all texts.

        TODO: Implement this method

        Requirements:
        1. Read the CSV file. It has a header row with columns 'text' and 'label'.
        2. Store the raw texts and integer labels.
        3. Tokenize each text using tokenizer.encode(text).
        4. For each encoded sequence:
           a. Truncate to max_length if longer.
           b. Record the actual length (before padding) for the attention mask.
           c. Pad with 0s to max_length if shorter.
        5. Store the processed sequences for use in __getitem__.

        HINT: Use csv.DictReader to parse the CSV file. Each row will be
        a dict like {'text': '...', 'label': '0'}.

        HINT: The attention mask should be 1 for real tokens and 0 for padding:
        mask = [1] * actual_length + [0] * (max_length - actual_length)

        Args:
            csv_path: Path to the CSV file.
            tokenizer: Tokenizer with an encode(text) -> list[int] method.
            max_length: Maximum sequence length.
        """
        # TODO: Implement dataset initialization
        # Step 1: Read CSV file and extract texts and labels
        # Step 2: Tokenize each text with tokenizer.encode()
        # Step 3: Truncate or pad each sequence to max_length
        # Step 4: Create attention masks (1 for real tokens, 0 for padding)
        # Step 5: Store input_ids, attention_masks, and labels as lists
        raise NotImplementedError("TODO: Implement SpamDataset.__init__")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample as a dictionary.

        TODO: Implement this method

        Requirements:
        1. Return a dict with three keys:
           - 'input_ids': LongTensor of shape (max_length,)
           - 'attention_mask': LongTensor of shape (max_length,)
           - 'label': LongTensor scalar (shape ())
        2. All tensors should be on CPU (default).

        HINT: Use torch.tensor(..., dtype=torch.long) to create each tensor.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dict with 'input_ids', 'attention_mask', and 'label' tensors.
        """
        # TODO: Implement __getitem__
        # Step 1: Get the input_ids, attention_mask, and label at index idx
        # Step 2: Convert each to a torch.long tensor
        # Step 3: Return as a dict
        raise NotImplementedError("TODO: Implement SpamDataset.__getitem__")


# ---------------------------------------------------------------------------
# DataLoader Creation
# ---------------------------------------------------------------------------

def create_classification_dataloaders(
    csv_path: str | Path,
    tokenizer,
    max_length: int = 128,
    batch_size: int = 8,
    train_split: float = 0.7,
    val_split: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders from a CSV file.

    TODO: Implement this function

    Requirements:
    1. Create a SpamDataset from the CSV file.
    2. Split into train/val/test using the given proportions.
       test_split = 1.0 - train_split - val_split
    3. Use torch.utils.data.random_split for the split.
    4. Create a DataLoader for each split.
       - Train loader: shuffle=True
       - Val and test loaders: shuffle=False
    5. Return (train_loader, val_loader, test_loader).

    HINT: random_split takes a dataset and a list of lengths:
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    HINT: Compute split sizes carefully -- they must sum to len(dataset).
    Use int() for train and val sizes, and give the remainder to test.

    Args:
        csv_path: Path to the CSV file.
        tokenizer: Tokenizer with encode() method.
        max_length: Maximum sequence length.
        batch_size: Batch size for all loaders.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # TODO: Implement DataLoader creation
    # Step 1: Create SpamDataset
    # Step 2: Compute split sizes (train, val, test)
    # Step 3: Use random_split to create subsets
    # Step 4: Create DataLoaders for each split
    # Step 5: Return (train_loader, val_loader, test_loader)
    raise NotImplementedError("TODO: Implement create_classification_dataloaders")
