"""
Lab 2: Sliding-Window Data Loader

Build a PyTorch Dataset and DataLoader for next-token prediction training.
The dataset creates overlapping windows of tokens where the target is the
input shifted right by one position.

Students implement the TODO sections below.
"""

import torch
from torch.utils.data import Dataset, DataLoader

from phase_1.tokenizer import BPETokenizer


class TextDataset(Dataset):
    """
    A PyTorch Dataset that produces (input, target) pairs from tokenized text
    using a sliding window.

    For a sequence of tokens [t0, t1, t2, t3, t4] and block_size=3:
        Index 0: x = [t0, t1, t2],  y = [t1, t2, t3]
        Index 1: x = [t1, t2, t3],  y = [t2, t3, t4]
    """

    # ------------------------------------------------------------------
    # TODO 1: Initialize the dataset
    # ------------------------------------------------------------------
    def __init__(self, text: str, tokenizer: BPETokenizer, block_size: int) -> None:
        """
        Tokenize the text and prepare sliding window data.

        TODO: Implement this method

        Requirements:
        1. Store the block_size as an instance attribute.
        2. Encode the text using the tokenizer.
        3. Convert the encoded token IDs to a PyTorch LongTensor (store as self.data).

        HINT: torch.tensor(token_ids, dtype=torch.long) creates the tensor.

        Args:
            text: Raw text string.
            tokenizer: A trained BPETokenizer instance.
            block_size: Number of tokens in each input window.
        """
        # TODO: Implement
        # Step 1: Store block_size
        # Step 2: Encode text with tokenizer
        # Step 3: Convert to torch.LongTensor and store as self.data
        raise NotImplementedError("TODO: Implement TextDataset.__init__")

    # ------------------------------------------------------------------
    # TODO 2: Return the number of windows
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """
        Return the number of sliding windows in the dataset.

        TODO: Implement this method

        Requirements:
        1. The number of valid windows is len(data) - block_size.
           (We need block_size tokens for x, plus one more for the last target.)

        HINT: If data has N tokens and block_size is B, there are N - B windows.

        Returns:
            Number of (x, y) pairs available.
        """
        # TODO: Implement
        raise NotImplementedError("TODO: Implement TextDataset.__len__")

    # ------------------------------------------------------------------
    # TODO 3: Get a single (input, target) pair
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the (input, target) pair at the given index.

        TODO: Implement this method

        Requirements:
        1. x = data[idx : idx + block_size]
        2. y = data[idx + 1 : idx + block_size + 1]
        3. Both x and y should have shape (block_size,).

        HINT: Standard Python slicing on a tensor works perfectly here.

        Args:
            idx: Index of the window.

        Returns:
            Tuple of (x, y) tensors, each of shape (block_size,).
        """
        # TODO: Implement
        # Step 1: Slice x from data[idx : idx + block_size]
        # Step 2: Slice y from data[idx+1 : idx + block_size + 1]
        # Step 3: Return (x, y)
        raise NotImplementedError("TODO: Implement TextDataset.__getitem__")


def create_dataloaders(
    text: str,
    tokenizer: BPETokenizer,
    block_size: int,
    batch_size: int,
    train_split: float = 0.9,
) -> tuple[DataLoader, DataLoader]:
    """
    Split text into train/val and create PyTorch DataLoaders.

    TODO: Implement this function

    Requirements:
    1. Split the text into training and validation portions at the character level.
       Use int(len(text) * train_split) as the split point.
    2. Create a TextDataset for each portion.
    3. Wrap each dataset in a PyTorch DataLoader.
       - Training loader should shuffle; validation loader should not.
       - Both should drop the last incomplete batch (drop_last=True).
    4. Return (train_loader, val_loader).

    HINT: DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    Args:
        text: Full text corpus.
        tokenizer: A trained BPETokenizer instance.
        block_size: Tokens per window.
        batch_size: Batch size for the DataLoaders.
        train_split: Fraction of text for training (default 0.9).

    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    # TODO: Implement
    # Step 1: Calculate split index
    # Step 2: Split text into train_text and val_text
    # Step 3: Create TextDataset for each
    # Step 4: Wrap in DataLoader (shuffle=True for train, False for val)
    # Step 5: Return (train_loader, val_loader)
    raise NotImplementedError("TODO: Implement create_dataloaders")
