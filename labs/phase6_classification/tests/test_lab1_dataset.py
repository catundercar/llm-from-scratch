"""
Phase 6 · Lab 1: Classification Dataset
=========================================

Tests for:
- SpamDataset loading and tokenization
- Proper padding and truncation
- Attention mask generation
- DataLoader creation and splitting

Difficulty: ** (Basic)
"""

import pytest
import torch

from phase6_classification.phase_6.dataset import (
    SpamDataset,
    create_classification_dataloaders,
)


# ===================================================================
# Tests: SpamDataset
# ===================================================================


class TestSpamDataset:
    """Tests for the SpamDataset class."""

    def test_loads_correct_number_of_samples(self, sample_csv, mock_tokenizer):
        """Dataset length should match number of rows in CSV."""
        dataset = SpamDataset(sample_csv, mock_tokenizer, max_length=32)
        assert len(dataset) == 20

    def test_getitem_returns_dict(self, sample_csv, mock_tokenizer):
        """Each sample should be a dict with the right keys."""
        dataset = SpamDataset(sample_csv, mock_tokenizer, max_length=32)
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "label" in sample

    def test_input_ids_shape(self, sample_csv, mock_tokenizer):
        """input_ids should have shape (max_length,)."""
        max_length = 32
        dataset = SpamDataset(sample_csv, mock_tokenizer, max_length=max_length)
        sample = dataset[0]
        assert sample["input_ids"].shape == (max_length,)

    def test_attention_mask_shape(self, sample_csv, mock_tokenizer):
        """attention_mask should have shape (max_length,)."""
        max_length = 32
        dataset = SpamDataset(sample_csv, mock_tokenizer, max_length=max_length)
        sample = dataset[0]
        assert sample["attention_mask"].shape == (max_length,)

    def test_tensors_are_long_type(self, sample_csv, mock_tokenizer):
        """All tensors should be torch.long (int64)."""
        dataset = SpamDataset(sample_csv, mock_tokenizer, max_length=32)
        sample = dataset[0]
        assert sample["input_ids"].dtype == torch.long
        assert sample["attention_mask"].dtype == torch.long
        assert sample["label"].dtype == torch.long

    def test_label_values(self, sample_csv, mock_tokenizer):
        """Labels should be 0 or 1."""
        dataset = SpamDataset(sample_csv, mock_tokenizer, max_length=32)
        for i in range(len(dataset)):
            label = dataset[i]["label"].item()
            assert label in (0, 1), f"Invalid label at index {i}: {label}"

    def test_attention_mask_has_ones_and_zeros(self, sample_csv, mock_tokenizer):
        """Attention mask should have 1s for real tokens and 0s for padding."""
        dataset = SpamDataset(sample_csv, mock_tokenizer, max_length=64)
        sample = dataset[0]
        mask = sample["attention_mask"]
        # Should have at least some 1s (for real tokens)
        assert mask.sum() > 0
        # All values should be 0 or 1
        assert ((mask == 0) | (mask == 1)).all()

    def test_padding_at_end(self, sample_csv, mock_tokenizer):
        """Padding (0s in mask) should be at the end, not the beginning."""
        dataset = SpamDataset(sample_csv, mock_tokenizer, max_length=64)
        sample = dataset[0]
        mask = sample["attention_mask"]
        n_real = mask.sum().item()
        # First n_real values should be 1, rest should be 0
        assert (mask[:n_real] == 1).all()
        if n_real < 64:
            assert (mask[n_real:] == 0).all()

    def test_truncation(self, sample_csv, mock_tokenizer):
        """Sequences longer than max_length should be truncated."""
        # Use a very short max_length to force truncation
        dataset = SpamDataset(sample_csv, mock_tokenizer, max_length=5)
        sample = dataset[0]
        assert sample["input_ids"].shape == (5,)
        assert sample["attention_mask"].shape == (5,)
        # With truncation, all positions should be real tokens
        assert sample["attention_mask"].sum() == 5


# ===================================================================
# Tests: create_classification_dataloaders
# ===================================================================


class TestCreateDataloaders:
    """Tests for the DataLoader creation utility."""

    def test_returns_three_loaders(self, sample_csv, mock_tokenizer):
        """Should return a tuple of three DataLoaders."""
        train_dl, val_dl, test_dl = create_classification_dataloaders(
            sample_csv, mock_tokenizer, max_length=32, batch_size=4,
        )
        assert isinstance(train_dl, torch.utils.data.DataLoader)
        assert isinstance(val_dl, torch.utils.data.DataLoader)
        assert isinstance(test_dl, torch.utils.data.DataLoader)

    def test_total_samples_preserved(self, sample_csv, mock_tokenizer):
        """Total samples across all splits should equal dataset size."""
        train_dl, val_dl, test_dl = create_classification_dataloaders(
            sample_csv, mock_tokenizer, max_length=32, batch_size=2,
        )
        total = sum(len(dl.dataset) for dl in [train_dl, val_dl, test_dl])
        assert total == 20

    def test_batch_yields_correct_keys(self, sample_csv, mock_tokenizer):
        """Batches should have the expected keys and shapes."""
        train_dl, _, _ = create_classification_dataloaders(
            sample_csv, mock_tokenizer, max_length=32, batch_size=4,
        )
        batch = next(iter(train_dl))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "label" in batch
        assert batch["input_ids"].shape[1] == 32  # max_length
