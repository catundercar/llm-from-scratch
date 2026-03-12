"""Tests for Phase 1, Lab 2: Sliding Window DataLoader."""

import pytest
import torch
from phase_1.tokenizer import BPETokenizer
from phase_1.dataloader import TextDataset, create_dataloaders


SAMPLE_TEXT = "the cat sat on the mat " * 20  # repeat for enough data


@pytest.fixture
def tokenizer():
    tok = BPETokenizer()
    tok.train(SAMPLE_TEXT, vocab_size=20)
    return tok


class TestTextDataset:
    def test_dataset_creation(self, tokenizer):
        ds = TextDataset(SAMPLE_TEXT, tokenizer, block_size=8)
        assert len(ds) > 0

    def test_xy_shapes(self, tokenizer):
        ds = TextDataset(SAMPLE_TEXT, tokenizer, block_size=8)
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)

    def test_xy_offset(self, tokenizer):
        ds = TextDataset(SAMPLE_TEXT, tokenizer, block_size=8)
        x, y = ds[0]
        # y should be x shifted right by 1
        x2, y2 = ds[0]
        # The last element of x should equal the second-to-last of y's context
        assert torch.equal(x[1:], y[:-1])

    def test_dtype(self, tokenizer):
        ds = TextDataset(SAMPLE_TEXT, tokenizer, block_size=4)
        x, y = ds[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_length_formula(self, tokenizer):
        ds = TextDataset(SAMPLE_TEXT, tokenizer, block_size=4)
        token_count = len(tokenizer.encode(SAMPLE_TEXT))
        assert len(ds) == token_count - 4


class TestCreateDataloaders:
    def test_returns_two_loaders(self, tokenizer):
        train_dl, val_dl = create_dataloaders(
            SAMPLE_TEXT, tokenizer, block_size=4, batch_size=2
        )
        assert train_dl is not None
        assert val_dl is not None

    def test_batch_shape(self, tokenizer):
        train_dl, _ = create_dataloaders(
            SAMPLE_TEXT, tokenizer, block_size=4, batch_size=2
        )
        x, y = next(iter(train_dl))
        assert x.shape == (2, 4)
        assert y.shape == (2, 4)

    def test_train_val_split(self, tokenizer):
        train_dl, val_dl = create_dataloaders(
            SAMPLE_TEXT, tokenizer, block_size=4, batch_size=2, train_split=0.8
        )
        # Train should have more batches than val
        train_batches = sum(1 for _ in train_dl)
        val_batches = sum(1 for _ in val_dl)
        assert train_batches > val_batches
