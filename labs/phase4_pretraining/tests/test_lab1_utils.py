"""
Phase 4 · Lab 1: Training Utilities
=====================================

Tests for:
- Cosine learning rate schedule with linear warmup
- Loss estimation
- Checkpoint save/load roundtrip

Difficulty: *** (Intermediate)
"""

import math
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from phase4_pretraining.phase_4.utils import (
    estimate_loss,
    get_lr,
    load_checkpoint,
    save_checkpoint,
)


# ===================================================================
# Tests: get_lr — Cosine Learning Rate Schedule
# ===================================================================


class TestGetLR:
    """Tests for the cosine learning rate schedule with linear warmup."""

    def test_warmup_starts_at_zero(self):
        """At step 0, learning rate should be 0 (or very close to 0)."""
        lr = get_lr(step=0, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-5)
        assert lr == pytest.approx(0.0, abs=1e-9)

    def test_warmup_increases_linearly(self):
        """During warmup, LR should increase linearly."""
        warmup_steps = 100
        max_lr = 1e-3

        lrs = [
            get_lr(step=s, warmup_steps=warmup_steps, max_steps=1000,
                   max_lr=max_lr, min_lr=1e-5)
            for s in range(warmup_steps)
        ]

        # Each step should be larger than the previous
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], f"LR should increase during warmup: step {i}"

        # Halfway through warmup should be ~half the max LR
        mid = lrs[warmup_steps // 2]
        assert mid == pytest.approx(max_lr * 0.5, rel=0.01)

    def test_warmup_reaches_max_lr(self):
        """At the end of warmup, LR should equal max_lr."""
        lr = get_lr(step=100, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-5)
        # At step=warmup_steps, we enter cosine decay; the value should be max_lr
        # (cosine at progress=0 gives max_lr)
        assert lr == pytest.approx(1e-3, rel=0.01)

    def test_cosine_decay_decreases(self):
        """After warmup, LR should generally decrease."""
        warmup = 100
        max_steps = 1000
        max_lr = 1e-3
        min_lr = 1e-5

        lrs = [
            get_lr(step=s, warmup_steps=warmup, max_steps=max_steps,
                   max_lr=max_lr, min_lr=min_lr)
            for s in range(warmup, max_steps + 1)
        ]

        # Overall should decrease (first > last)
        assert lrs[0] > lrs[-1]

        # Final value should be close to min_lr
        assert lrs[-1] == pytest.approx(min_lr, rel=0.01)

    def test_cosine_midpoint(self):
        """At the midpoint of cosine decay, LR should be mean of max and min."""
        warmup = 0
        max_steps = 100
        max_lr = 1.0
        min_lr = 0.0

        mid_step = max_steps // 2
        lr = get_lr(step=mid_step, warmup_steps=warmup, max_steps=max_steps,
                    max_lr=max_lr, min_lr=min_lr)
        expected = 0.5  # mean of max_lr and min_lr at cosine midpoint
        assert lr == pytest.approx(expected, rel=0.02)

    def test_beyond_max_steps_returns_min_lr(self):
        """After max_steps, LR should be min_lr."""
        min_lr = 1e-5
        lr = get_lr(step=2000, warmup_steps=100, max_steps=1000,
                    max_lr=1e-3, min_lr=min_lr)
        assert lr == pytest.approx(min_lr)

    def test_no_warmup(self):
        """With warmup_steps=0, should start directly at max_lr in cosine phase."""
        max_lr = 1e-3
        lr = get_lr(step=0, warmup_steps=0, max_steps=1000,
                    max_lr=max_lr, min_lr=1e-5)
        assert lr == pytest.approx(max_lr, rel=0.01)

    def test_lr_is_always_non_negative(self):
        """LR should never be negative."""
        for step in range(0, 1200, 50):
            lr = get_lr(step=step, warmup_steps=100, max_steps=1000,
                        max_lr=1e-3, min_lr=1e-5)
            assert lr >= 0, f"Negative LR at step {step}: {lr}"


# ===================================================================
# Tests: estimate_loss
# ===================================================================


class TestEstimateLoss:
    """Tests for the loss estimation utility."""

    def test_returns_float(self, mock_model, synthetic_dataloader):
        """estimate_loss should return a Python float."""
        loss = estimate_loss(mock_model, synthetic_dataloader, eval_steps=3)
        assert isinstance(loss, float)

    def test_positive_loss(self, mock_model, synthetic_dataloader):
        """Cross-entropy loss on random data should be positive."""
        loss = estimate_loss(mock_model, synthetic_dataloader, eval_steps=3)
        assert loss > 0

    def test_respects_eval_steps_limit(self, mock_model, synthetic_dataloader):
        """Should evaluate at most eval_steps batches, not the whole loader."""
        # With 32 samples and batch_size=4 there are 8 batches.
        # Asking for 2 eval_steps should only process 2 batches.
        loss = estimate_loss(mock_model, synthetic_dataloader, eval_steps=2)
        assert isinstance(loss, float)

    def test_model_stays_in_train_mode(self, mock_model, synthetic_dataloader):
        """After estimation, the model should be back in training mode."""
        mock_model.train()
        estimate_loss(mock_model, synthetic_dataloader, eval_steps=2)
        assert mock_model.training, "Model should be restored to train mode"

    def test_no_gradients_computed(self, mock_model, synthetic_dataloader):
        """Evaluation should not accumulate gradients."""
        estimate_loss(mock_model, synthetic_dataloader, eval_steps=2)
        for p in mock_model.parameters():
            assert p.grad is None or (p.grad == 0).all(), \
                "No gradients should be accumulated during eval"


# ===================================================================
# Tests: Checkpoint Save/Load
# ===================================================================


class TestCheckpoints:
    """Tests for checkpoint saving and loading."""

    def test_save_creates_file(self, mock_model):
        """save_checkpoint should create a file at the specified path."""
        optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "ckpt.pt")
            save_checkpoint(mock_model, optimizer, step=42, loss=2.5, path=path)
            assert os.path.exists(path)

    def test_save_load_roundtrip_model_weights(self, mock_model):
        """Model weights should be identical after save -> load."""
        optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            save_checkpoint(mock_model, optimizer, step=10, loss=1.5, path=path)

            # Create a fresh model with same architecture
            from conftest import MockGPT, MockGPTConfig
            new_model = MockGPT(MockGPTConfig())
            ckpt = load_checkpoint(path, new_model)

            # Weights should match
            for (n1, p1), (n2, p2) in zip(
                mock_model.named_parameters(), new_model.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    def test_load_returns_step_and_loss(self, mock_model):
        """The returned checkpoint dict should contain 'step' and 'loss'."""
        optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            save_checkpoint(mock_model, optimizer, step=99, loss=0.42, path=path)

            from conftest import MockGPT, MockGPTConfig
            new_model = MockGPT(MockGPTConfig())
            ckpt = load_checkpoint(path, new_model)

            assert ckpt["step"] == 99
            assert ckpt["loss"] == pytest.approx(0.42)

    def test_load_restores_optimizer_state(self, mock_model):
        """When an optimizer is provided, its state should be restored."""
        optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-3)

        # Run a fake step to populate optimizer state
        x = torch.randint(0, 64, (2, 16))
        _, loss = mock_model(x, x)
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            save_checkpoint(mock_model, optimizer, step=1, loss=loss.item(), path=path)

            from conftest import MockGPT, MockGPTConfig
            new_model = MockGPT(MockGPTConfig())
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            ckpt = load_checkpoint(path, new_model, new_optimizer)

            # Optimizer should have state now
            assert len(new_optimizer.state) > 0
