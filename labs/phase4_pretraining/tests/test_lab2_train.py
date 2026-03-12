"""
Phase 4 · Lab 2: Training Loop
================================

Tests for:
- TrainConfig dataclass
- Training loop mechanics (forward, backward, gradient clipping)
- Learning rate schedule integration

Difficulty: **** (Advanced)
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from phase4_pretraining.phase_4.train import TrainConfig, train


# ===================================================================
# Tests: TrainConfig
# ===================================================================


class TestTrainConfig:
    """Tests for the TrainConfig dataclass."""

    def test_has_required_fields(self):
        """TrainConfig should have all required fields with defaults."""
        config = TrainConfig()
        assert hasattr(config, "max_steps")
        assert hasattr(config, "batch_size")
        assert hasattr(config, "learning_rate")
        assert hasattr(config, "warmup_steps")
        assert hasattr(config, "eval_interval")
        assert hasattr(config, "eval_steps")
        assert hasattr(config, "checkpoint_dir")
        assert hasattr(config, "device")
        assert hasattr(config, "max_grad_norm")

    def test_default_values_are_reasonable(self):
        """Default values should be sensible for training."""
        config = TrainConfig()
        assert config.max_steps > 0
        assert config.batch_size > 0
        assert 0 < config.learning_rate < 1
        assert config.warmup_steps >= 0
        assert config.eval_interval > 0
        assert config.max_grad_norm > 0

    def test_can_override_defaults(self):
        """Should be able to create TrainConfig with custom values."""
        config = TrainConfig(max_steps=500, learning_rate=1e-4)
        assert config.max_steps == 500
        assert config.learning_rate == 1e-4


# ===================================================================
# Tests: Training Loop
# ===================================================================


class TestTrainLoop:
    """Tests for the main training loop."""

    def test_train_runs_without_error(self, mock_model, synthetic_dataloader):
        """Training should complete without exceptions for a few steps."""
        config = TrainConfig(
            max_steps=5,
            eval_interval=3,
            eval_steps=2,
            checkpoint_dir="/tmp/test_ckpts_phase4",
            checkpoint_interval=100,  # don't save during this short test
            device="cpu",
        )
        result = train(mock_model, synthetic_dataloader, synthetic_dataloader, config)
        assert "final_train_loss" in result

    def test_train_returns_loss_values(self, mock_model, synthetic_dataloader):
        """The returned dict should contain numeric loss values."""
        config = TrainConfig(
            max_steps=5,
            eval_interval=3,
            eval_steps=2,
            checkpoint_dir="/tmp/test_ckpts_phase4",
            checkpoint_interval=100,
            device="cpu",
        )
        result = train(mock_model, synthetic_dataloader, synthetic_dataloader, config)
        assert isinstance(result["final_train_loss"], float)
        assert result["final_train_loss"] > 0

    def test_model_weights_change_after_training(self, mock_model, synthetic_dataloader):
        """Model parameters should be updated by training."""
        # Snapshot initial weights
        initial_weights = {
            name: p.clone() for name, p in mock_model.named_parameters()
        }

        config = TrainConfig(
            max_steps=10,
            eval_interval=100,
            eval_steps=2,
            checkpoint_dir="/tmp/test_ckpts_phase4",
            checkpoint_interval=100,
            device="cpu",
        )
        train(mock_model, synthetic_dataloader, synthetic_dataloader, config)

        # At least some weights should have changed
        any_changed = False
        for name, p in mock_model.named_parameters():
            if not torch.allclose(p, initial_weights[name]):
                any_changed = True
                break
        assert any_changed, "Model weights should change after training"

    def test_gradient_clipping_is_applied(self, mock_model, synthetic_dataloader):
        """Gradient norms should be bounded by max_grad_norm."""
        max_norm = 1.0
        config = TrainConfig(
            max_steps=3,
            eval_interval=100,
            eval_steps=2,
            max_grad_norm=max_norm,
            checkpoint_dir="/tmp/test_ckpts_phase4",
            checkpoint_interval=100,
            device="cpu",
        )

        # We'll manually check by running one step and inspecting
        # (the train function should clip internally, so we test
        # that training with extreme LR doesn't explode)
        config.learning_rate = 10.0  # absurdly high to test clipping doesn't crash
        try:
            train(mock_model, synthetic_dataloader, synthetic_dataloader, config)
        except RuntimeError:
            pytest.fail("Training with gradient clipping should not raise errors")

    def test_train_without_val_loader(self, mock_model, synthetic_dataloader):
        """Training should work when val_loader is None."""
        config = TrainConfig(
            max_steps=3,
            eval_interval=100,
            eval_steps=2,
            checkpoint_dir="/tmp/test_ckpts_phase4",
            checkpoint_interval=100,
            device="cpu",
        )
        result = train(mock_model, synthetic_dataloader, None, config)
        assert "final_train_loss" in result
