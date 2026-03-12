"""
Tests for Phase 7, Lab 2: LoRA Training Pipeline

These tests verify the training utilities work correctly.

Tests are PROVIDED -- students do not modify this file.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from phase_7.lora import apply_lora_to_model, LoRALinear
from phase_7.train_lora import train_with_lora, compare_param_counts, create_dummy_data
from phase_7.sample_model import SampleGPT
from phase_7.types import GPTConfig, LoRATrainConfig


@pytest.fixture
def tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=50, n_layer=1, n_head=2, n_embd=16, block_size=32, bias=False
    )


@pytest.fixture
def tiny_loaders() -> tuple[DataLoader, DataLoader]:
    """Create tiny data loaders for testing."""
    ids = torch.randint(0, 50, (16, 8))
    labels = torch.randint(0, 50, (16, 8))
    ds = TensorDataset(ids, labels)
    train_ds, val_ds = torch.utils.data.random_split(ds, [12, 4])
    return DataLoader(train_ds, batch_size=4), DataLoader(val_ds, batch_size=4)


class TestTrainWithLoRA:
    """Tests for the train_with_lora function."""

    def test_returns_loss_list(
        self, tiny_config: GPTConfig, tiny_loaders: tuple
    ) -> None:
        """Should return a list of losses, one per epoch."""
        torch.manual_seed(0)
        model = SampleGPT(tiny_config)
        apply_lora_to_model(model, rank=2, alpha=1.0)

        config = LoRATrainConfig(epochs=2, lr=1e-3, log_interval=100)
        train_loader, val_loader = tiny_loaders
        losses = train_with_lora(model, train_loader, val_loader, config)

        assert isinstance(losses, list), "Should return a list"
        assert len(losses) == 2, f"Expected 2 epoch losses, got {len(losses)}"

    def test_loss_decreases(
        self, tiny_config: GPTConfig, tiny_loaders: tuple
    ) -> None:
        """Training loss should decrease over epochs (with enough epochs)."""
        torch.manual_seed(0)
        model = SampleGPT(tiny_config)
        apply_lora_to_model(model, rank=2, alpha=1.0)

        config = LoRATrainConfig(epochs=5, lr=1e-3, log_interval=100)
        train_loader, val_loader = tiny_loaders
        losses = train_with_lora(model, train_loader, val_loader, config)

        assert losses[-1] < losses[0], (
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_only_lora_params_updated(
        self, tiny_config: GPTConfig, tiny_loaders: tuple
    ) -> None:
        """Only LoRA parameters should change during training."""
        torch.manual_seed(0)
        model = SampleGPT(tiny_config)
        apply_lora_to_model(model, rank=2, alpha=1.0)

        # Snapshot frozen params
        frozen_before = {
            name: p.clone()
            for name, p in model.named_parameters()
            if not p.requires_grad
        }

        config = LoRATrainConfig(epochs=1, lr=1e-2, log_interval=100)
        train_loader, val_loader = tiny_loaders
        train_with_lora(model, train_loader, val_loader, config)

        # Verify frozen params unchanged
        for name, p in model.named_parameters():
            if not p.requires_grad:
                assert torch.equal(p, frozen_before[name]), (
                    f"Frozen param {name} was modified during training"
                )


class TestCompareParamCounts:
    """Tests for compare_param_counts."""

    def test_returns_dict_with_expected_keys(self, tiny_config: GPTConfig) -> None:
        """Should return a dict with the four expected keys."""
        original = SampleGPT(tiny_config)
        lora_model = SampleGPT(tiny_config)
        apply_lora_to_model(lora_model, rank=2, alpha=1.0)

        result = compare_param_counts(original, lora_model)

        assert isinstance(result, dict)
        for key in ["original_total", "original_trainable", "lora_total", "lora_trainable"]:
            assert key in result, f"Missing key: {key}"

    def test_lora_trainable_less_than_original(self, tiny_config: GPTConfig) -> None:
        """LoRA trainable params should be much less than original total."""
        original = SampleGPT(tiny_config)
        lora_model = SampleGPT(tiny_config)
        apply_lora_to_model(lora_model, rank=2, alpha=1.0)

        result = compare_param_counts(original, lora_model)
        assert result["lora_trainable"] < result["original_trainable"]
