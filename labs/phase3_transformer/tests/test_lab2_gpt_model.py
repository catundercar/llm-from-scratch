"""Tests for Phase 3, Lab 2: GPT Model."""

import pytest
import torch
from phase_3.gpt_model import GPT, GPTConfig


@pytest.fixture
def small_config():
    return GPTConfig(
        vocab_size=100,
        block_size=16,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )


@pytest.fixture
def model(small_config):
    return GPT(small_config)


class TestGPTConfig:
    def test_defaults(self):
        config = GPTConfig()
        assert config.vocab_size == 50257
        assert config.d_model == 768
        assert config.n_heads == 12
        assert config.n_layers == 12


class TestGPT:
    def test_output_shape(self, model, small_config):
        idx = torch.randint(0, small_config.vocab_size, (2, 8))
        logits, loss = model(idx)
        assert logits.shape == (2, 8, small_config.vocab_size)
        assert loss is None

    def test_loss_computation(self, model, small_config):
        idx = torch.randint(0, small_config.vocab_size, (2, 8))
        targets = torch.randint(0, small_config.vocab_size, (2, 8))
        logits, loss = model(idx, targets)
        assert loss is not None
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # cross-entropy is always positive

    def test_single_token(self, model, small_config):
        idx = torch.randint(0, small_config.vocab_size, (1, 1))
        logits, _ = model(idx)
        assert logits.shape == (1, 1, small_config.vocab_size)

    def test_max_seq_length(self, model, small_config):
        idx = torch.randint(0, small_config.vocab_size, (1, small_config.block_size))
        logits, _ = model(idx)
        assert logits.shape == (1, small_config.block_size, small_config.vocab_size)

    def test_count_parameters(self, model):
        count = model.count_parameters()
        assert isinstance(count, int)
        assert count > 0

    def test_gradient_flow(self, model, small_config):
        idx = torch.randint(0, small_config.vocab_size, (1, 4))
        targets = torch.randint(0, small_config.vocab_size, (1, 4))
        _, loss = model(idx, targets)
        loss.backward()
        # At least some parameters should have gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
