"""Shared pytest fixtures for Phase 9 labs."""

import pytest
import torch
from phase_9.types import MoEConfig


@pytest.fixture
def small_moe_config() -> MoEConfig:
    """A small MoE config for fast testing."""
    return MoEConfig(
        vocab_size=100,
        n_layer=4,
        n_head=2,
        n_embd=32,
        block_size=64,
        dropout=0.0,
        bias=False,
        n_experts=4,
        top_k=2,
        moe_every_n_layers=2,
        aux_loss_weight=0.01,
    )


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """A dummy input tensor of token IDs."""
    torch.manual_seed(42)
    return torch.randint(0, 100, (2, 16))


@pytest.fixture
def dummy_hidden() -> torch.Tensor:
    """A dummy hidden state tensor."""
    torch.manual_seed(42)
    return torch.randn(2, 16, 32)
