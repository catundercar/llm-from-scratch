"""Shared pytest fixtures for Phase 7 labs."""

import pytest
import torch
from phase_7.types import GPTConfig
from phase_7.sample_model import SampleGPT


@pytest.fixture
def small_config() -> GPTConfig:
    """A small GPT config for fast testing."""
    return GPTConfig(
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=64,
        dropout=0.0,
        bias=False,
    )


@pytest.fixture
def small_model(small_config: GPTConfig) -> SampleGPT:
    """A small GPT model for testing."""
    torch.manual_seed(42)
    return SampleGPT(small_config)


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """A dummy input tensor of token IDs."""
    torch.manual_seed(42)
    return torch.randint(0, 100, (2, 16))
