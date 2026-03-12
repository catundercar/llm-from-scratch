"""
Shared pytest fixtures for Phase 4 tests.
"""

import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# A minimal GPT-like model for testing (no dependency on Phase 3)
# ---------------------------------------------------------------------------

class MockGPTConfig:
    """Minimal config that mirrors GPTConfig from Phase 3."""
    def __init__(
        self,
        vocab_size: int = 64,
        n_layer: int = 1,
        n_head: int = 2,
        n_embd: int = 32,
        block_size: int = 16,
        dropout: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout


class MockGPT(nn.Module):
    """A simplified GPT-like model for testing the training loop.

    This mock avoids depending on Phase 3 code so that Phase 4 tests
    can run independently. It has the same forward signature:
        forward(input_ids, targets=None) -> (logits, loss | None)
    """

    def __init__(self, config: MockGPTConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.ln = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)          # (B, T, C)
        x = self.ln(x)
        logits = self.head(x)                  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
            )
        return logits, loss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """A small MockGPTConfig for fast tests."""
    return MockGPTConfig(vocab_size=64, n_layer=1, n_head=2, n_embd=32, block_size=16)


@pytest.fixture
def mock_model(mock_config):
    """A small MockGPT model for fast tests."""
    return MockGPT(mock_config)


@pytest.fixture
def synthetic_dataloader(mock_config):
    """DataLoader that yields (input_ids, targets) of random token IDs."""
    seq_len = mock_config.block_size
    vocab = mock_config.vocab_size
    n_samples = 32

    data = torch.randint(0, vocab, (n_samples, seq_len + 1))
    input_ids = data[:, :-1]
    targets = data[:, 1:]

    dataset = TensorDataset(input_ids, targets)
    return DataLoader(dataset, batch_size=4, shuffle=True)
