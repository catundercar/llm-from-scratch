"""
Shared pytest fixtures for Phase 5 tests.
"""

import torch
import torch.nn as nn
import pytest


# ---------------------------------------------------------------------------
# Mock model and tokenizer (no dependency on Phase 3)
# ---------------------------------------------------------------------------

class MockGPT(nn.Module):
    """Minimal GPT-like model for testing generation strategies.

    Has the same forward signature: (input_ids, targets=None) -> (logits, loss).
    Uses a simple embedding + linear head so outputs are deterministic.
    """

    def __init__(self, vocab_size: int = 32, n_embd: int = 16, block_size: int = 64):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
            )
        return logits, loss


class MockTokenizer:
    """Minimal tokenizer for testing the generate() interface.

    Maps characters to integer IDs (0-255) using ord/chr.
    """

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 for c in text]  # keep in vocab range

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 65) for i in ids)  # map to A-Z...


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """A small deterministic model for testing generation."""
    torch.manual_seed(42)
    model = MockGPT(vocab_size=32, n_embd=16, block_size=64)
    model.eval()
    return model


@pytest.fixture
def mock_tokenizer():
    """A simple character-level tokenizer mock."""
    return MockTokenizer()


@pytest.fixture
def sample_input():
    """A sample input tensor of shape (1, 5) with token IDs."""
    return torch.tensor([[1, 2, 3, 4, 5]])
