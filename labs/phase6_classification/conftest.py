"""
Shared pytest fixtures for Phase 6 tests.
"""

import csv
import os
import tempfile
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Mock GPT model for testing (no Phase 3 dependency)
# ---------------------------------------------------------------------------

@dataclass
class MockGPTConfig:
    vocab_size: int = 64
    n_layer: int = 1
    n_head: int = 2
    n_embd: int = 32
    block_size: int = 64
    dropout: float = 0.0


class MockGPT(nn.Module):
    """Simplified GPT that exposes hidden states via get_hidden_states().

    forward(input_ids, targets=None) -> (logits, loss)
    get_hidden_states(input_ids) -> hidden states of shape (B, T, n_embd)
    """

    def __init__(self, config: MockGPTConfig = None):
        super().__init__()
        if config is None:
            config = MockGPTConfig()
        self.config = config
        self.n_embd = config.n_embd
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.ln = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def get_hidden_states(self, input_ids):
        """Return hidden states before the LM head."""
        x = self.embedding(input_ids)
        x = self.ln(x)
        return x  # (B, T, n_embd)

    def forward(self, input_ids, targets=None):
        x = self.get_hidden_states(input_ids)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(B * T, C), targets.view(B * T)
            )
        return logits, loss


class MockTokenizer:
    """Simple tokenizer that maps characters to integer IDs."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 64 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gpt_config():
    return MockGPTConfig()


@pytest.fixture
def mock_gpt(mock_gpt_config):
    torch.manual_seed(42)
    return MockGPT(mock_gpt_config)


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small CSV file for testing the dataset."""
    csv_path = tmp_path / "test_spam.csv"
    rows = [
        {"text": "Hey how are you doing today?", "label": "0"},
        {"text": "FREE MONEY click here now!!!", "label": "1"},
        {"text": "Can we meet for lunch tomorrow?", "label": "0"},
        {"text": "You won a $1000 gift card!", "label": "1"},
        {"text": "The meeting is at 3pm", "label": "0"},
        {"text": "URGENT: verify your account", "label": "1"},
        {"text": "Thanks for dinner last night", "label": "0"},
        {"text": "Claim your prize now!!!", "label": "1"},
        {"text": "Happy birthday to you!", "label": "0"},
        {"text": "Make money fast online", "label": "1"},
        {"text": "See you at the party", "label": "0"},
        {"text": "Win a free iPhone today", "label": "1"},
        {"text": "The report is attached", "label": "0"},
        {"text": "Investment returns guaranteed", "label": "1"},
        {"text": "Pick up milk please", "label": "0"},
        {"text": "Double your income now", "label": "1"},
        {"text": "Good morning everyone", "label": "0"},
        {"text": "Lose weight in one week", "label": "1"},
        {"text": "Let me know if you need help", "label": "0"},
        {"text": "Free cruise vacation winner", "label": "1"},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


@pytest.fixture
def classification_dataloader():
    """A simple DataLoader that mimics SpamDataset output format."""
    n_samples = 16
    max_length = 32
    n_classes = 2

    input_ids = torch.randint(0, 64, (n_samples, max_length))
    attention_mask = torch.ones(n_samples, max_length, dtype=torch.long)
    labels = torch.randint(0, n_classes, (n_samples,))

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "label": self.labels[idx],
            }

    dataset = DictDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=4, shuffle=True)
