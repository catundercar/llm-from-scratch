"""Shared pytest fixtures for Phase 1 labs."""

import pytest
import os

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "sample_data")


@pytest.fixture
def sample_text() -> str:
    """Load the sample text from sample_data/input.txt."""
    path = os.path.join(SAMPLE_DATA_DIR, "input.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def short_text() -> str:
    """A short text for quick tokenizer tests."""
    return "the cat sat on the mat. the cat sat."


@pytest.fixture
def tiny_text() -> str:
    """A tiny text for minimal tests."""
    return "aaabdaaabac"
