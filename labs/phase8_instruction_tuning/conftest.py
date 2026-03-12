"""Shared pytest fixtures for Phase 8 labs."""

import pytest
import os
from phase_8.sample_tokenizer import SampleTokenizer


SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "sample_data")


@pytest.fixture
def sample_tokenizer() -> SampleTokenizer:
    """A simple character-level tokenizer for testing."""
    return SampleTokenizer(vocab_size=256)


@pytest.fixture
def instructions_path() -> str:
    """Path to the sample instructions JSONL file."""
    return os.path.join(SAMPLE_DATA_DIR, "instructions.jsonl")


@pytest.fixture
def preferences_path() -> str:
    """Path to the sample preferences JSONL file."""
    return os.path.join(SAMPLE_DATA_DIR, "preferences.jsonl")
