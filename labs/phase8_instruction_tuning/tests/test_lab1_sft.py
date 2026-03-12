"""
Tests for Phase 8, Lab 1: Supervised Fine-Tuning

These tests verify:
    - InstructionDataset formatting and label masking
    - SFT training loop basics

Tests are PROVIDED -- students do not modify this file.
"""

import pytest
import json
import os
import tempfile
import torch
from phase_8.sft import InstructionDataset, format_instruction
from phase_8.sample_tokenizer import SampleTokenizer


@pytest.fixture
def tiny_instructions_file() -> str:
    """Create a tiny instructions JSONL file for testing."""
    samples = [
        {
            "instruction": "Say hello.",
            "input": "",
            "output": "Hello there!",
        },
        {
            "instruction": "Translate to French.",
            "input": "Good morning",
            "output": "Bonjour",
        },
        {
            "instruction": "Add the numbers.",
            "input": "2 + 3",
            "output": "5",
        },
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    yield path
    os.unlink(path)


class TestFormatInstruction:
    """Tests for the format_instruction helper."""

    def test_format_without_input(self) -> None:
        """When input is empty, the Input section should be omitted."""
        prompt, full = format_instruction("Say hello.", "", "Hello!")
        assert "### Instruction:" in prompt
        assert "### Response:" in prompt
        assert "### Input:" not in prompt
        assert "Hello!" in full
        assert "Hello!" not in prompt

    def test_format_with_input(self) -> None:
        """When input is provided, the Input section should be included."""
        prompt, full = format_instruction("Translate.", "Hello", "Bonjour")
        assert "### Input:" in prompt
        assert "Hello" in prompt
        assert "Bonjour" in full
        assert "Bonjour" not in prompt

    def test_full_text_starts_with_prompt(self) -> None:
        """The full text should start with the prompt."""
        prompt, full = format_instruction("Do something.", "", "Done.")
        assert full.startswith(prompt)


class TestInstructionDataset:
    """Tests for the InstructionDataset class."""

    def test_length(
        self, tiny_instructions_file: str, sample_tokenizer: SampleTokenizer
    ) -> None:
        """Dataset should have the correct number of samples."""
        ds = InstructionDataset(tiny_instructions_file, sample_tokenizer, max_length=128)
        assert len(ds) == 3

    def test_returns_correct_keys(
        self, tiny_instructions_file: str, sample_tokenizer: SampleTokenizer
    ) -> None:
        """Each item should have input_ids, attention_mask, and labels."""
        ds = InstructionDataset(tiny_instructions_file, sample_tokenizer, max_length=128)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_output_shapes(
        self, tiny_instructions_file: str, sample_tokenizer: SampleTokenizer
    ) -> None:
        """All tensors should have shape (max_length,)."""
        max_len = 128
        ds = InstructionDataset(tiny_instructions_file, sample_tokenizer, max_length=max_len)
        item = ds[0]
        assert item["input_ids"].shape == (max_len,)
        assert item["attention_mask"].shape == (max_len,)
        assert item["labels"].shape == (max_len,)

    def test_label_masking(
        self, tiny_instructions_file: str, sample_tokenizer: SampleTokenizer
    ) -> None:
        """Instruction/prompt tokens should be masked with -100 in labels."""
        ds = InstructionDataset(tiny_instructions_file, sample_tokenizer, max_length=256)
        item = ds[0]
        labels = item["labels"]

        # There should be some -100 values (masked prompt tokens)
        masked_count = (labels == -100).sum().item()
        assert masked_count > 0, "Some tokens should be masked with -100"

        # There should also be some non-(-100) values (response tokens)
        unmasked_count = (labels != -100).sum().item()
        # At least need padding tokens that could be -100 too, but response tokens should exist
        # Check that not ALL are masked
        total_tokens = labels.shape[0]
        assert masked_count < total_tokens, "Not all tokens should be masked"

    def test_prompt_tokens_are_masked(
        self, tiny_instructions_file: str, sample_tokenizer: SampleTokenizer
    ) -> None:
        """The first tokens (prompt) should all be -100, followed by response tokens."""
        ds = InstructionDataset(tiny_instructions_file, sample_tokenizer, max_length=256)
        item = ds[0]
        labels = item["labels"]

        # Find first non-(-100) label
        non_masked = (labels != -100).nonzero(as_tuple=True)[0]
        if len(non_masked) > 0:
            first_response_idx = non_masked[0].item()
            # All tokens before first_response_idx should be -100
            assert torch.all(labels[:first_response_idx] == -100), (
                "All prompt tokens before the response should be masked with -100"
            )

    def test_loads_from_sample_data(
        self, instructions_path: str, sample_tokenizer: SampleTokenizer
    ) -> None:
        """Should successfully load the provided sample data."""
        ds = InstructionDataset(instructions_path, sample_tokenizer, max_length=512)
        assert len(ds) == 30
