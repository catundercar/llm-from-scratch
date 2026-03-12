"""
Phase 5 · Lab 1: Text Generation Strategies
=============================================

Tests for:
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Unified generate() interface

Difficulty: *** (Intermediate)
"""

import pytest
import torch
import torch.nn.functional as F

from phase5_generation.phase_5.generate import (
    generate,
    greedy_decode,
    temperature_sample,
    top_k_sample,
    top_p_sample,
)


# ===================================================================
# Tests: Greedy Decoding
# ===================================================================


class TestGreedyDecode:
    """Tests for greedy (argmax) decoding."""

    def test_output_shape(self, mock_model, sample_input):
        """Output should have shape (1, T + max_new_tokens)."""
        max_new = 10
        result = greedy_decode(mock_model, sample_input, max_new_tokens=max_new)
        assert result.shape == (1, sample_input.shape[1] + max_new)

    def test_preserves_input_prefix(self, mock_model, sample_input):
        """The original input tokens should be preserved at the start."""
        result = greedy_decode(mock_model, sample_input, max_new_tokens=5)
        assert torch.equal(result[:, :sample_input.shape[1]], sample_input)

    def test_deterministic(self, mock_model, sample_input):
        """Greedy decoding should produce the same output every time."""
        r1 = greedy_decode(mock_model, sample_input, max_new_tokens=10)
        r2 = greedy_decode(mock_model, sample_input, max_new_tokens=10)
        assert torch.equal(r1, r2)

    def test_output_contains_valid_token_ids(self, mock_model, sample_input):
        """All generated token IDs should be in [0, vocab_size)."""
        result = greedy_decode(mock_model, sample_input, max_new_tokens=10)
        assert (result >= 0).all()
        assert (result < mock_model.vocab_size).all()

    def test_zero_new_tokens(self, mock_model, sample_input):
        """With max_new_tokens=0, output should equal input."""
        result = greedy_decode(mock_model, sample_input, max_new_tokens=0)
        assert torch.equal(result, sample_input)


# ===================================================================
# Tests: Temperature Sampling
# ===================================================================


class TestTemperatureSample:
    """Tests for temperature-scaled sampling."""

    def test_output_shape(self, mock_model, sample_input):
        """Output should have the correct shape."""
        result = temperature_sample(mock_model, sample_input, max_new_tokens=10,
                                    temperature=1.0)
        assert result.shape == (1, sample_input.shape[1] + 10)

    def test_temperature_zero_equals_greedy(self, mock_model, sample_input):
        """Temperature ~0 should behave like greedy decoding."""
        greedy_result = greedy_decode(mock_model, sample_input, max_new_tokens=10)
        temp_result = temperature_sample(mock_model, sample_input,
                                         max_new_tokens=10, temperature=1e-10)
        assert torch.equal(greedy_result, temp_result)

    def test_preserves_input_prefix(self, mock_model, sample_input):
        """Original tokens should be preserved."""
        result = temperature_sample(mock_model, sample_input,
                                    max_new_tokens=5, temperature=1.0)
        assert torch.equal(result[:, :sample_input.shape[1]], sample_input)

    def test_valid_token_ids(self, mock_model, sample_input):
        """Generated tokens should be valid IDs."""
        result = temperature_sample(mock_model, sample_input,
                                    max_new_tokens=10, temperature=0.5)
        assert (result >= 0).all()
        assert (result < mock_model.vocab_size).all()

    def test_high_temperature_adds_diversity(self, mock_model, sample_input):
        """With high temperature, multiple runs should sometimes differ."""
        results = set()
        for _ in range(20):
            r = temperature_sample(mock_model, sample_input,
                                   max_new_tokens=10, temperature=2.0)
            results.add(tuple(r[0].tolist()))
        # With temperature=2.0 over 20 runs, we expect some variation
        assert len(results) > 1, "High temperature should produce diverse outputs"


# ===================================================================
# Tests: Top-k Sampling
# ===================================================================


class TestTopKSample:
    """Tests for top-k sampling."""

    def test_output_shape(self, mock_model, sample_input):
        """Output should have the correct shape."""
        result = top_k_sample(mock_model, sample_input, max_new_tokens=10, k=5)
        assert result.shape == (1, sample_input.shape[1] + 10)

    def test_valid_token_ids(self, mock_model, sample_input):
        """All generated tokens should be valid."""
        result = top_k_sample(mock_model, sample_input, max_new_tokens=10, k=5)
        assert (result >= 0).all()
        assert (result < mock_model.vocab_size).all()

    def test_k_equals_1_is_greedy(self, mock_model, sample_input):
        """Top-k with k=1 should produce the same result as greedy."""
        greedy_result = greedy_decode(mock_model, sample_input, max_new_tokens=10)
        topk_result = top_k_sample(mock_model, sample_input,
                                   max_new_tokens=10, k=1, temperature=1.0)
        assert torch.equal(greedy_result, topk_result)

    def test_preserves_input_prefix(self, mock_model, sample_input):
        """Original tokens should be preserved."""
        result = top_k_sample(mock_model, sample_input, max_new_tokens=5, k=10)
        assert torch.equal(result[:, :sample_input.shape[1]], sample_input)

    def test_k_larger_than_vocab(self, mock_model, sample_input):
        """k larger than vocab_size should not crash (clamp to vocab_size)."""
        result = top_k_sample(mock_model, sample_input, max_new_tokens=5,
                              k=1000, temperature=1.0)
        assert result.shape == (1, sample_input.shape[1] + 5)


# ===================================================================
# Tests: Top-p (Nucleus) Sampling
# ===================================================================


class TestTopPSample:
    """Tests for nucleus (top-p) sampling."""

    def test_output_shape(self, mock_model, sample_input):
        """Output should have the correct shape."""
        result = top_p_sample(mock_model, sample_input, max_new_tokens=10, p=0.9)
        assert result.shape == (1, sample_input.shape[1] + 10)

    def test_valid_token_ids(self, mock_model, sample_input):
        """All generated tokens should be valid."""
        result = top_p_sample(mock_model, sample_input, max_new_tokens=10, p=0.9)
        assert (result >= 0).all()
        assert (result < mock_model.vocab_size).all()

    def test_preserves_input_prefix(self, mock_model, sample_input):
        """Original tokens should be preserved."""
        result = top_p_sample(mock_model, sample_input, max_new_tokens=5, p=0.9)
        assert torch.equal(result[:, :sample_input.shape[1]], sample_input)

    def test_p_equals_1_includes_all_tokens(self, mock_model, sample_input):
        """With p=1.0, all tokens should be candidates (no filtering)."""
        # Should not crash and should produce valid output
        result = top_p_sample(mock_model, sample_input, max_new_tokens=5,
                              p=1.0, temperature=1.0)
        assert result.shape == (1, sample_input.shape[1] + 5)

    def test_very_small_p_is_selective(self, mock_model, sample_input):
        """With very small p, sampling should be very selective (near greedy)."""
        greedy_result = greedy_decode(mock_model, sample_input, max_new_tokens=10)

        # Run multiple times -- with p=0.01 most runs should match greedy
        matches = 0
        for _ in range(10):
            result = top_p_sample(mock_model, sample_input, max_new_tokens=10,
                                  p=0.01, temperature=1.0)
            if torch.equal(result, greedy_result):
                matches += 1

        assert matches >= 5, "Very small p should produce near-greedy results"


# ===================================================================
# Tests: Unified generate() Interface
# ===================================================================


class TestGenerate:
    """Tests for the unified generate() function."""

    def test_returns_string(self, mock_model, mock_tokenizer):
        """generate() should return a string."""
        result = generate(mock_model, mock_tokenizer, "hello",
                          max_new_tokens=5, strategy="greedy")
        assert isinstance(result, str)

    def test_greedy_strategy(self, mock_model, mock_tokenizer):
        """generate() with strategy='greedy' should work."""
        result = generate(mock_model, mock_tokenizer, "hi",
                          max_new_tokens=5, strategy="greedy")
        assert len(result) > 0

    def test_temperature_strategy(self, mock_model, mock_tokenizer):
        """generate() with strategy='temperature' should work."""
        result = generate(mock_model, mock_tokenizer, "hi",
                          max_new_tokens=5, strategy="temperature",
                          temperature=0.8)
        assert len(result) > 0

    def test_top_k_strategy(self, mock_model, mock_tokenizer):
        """generate() with strategy='top_k' should work."""
        result = generate(mock_model, mock_tokenizer, "hi",
                          max_new_tokens=5, strategy="top_k", k=10)
        assert len(result) > 0

    def test_top_p_strategy(self, mock_model, mock_tokenizer):
        """generate() with strategy='top_p' should work."""
        result = generate(mock_model, mock_tokenizer, "hi",
                          max_new_tokens=5, strategy="top_p", p=0.9)
        assert len(result) > 0

    def test_invalid_strategy_raises(self, mock_model, mock_tokenizer):
        """Unknown strategy should raise an error."""
        with pytest.raises((ValueError, KeyError, NotImplementedError)):
            generate(mock_model, mock_tokenizer, "hi",
                     max_new_tokens=5, strategy="beam_search")
