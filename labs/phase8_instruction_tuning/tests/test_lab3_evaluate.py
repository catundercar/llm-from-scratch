"""
Tests for Phase 8, Lab 3: Evaluation Utilities

These tests verify:
    - compute_metrics produces correct overlap metrics
    - llm_as_judge formats evaluation prompts correctly
    - evaluate_model returns the expected structure

Tests are PROVIDED -- students do not modify this file.
"""

import pytest
from phase_8.evaluate import compute_metrics, llm_as_judge


class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_exact_match(self) -> None:
        """Exact matches should yield precision=1, recall=1, f1=1, exact_match=1."""
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        metrics = compute_metrics(preds, refs)
        assert metrics["exact_match"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """No word overlap should give zero precision, recall, f1."""
        preds = ["hello world"]
        refs = ["foo bar baz"]
        metrics = compute_metrics(preds, refs)
        assert metrics["precision"] == pytest.approx(0.0)
        assert metrics["recall"] == pytest.approx(0.0)
        assert metrics["f1"] == pytest.approx(0.0)
        assert metrics["exact_match"] == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Partial overlap should produce values between 0 and 1."""
        preds = ["the cat sat"]
        refs = ["the cat ran away"]
        metrics = compute_metrics(preds, refs)
        # "the" and "cat" overlap -> precision=2/3, recall=2/4
        assert 0 < metrics["precision"] < 1
        assert 0 < metrics["recall"] < 1
        assert 0 < metrics["f1"] < 1

    def test_multiple_samples_averaged(self) -> None:
        """Metrics should be averaged across multiple samples."""
        preds = ["hello world", "foo bar"]
        refs = ["hello world", "baz qux"]
        metrics = compute_metrics(preds, refs)
        # First pair: exact match (precision=1, recall=1)
        # Second pair: no overlap (precision=0, recall=0)
        # Average: 0.5
        assert metrics["precision"] == pytest.approx(0.5)
        assert metrics["exact_match"] == pytest.approx(0.5)

    def test_returns_all_keys(self) -> None:
        """Should return dict with precision, recall, f1, exact_match."""
        metrics = compute_metrics(["a b"], ["a b"])
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "exact_match" in metrics

    def test_empty_prediction(self) -> None:
        """Empty prediction should produce precision=0, recall=0, f1=0."""
        metrics = compute_metrics([""], ["hello world"])
        assert metrics["precision"] == pytest.approx(0.0)
        assert metrics["f1"] == pytest.approx(0.0)


class TestLLMAsJudge:
    """Tests for the llm_as_judge function."""

    def test_returns_string(self) -> None:
        """Should return a string."""
        prompt = llm_as_judge("Good response", "Expected answer")
        assert isinstance(prompt, str)

    def test_contains_response(self) -> None:
        """The evaluation prompt should contain the model response."""
        prompt = llm_as_judge("My specific response text", "Reference answer")
        assert "My specific response text" in prompt

    def test_contains_reference(self) -> None:
        """The evaluation prompt should contain the reference answer."""
        prompt = llm_as_judge("Response", "The expected reference answer")
        assert "The expected reference answer" in prompt

    def test_contains_criteria(self) -> None:
        """The evaluation prompt should contain the evaluation criteria."""
        prompt = llm_as_judge("Response", "Reference", criteria="factual accuracy")
        assert "factual accuracy" in prompt

    def test_asks_for_score(self) -> None:
        """The evaluation prompt should ask for a numerical score."""
        prompt = llm_as_judge("Response", "Reference")
        prompt_lower = prompt.lower()
        assert "score" in prompt_lower or "rating" in prompt_lower or "rate" in prompt_lower
