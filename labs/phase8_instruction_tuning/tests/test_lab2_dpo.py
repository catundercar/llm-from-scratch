"""
Tests for Phase 8, Lab 2: Direct Preference Optimization

These tests verify:
    - PreferenceDataset loading
    - DPO loss computation
    - DPO loss behavior (prefers chosen over rejected)

Tests are PROVIDED -- students do not modify this file.
"""

import pytest
import json
import os
import tempfile
import torch
import torch.nn.functional as F

from phase_8.dpo import PreferenceDataset, dpo_loss
from phase_8.sample_tokenizer import SampleTokenizer


@pytest.fixture
def tiny_preferences_file() -> str:
    """Create a tiny preferences JSONL file for testing."""
    samples = [
        {
            "prompt": "What is 2+2?",
            "chosen": "2+2 equals 4.",
            "rejected": "The answer is maybe 5.",
        },
        {
            "prompt": "Say hello.",
            "chosen": "Hello! How can I help you today?",
            "rejected": "hey",
        },
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    yield path
    os.unlink(path)


class TestPreferenceDataset:
    """Tests for the PreferenceDataset class."""

    def test_length(
        self, tiny_preferences_file: str, sample_tokenizer: SampleTokenizer
    ) -> None:
        """Dataset should have the correct number of samples."""
        ds = PreferenceDataset(tiny_preferences_file, sample_tokenizer, max_length=128)
        assert len(ds) == 2

    def test_returns_correct_keys(
        self, tiny_preferences_file: str, sample_tokenizer: SampleTokenizer
    ) -> None:
        """Each item should have chosen_ids, rejected_ids, masks, and prompt_length."""
        ds = PreferenceDataset(tiny_preferences_file, sample_tokenizer, max_length=128)
        item = ds[0]
        assert "chosen_ids" in item
        assert "chosen_mask" in item
        assert "rejected_ids" in item
        assert "rejected_mask" in item
        assert "prompt_length" in item

    def test_output_shapes(
        self, tiny_preferences_file: str, sample_tokenizer: SampleTokenizer
    ) -> None:
        """Token ID tensors should have shape (max_length,)."""
        max_len = 128
        ds = PreferenceDataset(tiny_preferences_file, sample_tokenizer, max_length=max_len)
        item = ds[0]
        assert item["chosen_ids"].shape == (max_len,)
        assert item["rejected_ids"].shape == (max_len,)
        assert item["chosen_mask"].shape == (max_len,)
        assert item["rejected_mask"].shape == (max_len,)

    def test_prompt_length_positive(
        self, tiny_preferences_file: str, sample_tokenizer: SampleTokenizer
    ) -> None:
        """Prompt length should be a positive integer."""
        ds = PreferenceDataset(tiny_preferences_file, sample_tokenizer, max_length=128)
        item = ds[0]
        prompt_len = item["prompt_length"]
        if isinstance(prompt_len, torch.Tensor):
            prompt_len = prompt_len.item()
        assert prompt_len > 0, "Prompt length should be positive"

    def test_loads_from_sample_data(
        self, preferences_path: str, sample_tokenizer: SampleTokenizer
    ) -> None:
        """Should successfully load the provided sample data."""
        ds = PreferenceDataset(preferences_path, sample_tokenizer, max_length=512)
        assert len(ds) == 20


class TestDPOLoss:
    """Tests for the dpo_loss function."""

    def test_loss_is_scalar(self) -> None:
        """DPO loss should return a scalar tensor."""
        policy_chosen = torch.tensor([-1.0, -2.0])
        policy_rejected = torch.tensor([-3.0, -4.0])
        ref_chosen = torch.tensor([-1.5, -2.5])
        ref_rejected = torch.tensor([-3.5, -4.5])

        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
        assert loss.dim() == 0, "Loss should be a scalar"

    def test_loss_is_finite(self) -> None:
        """DPO loss should be finite for reasonable inputs."""
        policy_chosen = torch.tensor([-1.0, -2.0])
        policy_rejected = torch.tensor([-3.0, -4.0])
        ref_chosen = torch.tensor([-1.5, -2.5])
        ref_rejected = torch.tensor([-3.5, -4.5])

        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
        assert torch.isfinite(loss), "Loss should be finite"

    def test_loss_with_known_values(self) -> None:
        """DPO loss should match manual computation for simple inputs."""
        # When policy strongly prefers chosen over rejected (compared to ref),
        # the loss should be low.
        policy_chosen = torch.tensor([-1.0])
        policy_rejected = torch.tensor([-5.0])
        ref_chosen = torch.tensor([-2.0])
        ref_rejected = torch.tensor([-3.0])
        beta = 0.1

        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)

        # Manual:
        # log_ratio_chosen = -1.0 - (-2.0) = 1.0
        # log_ratio_rejected = -5.0 - (-3.0) = -2.0
        # inside_sigmoid = 0.1 * (1.0 - (-2.0)) = 0.1 * 3.0 = 0.3
        # loss = -log(sigmoid(0.3))
        expected = -F.logsigmoid(torch.tensor(0.3))
        assert torch.allclose(loss, expected, atol=1e-5), (
            f"Expected loss {expected.item():.6f}, got {loss.item():.6f}"
        )

    def test_loss_lower_when_policy_prefers_chosen(self) -> None:
        """Loss should be lower when the policy model prefers chosen over rejected."""
        ref_chosen = torch.tensor([-2.0])
        ref_rejected = torch.tensor([-2.0])

        # Policy that prefers chosen
        loss_good = dpo_loss(
            policy_chosen_logprobs=torch.tensor([-1.0]),   # high prob for chosen
            policy_rejected_logprobs=torch.tensor([-5.0]), # low prob for rejected
            ref_chosen_logprobs=ref_chosen,
            ref_rejected_logprobs=ref_rejected,
            beta=0.1,
        )

        # Policy that prefers rejected
        loss_bad = dpo_loss(
            policy_chosen_logprobs=torch.tensor([-5.0]),   # low prob for chosen
            policy_rejected_logprobs=torch.tensor([-1.0]), # high prob for rejected
            ref_chosen_logprobs=ref_chosen,
            ref_rejected_logprobs=ref_rejected,
            beta=0.1,
        )

        assert loss_good < loss_bad, (
            f"Loss should be lower when policy prefers chosen: "
            f"good={loss_good.item():.4f}, bad={loss_bad.item():.4f}"
        )

    def test_loss_sensitive_to_beta(self) -> None:
        """Higher beta should amplify the loss difference."""
        policy_chosen = torch.tensor([-1.0])
        policy_rejected = torch.tensor([-3.0])
        ref_chosen = torch.tensor([-2.0])
        ref_rejected = torch.tensor([-2.0])

        loss_low_beta = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.01
        )
        loss_high_beta = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0
        )

        # With the policy preferring chosen, higher beta should give lower loss
        # (stronger signal)
        assert loss_high_beta < loss_low_beta, (
            "Higher beta should amplify the preference signal"
        )
