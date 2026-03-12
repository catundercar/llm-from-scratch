"""
Tests for Phase 7, Lab 1: LoRA Implementation

These tests verify the core LoRA building blocks:
    - LoRALinear layer behavior and initialization
    - apply_lora_to_model correctly injects LoRA
    - merge_lora_weights produces equivalent outputs
    - Parameter counting is correct

Tests are PROVIDED -- students do not modify this file.
"""

import pytest
import torch
import torch.nn as nn
import copy
from phase_7.lora import (
    LoRALinear,
    apply_lora_to_model,
    merge_lora_weights,
    count_trainable_params,
)
from phase_7.sample_model import SampleGPT
from phase_7.types import GPTConfig


# ---------------------------------------------------------------------------
# LoRALinear basics
# ---------------------------------------------------------------------------

class TestLoRALinear:
    """Tests for the LoRALinear module."""

    def test_output_shape_matches_linear(self) -> None:
        """LoRALinear should produce the same output shape as nn.Linear."""
        torch.manual_seed(0)
        in_f, out_f = 64, 128
        lora = LoRALinear(in_f, out_f, rank=4, alpha=1.0)
        x = torch.randn(2, 10, in_f)
        out = lora(x)
        assert out.shape == (2, 10, out_f), (
            f"Expected shape (2, 10, {out_f}), got {out.shape}"
        )

    def test_b_initialized_to_zeros(self) -> None:
        """lora_B should be initialized to zeros so LoRA has no effect at init."""
        lora = LoRALinear(32, 64, rank=4, alpha=1.0)
        assert torch.all(lora.lora_B == 0), "lora_B should be initialized to zeros"

    def test_a_is_not_zero(self) -> None:
        """lora_A should be initialized with Kaiming uniform (not zeros)."""
        torch.manual_seed(42)
        lora = LoRALinear(32, 64, rank=4, alpha=1.0)
        assert not torch.all(lora.lora_A == 0), "lora_A should not be all zeros"

    def test_initial_output_matches_base(self) -> None:
        """At initialization (B=0), LoRALinear should produce same output as base Linear."""
        torch.manual_seed(0)
        in_f, out_f = 32, 64

        # Create a regular linear and a LoRA linear with same weights
        base_linear = nn.Linear(in_f, out_f, bias=True)
        lora = LoRALinear.from_linear(base_linear, rank=4, alpha=1.0)

        x = torch.randn(2, 8, in_f)
        base_out = base_linear(x)
        lora_out = lora(x)

        assert torch.allclose(base_out, lora_out, atol=1e-6), (
            "With B=0, LoRA output should match base linear output"
        )

    def test_lora_a_shape(self) -> None:
        """lora_A should have shape (in_features, rank)."""
        lora = LoRALinear(64, 128, rank=8, alpha=2.0)
        assert lora.lora_A.shape == (64, 8), (
            f"Expected lora_A shape (64, 8), got {lora.lora_A.shape}"
        )

    def test_lora_b_shape(self) -> None:
        """lora_B should have shape (rank, out_features)."""
        lora = LoRALinear(64, 128, rank=8, alpha=2.0)
        assert lora.lora_B.shape == (8, 128), (
            f"Expected lora_B shape (8, 128), got {lora.lora_B.shape}"
        )

    def test_frozen_weight_not_trainable(self) -> None:
        """The base weight W should not require gradients."""
        lora = LoRALinear(32, 64, rank=4, alpha=1.0)
        assert not lora.weight.requires_grad, "Base weight should be frozen"

    def test_lora_params_are_trainable(self) -> None:
        """lora_A and lora_B should require gradients."""
        lora = LoRALinear(32, 64, rank=4, alpha=1.0)
        assert lora.lora_A.requires_grad, "lora_A should be trainable"
        assert lora.lora_B.requires_grad, "lora_B should be trainable"

    def test_scaling_factor(self) -> None:
        """The scaling factor should be alpha / rank."""
        lora = LoRALinear(32, 64, rank=8, alpha=4.0)
        assert lora.scaling == pytest.approx(0.5), (
            f"Expected scaling 4.0/8=0.5, got {lora.scaling}"
        )

    def test_lora_changes_output_after_training(self) -> None:
        """After modifying lora_B, output should differ from base."""
        torch.manual_seed(0)
        base_linear = nn.Linear(32, 64)
        lora = LoRALinear.from_linear(base_linear, rank=4, alpha=1.0)
        x = torch.randn(1, 4, 32)

        out_before = lora(x).clone()
        # Simulate training: set B to non-zero
        with torch.no_grad():
            lora.lora_B.fill_(0.1)
        out_after = lora(x)

        assert not torch.allclose(out_before, out_after, atol=1e-6), (
            "Output should change after modifying lora_B"
        )


# ---------------------------------------------------------------------------
# apply_lora_to_model
# ---------------------------------------------------------------------------

class TestApplyLoRA:
    """Tests for the apply_lora_to_model function."""

    def test_replaces_target_modules(self, small_model: SampleGPT) -> None:
        """Target modules should be replaced with LoRALinear."""
        apply_lora_to_model(small_model, rank=4, alpha=1.0, target_modules=["q_proj", "v_proj"])

        for name, module in small_model.named_modules():
            if name.endswith("q_proj") or name.endswith("v_proj"):
                assert isinstance(module, LoRALinear), (
                    f"Module {name} should be LoRALinear, got {type(module)}"
                )

    def test_does_not_replace_non_target_modules(self, small_model: SampleGPT) -> None:
        """Non-target Linear modules should remain as nn.Linear."""
        apply_lora_to_model(small_model, rank=4, alpha=1.0, target_modules=["q_proj", "v_proj"])

        for name, module in small_model.named_modules():
            if name.endswith("k_proj") or name.endswith("out_proj"):
                assert isinstance(module, nn.Linear), (
                    f"Module {name} should remain nn.Linear, got {type(module)}"
                )
                assert not isinstance(module, LoRALinear), (
                    f"Module {name} should not be LoRALinear"
                )

    def test_only_lora_params_trainable(self, small_model: SampleGPT) -> None:
        """After applying LoRA, only lora_A and lora_B should be trainable."""
        apply_lora_to_model(small_model, rank=4, alpha=1.0)

        for name, param in small_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.requires_grad, f"LoRA param {name} should be trainable"
            else:
                assert not param.requires_grad, (
                    f"Non-LoRA param {name} should be frozen"
                )

    def test_model_still_produces_output(
        self, small_model: SampleGPT, dummy_input: torch.Tensor
    ) -> None:
        """Model should still work after LoRA injection."""
        apply_lora_to_model(small_model, rank=4, alpha=1.0)
        logits, _ = small_model(dummy_input)
        assert logits.shape == (2, 16, 100), (
            f"Expected logits shape (2, 16, 100), got {logits.shape}"
        )

    def test_trainable_params_reduced(self, small_model: SampleGPT) -> None:
        """Trainable params should be less than 5% of total after LoRA."""
        total_before = sum(p.numel() for p in small_model.parameters())
        apply_lora_to_model(small_model, rank=4, alpha=1.0)
        trainable_after = sum(
            p.numel() for p in small_model.parameters() if p.requires_grad
        )
        ratio = trainable_after / total_before
        assert ratio < 0.05, (
            f"Trainable params should be < 5% of total, got {ratio:.2%}"
        )


# ---------------------------------------------------------------------------
# merge_lora_weights
# ---------------------------------------------------------------------------

class TestMergeLoRA:
    """Tests for the merge_lora_weights function."""

    def test_merged_output_matches_lora_output(
        self, small_model: SampleGPT, dummy_input: torch.Tensor
    ) -> None:
        """After merging, the model should produce the same output as with LoRA layers."""
        apply_lora_to_model(small_model, rank=4, alpha=1.0)

        # Simulate some training by setting B to non-zero
        with torch.no_grad():
            for module in small_model.modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.normal_(0, 0.01)

        # Get output before merge
        with torch.no_grad():
            lora_out, _ = small_model(dummy_input)

        # Merge and get output
        merge_lora_weights(small_model)
        with torch.no_grad():
            merged_out, _ = small_model(dummy_input)

        assert torch.allclose(lora_out, merged_out, atol=1e-5), (
            "Merged model output should match LoRA model output"
        )

    def test_no_lora_modules_after_merge(self, small_model: SampleGPT) -> None:
        """After merging, there should be no LoRALinear modules left."""
        apply_lora_to_model(small_model, rank=4, alpha=1.0)
        merge_lora_weights(small_model)

        for name, module in small_model.named_modules():
            assert not isinstance(module, LoRALinear), (
                f"Module {name} is still LoRALinear after merge"
            )


# ---------------------------------------------------------------------------
# count_trainable_params
# ---------------------------------------------------------------------------

class TestCountParams:
    """Tests for the count_trainable_params function."""

    def test_returns_correct_counts(self) -> None:
        """Should return (trainable, total) parameter counts."""
        model = nn.Linear(10, 5)  # 50 weight + 5 bias = 55 total, all trainable
        trainable, total = count_trainable_params(model)
        assert total == 55, f"Expected 55 total params, got {total}"
        assert trainable == 55, f"Expected 55 trainable params, got {trainable}"

    def test_frozen_params_not_counted_as_trainable(self) -> None:
        """Frozen parameters should not be counted as trainable."""
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        trainable, total = count_trainable_params(model)
        assert total == 55
        assert trainable == 0
