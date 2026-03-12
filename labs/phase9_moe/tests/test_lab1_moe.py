"""
Tests for Phase 9, Lab 1: Mixture of Experts Layer

These tests verify:
    - Expert FFN basics
    - Router produces valid distributions and top-k selection
    - MoELayer output shape and routing behavior
    - Load balancing loss properties

Tests are PROVIDED -- students do not modify this file.
"""

import pytest
import torch
import torch.nn.functional as F
from phase_9.moe import Expert, Router, MoELayer, load_balancing_loss
from phase_9.types import MoEConfig


class TestExpert:
    """Tests for the Expert module."""

    def test_output_shape(self) -> None:
        """Expert output shape should match input shape."""
        torch.manual_seed(0)
        expert = Expert(d_model=32, d_ff=128)
        x = torch.randn(4, 16, 32)
        out = expert(x)
        assert out.shape == (4, 16, 32), f"Expected (4, 16, 32), got {out.shape}"

    def test_different_experts_produce_different_output(self) -> None:
        """Two independently initialized experts should produce different outputs."""
        torch.manual_seed(0)
        expert1 = Expert(d_model=32, d_ff=128)
        torch.manual_seed(1)
        expert2 = Expert(d_model=32, d_ff=128)

        x = torch.randn(1, 4, 32)
        out1 = expert1(x)
        out2 = expert2(x)
        assert not torch.allclose(out1, out2, atol=1e-4), (
            "Different experts should produce different outputs"
        )


class TestRouter:
    """Tests for the Router module."""

    def test_router_probs_sum_to_one(self) -> None:
        """Router probabilities should sum to 1 for each token."""
        torch.manual_seed(0)
        router = Router(d_model=32, n_experts=4, top_k=2)
        x = torch.randn(2, 8, 32)
        router_probs, _, _ = router(x)

        sums = router_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"Router probs should sum to 1, got sums: {sums}"
        )

    def test_router_probs_non_negative(self) -> None:
        """All routing probabilities should be non-negative."""
        torch.manual_seed(0)
        router = Router(d_model=32, n_experts=4, top_k=2)
        x = torch.randn(2, 8, 32)
        router_probs, _, _ = router(x)

        assert (router_probs >= 0).all(), "Router probs should be non-negative"

    def test_top_k_selection_count(self) -> None:
        """Router should select exactly top_k experts per token."""
        torch.manual_seed(0)
        top_k = 2
        router = Router(d_model=32, n_experts=8, top_k=top_k)
        x = torch.randn(2, 8, 32)
        _, top_k_weights, top_k_indices = router(x)

        assert top_k_indices.shape[-1] == top_k, (
            f"Expected top_k={top_k} indices, got {top_k_indices.shape[-1]}"
        )
        assert top_k_weights.shape[-1] == top_k, (
            f"Expected top_k={top_k} weights, got {top_k_weights.shape[-1]}"
        )

    def test_top_k_indices_valid(self) -> None:
        """Top-k expert indices should be in valid range [0, n_experts)."""
        torch.manual_seed(0)
        n_experts = 4
        router = Router(d_model=32, n_experts=n_experts, top_k=2)
        x = torch.randn(2, 8, 32)
        _, _, top_k_indices = router(x)

        assert (top_k_indices >= 0).all() and (top_k_indices < n_experts).all(), (
            f"Expert indices should be in [0, {n_experts}), got min={top_k_indices.min()}, max={top_k_indices.max()}"
        )

    def test_top_k_weights_normalized(self) -> None:
        """Top-k weights should sum to 1 per token (after renormalization)."""
        torch.manual_seed(0)
        router = Router(d_model=32, n_experts=4, top_k=2)
        x = torch.randn(2, 8, 32)
        _, top_k_weights, _ = router(x)

        sums = top_k_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"Top-k weights should sum to 1 after normalization, got: {sums}"
        )

    def test_router_probs_shape(self) -> None:
        """Router probs should have shape (batch*seq, n_experts)."""
        torch.manual_seed(0)
        n_experts = 4
        router = Router(d_model=32, n_experts=n_experts, top_k=2)
        x = torch.randn(2, 8, 32)
        router_probs, _, _ = router(x)

        assert router_probs.shape == (16, n_experts), (
            f"Expected shape (16, {n_experts}), got {router_probs.shape}"
        )


class TestMoELayer:
    """Tests for the MoELayer module."""

    def test_output_shape(self, dummy_hidden: torch.Tensor) -> None:
        """MoE layer output should have same shape as input."""
        torch.manual_seed(0)
        moe = MoELayer(d_model=32, d_ff=128, n_experts=4, top_k=2)
        output, router_probs = moe(dummy_hidden)
        assert output.shape == dummy_hidden.shape, (
            f"Expected shape {dummy_hidden.shape}, got {output.shape}"
        )

    def test_returns_router_probs(self, dummy_hidden: torch.Tensor) -> None:
        """MoE layer should return router probabilities."""
        torch.manual_seed(0)
        moe = MoELayer(d_model=32, d_ff=128, n_experts=4, top_k=2)
        output, router_probs = moe(dummy_hidden)
        assert router_probs is not None, "Should return router probabilities"
        assert router_probs.shape[-1] == 4, (
            f"Router probs last dim should be n_experts=4, got {router_probs.shape[-1]}"
        )

    def test_output_not_all_zeros(self, dummy_hidden: torch.Tensor) -> None:
        """MoE output should not be all zeros."""
        torch.manual_seed(0)
        moe = MoELayer(d_model=32, d_ff=128, n_experts=4, top_k=2)
        output, _ = moe(dummy_hidden)
        assert not torch.all(output == 0), "MoE output should not be all zeros"

    def test_gradient_flows(self, dummy_hidden: torch.Tensor) -> None:
        """Gradients should flow through the MoE layer."""
        torch.manual_seed(0)
        moe = MoELayer(d_model=32, d_ff=128, n_experts=4, top_k=2)
        hidden = dummy_hidden.clone().requires_grad_(True)
        output, _ = moe(hidden)
        loss = output.sum()
        loss.backward()
        assert hidden.grad is not None, "Gradients should flow through MoE layer"
        assert not torch.all(hidden.grad == 0), "Gradients should be non-zero"


class TestLoadBalancingLoss:
    """Tests for the load_balancing_loss function."""

    def test_returns_scalar(self) -> None:
        """Load balancing loss should be a scalar."""
        n_experts = 4
        router_probs = torch.ones(16, n_experts) / n_experts
        top_k_indices = torch.randint(0, n_experts, (16, 2))
        loss = load_balancing_loss(router_probs, top_k_indices, n_experts)
        assert loss.dim() == 0, "Loss should be a scalar"

    def test_loss_is_finite(self) -> None:
        """Load balancing loss should be finite."""
        n_experts = 4
        router_probs = torch.ones(16, n_experts) / n_experts
        top_k_indices = torch.randint(0, n_experts, (16, 2))
        loss = load_balancing_loss(router_probs, top_k_indices, n_experts)
        assert torch.isfinite(loss), "Loss should be finite"

    def test_uniform_routing_gives_minimum(self) -> None:
        """Uniform routing should give lower loss than skewed routing."""
        n_experts = 4
        n_tokens = 100

        # Uniform routing: each expert gets equal share
        uniform_probs = torch.ones(n_tokens, n_experts) / n_experts
        # Each token goes to exactly top_k=2 experts, evenly distributed
        uniform_indices = torch.stack([
            torch.arange(n_tokens) % n_experts,
            (torch.arange(n_tokens) + 1) % n_experts,
        ], dim=1)

        uniform_loss = load_balancing_loss(uniform_probs, uniform_indices, n_experts)

        # Skewed routing: all tokens go to expert 0
        skewed_probs = torch.zeros(n_tokens, n_experts)
        skewed_probs[:, 0] = 0.9
        skewed_probs[:, 1:] = 0.1 / (n_experts - 1)
        skewed_indices = torch.zeros(n_tokens, 2, dtype=torch.long)
        skewed_indices[:, 1] = 1

        skewed_loss = load_balancing_loss(skewed_probs, skewed_indices, n_experts)

        assert uniform_loss < skewed_loss, (
            f"Uniform loss ({uniform_loss.item():.4f}) should be less than "
            f"skewed loss ({skewed_loss.item():.4f})"
        )

    def test_loss_non_negative(self) -> None:
        """Load balancing loss should always be non-negative."""
        n_experts = 4
        router_probs = torch.softmax(torch.randn(32, n_experts), dim=-1)
        top_k_indices = torch.randint(0, n_experts, (32, 2))
        loss = load_balancing_loss(router_probs, top_k_indices, n_experts)
        assert loss >= 0, f"Loss should be non-negative, got {loss.item()}"
