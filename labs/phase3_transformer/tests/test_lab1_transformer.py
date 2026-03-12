"""Tests for Phase 3, Lab 1: Transformer Components."""

import pytest
import torch
from phase_3.transformer import LayerNorm, GELU, FeedForward, TransformerBlock


class TestLayerNorm:
    def test_output_shape(self):
        ln = LayerNorm(32)
        x = torch.randn(2, 5, 32)
        out = ln(x)
        assert out.shape == (2, 5, 32)

    def test_zero_mean(self):
        ln = LayerNorm(64)
        x = torch.randn(2, 4, 64)
        out = ln(x)
        means = out.mean(dim=-1)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)

    def test_unit_variance(self):
        ln = LayerNorm(64)
        x = torch.randn(2, 4, 64)
        out = ln(x)
        # Before affine transform, variance should be ~1
        # After affine, gamma=1 beta=0, so still ~1
        vars_ = out.var(dim=-1, unbiased=False)
        assert torch.allclose(vars_, torch.ones_like(vars_), atol=1e-4)

    def test_learnable_params(self):
        ln = LayerNorm(32)
        param_names = [n for n, _ in ln.named_parameters()]
        assert len(param_names) == 2  # gamma and beta


class TestGELU:
    def test_output_shape(self):
        gelu = GELU()
        x = torch.randn(2, 3, 16)
        out = gelu(x)
        assert out.shape == (2, 3, 16)

    def test_zero_input(self):
        gelu = GELU()
        x = torch.zeros(1)
        out = gelu(x)
        assert torch.allclose(out, torch.zeros(1), atol=1e-6)

    def test_positive_for_large_input(self):
        gelu = GELU()
        x = torch.tensor([5.0])
        out = gelu(x)
        assert out.item() > 0


class TestFeedForward:
    def test_output_shape(self):
        ff = FeedForward(32)
        x = torch.randn(2, 5, 32)
        out = ff(x)
        assert out.shape == (2, 5, 32)

    def test_custom_d_ff(self):
        ff = FeedForward(32, d_ff=64)
        x = torch.randn(1, 3, 32)
        out = ff(x)
        assert out.shape == (1, 3, 32)

    def test_gradient_flow(self):
        ff = FeedForward(16)
        x = torch.randn(1, 2, 16, requires_grad=True)
        out = ff(x)
        out.sum().backward()
        assert x.grad is not None


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=32, n_heads=4, block_size=10)
        x = torch.randn(2, 5, 32)
        out = block(x)
        assert out.shape == (2, 5, 32)

    def test_residual_connection(self):
        """Output should not be zero even with zero-initialized sub-layers."""
        block = TransformerBlock(d_model=16, n_heads=2, block_size=8)
        x = torch.randn(1, 4, 16)
        out = block(x)
        # Due to residual, output should be close to input (not zero)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_different_seq_lengths(self):
        block = TransformerBlock(d_model=32, n_heads=4, block_size=20)
        for seq_len in [1, 5, 10, 20]:
            x = torch.randn(1, seq_len, 32)
            out = block(x)
            assert out.shape == (1, seq_len, 32)
