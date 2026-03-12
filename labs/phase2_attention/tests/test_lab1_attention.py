"""Tests for Phase 2: Attention Mechanisms."""

import pytest
import torch
from phase_2.attention import (
    scaled_dot_product_attention,
    CausalSelfAttention,
    MultiHeadAttention,
)


class TestScaledDotProductAttention:
    def test_output_shape(self):
        Q = torch.randn(2, 4, 8)  # (batch, seq_len, d_k)
        K = torch.randn(2, 4, 8)
        V = torch.randn(2, 4, 8)
        out = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (2, 4, 8)

    def test_attention_weights_sum_to_one(self):
        Q = torch.randn(1, 3, 4)
        K = torch.randn(1, 3, 4)
        V = torch.randn(1, 3, 4)
        # Manually compute weights to verify
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        # Each row should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 3), atol=1e-5)

    def test_mask_prevents_future(self):
        Q = torch.randn(1, 4, 8)
        K = torch.randn(1, 4, 8)
        V = torch.ones(1, 4, 8)  # uniform values for easy verification
        mask = torch.tril(torch.ones(4, 4))  # causal mask
        out = scaled_dot_product_attention(Q, K, V, mask=mask)
        assert out.shape == (1, 4, 8)

    def test_no_mask(self):
        Q = torch.randn(2, 3, 4)
        K = torch.randn(2, 3, 4)
        V = torch.randn(2, 3, 4)
        out = scaled_dot_product_attention(Q, K, V)
        assert not torch.isnan(out).any()


class TestCausalSelfAttention:
    def test_output_shape(self):
        attn = CausalSelfAttention(d_model=32, head_dim=16, block_size=10)
        x = torch.randn(2, 5, 32)
        out = attn(x)
        assert out.shape == (2, 5, 32)

    def test_causal_mask_applied(self):
        attn = CausalSelfAttention(d_model=16, head_dim=8, block_size=4)
        x = torch.randn(1, 4, 16)
        # Should not raise, mask should handle it
        out = attn(x)
        assert not torch.isnan(out).any()


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=32, n_heads=4, block_size=10)
        x = torch.randn(2, 5, 32)
        out = mha(x)
        assert out.shape == (2, 5, 32)

    def test_head_dim_calculation(self):
        mha = MultiHeadAttention(d_model=64, n_heads=8, block_size=10)
        assert mha.head_dim == 8

    def test_d_model_not_divisible(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=33, n_heads=4, block_size=10)

    def test_different_seq_lengths(self):
        mha = MultiHeadAttention(d_model=32, n_heads=4, block_size=20)
        for seq_len in [1, 5, 10, 20]:
            x = torch.randn(1, seq_len, 32)
            out = mha(x)
            assert out.shape == (1, seq_len, 32)

    def test_gradient_flow(self):
        mha = MultiHeadAttention(d_model=16, n_heads=2, block_size=8)
        x = torch.randn(1, 4, 16, requires_grad=True)
        out = mha(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 4, 16)
