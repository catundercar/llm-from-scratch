"""
Phase 2: Attention Mechanisms
=============================

Implement the core attention mechanisms used in Transformer models, from
basic scaled dot-product attention to full multi-head causal attention.

Students implement:
    - scaled_dot_product_attention: The fundamental attention computation
    - CausalSelfAttention: Single-head causal (masked) self-attention
    - MultiHeadAttention: Multi-head attention with parallel heads

Reference: Vaswani et al., "Attention Is All You Need" (2017)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TODO 1: Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: float = 0.0,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    TODO: Implement this function

    Requirements:
    1. Compute attention scores: Q @ K^T (transpose last two dims of K).
    2. Scale by 1 / sqrt(d_k) where d_k = Q.shape[-1].
    3. If mask is provided, fill positions where mask == 0 with -inf.
    4. Apply softmax along the last dimension.
    5. Apply dropout if dropout > 0 (use F.dropout in training mode).
    6. Multiply attention weights by V.

    HINT: Use torch.matmul for batched matrix multiplication.
    HINT: Use scores.masked_fill(mask == 0, float('-inf')) for masking.

    Args:
        Q: Query tensor of shape (..., seq_len, d_k).
        K: Key tensor of shape (..., seq_len, d_k).
        V: Value tensor of shape (..., seq_len, d_v).
        mask: Optional boolean mask of shape (..., seq_len, seq_len).
              1 = attend, 0 = mask out.
        dropout: Dropout probability (default 0.0).

    Returns:
        Attention output of shape (..., seq_len, d_v).
    """
    # TODO: Implement
    # Step 1: Compute Q @ K^T
    # Step 2: Scale by 1/sqrt(d_k)
    # Step 3: Apply mask if provided
    # Step 4: Softmax
    # Step 5: Optional dropout
    # Step 6: Multiply by V
    raise NotImplementedError("TODO: Implement scaled_dot_product_attention")


# ---------------------------------------------------------------------------
# TODO 2 & 3: Single-Head Causal Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Single-head causal (autoregressive) self-attention.

    This module projects the input into Q, K, V, applies a causal mask
    to prevent attending to future tokens, computes attention, and projects
    the output.
    """

    def __init__(self, d_model: int, head_dim: int, block_size: int, dropout: float = 0.0) -> None:
        """
        Initialize CausalSelfAttention.

        TODO: Implement this method

        Requirements:
        1. Create linear projection layers W_q, W_k, W_v: (d_model -> head_dim)
        2. Create output projection W_o: (head_dim -> d_model)
        3. Register a causal mask as a buffer (not a parameter):
           mask = torch.tril(torch.ones(block_size, block_size))
           Use self.register_buffer('mask', mask)
        4. Store dropout rate.

        HINT: Use nn.Linear(d_model, head_dim, bias=False) for projections.
        HINT: register_buffer makes the mask part of the module state but
              not a learnable parameter.

        Args:
            d_model: Model embedding dimension.
            head_dim: Dimension of Q, K, V for this head.
            block_size: Maximum sequence length (for causal mask size).
            dropout: Dropout probability.
        """
        super().__init__()
        # TODO: Implement
        # Step 1: Create Q, K, V projection layers
        # Step 2: Create output projection layer
        # Step 3: Register causal mask buffer
        # Step 4: Store dropout
        raise NotImplementedError("TODO: Implement CausalSelfAttention.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for single-head causal self-attention.

        TODO: Implement this method

        Requirements:
        1. Project input x to Q, K, V using the linear layers.
        2. Extract the causal mask for the current sequence length:
           mask = self.mask[:seq_len, :seq_len]
        3. Compute attention using scaled_dot_product_attention with mask.
        4. Project the output through W_o.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # TODO: Implement
        # Step 1: Get batch, seq_len from x.shape
        # Step 2: Project to Q, K, V
        # Step 3: Get causal mask for current seq_len
        # Step 4: Compute attention
        # Step 5: Project output
        raise NotImplementedError("TODO: Implement CausalSelfAttention.forward")


# ---------------------------------------------------------------------------
# TODO 4 & 5: Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Splits the model dimension into multiple heads, computes attention
    in parallel across heads, concatenates, and projects the output.

    d_model must be divisible by n_heads.
    head_dim = d_model // n_heads
    """

    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float = 0.0) -> None:
        """
        Initialize MultiHeadAttention.

        TODO: Implement this method

        Requirements:
        1. Verify d_model is divisible by n_heads.
        2. Compute head_dim = d_model // n_heads.
        3. Create combined Q, K, V projections: nn.Linear(d_model, d_model)
           (this projects to all heads at once, then we reshape).
        4. Create output projection: nn.Linear(d_model, d_model).
        5. Register causal mask buffer.
        6. Store n_heads, head_dim, dropout.

        HINT: Using a single nn.Linear(d_model, d_model) for all Q projections
              (and similarly for K, V) is more efficient than n_heads separate
              nn.Linear(d_model, head_dim) layers.

        Args:
            d_model: Model embedding dimension.
            n_heads: Number of attention heads.
            block_size: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        # TODO: Implement
        # Step 1: Store n_heads, head_dim
        # Step 2: Create W_q, W_k, W_v as nn.Linear(d_model, d_model)
        # Step 3: Create W_o as nn.Linear(d_model, d_model)
        # Step 4: Register causal mask buffer
        # Step 5: Store dropout
        raise NotImplementedError("TODO: Implement MultiHeadAttention.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head causal self-attention.

        TODO: Implement this method

        Requirements:
        1. Get batch size B, sequence length T, and d_model C from x.shape.
        2. Project x to Q, K, V (each of shape B, T, d_model).
        3. Reshape each to (B, T, n_heads, head_dim) then transpose to
           (B, n_heads, T, head_dim).
        4. Create causal mask of shape (1, 1, T, T) for broadcasting
           across batch and head dimensions.
        5. Compute scaled_dot_product_attention with the mask.
        6. Transpose back to (B, T, n_heads, head_dim).
        7. Reshape (concatenate heads) to (B, T, d_model).
        8. Apply output projection W_o.

        HINT: The reshape for splitting heads:
              Q = Q.view(B, T, n_heads, head_dim).transpose(1, 2)
              This gives shape (B, n_heads, T, head_dim).

        HINT: The reshape for concatenating heads:
              out = out.transpose(1, 2).contiguous().view(B, T, C)

        HINT: The mask should be (1, 1, T, T) so it broadcasts over batch
              and head dimensions.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, C = x.shape
        # TODO: Implement
        # Step 1: Project to Q, K, V
        # Step 2: Reshape to (B, n_heads, T, head_dim)
        # Step 3: Create causal mask (1, 1, T, T)
        # Step 4: Compute attention
        # Step 5: Reshape back to (B, T, d_model)
        # Step 6: Apply output projection
        raise NotImplementedError("TODO: Implement MultiHeadAttention.forward")
