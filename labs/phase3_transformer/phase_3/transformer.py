"""
Phase 3, Lab 1: Transformer Components
=======================================

Implement the building blocks of the Transformer architecture:
- LayerNorm: Layer normalization from scratch
- GELU: Gaussian Error Linear Unit activation
- FeedForward: Two-layer feed-forward network with GELU
- TransformerBlock: Pre-norm Transformer block with residual connections

Reference: Vaswani et al., "Attention Is All You Need" (2017)
           Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2)
"""

import torch
import torch.nn as nn
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from phase2_attention.phase_2.attention import MultiHeadAttention


# ---------------------------------------------------------------------------
# TODO 1: Layer Normalization
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """
    Layer Normalization (from scratch, without using nn.LayerNorm).

    Normalizes each sample across the feature dimension to zero mean and
    unit variance, then applies a learnable affine transformation.

    y = gamma * (x - mean) / (std + eps) + beta
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        """
        Initialize LayerNorm.

        TODO: Implement this method

        Requirements:
        1. Create learnable parameter gamma (scale) of shape (d_model,),
           initialized to ones.
        2. Create learnable parameter beta (shift) of shape (d_model,),
           initialized to zeros.
        3. Store eps.

        HINT: Use nn.Parameter(torch.ones(d_model)) for gamma.

        Args:
            d_model: Feature dimension to normalize over.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        # TODO: Implement
        # Step 1: Create gamma parameter (ones)
        # Step 2: Create beta parameter (zeros)
        # Step 3: Store eps
        raise NotImplementedError("TODO: Implement LayerNorm.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        TODO: Implement this method

        Requirements:
        1. Compute mean across the last dimension (keepdim=True).
        2. Compute variance across the last dimension (keepdim=True).
           Use x.var(dim=-1, keepdim=True, unbiased=False) for unbiased=False.
        3. Normalize: x_norm = (x - mean) / sqrt(var + eps)
        4. Apply affine: output = gamma * x_norm + beta
        5. Return output.

        HINT: torch.sqrt(var + self.eps) for the denominator.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Normalized tensor of same shape.
        """
        # TODO: Implement
        # Step 1: Compute mean
        # Step 2: Compute variance
        # Step 3: Normalize
        # Step 4: Apply affine transform
        raise NotImplementedError("TODO: Implement LayerNorm.forward")


# ---------------------------------------------------------------------------
# TODO 2: GELU Activation
# ---------------------------------------------------------------------------

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    This is the approximate form used by GPT-2.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.

        TODO: Implement this method

        Requirements:
        1. Compute the approximate GELU:
           0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

        HINT: math.sqrt(2.0 / math.pi) ≈ 0.7978845608

        Args:
            x: Input tensor.

        Returns:
            GELU-activated tensor.
        """
        # TODO: Implement the GELU formula
        raise NotImplementedError("TODO: Implement GELU.forward")


# ---------------------------------------------------------------------------
# TODO 3: Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    FFN(x) = Linear2(GELU(Linear1(x)))

    The inner dimension is typically 4 * d_model.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.0) -> None:
        """
        Initialize FeedForward network.

        TODO: Implement this method

        Requirements:
        1. If d_ff is None, set it to 4 * d_model.
        2. Create first linear layer: d_model -> d_ff.
        3. Create GELU activation (use your GELU class above).
        4. Create second linear layer: d_ff -> d_model.
        5. Create dropout layer.

        Args:
            d_model: Input/output dimension.
            d_ff: Inner dimension (default: 4 * d_model).
            dropout: Dropout probability.
        """
        super().__init__()
        # TODO: Implement
        # Step 1: Set d_ff default
        # Step 2: Create layers: Linear -> GELU -> Linear -> Dropout
        raise NotImplementedError("TODO: Implement FeedForward.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        TODO: Implement this method

        Requirements:
        1. Pass x through: Linear1 -> GELU -> Linear2 -> Dropout

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., d_model).
        """
        # TODO: Implement
        raise NotImplementedError("TODO: Implement FeedForward.forward")


# ---------------------------------------------------------------------------
# TODO 4: Transformer Block (Pre-Norm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single Transformer block using Pre-Norm architecture (GPT-2 style).

    Pre-Norm:
        x = x + MHA(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    This differs from the original Transformer which uses Post-Norm.
    """

    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float = 0.0) -> None:
        """
        Initialize TransformerBlock.

        TODO: Implement this method

        Requirements:
        1. Create LayerNorm for attention sub-layer (ln1).
        2. Create MultiHeadAttention module (attn).
        3. Create LayerNorm for FFN sub-layer (ln2).
        4. Create FeedForward module (ffn).
        5. Create dropout layer.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            block_size: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        # TODO: Implement
        # Step 1: Create ln1 (LayerNorm)
        # Step 2: Create attn (MultiHeadAttention)
        # Step 3: Create ln2 (LayerNorm)
        # Step 4: Create ffn (FeedForward)
        # Step 5: Create dropout
        raise NotImplementedError("TODO: Implement TransformerBlock.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Pre-Norm and residual connections.

        TODO: Implement this method

        Requirements:
        1. Attention sub-layer: x = x + dropout(attn(ln1(x)))
        2. FFN sub-layer: x = x + dropout(ffn(ln2(x)))
        3. Return x.

        HINT: The residual connection adds the original x to the sub-layer output.
              Pre-Norm means LayerNorm is applied BEFORE the sub-layer, not after.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of same shape.
        """
        # TODO: Implement
        # Step 1: x = x + dropout(attn(ln1(x)))
        # Step 2: x = x + dropout(ffn(ln2(x)))
        raise NotImplementedError("TODO: Implement TransformerBlock.forward")
