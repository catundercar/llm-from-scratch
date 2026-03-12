"""
Phase 9, Lab 2: MoE-Enhanced Transformer

This module builds a full transformer model that uses MoE layers in place
of some feed-forward networks. This mirrors real architectures like Mixtral,
where every other transformer block uses an MoE layer instead of a dense FFN.

Students implement:
    - MoETransformerBlock: A transformer block with MoE instead of FFN
    - MoEGPT: Full GPT model with interleaved MoE and dense blocks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from phase_9.types import MoEConfig
from phase_9.moe import MoELayer, load_balancing_loss


class SampleAttention(nn.Module):
    """
    Multi-head self-attention.

    This is PROVIDED -- students do not need to modify this class.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class DenseFeedForward(nn.Module):
    """
    Standard dense feed-forward network.

    This is PROVIDED -- used in non-MoE transformer blocks.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseTransformerBlock(nn.Module):
    """
    Standard transformer block with dense FFN.

    This is PROVIDED -- used for non-MoE layers.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SampleAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = DenseFeedForward(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x, None  # None for router_probs (no MoE)


class MoETransformerBlock(nn.Module):
    """
    Transformer block with Mixture of Experts replacing the FFN.

    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> residual
          -> LayerNorm -> MoELayer -> residual

    This is identical to a standard transformer block except the dense FFN
    is replaced with an MoELayer.
    """

    def __init__(self, config: MoEConfig) -> None:
        """
        Initialize the MoE transformer block.

        TODO: Implement this method

        Requirements:
        1. Create two LayerNorm layers (one before attention, one before MoE).
        2. Create a SampleAttention module.
        3. Create an MoELayer with n_experts and top_k from config.
        4. The MoE hidden dimension should be 4 * n_embd (same as dense FFN).

        Args:
            config: Model configuration.

        HINT: MoELayer(config.n_embd, 4 * config.n_embd, config.n_experts, config.top_k)
        """
        super().__init__()
        # TODO: Implement
        # Step 1: self.ln1 = nn.LayerNorm(config.n_embd)
        # Step 2: self.attn = SampleAttention(config)
        # Step 3: self.ln2 = nn.LayerNorm(config.n_embd)
        # Step 4: self.moe = MoELayer(config.n_embd, 4*config.n_embd, config.n_experts, config.top_k)
        raise NotImplementedError("TODO: Implement MoETransformerBlock.__init__")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the MoE transformer block.

        TODO: Implement this method

        Requirements:
        1. Apply LayerNorm + attention + residual connection.
        2. Apply LayerNorm + MoE layer + residual connection.
        3. Return the output and the router probabilities (from MoELayer).

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tuple of (output, router_probs) where router_probs can be used
            for computing the load balancing loss.

        HINT: The MoE layer returns (output, router_probs).
        """
        # TODO: Implement
        # Step 1: x = x + self.attn(self.ln1(x))
        # Step 2: moe_out, router_probs = self.moe(self.ln2(x))
        # Step 3: x = x + moe_out
        # Step 4: return x, router_probs
        raise NotImplementedError("TODO: Implement MoETransformerBlock.forward")


class MoEGPT(nn.Module):
    """
    GPT model with interleaved MoE and dense transformer blocks.

    Architecture mirrors Mixtral: every moe_every_n_layers-th block uses
    an MoE layer instead of a dense FFN. For example, with 6 layers and
    moe_every_n_layers=2, layers 1, 3, 5 are MoE and layers 0, 2, 4 are dense.
    """

    def __init__(self, config: MoEConfig) -> None:
        """
        Initialize the MoE GPT model.

        TODO: Implement this method

        Requirements:
        1. Create token and position embeddings.
        2. Create a list of transformer blocks:
           - Every moe_every_n_layers-th block is an MoETransformerBlock.
           - All other blocks are DenseTransformerBlocks.
           - Specifically: block i is MoE if (i + 1) % moe_every_n_layers == 0.
        3. Create a final LayerNorm and output head (Linear to vocab_size).
        4. Store the config and aux_loss_weight.

        Args:
            config: MoE model configuration.

        HINT: The pattern (i+1) % moe_every_n_layers == 0 makes layers
              1, 3, 5, ... into MoE blocks (0-indexed), which means the
              second, fourth, sixth, ... blocks use MoE. This is the Mixtral
              pattern where MoE blocks are interleaved with dense blocks.
        """
        super().__init__()
        # TODO: Implement
        # Step 1: Token and position embeddings
        # Step 2: Build block list with interleaved MoE and dense blocks
        # Step 3: Final LayerNorm and output head
        # Step 4: Store config
        raise NotImplementedError("TODO: Implement MoEGPT.__init__")

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the MoE GPT model.

        TODO: Implement this method

        Requirements:
        1. Compute token + position embeddings.
        2. Pass through all transformer blocks, collecting router_probs
           from MoE blocks.
        3. Apply final LayerNorm and output head.
        4. If targets are provided:
           a. Compute cross-entropy loss.
           b. Compute auxiliary load balancing loss from all MoE blocks.
           c. Total loss = CE loss + aux_loss_weight * aux_loss.
        5. Return (logits, loss).

        Args:
            idx: Input token IDs, shape (batch, seq_len).
            targets: Target token IDs, shape (batch, seq_len). Optional.

        Returns:
            Tuple of (logits, loss) where loss is None if targets not provided.

        HINT: Collect router_probs from each MoE block's forward pass.
              After the forward pass, compute aux_loss as the mean of
              load_balancing_loss over all MoE blocks.

        HINT: For load_balancing_loss, you need the top_k_indices too.
              The Router returns them, so you may need to modify MoELayer
              to also return them, or re-derive them from router_probs.
              A simpler approach: store them in the MoELayer during forward.
        """
        # TODO: Implement
        # Step 1: Embeddings
        # Step 2: Pass through blocks, collect router info from MoE blocks
        # Step 3: Final LayerNorm + head -> logits
        # Step 4: Compute loss if targets provided
        # Step 5: Return (logits, loss)
        raise NotImplementedError("TODO: Implement MoEGPT.forward")
