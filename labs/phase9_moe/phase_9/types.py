"""
Type definitions for Phase 9: Mixture of Experts.

These types are PROVIDED -- students do not need to modify this file.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MoEConfig:
    """Configuration for a Mixture of Experts model."""
    # Base transformer config
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.1
    bias: bool = False

    # MoE-specific config
    n_experts: int = 8
    top_k: int = 2
    moe_every_n_layers: int = 2
    aux_loss_weight: float = 0.01
