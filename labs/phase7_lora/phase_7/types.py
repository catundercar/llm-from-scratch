"""
Type definitions for Phase 7: LoRA.

These types are PROVIDED -- students do not need to modify this file.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    rank: int = 4
    alpha: float = 1.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    dropout: float = 0.0


@dataclass
class LoRATrainConfig:
    """Configuration for LoRA training."""
    epochs: int = 3
    lr: float = 1e-4
    batch_size: int = 8
    max_length: int = 256
    device: str = "cpu"
    save_path: Optional[str] = None
    log_interval: int = 10


@dataclass
class GPTConfig:
    """
    Minimal GPT configuration for use in Phase 7+ labs.

    This mirrors the GPTConfig from Phase 3. If you have completed Phase 3,
    you can import from there instead.
    """
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.1
    bias: bool = False
