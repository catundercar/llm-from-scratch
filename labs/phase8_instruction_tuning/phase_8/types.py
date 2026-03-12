"""
Type definitions for Phase 8: Instruction Fine-Tuning.

These types are PROVIDED -- students do not need to modify this file.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning."""
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 4
    max_length: int = 512
    device: str = "cpu"
    save_path: Optional[str] = None
    log_interval: int = 10


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""
    epochs: int = 1
    lr: float = 1e-6
    beta: float = 0.1
    batch_size: int = 2
    max_length: int = 512
    device: str = "cpu"
    save_path: Optional[str] = None
    log_interval: int = 10


@dataclass
class InstructionSample:
    """A single instruction-tuning sample."""
    instruction: str
    input: str  # can be empty string
    output: str


@dataclass
class PreferenceSample:
    """A single preference pair for DPO."""
    prompt: str
    chosen: str
    rejected: str
