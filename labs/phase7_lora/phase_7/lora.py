"""
Phase 7, Lab 1: LoRA (Low-Rank Adaptation)

This module implements LoRA, a parameter-efficient fine-tuning method that
freezes the pretrained model weights and injects trainable low-rank
decomposition matrices into each target layer.

Key idea: Instead of fine-tuning W directly, we learn a low-rank update
    W' = W + (alpha/r) * B @ A
where A is (in_features x rank) and B is (rank x out_features), with
rank << min(in_features, out_features).

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)

Students implement:
    - LoRALinear: A drop-in replacement for nn.Linear with LoRA parameters
    - apply_lora_to_model: Inject LoRA into specific layers of a model
    - merge_lora_weights: Fold LoRA weights back into the base model
    - count_trainable_params: Report parameter efficiency
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class LoRALinear(nn.Module):
    """
    A Linear layer augmented with LoRA (Low-Rank Adaptation).

    This replaces a standard nn.Linear layer. The original weight W is frozen,
    and two low-rank matrices A and B are trained such that the effective
    weight becomes W + (alpha/rank) * B @ A.

    Attributes:
        weight: The original (frozen) weight matrix from the pretrained model.
        bias: The original (frozen) bias, if any.
        lora_A: Low-rank matrix A of shape (in_features, rank).
        lora_B: Low-rank matrix B of shape (rank, out_features).
        scaling: The scaling factor alpha / rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        bias: bool = True,
    ) -> None:
        """
        Initialize LoRALinear.

        TODO: Implement this method

        Requirements:
        1. Create a frozen linear weight W of shape (out_features, in_features)
           and optional bias -- these represent the pretrained weights.
           Set requires_grad=False on W and bias.
        2. Create lora_A as a Parameter of shape (in_features, rank),
           initialized with Kaiming uniform (fan_out mode).
        3. Create lora_B as a Parameter of shape (rank, out_features),
           initialized to zeros.
        4. Store the scaling factor alpha / rank.

        HINT: Use nn.Parameter for lora_A and lora_B so they are registered
              as trainable parameters. Use a plain tensor (or Parameter with
              requires_grad=False) for the frozen weight.

        HINT: nn.init.kaiming_uniform_(tensor, a=math.sqrt(5)) matches the
              default Linear initialization.

        HINT: B is initialized to zeros so that the LoRA contribution starts
              at zero -- the model behaves identically to the original at init.
        """
        super().__init__()
        # TODO: Implement
        # Step 1: Create frozen weight (and optional bias)
        # Step 2: Create lora_A with Kaiming uniform init
        # Step 3: Create lora_B initialized to zeros
        # Step 4: Compute and store scaling = alpha / rank
        raise NotImplementedError("TODO: Implement LoRALinear.__init__")

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, rank: int = 4, alpha: float = 1.0
    ) -> "LoRALinear":
        """
        Create a LoRALinear from an existing nn.Linear layer.

        This is a convenience method used by apply_lora_to_model. It copies
        the pretrained weights into the new LoRALinear layer.

        This method is PROVIDED -- students do not need to modify it,
        but it depends on __init__ being correctly implemented.
        """
        has_bias = linear.bias is not None
        lora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=has_bias,
        )
        # Copy pretrained weights (frozen)
        lora.weight.data.copy_(linear.weight.data)
        if has_bias and linear.bias is not None:
            lora.bias.data.copy_(linear.bias.data)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.

        TODO: Implement this method

        Requirements:
        1. Compute the base linear output: x @ W^T + bias
        2. Compute the LoRA update: x @ A @ B^T * scaling
           (Note: W is stored as (out_features, in_features) per PyTorch convention,
            so the base output is F.linear(x, W, bias).)
        3. Return base_output + lora_output

        HINT: Use torch.nn.functional.linear(x, self.weight, self.bias) for the
              base computation to handle the transpose correctly.

        HINT: The LoRA path is: x @ self.lora_A @ self.lora_B.T * self.scaling
              This works because A is (in, rank) and B is (rank, out), so
              x @ A gives (batch, seq, rank) and then @ B.T gives (batch, seq, out).
        """
        # TODO: Implement
        # Step 1: Compute base output using frozen weights
        # Step 2: Compute LoRA delta: x @ A @ B^T * scaling
        # Step 3: Return sum
        raise NotImplementedError("TODO: Implement LoRALinear.forward")


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """
    Replace specified Linear layers in a model with LoRALinear layers.

    TODO: Implement this function

    Requirements:
    1. Walk through all named modules in the model.
    2. For each module whose name matches one of target_modules, replace
       the nn.Linear layer with a LoRALinear created via from_linear().
    3. Freeze ALL original model parameters (requires_grad = False).
    4. Only LoRA parameters (lora_A, lora_B) should remain trainable.
    5. Return the modified model.

    Args:
        model: The pretrained model to adapt.
        rank: LoRA rank (controls parameter count vs expressiveness).
        alpha: LoRA scaling factor.
        target_modules: List of module name suffixes to target
                       (e.g., ["q_proj", "v_proj"]).
                       Defaults to ["q_proj", "v_proj"] if None.

    Returns:
        The model with LoRA layers injected (modified in-place).

    HINT: Use model.named_modules() to iterate. For each (name, module),
          check if the name ends with any of the target_modules strings.

    HINT: To replace a submodule, you need to find its parent and use setattr.
          If name is "blocks.0.attn.q_proj", the parent is accessed as
          "blocks.0.attn" and the attribute is "q_proj".

    HINT: First freeze everything, then create LoRA layers (which will have
          trainable A and B by default).
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # TODO: Implement
    # Step 1: Freeze all existing parameters
    # Step 2: Collect modules to replace (don't modify dict during iteration)
    # Step 3: For each target module, replace with LoRALinear.from_linear()
    # Step 4: Return modified model
    raise NotImplementedError("TODO: Implement apply_lora_to_model")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights back into the base model for efficient deployment.

    After merging, the model has no LoRA parameters -- each LoRALinear is
    replaced with a standard nn.Linear whose weight is W + (alpha/r) * B @ A.

    TODO: Implement this function

    Requirements:
    1. Find all LoRALinear modules in the model.
    2. For each, compute the merged weight: W + scaling * (B.T @ A.T)^T
       which simplifies to: W + scaling * A @ B  ... wait, let's be precise:
       W is (out, in). LoRA adds scaling * (B @ A)^T to it.
       Actually: W_new = W + scaling * (lora_B.T @ lora_A.T)
       where lora_A is (in, rank) and lora_B is (rank, out).
       So lora_B.T is (out, rank) and lora_A.T is (rank, in).
       Result: (out, rank) @ (rank, in) = (out, in). Correct!
    3. Create a new nn.Linear with the merged weight and set it in the model.
    4. Return the modified model.

    HINT: merged_weight = lora_layer.weight + lora_layer.scaling * (lora_layer.lora_B.T @ lora_layer.lora_A.T)

    HINT: Use the same parent-module setattr pattern as apply_lora_to_model.
    """
    # TODO: Implement
    # Step 1: Find all LoRALinear modules
    # Step 2: For each, compute merged weight
    # Step 3: Create nn.Linear with merged weight (and bias if present)
    # Step 4: Replace LoRALinear with merged nn.Linear in model
    # Step 5: Return model
    raise NotImplementedError("TODO: Implement merge_lora_weights")


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """
    Count trainable and total parameters in a model.

    TODO: Implement this function

    Requirements:
    1. Count total parameters (all p.numel() for p in model.parameters()).
    2. Count trainable parameters (only those with requires_grad=True).
    3. Print a summary: total, trainable, percentage, and frozen count.
    4. Return (trainable_count, total_count).

    HINT: Use model.parameters() and check p.requires_grad for each.
    """
    # TODO: Implement
    # Step 1: Sum all parameter counts
    # Step 2: Sum only trainable parameter counts
    # Step 3: Print summary
    # Step 4: Return (trainable, total)
    raise NotImplementedError("TODO: Implement count_trainable_params")
