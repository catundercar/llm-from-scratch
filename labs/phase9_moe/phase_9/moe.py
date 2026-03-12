"""
Phase 9, Lab 1: Mixture of Experts Layer

This module implements the core MoE components:
    - Expert: A single feed-forward network (same architecture as a standard FFN)
    - Router: A gating network that selects which experts process each token
    - MoELayer: Combines routing with expert computation
    - load_balancing_loss: Auxiliary loss for balanced expert utilization

Key idea: Instead of one large FFN, use N smaller expert FFNs and route each
token to the top-k experts. This allows the model to have many more parameters
(total) while only activating a fraction of them for each token (active params).

Reference: Shazeer et al., "Outrageously Large Neural Networks" (2017)
           Fedus et al., "Switch Transformers" (2022)
           Jiang et al., "Mixtral of Experts" (2024)

Students implement:
    - Expert: single FFN module
    - Router: gating network with top-k selection
    - MoELayer: full MoE forward pass
    - load_balancing_loss: auxiliary loss for balanced routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Expert(nn.Module):
    """
    A single expert network.

    This is identical to a standard feed-forward network (FFN) used in
    transformers: Linear -> GELU -> Linear, with an expansion factor of 4x.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        """
        Initialize a single expert.

        TODO: Implement this method

        Requirements:
        1. Create a feed-forward network: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)
        2. This is the same architecture as a standard transformer FFN.

        Args:
            d_model: Model dimension (input and output size).
            d_ff: Hidden dimension of the FFN (typically 4 * d_model).

        HINT: Use nn.Sequential for a clean implementation.
        """
        super().__init__()
        # TODO: Implement
        # Step 1: self.net = nn.Sequential(
        #     nn.Linear(d_model, d_ff),
        #     nn.GELU(),
        #     nn.Linear(d_ff, d_model),
        # )
        raise NotImplementedError("TODO: Implement Expert.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert.

        TODO: Implement this method

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of same shape as input.
        """
        # TODO: Implement
        raise NotImplementedError("TODO: Implement Expert.forward")


class Router(nn.Module):
    """
    Gating network that routes tokens to experts.

    The router computes a probability distribution over experts for each token,
    then selects the top-k experts with the highest probabilities.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2) -> None:
        """
        Initialize the router.

        TODO: Implement this method

        Requirements:
        1. Create a linear layer: Linear(d_model, n_experts) -- the gate.
        2. Store top_k and n_experts.

        Args:
            d_model: Model dimension.
            n_experts: Number of experts to route to.
            top_k: Number of experts to select per token.

        HINT: The gate produces logits over experts, which are converted to
              probabilities via softmax in the forward pass.
        """
        super().__init__()
        # TODO: Implement
        # Step 1: self.gate = nn.Linear(d_model, n_experts)
        # Step 2: Store top_k, n_experts
        raise NotImplementedError("TODO: Implement Router.__init__")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        TODO: Implement this method

        Requirements:
        1. Compute gate logits: x @ gate_weight (via self.gate(x)).
        2. Apply softmax to get routing probabilities over experts.
        3. Select top-k experts per token using torch.topk.
        4. Return:
           - router_probs: full probability distribution, shape (batch*seq, n_experts)
           - top_k_weights: probabilities of selected experts, shape (batch*seq, top_k)
           - top_k_indices: indices of selected experts, shape (batch*seq, top_k)

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tuple of (router_probs, top_k_weights, top_k_indices).

        HINT: Reshape x to (batch*seq_len, d_model) before routing, then
              reshape back as needed.

        HINT: Normalize top_k_weights so they sum to 1 per token:
              top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        """
        # TODO: Implement
        # Step 1: Reshape x to (batch*seq, d_model)
        # Step 2: Compute gate logits and softmax
        # Step 3: Select top-k experts
        # Step 4: Normalize top-k weights
        # Step 5: Return (router_probs, top_k_weights, top_k_indices)
        raise NotImplementedError("TODO: Implement Router.forward")


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.

    Replaces the standard FFN in a transformer block. Each token is routed
    to the top-k experts, and their outputs are combined using the routing
    weights.
    """

    def __init__(
        self, d_model: int, d_ff: int, n_experts: int, top_k: int = 2
    ) -> None:
        """
        Initialize the MoE layer.

        TODO: Implement this method

        Requirements:
        1. Create n_experts Expert modules.
        2. Create one Router module.
        3. Store d_model, n_experts, top_k.

        Args:
            d_model: Model dimension.
            d_ff: Hidden dimension for each expert FFN.
            n_experts: Number of experts.
            top_k: Number of experts per token.

        HINT: Use nn.ModuleList to store experts so they are registered
              as submodules.
        """
        super().__init__()
        # TODO: Implement
        # Step 1: self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        # Step 2: self.router = Router(d_model, n_experts, top_k)
        # Step 3: Store config values
        raise NotImplementedError("TODO: Implement MoELayer.__init__")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MoE layer.

        TODO: Implement this method

        Requirements:
        1. Get routing decisions from the router.
        2. Initialize output tensor of same shape as input.
        3. For each expert, find which tokens are routed to it.
        4. Process those tokens through the expert.
        5. Weight the expert output by the routing weight and accumulate.
        6. Return the combined output and the router probabilities
           (for load balancing loss computation).

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tuple of:
            - output: Combined expert outputs, shape (batch, seq_len, d_model).
            - router_probs: Routing probabilities for load balancing loss.

        HINT: The basic approach is:
              output = zeros_like(x_flat)
              for expert_idx in range(n_experts):
                  # Find tokens routed to this expert
                  mask = (top_k_indices == expert_idx).any(dim=-1)
                  if mask.any():
                      expert_input = x_flat[mask]
                      expert_output = self.experts[expert_idx](expert_input)
                      # Get the weight for this expert for these tokens
                      # Accumulate: output[mask] += weight * expert_output

        HINT: For each token routed to expert_idx, find which of its top_k slots
              selected that expert to get the correct weight:
              weight_mask = (top_k_indices[mask] == expert_idx)
              weights = (top_k_weights[mask] * weight_mask.float()).sum(dim=-1)
        """
        # TODO: Implement
        # Step 1: Get routing: router_probs, top_k_weights, top_k_indices
        # Step 2: Flatten x to (batch*seq, d_model)
        # Step 3: Initialize output tensor
        # Step 4: For each expert, process routed tokens
        # Step 5: Reshape output back to (batch, seq, d_model)
        # Step 6: Return (output, router_probs)
        raise NotImplementedError("TODO: Implement MoELayer.forward")


def load_balancing_loss(
    router_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """
    Compute auxiliary load balancing loss.

    Without this loss, the router tends to collapse: it sends all tokens to
    one or two experts, leaving the rest unused. The auxiliary loss encourages
    uniform expert utilization.

    TODO: Implement this function

    The load balancing loss is:
        L_aux = n_experts * sum_i(f_i * p_i)

    where:
        f_i = fraction of tokens routed to expert i
        p_i = mean routing probability for expert i (across all tokens)

    This loss is minimized when both f_i and p_i are uniform (= 1/n_experts).

    Requirements:
    1. Compute f_i: for each expert, count how many tokens have it in their
       top-k selection, divided by total number of tokens.
    2. Compute p_i: for each expert, compute the mean routing probability
       across all tokens.
    3. Compute L_aux = n_experts * sum(f_i * p_i).

    Args:
        router_probs: Full routing probability distribution,
                     shape (n_tokens, n_experts).
        top_k_indices: Indices of selected experts per token,
                      shape (n_tokens, top_k).
        n_experts: Total number of experts.

    Returns:
        Scalar auxiliary loss.

    HINT: f_i can be computed using one-hot encoding:
          expert_mask = F.one_hot(top_k_indices, n_experts).sum(dim=1)  # (n_tokens, n_experts)
          f = expert_mask.float().mean(dim=0)  # (n_experts,)

    HINT: p_i = router_probs.mean(dim=0)  # (n_experts,)

    HINT: The loss is minimized (equal to 1/n_experts * top_k) when routing
          is perfectly uniform.
    """
    # TODO: Implement
    # Step 1: Compute fraction of tokens per expert (f_i)
    # Step 2: Compute mean routing probability per expert (p_i)
    # Step 3: Return n_experts * sum(f_i * p_i)
    raise NotImplementedError("TODO: Implement load_balancing_loss")
