# Phase 9: Mixture of Experts (MoE)

## Overview

Scaling language models improves performance, but dense models require
activating all parameters for every token. Mixture of Experts (MoE) breaks
this constraint: the model has many parameters in total, but only a subset
(the "active" parameters) are used for any given token.

This enables models with 8x more parameters at roughly 2x the compute cost.

## Architecture

### Expert

Each expert is a standard feed-forward network (FFN):

    Expert(x) = Linear2(GELU(Linear1(x)))

where Linear1 expands the dimension (d_model -> d_ff) and Linear2 projects
back (d_ff -> d_model). This is identical to the FFN in a standard transformer.

### Router (Gating Network)

The router decides which experts process each token:

    gate_logits = Linear(x)        # shape: (n_tokens, n_experts)
    probs = softmax(gate_logits)   # probability distribution over experts
    top_k_probs, top_k_idx = topk(probs, k)
    weights = top_k_probs / sum(top_k_probs)  # renormalize

Each token is sent to its top-k experts (typically k=2).

### MoE Layer

The MoE layer replaces the dense FFN in a transformer block:

    For each token x_i:
        experts_i = Router.select_top_k(x_i)
        output_i = sum(weight_j * Expert_j(x_i) for j in experts_i)

The output is a weighted combination of the selected experts' outputs.

### Interleaved Architecture

Modern MoE models (like Mixtral) do not make every layer an MoE layer.
Instead, they interleave MoE and dense layers:

    Layer 0: Dense FFN
    Layer 1: MoE (8 experts, top-2)
    Layer 2: Dense FFN
    Layer 3: MoE (8 experts, top-2)
    ...

This provides a good balance between specialized (MoE) and shared (dense)
computation.

## Load Balancing

### The Collapse Problem

Without intervention, routers tend to collapse: they learn to send all tokens
to one or two "favorite" experts, leaving others unused. This wastes capacity.

### Auxiliary Loss

The load balancing loss encourages uniform expert utilization:

    L_aux = N * sum_i(f_i * p_i)

where:
- N = number of experts
- f_i = fraction of tokens routed to expert i
- p_i = mean routing probability assigned to expert i

This loss is minimized when both f_i and p_i are uniform (= 1/N for fractions,
accounting for top-k).

The auxiliary loss is added to the main cross-entropy loss with a small weight
(typically 0.01):

    L_total = L_CE + alpha * L_aux

## Parameter Counting

For a model with d_model=4096, d_ff=16384, 8 experts, top-2:

- Dense FFN: 2 * 4096 * 16384 = 134M params (all active)
- MoE layer: 8 * 134M = 1.07B params total, but only 2/8 = 268M active

The MoE model has 8x more parameters but only 2x the compute per token.

## References

- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer" (2017)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models
  with Simple and Efficient Sparsity" (2022)
- Jiang et al., "Mixtral of Experts" (2024)
