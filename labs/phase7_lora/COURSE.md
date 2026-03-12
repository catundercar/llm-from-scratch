# Phase 7: LoRA (Low-Rank Adaptation)

## Overview

Fine-tuning large language models is expensive. A 7B-parameter model requires
storing a full copy of all gradients and optimizer states. LoRA (Low-Rank
Adaptation) solves this by freezing the pretrained weights and injecting small,
trainable matrices into specific layers.

## Key Insight

Weight updates during fine-tuning have low intrinsic rank. Instead of learning
a full-rank update matrix dW of shape (d x d), we decompose it as:

    dW = (alpha / r) * B @ A

where A is (d_in x r), B is (r x d_out), and r << d. This reduces the number
of trainable parameters by orders of magnitude while maintaining performance
close to full fine-tuning.

## Mathematical Foundation

For a pretrained weight matrix W in R^{d_out x d_in}:

    h = W @ x                          (original)
    h = W @ x + (alpha/r) * B @ A @ x  (with LoRA)

- A in R^{d_in x r}: initialized with Kaiming uniform
- B in R^{r x d_out}: initialized to zeros
- alpha: scaling hyperparameter
- r: rank (typically 4-64)

Because B is initialized to zero, the LoRA contribution starts at zero. The
model begins training from the pretrained weights with no perturbation.

## Why Target q_proj and v_proj?

Empirically, applying LoRA to the query and value projection matrices in
attention gives the best efficiency/quality tradeoff. The key projection
benefits less because keys are used for matching (which the pretrained model
already does well), while query and value projections control *what to attend
to* and *what information to extract*, which are more task-specific.

## Parameter Efficiency

For a model with d=4096 and LoRA rank r=8:

- Full linear layer: 4096 x 4096 = 16.7M params
- LoRA matrices: (4096 x 8) + (8 x 4096) = 65.5K params
- Reduction: 99.6%

Across all attention layers in a 7B model, LoRA typically trains ~0.1-1% of
parameters while achieving 90-99% of full fine-tuning performance.

## Merging for Deployment

After training, the LoRA weights can be merged back into the base weights:

    W' = W + (alpha/r) * B^T @ A^T

(accounting for PyTorch's weight transposition convention)

This produces a standard model with no inference overhead -- the same
architecture, same speed, but updated weights.

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
