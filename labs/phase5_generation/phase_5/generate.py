"""
Phase 5 - Text Generation Strategies
======================================

This module implements several decoding strategies for autoregressive text
generation with a GPT model:

- **Greedy decoding**: Always pick the most probable next token.
- **Temperature sampling**: Scale logits to control randomness.
- **Top-k sampling**: Restrict sampling to the k most probable tokens.
- **Top-p (nucleus) sampling**: Restrict sampling to the smallest set of
  tokens whose cumulative probability exceeds p.
- **generate()**: Unified interface that ties encoding/decoding to strategies.

Dependencies:
- Phase 3: GPT model (for the model's forward pass)
- Phase 1: Tokenizer (for encode/decode in the generate() wrapper)
"""

import sys
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Cross-phase imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Greedy Decoding
# ---------------------------------------------------------------------------

def greedy_decode(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate tokens by always choosing the most probable next token.

    At each step:
    1. Feed the current sequence to the model to get logits.
    2. Take the logits for the last position.
    3. Pick the token with the highest logit (argmax).
    4. Append it to the sequence and repeat.

    TODO: Implement this function

    Requirements:
    1. Input idx has shape (1, T) -- a single sequence of token IDs.
    2. Generate exactly max_new_tokens new tokens.
    3. At each step, feed the full sequence (or truncated to block_size
       if the model has a block_size attribute) to the model.
    4. Use torch.argmax on the last-position logits to pick the next token.
    5. Concatenate the new token to idx along dimension 1.
    6. Return the full sequence (original + generated) of shape (1, T + max_new_tokens).

    HINT: The model returns (logits, loss). You only need logits here.
    logits has shape (B, T, vocab_size). You want logits[:, -1, :] for the
    last position.

    HINT: Use model.eval() and torch.no_grad() for inference efficiency.

    Args:
        model: The GPT model.
        idx: Starting token IDs, shape (1, T).
        max_new_tokens: Number of tokens to generate.
        device: Device to run on.

    Returns:
        Token IDs tensor of shape (1, T + max_new_tokens).
    """
    # TODO: Implement greedy decoding
    # Step 1: Move idx to device, set model to eval mode
    # Step 2: For each new token:
    #   a. Truncate idx to block_size if needed
    #   b. Forward pass to get logits
    #   c. Get logits for the last position: logits[:, -1, :]
    #   d. Take argmax to get next token
    #   e. Append to idx
    # Step 3: Return the full sequence
    raise NotImplementedError("TODO: Implement greedy_decode")


# ---------------------------------------------------------------------------
# Temperature Sampling
# ---------------------------------------------------------------------------

def temperature_sample(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate tokens by sampling from a temperature-scaled distribution.

    Temperature controls the "sharpness" of the probability distribution:
    - temperature < 1.0: sharper, more deterministic (peaks get amplified)
    - temperature = 1.0: standard softmax sampling
    - temperature > 1.0: flatter, more random (uniform-like)

    TODO: Implement this function

    Requirements:
    1. At each step, get logits for the last position.
    2. Divide logits by temperature: scaled_logits = logits / temperature
    3. Apply softmax to get probabilities.
    4. Sample from the distribution using torch.multinomial.
    5. If temperature is 0 (or very close), fall back to greedy (argmax).

    HINT: torch.multinomial(probs, num_samples=1) samples one token index
    from the probability distribution.

    HINT: Handle temperature=0 as a special case equivalent to greedy
    decoding. Check if temperature < 1e-8 to avoid division by zero.

    Args:
        model: The GPT model.
        idx: Starting token IDs, shape (1, T).
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature (> 0).
        device: Device to run on.

    Returns:
        Token IDs tensor of shape (1, T + max_new_tokens).
    """
    # TODO: Implement temperature sampling
    # Step 1: Move idx to device, set model to eval
    # Step 2: For each new token:
    #   a. Forward pass -> logits[:, -1, :]
    #   b. If temperature ~= 0, use argmax (greedy fallback)
    #   c. Otherwise: divide logits by temperature
    #   d. Apply softmax to get probabilities
    #   e. Sample with torch.multinomial(probs, num_samples=1)
    #   f. Append sampled token to idx
    # Step 3: Return full sequence
    raise NotImplementedError("TODO: Implement temperature_sample")


# ---------------------------------------------------------------------------
# Top-k Sampling
# ---------------------------------------------------------------------------

def top_k_sample(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    k: int = 50,
    temperature: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate tokens by sampling from the top-k most probable tokens.

    Top-k sampling restricts the candidate set to the k tokens with the
    highest logits, then samples from that restricted distribution. This
    prevents sampling rare, low-quality tokens while maintaining diversity.

    TODO: Implement this function

    Requirements:
    1. At each step, get logits for the last position.
    2. Apply temperature scaling.
    3. Find the top-k logit values and set all others to -infinity.
    4. Apply softmax and sample with torch.multinomial.
    5. k should be clamped to at most vocab_size.

    HINT: Use torch.topk(logits, k) to get the top-k values and indices.
    Then create a mask: set logits below the k-th largest value to -inf.

    HINT: A simpler approach: use torch.topk to get the k-th value, then
    logits[logits < top_k_value] = float('-inf')

    Args:
        model: The GPT model.
        idx: Starting token IDs, shape (1, T).
        max_new_tokens: Number of tokens to generate.
        k: Number of top tokens to consider.
        temperature: Sampling temperature.
        device: Device to run on.

    Returns:
        Token IDs tensor of shape (1, T + max_new_tokens).
    """
    # TODO: Implement top-k sampling
    # Step 1: Move idx to device, set model to eval
    # Step 2: For each new token:
    #   a. Forward pass -> logits[:, -1, :]
    #   b. Divide logits by temperature
    #   c. Find the k-th largest value using torch.topk
    #   d. Set all logits below that threshold to -inf
    #   e. Apply softmax to get probabilities
    #   f. Sample with torch.multinomial
    #   g. Append sampled token to idx
    # Step 3: Return full sequence
    raise NotImplementedError("TODO: Implement top_k_sample")


# ---------------------------------------------------------------------------
# Top-p (Nucleus) Sampling
# ---------------------------------------------------------------------------

def top_p_sample(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    p: float = 0.9,
    temperature: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate tokens using nucleus (top-p) sampling.

    Top-p sampling dynamically selects the smallest set of tokens whose
    cumulative probability mass exceeds the threshold p. Unlike top-k,
    the number of candidate tokens varies per step based on the
    distribution shape.

    TODO: Implement this function

    Requirements:
    1. At each step, get logits for the last position.
    2. Apply temperature scaling.
    3. Sort logits in descending order.
    4. Compute cumulative softmax probabilities of the sorted logits.
    5. Create a mask for tokens where cumulative probability exceeds p.
    6. Set masked logits to -infinity (keep the nucleus).
    7. Sample from the resulting distribution.

    HINT: The tricky part is the cumulative probability mask. After sorting
    logits descending and computing cumulative_sum of softmax:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    Then mask where cumulative_probs > p, BUT shift the mask right by 1 so
    you always keep at least the top token.

    HINT: After masking in sorted order, you need to scatter the -inf values
    back to the original positions. Use sorted_indices to do this:
        sorted_logits[mask] = float('-inf')
        logits.scatter_(1, sorted_indices, sorted_logits)

    Args:
        model: The GPT model.
        idx: Starting token IDs, shape (1, T).
        max_new_tokens: Number of tokens to generate.
        p: Cumulative probability threshold (0 < p <= 1).
        temperature: Sampling temperature.
        device: Device to run on.

    Returns:
        Token IDs tensor of shape (1, T + max_new_tokens).
    """
    # TODO: Implement top-p (nucleus) sampling
    # Step 1: Move idx to device, set model to eval
    # Step 2: For each new token:
    #   a. Forward pass -> logits[:, -1, :]
    #   b. Divide logits by temperature
    #   c. Sort logits descending
    #   d. Compute cumulative softmax probabilities
    #   e. Create mask: cumulative_probs > p (shifted right by 1)
    #   f. Set masked sorted_logits to -inf
    #   g. Scatter back to original positions
    #   h. Apply softmax and sample
    #   i. Append sampled token to idx
    # Step 3: Return full sequence
    raise NotImplementedError("TODO: Implement top_p_sample")


# ---------------------------------------------------------------------------
# Unified Generate Interface
# ---------------------------------------------------------------------------

def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    strategy: Literal["greedy", "temperature", "top_k", "top_p"] = "greedy",
    device: str = "cpu",
    **kwargs,
) -> str:
    """Unified text generation interface.

    Encodes the prompt, runs the chosen decoding strategy, and decodes
    the result back to text.

    TODO: Implement this function

    Requirements:
    1. Encode the prompt string to token IDs using tokenizer.encode().
    2. Convert to a tensor of shape (1, T) on the correct device.
    3. Call the appropriate strategy function based on the `strategy` arg.
    4. Decode the output token IDs back to text using tokenizer.decode().
    5. Return the decoded text (full sequence including prompt).
    6. Pass **kwargs through to the strategy function (e.g., temperature, k, p).

    HINT: Use a dictionary to map strategy names to functions:
    strategies = {"greedy": greedy_decode, "temperature": temperature_sample, ...}

    Args:
        model: The GPT model.
        tokenizer: Tokenizer with encode() and decode() methods.
        prompt: Input text string.
        max_new_tokens: Number of new tokens to generate.
        strategy: Decoding strategy name.
        device: Device to run on.
        **kwargs: Additional arguments passed to the strategy function
                  (e.g., temperature=0.8, k=50, p=0.9).

    Returns:
        The generated text as a string.
    """
    # TODO: Implement unified generate interface
    # Step 1: Encode prompt to token IDs
    # Step 2: Convert to tensor of shape (1, T)
    # Step 3: Dispatch to the correct strategy function
    # Step 4: Decode output token IDs to text
    # Step 5: Return the generated text
    raise NotImplementedError("TODO: Implement generate")
