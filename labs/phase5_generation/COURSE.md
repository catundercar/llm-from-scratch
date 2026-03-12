# Phase 5: Text Generation

## Overview

Now that you have a pretrained GPT model, you need a way to generate text from
it. This phase covers several **decoding strategies** that control the
trade-off between quality (coherence) and diversity (creativity).

## Key Concepts

### Autoregressive Generation

GPT generates text one token at a time. At each step:
1. Feed the current sequence to the model.
2. Get the logits (unnormalized scores) for the next token.
3. Choose the next token using some strategy.
4. Append it to the sequence and repeat.

### Greedy Decoding

Always pick the token with the highest probability:

    next_token = argmax(logits)

Pros: Deterministic, fast.
Cons: Can produce repetitive, boring text. Gets stuck in loops.

### Temperature Sampling

Scale logits before applying softmax to control randomness:

    probs = softmax(logits / temperature)
    next_token = sample(probs)

- **temperature = 1.0**: Standard sampling.
- **temperature < 1.0**: Sharper distribution, more deterministic.
- **temperature > 1.0**: Flatter distribution, more random.
- **temperature -> 0**: Equivalent to greedy decoding.

### Top-k Sampling

Restrict sampling to the k most probable tokens:

1. Find the top-k logits.
2. Set all other logits to -infinity.
3. Apply softmax and sample.

This prevents sampling from the long tail of unlikely tokens while
maintaining diversity among the top candidates.

### Top-p (Nucleus) Sampling

Dynamically select the smallest set of tokens whose cumulative probability
exceeds a threshold p:

1. Sort tokens by probability (descending).
2. Compute cumulative probabilities.
3. Keep tokens until cumulative probability exceeds p.
4. Sample from this dynamic set.

Top-p adapts the candidate set size based on the distribution shape:
- When the model is confident (peaked distribution), fewer tokens are kept.
- When the model is uncertain (flat distribution), more tokens are kept.

## Lab Structure

| File | What you implement |
|---|---|
| `phase_5/generate.py` | `greedy_decode`, `temperature_sample`, `top_k_sample`, `top_p_sample`, `generate` |

## Running Tests

```bash
python -m pytest tests/ -v
python scripts/grade.py
```

## Tips

- Start with `greedy_decode` -- it is the simplest strategy.
- Once greedy works, `temperature_sample` is a small modification (divide
  logits by temperature, then sample instead of argmax).
- `top_k_sample` builds on temperature sampling by masking logits.
- `top_p_sample` is the most complex -- pay careful attention to the
  cumulative probability mask and the index scatter operation.
- Always use `torch.no_grad()` during generation for efficiency.
