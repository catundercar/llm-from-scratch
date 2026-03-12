# Phase 5: Text Generation

**Difficulty: Intermediate**

Implement multiple decoding strategies for autoregressive text generation.

## Labs

| Lab | File | Difficulty | Description |
|-----|------|-----------|-------------|
| 1 | `phase_5/generate.py` | *** | Greedy, temperature, top-k, and top-p sampling |

## Prerequisites

- Phase 3 (GPT model architecture)
- Phase 4 (Pretrained model weights)
- Understanding of softmax and probability distributions

## Quick Start

```bash
pip install torch pytest
python -m pytest tests/ -v
python scripts/grade.py
```

## What You Will Learn

1. How greedy decoding works and why it can produce repetitive text
2. How temperature controls the randomness of sampling
3. How top-k sampling restricts the candidate token set
4. How nucleus (top-p) sampling dynamically adjusts candidate set size
5. How to build a unified generation interface
