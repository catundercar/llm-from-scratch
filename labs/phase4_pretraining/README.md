# Phase 4: Pretraining

**Difficulty: Intermediate-Advanced**

Implement the complete pretraining pipeline for a GPT language model.

## Labs

| Lab | File | Difficulty | Description |
|-----|------|-----------|-------------|
| 1 | `phase_4/utils.py` | *** | Learning rate schedule, loss estimation, checkpoints |
| 2 | `phase_4/train.py` | **** | Full training loop with LR scheduling and gradient clipping |

## Prerequisites

- Phase 1 (Tokenizer + DataLoader)
- Phase 3 (GPT model)
- Understanding of gradient descent and backpropagation

## Quick Start

```bash
# Install dependencies
pip install torch pytest

# Run tests
python -m pytest tests/ -v

# Grade your work
python scripts/grade.py
```

## What You Will Learn

1. How to implement a cosine learning rate schedule with linear warmup
2. How to write a robust training loop with gradient clipping
3. How to save and load training checkpoints
4. How to estimate validation loss during training
