# Phase 4: Pretraining

## Overview

In this phase you will implement the training infrastructure needed to pretrain
a GPT model from scratch. You already have the model architecture (Phase 3) and
the data pipeline (Phase 1). Now you need the training loop, learning rate
scheduling, and checkpoint management.

## Key Concepts

### Cross-Entropy Loss

Language model pretraining uses **next-token prediction**. Given a sequence of
tokens `[t_0, t_1, ..., t_n]`, the model predicts the probability distribution
over the vocabulary for each position, and the loss measures how well those
predictions match the actual next tokens.

Cross-entropy loss:

    L = -(1/N) * sum( log P(t_{i+1} | t_0, ..., t_i) )

PyTorch computes this via `F.cross_entropy(logits, targets)` where logits have
shape `(B*T, vocab_size)` and targets have shape `(B*T,)`.

### Learning Rate Schedule

Modern LLM training uses a **cosine learning rate schedule with linear warmup**:

1. **Warmup phase**: LR increases linearly from 0 to `max_lr` over the first
   `warmup_steps` steps. This prevents large, destabilizing gradient updates
   early in training when the model weights are random.

2. **Cosine decay phase**: LR decreases following a cosine curve from `max_lr`
   to `min_lr`. The formula is:

       lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))

   where `progress` ranges from 0 to 1 over the decay phase.

### Gradient Clipping

Gradient clipping prevents exploding gradients by scaling the gradient vector
if its norm exceeds a threshold:

    if ||g|| > max_norm:
        g = g * (max_norm / ||g||)

PyTorch provides `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`.

### AdamW Optimizer

AdamW is the standard optimizer for transformer training. It combines:
- Adaptive learning rates per parameter (Adam)
- Decoupled weight decay (the "W" in AdamW)

Use `torch.optim.AdamW(model.parameters(), lr=learning_rate)`.

### Checkpointing

Save training state periodically so you can resume after interruption.
A checkpoint typically includes:
- Model weights (`model.state_dict()`)
- Optimizer state (`optimizer.state_dict()`)
- Current step number
- Current loss value

## Lab Structure

| File | What you implement |
|---|---|
| `phase_4/utils.py` | `get_lr`, `estimate_loss`, `save_checkpoint`, `load_checkpoint` |
| `phase_4/train.py` | `TrainConfig`, `train()`, `main()` |

## Running Tests

```bash
# Run all Phase 4 tests
python -m pytest tests/ -v

# Run individual labs
python -m pytest tests/test_lab1_utils.py -v
python -m pytest tests/test_lab2_train.py -v

# Auto-grade
python scripts/grade.py
```

## Tips

- Start with `utils.py` (Lab 1) since `train.py` depends on it.
- Test `get_lr` by plotting the schedule: you should see a linear ramp
  followed by a smooth cosine curve.
- For the training loop, use `itertools.cycle(dataloader)` to loop over
  the dataset indefinitely.
- The training loop does not need to converge -- the tests only check
  that it runs correctly and updates weights.
