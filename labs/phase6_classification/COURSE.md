# Phase 6: Fine-Tuning for Classification

## Overview

You have a pretrained GPT model that understands language. Now you will adapt
it for a specific downstream task: **spam classification**. This is the core
idea behind transfer learning -- take a general-purpose model and specialize
it with a small amount of labeled data.

## Key Concepts

### Transfer Learning

Training a language model from scratch on billions of tokens is expensive.
Transfer learning reuses those learned representations:

1. **Pretraining**: Train a large model on a massive unlabeled corpus (Phase 4).
2. **Fine-tuning**: Adapt the pretrained model to a specific task with a small
   labeled dataset (this phase).

The pretrained model has already learned syntax, semantics, and world knowledge.
Fine-tuning only needs to teach it the task-specific mapping.

### Feature Extraction vs. Full Fine-Tuning

Two main approaches:

**Feature extraction** (freeze backbone):
- Freeze all pretrained weights.
- Only train the new classification head.
- Fast, requires less data, lower risk of overfitting.
- Works well when the pretrained model is large relative to the dataset.

**Full fine-tuning** (unfreeze backbone):
- Update all weights, including the pretrained backbone.
- Can achieve better performance but needs more data.
- Risk of catastrophic forgetting (losing pretrained knowledge).
- Usually uses a smaller learning rate for the backbone.

### Classification Head Design

For GPT (decoder-only), we use the **last token's hidden state** as the
sequence representation:

    input_ids -> GPT -> hidden_states (B, T, n_embd) -> last_token (B, n_embd) -> Linear -> (B, n_classes)

Why the last token? In a causal (left-to-right) model, the last token has
attended to all previous tokens, so its representation contains information
about the entire sequence.

### Attention Mask and Padding

When sequences have different lengths, we pad shorter ones with zeros. The
attention mask tells the model (and our classifier) which positions are real
tokens vs. padding:

    attention_mask = [1, 1, 1, 1, 1, 0, 0, 0]  # 5 real tokens, 3 padding

When extracting the "last token," we need the last **real** token, not the
last padding token.

### Early Stopping

To prevent overfitting, we monitor validation accuracy and stop training
when it stops improving:

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        stop_training()

## Lab Structure

| File | What you implement |
|---|---|
| `phase_6/dataset.py` | `SpamDataset`, `create_classification_dataloaders` |
| `phase_6/classifier.py` | `GPTClassifier`, `freeze_backbone`, `unfreeze_backbone` |
| `phase_6/train_classifier.py` | `train_epoch`, `evaluate`, `train_classifier`, `main` |

## Running Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/test_lab1_dataset.py -v
python -m pytest tests/test_lab2_classifier.py -v
python -m pytest tests/test_lab3_train_classifier.py -v
python scripts/grade.py
```

## Tips

- Start with Lab 1 (dataset.py) since the other labs depend on it.
- For the classifier, pay attention to how you extract hidden states from
  the GPT backbone. The mock model provides `get_hidden_states()` for
  convenience, but a real GPT would require accessing internal layers.
- The training loop is simpler than Phase 4 because we train by epochs
  (full passes through the data) rather than by steps.
- Early stopping patience of 3 is reasonable for this task.
