# Phase 6: Fine-Tuning for Classification

**Difficulty: Intermediate**

Fine-tune a pretrained GPT model for spam/ham text classification.

## Labs

| Lab | File | Difficulty | Description |
|-----|------|-----------|-------------|
| 1 | `phase_6/dataset.py` | ** | Load CSV, tokenize, pad/truncate, create DataLoaders |
| 2 | `phase_6/classifier.py` | *** | Wrap GPT with classification head, freeze/unfreeze backbone |
| 3 | `phase_6/train_classifier.py` | *** | Training loop, evaluation, early stopping |

## Prerequisites

- Phase 3 (GPT model architecture)
- Phase 4 (Pretrained weights)
- Understanding of transfer learning concepts

## Quick Start

```bash
pip install torch pytest
python -m pytest tests/ -v
python scripts/grade.py
```

## Sample Data

The `sample_data/spam.csv` file contains 100 examples (50 ham, 50 spam)
for development and testing. Each row has `text` and `label` columns.

## What You Will Learn

1. How to prepare text data for classification (tokenize, pad, create masks)
2. How to add a classification head to a pretrained language model
3. The difference between feature extraction and full fine-tuning
4. How to implement early stopping based on validation metrics
5. How to evaluate a classifier with accuracy and loss
