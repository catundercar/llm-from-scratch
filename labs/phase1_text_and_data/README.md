# Phase 1: Text and Data

Build a BPE tokenizer and a sliding-window data loader from scratch.

## Labs

| Lab | File | Difficulty | Description |
|-----|------|------------|-------------|
| 1 | `phase_1/tokenizer.py` | Medium | Byte Pair Encoding tokenizer |
| 2 | `phase_1/dataloader.py` | Easy | Sliding-window dataset and DataLoader |

## Setup

```bash
cd labs/phase_1_text_and_data
pip install -e .
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run individual lab tests
python -m pytest tests/test_lab1_tokenizer.py -v
python -m pytest tests/test_lab2_dataloader.py -v
```

## Grading

```bash
python scripts/grade.py
```

## Instructions

1. Read `COURSE.md` for background on tokenization and data loading.
2. Open `phase_1/tokenizer.py` and fill in the TODO sections.
3. Run `python -m pytest tests/test_lab1_tokenizer.py -v` to check your work.
4. Open `phase_1/dataloader.py` and fill in the TODO sections.
5. Run `python -m pytest tests/test_lab2_dataloader.py -v` to check your work.
6. Run `python scripts/grade.py` for a full grading report.
