# Phase 8: Instruction Fine-Tuning

## Labs

### Lab 1: Supervised Fine-Tuning (`phase_8/sft.py`)
**Difficulty: Medium**

Implement instruction fine-tuning with label masking:
- `InstructionDataset` -- load, format, tokenize, and mask instruction data
- `train_sft` -- SFT training loop with cross-entropy loss

### Lab 2: Direct Preference Optimization (`phase_8/dpo.py`)
**Difficulty: Hard**

Implement DPO for preference alignment:
- `PreferenceDataset` -- load and tokenize preference pairs
- `get_log_probs` -- compute per-sequence log probabilities
- `dpo_loss` -- the DPO loss function
- `train_dpo` -- DPO training loop with frozen reference model

### Lab 3: Evaluation (`phase_8/evaluate.py`)
**Difficulty: Easy-Medium**

Implement evaluation utilities:
- `generate_responses` -- autoregressive text generation
- `llm_as_judge` -- format LLM-as-judge evaluation prompts
- `compute_metrics` -- simple word-overlap metrics
- `evaluate_model` -- full evaluation pipeline

## Provided Files (do NOT modify)
- `phase_8/types.py` -- Configuration dataclasses
- `phase_8/sample_tokenizer.py` -- Character-level tokenizer for testing
- `sample_data/instructions.jsonl` -- 30 instruction-response pairs
- `sample_data/preferences.jsonl` -- 20 preference pairs
- `tests/test_lab1_sft.py` -- Tests for Lab 1
- `tests/test_lab2_dpo.py` -- Tests for Lab 2
- `tests/test_lab3_evaluate.py` -- Tests for Lab 3
- `conftest.py` -- Shared test fixtures
- `scripts/grade.py` -- Grading script

## Running Tests

```bash
cd labs/phase8_instruction_tuning
python -m pytest tests/ -v

# Or run the grading script:
python scripts/grade.py
```

## Tips
- Read `COURSE.md` for background on SFT and DPO
- Start with Lab 1 (SFT) before attempting Lab 2 (DPO)
- The DPO loss test with known values is a great way to verify your implementation
- Label masking is the trickiest part of Lab 1 -- study the test carefully
