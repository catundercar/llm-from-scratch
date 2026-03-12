# Phase 7: LoRA (Low-Rank Adaptation)

## Labs

### Lab 1: LoRA Implementation (`phase_7/lora.py`)
**Difficulty: Medium**

Implement the core LoRA building blocks:
- `LoRALinear` -- a drop-in replacement for `nn.Linear` with low-rank adapters
- `apply_lora_to_model` -- inject LoRA into a pretrained model
- `merge_lora_weights` -- fold LoRA weights back for deployment
- `count_trainable_params` -- report parameter efficiency

### Lab 2: LoRA Training (`phase_7/train_lora.py`)
**Difficulty: Medium**

Implement the training pipeline:
- `train_with_lora` -- train only LoRA parameters
- `compare_param_counts` -- visualize parameter reduction
- `main` -- end-to-end fine-tuning pipeline

## Provided Files (do NOT modify)
- `phase_7/types.py` -- Configuration dataclasses
- `phase_7/sample_model.py` -- A minimal GPT model for testing
- `tests/test_lab1_lora.py` -- Tests for Lab 1
- `tests/test_lab2_train_lora.py` -- Tests for Lab 2
- `conftest.py` -- Shared test fixtures
- `scripts/grade.py` -- Grading script

## Running Tests

```bash
cd labs/phase7_lora
python -m pytest tests/ -v

# Or run the grading script:
python scripts/grade.py
```

## Tips
- Read `COURSE.md` for the mathematical background
- Start with `LoRALinear.__init__` and `forward`, then move to `apply_lora_to_model`
- The tests are ordered from simple to complex -- work through them in order
- Use the `sample_model.py` to understand the model structure you are adapting
