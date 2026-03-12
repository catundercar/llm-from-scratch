# Phase 9: Mixture of Experts (MoE)

## Labs

### Lab 1: MoE Layer (`phase_9/moe.py`)
**Difficulty: Hard**

Implement the core MoE building blocks:
- `Expert` -- single feed-forward network (same as standard transformer FFN)
- `Router` -- gating network with softmax and top-k selection
- `MoELayer` -- combines routing with expert computation
- `load_balancing_loss` -- auxiliary loss for balanced expert utilization

### Lab 2: MoE Transformer (`phase_9/moe_transformer.py`)
**Difficulty: Hard**

Build a full MoE-enhanced GPT model:
- `MoETransformerBlock` -- transformer block with MoE replacing FFN
- `MoEGPT` -- full model with interleaved MoE and dense blocks

## Provided Files (do NOT modify)
- `phase_9/types.py` -- Configuration dataclass
- `phase_9/moe_transformer.py` (partially) -- `SampleAttention`, `DenseFeedForward`, `DenseTransformerBlock`
- `tests/test_lab1_moe.py` -- Tests for Lab 1
- `tests/test_lab2_moe_transformer.py` -- Tests for Lab 2
- `conftest.py` -- Shared test fixtures
- `scripts/grade.py` -- Grading script

## Running Tests

```bash
cd labs/phase9_moe
python -m pytest tests/ -v

# Or run the grading script:
python scripts/grade.py
```

## Tips
- Read `COURSE.md` for the mathematical background on routing and load balancing
- Start with `Expert` and `Router` (simpler), then tackle `MoELayer`
- The trickiest part is the MoELayer forward pass -- study the hints carefully
- Use the load balancing loss tests to verify your understanding
- Lab 2 builds directly on Lab 1, so get Lab 1 passing first
