# LLM From Scratch

Build a large language model from scratch — a hands-on course that walks you through Python + PyTorch, from Tokenizer all the way to Mixture of Experts.

**[Course Website](https://catundercar.github.io/llm-from-scratch/)** · [中文版](README.zh-CN.md)

## Course Overview

| Phase | Topic | Key Concepts |
|-------|-------|--------------|
| 0 | Big Picture: Transformer & LLM | Overall architecture, data flow, core components |
| 1 | Text & Data | Tokenizer (BPE), DataLoader, Embedding |
| 2 | Attention Mechanism | Self-Attention, Multi-Head Attention, Causal Mask |
| 3 | Transformer Architecture | GPT Block, LayerNorm, Residual Connections |
| 4 | Pre-training | Language Modeling Loss, Training Loop, LR Scheduling |
| 5 | Text Generation | Temperature, Top-k, Top-p Sampling |
| 6 | Classification Fine-tuning | Transfer Learning, Classification Head, Freezing Strategy |
| 7 | LoRA | Low-Rank Adaptation, Parameter-Efficient Fine-tuning |
| 8 | Instruction Tuning | SFT, DPO, Conversation Format |
| 9 | Mixture of Experts | MoE Routing, Sparse Activation, Load Balancing |

## Quick Start

### Setup

```bash
# Clone the repo
git clone https://github.com/catundercar/llm-from-scratch.git
cd llm-from-scratch

# Install Python dependencies
pip install -r requirements.txt
```

### Run a Lab

Each Phase is a standalone Python package with skeleton code (TODO markers), tests, and a grading script:

```bash
cd labs/phase1_text_and_data

# Run tests
pytest tests/ -v

# Check your score
python scripts/grade.py
```

### Run the Course Website Locally

```bash
cd website
npm install
npm run dev
```

## Project Structure

```
llm-from-scratch/
├── labs/                        # Hands-on Labs (Python + PyTorch)
│   ├── shared/                  # Shared utilities across phases
│   ├── phase1_text_and_data/    # Phase 1: Tokenizer + DataLoader
│   ├── phase2_attention/        # Phase 2: Self-Attention
│   ├── phase3_transformer/      # Phase 3: GPT Block
│   ├── phase4_pretraining/      # Phase 4: Pre-training
│   ├── phase5_generation/       # Phase 5: Text Generation
│   ├── phase6_classification/   # Phase 6: Classification Fine-tuning
│   ├── phase7_lora/             # Phase 7: LoRA
│   ├── phase8_instruction_tuning/ # Phase 8: SFT + DPO
│   └── phase9_moe/              # Phase 9: MoE
├── website/                     # Course website (Vite + React + TypeScript)
├── COURSE.md                    # Full course content
├── course-roadmap.jsx           # Course roadmap prototype
└── requirements.txt             # Python dependencies
```

## Tech Stack

- **Labs**: Python 3.10+, PyTorch 2.0+, tiktoken
- **Website**: Vite, React 18, TypeScript, React Router
- **Deployment**: GitHub Pages (auto-deploy on push)

## Design Philosophy

1. **Bottom-up** — Start from the lowest-level Tokenizer and build layer by layer up to a full LLM
2. **Concept first** — Understand the WHY before implementing the HOW
3. **Skeleton fill-in** — Framework and tests are provided; you implement the core algorithms (TODO markers)
4. **Immediate feedback** — Each Phase produces a runnable artifact you can test

## References

- [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) — Sebastian Raschka
- [LLM Course](https://github.com/mlabonne/llm-course) — Maxime Labonne
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.

## License

MIT
