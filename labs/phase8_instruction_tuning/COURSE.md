# Phase 8: Instruction Fine-Tuning

## Overview

Pretraining teaches a language model to predict the next token, but it does not
teach it to follow instructions, answer questions helpfully, or refuse harmful
requests. Instruction fine-tuning bridges this gap through two stages:

1. **Supervised Fine-Tuning (SFT)** -- train on curated (instruction, response) pairs
2. **Preference Optimization (DPO)** -- align the model with human preferences

## Stage 1: Supervised Fine-Tuning (SFT)

### The Alpaca Format

The most common instruction format follows the Stanford Alpaca template:

    ### Instruction:
    {instruction}

    ### Input:
    {input}          (omitted if empty)

    ### Response:
    {output}

The model learns this pattern so that at inference time, given an instruction
and input, it generates the response.

### Label Masking

A critical detail: we only compute loss on the response tokens. The prompt
(instruction + input) is provided as context but should not contribute to the
training loss. This is implemented by setting labels to -100 for prompt tokens,
which PyTorch's cross_entropy ignores via `ignore_index=-100`.

Without label masking, the model wastes capacity learning to predict the
instruction tokens (which are provided at inference time anyway).

### SFT Training

SFT uses standard cross-entropy loss on the response tokens:

    L_SFT = -sum(log P(y_t | y_{<t}, x))

where x is the instruction and y is the response. This is the same training
objective as pretraining, just on curated data with label masking.

## Stage 2: Direct Preference Optimization (DPO)

### Motivation

SFT teaches the model to imitate reference responses, but it cannot express
preferences between responses of different quality. DPO uses human preference
data (chosen vs rejected response pairs) to further align the model.

### The DPO Objective

DPO directly optimizes the policy model using preference pairs, without
training a separate reward model (unlike RLHF with PPO).

The loss function is:

    L_DPO = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

where:
    log_ratio_chosen  = log P_policy(chosen) - log P_ref(chosen)
    log_ratio_rejected = log P_policy(rejected) - log P_ref(rejected)

### Intuition

- `log_ratio_chosen` measures how much more likely the policy makes the chosen
  response compared to the reference model.
- `log_ratio_rejected` measures the same for the rejected response.
- We want the policy to increase the probability of chosen responses MORE than
  it increases rejected responses, relative to the reference.
- `beta` controls how far the policy can deviate from the reference. Lower beta
  allows more deviation; higher beta keeps the policy closer to the reference.

### Reference Model

The reference model is a frozen copy of the SFT model. It serves as an anchor:
without it, the policy could collapse to always outputting the same high-reward
response regardless of the prompt.

## Stage 3: Evaluation

### LLM-as-Judge

Modern evaluation increasingly uses LLMs themselves as evaluators. Given a
model's response and a reference answer, an evaluator LLM rates the quality.
This scales better than human evaluation while maintaining reasonable
correlation with human judgments.

### Overlap Metrics

Simple word-overlap metrics (precision, recall, F1) provide a quick sanity
check but should not be the primary evaluation method for generative tasks.

## References

- Taori et al., "Stanford Alpaca: An Instruction-following LLaMA Model" (2023)
- Rafailov et al., "Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model" (2023)
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023)
