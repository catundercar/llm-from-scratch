"""
Phase 8, Lab 3: Evaluation Utilities

This module implements evaluation tools for instruction-tuned models:
    - Response generation from prompts
    - LLM-as-judge prompt formatting
    - Simple overlap metrics (BLEU-like)
    - Full evaluation pipeline

Students implement:
    - generate_responses: batch generation
    - llm_as_judge: format evaluation prompts
    - compute_metrics: compute text overlap metrics
    - evaluate_model: full pipeline
"""

import torch
import torch.nn as nn
from typing import Optional
from collections import Counter


def generate_responses(
    model: nn.Module,
    tokenizer: object,
    prompts: list[str],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    device: str = "cpu",
) -> list[str]:
    """
    Generate responses for a list of prompts.

    TODO: Implement this function

    Requirements:
    1. For each prompt, tokenize and generate tokens autoregressively.
    2. Use temperature sampling (divide logits by temperature before softmax).
    3. Stop generation at max_new_tokens or EOS token.
    4. Decode generated tokens back to text.
    5. Return only the generated response (not the prompt).

    Args:
        model: The language model.
        tokenizer: Tokenizer with encode/decode methods.
        prompts: List of prompt strings.
        max_new_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature (1.0 = normal, <1.0 = more deterministic).
        device: Device to run on.

    Returns:
        List of generated response strings.

    HINT: Basic autoregressive loop:
          for _ in range(max_new_tokens):
              logits = model(input_ids)[0]  # or model(input_ids) if returns logits directly
              next_logits = logits[:, -1, :] / temperature
              probs = torch.softmax(next_logits, dim=-1)
              next_token = torch.multinomial(probs, 1)
              input_ids = torch.cat([input_ids, next_token], dim=1)

    HINT: Use torch.no_grad() and model.eval() for generation.
    """
    # TODO: Implement
    # Step 1: model.eval(), move to device
    # Step 2: For each prompt, encode to ids
    # Step 3: Autoregressive generation loop
    # Step 4: Decode generated tokens (excluding prompt)
    # Step 5: Return list of response strings
    raise NotImplementedError("TODO: Implement generate_responses")


def llm_as_judge(
    response: str,
    reference: str,
    criteria: str = "helpfulness, accuracy, and clarity",
) -> str:
    """
    Format an evaluation prompt for LLM-as-judge.

    TODO: Implement this function

    Requirements:
    1. Create a structured prompt that asks an LLM to evaluate a response
       against a reference answer.
    2. The prompt should include:
       - The model's response
       - The reference/gold answer
       - The evaluation criteria
       - Instructions to provide a score (1-5) and explanation
    3. Return the formatted prompt string (do NOT call any API).

    Args:
        response: The model-generated response to evaluate.
        reference: The reference/gold answer.
        criteria: Evaluation criteria string.

    Returns:
        A formatted evaluation prompt string.

    HINT: A good template looks like:
          "You are an expert evaluator. Rate the following response on {criteria}.

           [Reference Answer]
           {reference}

           [Model Response]
           {response}

           Provide a score from 1-5 and a brief explanation."
    """
    # TODO: Implement
    # Step 1: Format the evaluation prompt template
    # Step 2: Return the prompt string
    raise NotImplementedError("TODO: Implement llm_as_judge")


def compute_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute simple text overlap metrics between predictions and references.

    TODO: Implement this function

    Requirements:
    1. Compute unigram precision: fraction of prediction unigrams found in reference.
    2. Compute unigram recall: fraction of reference unigrams found in prediction.
    3. Compute F1 score: harmonic mean of precision and recall.
    4. Compute exact match rate: fraction of predictions that exactly match reference.
    5. Average metrics across all prediction-reference pairs.
    6. Return dict with keys: "precision", "recall", "f1", "exact_match".

    Args:
        predictions: List of predicted/generated strings.
        references: List of reference/gold strings.

    Returns:
        Dict of metric name -> average score.

    HINT: Tokenize by splitting on whitespace: words = text.lower().split()

    HINT: Use collections.Counter for efficient unigram overlap computation:
          pred_counts = Counter(pred_words)
          ref_counts = Counter(ref_words)
          overlap = sum((pred_counts & ref_counts).values())
    """
    # TODO: Implement
    # Step 1: For each (pred, ref) pair:
    #   a. Tokenize into words
    #   b. Compute unigram overlap
    #   c. Compute precision, recall, F1
    #   d. Check exact match
    # Step 2: Average across all pairs
    # Step 3: Return dict
    raise NotImplementedError("TODO: Implement compute_metrics")


def evaluate_model(
    model: nn.Module,
    tokenizer: object,
    test_prompts: list[str],
    reference_responses: Optional[list[str]] = None,
    device: str = "cpu",
) -> dict[str, object]:
    """
    Run full evaluation pipeline on a model.

    TODO: Implement this function

    Requirements:
    1. Generate responses for all test prompts.
    2. If reference_responses are provided, compute overlap metrics.
    3. For each (response, reference) pair, generate an LLM-as-judge prompt.
    4. Return a dict with:
       - "responses": list of generated responses
       - "metrics": dict of computed metrics (or None if no references)
       - "judge_prompts": list of evaluation prompts (or None if no references)

    Args:
        model: The language model to evaluate.
        tokenizer: Tokenizer with encode/decode methods.
        test_prompts: List of prompts to generate from.
        reference_responses: Optional reference answers for metric computation.
        device: Device to run on.

    Returns:
        Dict with responses, metrics, and judge prompts.
    """
    # TODO: Implement
    # Step 1: Generate responses
    # Step 2: Compute metrics if references available
    # Step 3: Create judge prompts if references available
    # Step 4: Return results dict
    raise NotImplementedError("TODO: Implement evaluate_model")
