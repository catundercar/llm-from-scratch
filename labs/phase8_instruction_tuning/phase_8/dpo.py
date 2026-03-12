"""
Phase 8, Lab 2: Direct Preference Optimization (DPO)

This module implements DPO, a method for aligning language models with human
preferences without training a separate reward model.

Key concepts:
    - Preference data: pairs of (chosen, rejected) responses for each prompt
    - Reference model: a frozen copy of the SFT model used as baseline
    - DPO loss: directly optimizes the policy to prefer chosen over rejected
      responses, using the reference model to prevent distribution collapse

Reference: Rafailov et al., "Direct Preference Optimization: Your Language Model
           is Secretly a Reward Model" (2023)

Students implement:
    - PreferenceDataset: Load preference pairs
    - get_log_probs: Compute per-sequence log probabilities
    - dpo_loss: The DPO loss function
    - train_dpo: The DPO training loop
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from pathlib import Path

from phase_8.types import DPOConfig, PreferenceSample


class PreferenceDataset(Dataset):
    """
    Dataset for Direct Preference Optimization.

    Loads preference pairs from a JSONL file. Each line has:
        - prompt: the instruction/question
        - chosen: the preferred response
        - rejected: the worse response
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: object,
        max_length: int = 512,
    ) -> None:
        """
        Initialize the PreferenceDataset.

        TODO: Implement this method

        Requirements:
        1. Load all samples from the JSONL file. Each line is a JSON object
           with keys: "prompt", "chosen", "rejected".
        2. For each sample, tokenize:
           - chosen_text = prompt + chosen
           - rejected_text = prompt + rejected
        3. Pad or truncate to max_length.
        4. Also store the prompt length (in tokens) so we can mask prompt
           tokens during loss computation.
        5. Store processed samples.

        HINT: The prompt tokens are shared between chosen and rejected.
              Store the prompt token count so dpo_loss can ignore those tokens.
        """
        # TODO: Implement
        # Step 1: Load JSONL
        # Step 2: Tokenize chosen and rejected full texts
        # Step 3: Compute prompt token lengths
        # Step 4: Pad/truncate
        # Step 5: Store data
        raise NotImplementedError("TODO: Implement PreferenceDataset.__init__")

    def __len__(self) -> int:
        """Return the number of preference pairs."""
        # TODO: Implement
        raise NotImplementedError("TODO: Implement PreferenceDataset.__len__")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return a single preference pair.

        TODO: Implement this method

        Returns:
            Dict with keys:
                "chosen_ids": token IDs for prompt + chosen response
                "chosen_mask": attention mask for chosen
                "rejected_ids": token IDs for prompt + rejected response
                "rejected_mask": attention mask for rejected
                "prompt_length": number of prompt tokens (for masking)
        """
        # TODO: Implement
        raise NotImplementedError("TODO: Implement PreferenceDataset.__getitem__")


def get_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sequence log probabilities for the response tokens only.

    TODO: Implement this function

    Requirements:
    1. Run the model forward to get logits.
    2. Shift logits and labels for next-token prediction.
    3. Compute per-token log probabilities using log_softmax.
    4. Gather the log probs of the actual tokens.
    5. Mask out prompt tokens (only sum log probs of response tokens).
    6. Return the sum of log probs per sequence (shape: [batch_size]).

    Args:
        model: The language model.
        input_ids: Token IDs, shape (batch, seq_len).
        attention_mask: Attention mask, shape (batch, seq_len).
        prompt_lengths: Length of prompt portion per sample, shape (batch,).

    Returns:
        Per-sequence sum of response token log probs, shape (batch,).

    HINT: After shifting, token at position i predicts position i+1.
          So response tokens start at position prompt_length-1 in the
          shifted sequence.

    HINT: log_probs = logits.log_softmax(dim=-1)
          token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    """
    # TODO: Implement
    # Step 1: Forward pass to get logits
    # Step 2: Shift logits and input_ids for next-token prediction
    # Step 3: Compute log softmax
    # Step 4: Gather log probs of actual next tokens
    # Step 5: Create mask for response tokens only
    # Step 6: Sum masked log probs per sequence
    raise NotImplementedError("TODO: Implement get_log_probs")


def dpo_loss(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    ref_chosen_logprobs: torch.Tensor,
    ref_rejected_logprobs: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Compute the DPO loss.

    TODO: Implement this function

    The DPO loss is:
        L = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    where:
        log_ratio_chosen  = policy_chosen_logprobs  - ref_chosen_logprobs
        log_ratio_rejected = policy_rejected_logprobs - ref_rejected_logprobs

    Requirements:
    1. Compute log ratios for chosen and rejected.
    2. Compute the DPO loss using the formula above.
    3. Return the mean loss over the batch.

    Args:
        policy_chosen_logprobs: Log probs of chosen under policy, shape (batch,).
        policy_rejected_logprobs: Log probs of rejected under policy, shape (batch,).
        ref_chosen_logprobs: Log probs of chosen under reference, shape (batch,).
        ref_rejected_logprobs: Log probs of rejected under reference, shape (batch,).
        beta: Temperature parameter controlling deviation from reference.

    Returns:
        Scalar DPO loss.

    HINT: The log_ratio measures how much more likely the policy makes a
          response compared to the reference. We want the policy to increase
          this ratio more for chosen than rejected.

    HINT: F.logsigmoid is numerically more stable than log(sigmoid(...)).
          loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected)).mean()
    """
    # TODO: Implement
    # Step 1: log_ratio_chosen = policy_chosen - ref_chosen
    # Step 2: log_ratio_rejected = policy_rejected - ref_rejected
    # Step 3: loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    # Step 4: Return mean loss
    raise NotImplementedError("TODO: Implement dpo_loss")


def train_dpo(
    policy_model: nn.Module,
    ref_model: nn.Module,
    train_loader: DataLoader,
    config: DPOConfig,
) -> list[float]:
    """
    DPO training loop.

    TODO: Implement this function

    Requirements:
    1. Freeze the reference model (all requires_grad = False, eval mode).
    2. Create an optimizer for the policy model only.
    3. For each epoch:
       a. For each batch, compute log probs under both policy and reference.
       b. Compute DPO loss.
       c. Backward pass and optimizer step.
       d. Log loss every config.log_interval steps.
    4. Return list of average epoch losses.

    Args:
        policy_model: The model being trained.
        ref_model: Frozen reference model (typically the SFT checkpoint).
        train_loader: DataLoader yielding preference pair batches.
        config: DPO training configuration.

    Returns:
        List of average losses per epoch.

    HINT: Use torch.no_grad() when computing reference model log probs.

    HINT: The ref_model should never be updated. Call ref_model.eval()
          and set all its parameters to requires_grad=False.
    """
    # TODO: Implement
    # Step 1: Freeze ref_model
    # Step 2: Create optimizer for policy_model
    # Step 3: Training loop
    #   for batch in train_loader:
    #       policy_chosen_lp = get_log_probs(policy_model, chosen_ids, ...)
    #       policy_rejected_lp = get_log_probs(policy_model, rejected_ids, ...)
    #       with torch.no_grad():
    #           ref_chosen_lp = get_log_probs(ref_model, chosen_ids, ...)
    #           ref_rejected_lp = get_log_probs(ref_model, rejected_ids, ...)
    #       loss = dpo_loss(...)
    #       loss.backward()
    #       optimizer.step()
    # Step 4: Return losses
    raise NotImplementedError("TODO: Implement train_dpo")


def main() -> None:
    """
    End-to-end DPO pipeline.

    TODO: Implement this function

    Requirements:
    1. Load the SFT-trained model as the reference model.
    2. Create a deep copy as the policy model.
    3. Create PreferenceDataset from sample_data/preferences.jsonl.
    4. Train with DPO.
    5. Save the aligned model.

    HINT: Use copy.deepcopy(ref_model) to create the policy model.
    """
    # TODO: Implement
    raise NotImplementedError("TODO: Implement main")


if __name__ == "__main__":
    main()
