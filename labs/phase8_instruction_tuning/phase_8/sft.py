"""
Phase 8, Lab 1: Supervised Fine-Tuning (SFT)

This module implements instruction fine-tuning using supervised learning.
The model learns to follow instructions by training on (instruction, response)
pairs formatted in the Alpaca template.

Key concepts:
    - Instruction formatting: structuring prompts so the model learns the
      instruction-following pattern
    - Label masking: only computing loss on the response tokens, not the
      instruction/prompt tokens (labels = -100 for masked positions)
    - SFT training loop: standard cross-entropy training on formatted data

Students implement:
    - InstructionDataset: Load and format instruction data
    - train_sft: The SFT training loop
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from pathlib import Path

from phase_8.types import SFTConfig, InstructionSample


class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning.

    Loads instruction/input/output triples from a JSONL file and formats
    them using the Alpaca template. Labels are masked so that loss is only
    computed on the response tokens.

    Alpaca format:
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}

    When there is no input field (or it is empty), the "### Input:" section
    is omitted.
    """

    PROMPT_TEMPLATE_WITH_INPUT = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n"
    )

    PROMPT_TEMPLATE_NO_INPUT = (
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: object,
        max_length: int = 512,
    ) -> None:
        """
        Initialize the InstructionDataset.

        TODO: Implement this method

        Requirements:
        1. Load all samples from the JSONL file. Each line is a JSON object
           with keys: "instruction", "input" (optional, may be empty), "output".
        2. For each sample, format the full text using the Alpaca template above.
           If input is empty or missing, use PROMPT_TEMPLATE_NO_INPUT.
        3. Tokenize the full text (prompt + response) to get input_ids.
        4. Create labels: copy of input_ids, but with -100 for all tokens
           that correspond to the prompt (instruction + input) portion.
           Only response tokens should have real label values.
        5. Pad or truncate all sequences to max_length.
        6. Store the processed samples.

        Args:
            jsonl_path: Path to the JSONL file with instruction data.
            tokenizer: A tokenizer object with encode(text) -> list[int] method.
            max_length: Maximum sequence length for padding/truncation.

        HINT: To find where the response starts, tokenize just the prompt
              portion (without the output) and count its length. All tokens
              before that index get label -100.

        HINT: Use tokenizer.encode(text) to convert text to token IDs.
              Use tokenizer.pad_token_id for padding.
        """
        # TODO: Implement
        # Step 1: Load JSONL file into list of dicts
        # Step 2: For each sample, format full text and prompt-only text
        # Step 3: Tokenize both to find where response tokens start
        # Step 4: Create labels with -100 for prompt tokens
        # Step 5: Pad/truncate to max_length
        # Step 6: Store processed data
        raise NotImplementedError("TODO: Implement InstructionDataset.__init__")

    def __len__(self) -> int:
        """Return the number of samples."""
        # TODO: Implement
        raise NotImplementedError("TODO: Implement InstructionDataset.__len__")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return a single training example.

        TODO: Implement this method

        Requirements:
        1. Return a dict with keys: "input_ids", "attention_mask", "labels"
        2. All values should be torch.Tensors of shape (max_length,)
        3. attention_mask is 1 for real tokens, 0 for padding

        Returns:
            Dict with "input_ids", "attention_mask", "labels" tensors.
        """
        # TODO: Implement
        raise NotImplementedError("TODO: Implement InstructionDataset.__getitem__")


def format_instruction(
    instruction: str, input_text: str = "", output_text: str = ""
) -> tuple[str, str]:
    """
    Format an instruction sample into prompt and full text.

    This is a helper function that can be used by InstructionDataset.
    It is PROVIDED for reference.

    Returns:
        (prompt, full_text) where prompt is everything before the response
        and full_text includes the response.
    """
    if input_text.strip():
        prompt = InstructionDataset.PROMPT_TEMPLATE_WITH_INPUT.format(
            instruction=instruction, input=input_text
        )
    else:
        prompt = InstructionDataset.PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=instruction
        )
    full_text = prompt + output_text
    return prompt, full_text


def train_sft(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: SFTConfig,
) -> list[float]:
    """
    Supervised fine-tuning training loop.

    TODO: Implement this function

    Requirements:
    1. Move model to config.device.
    2. Create an AdamW optimizer with config.lr.
    3. For each epoch:
       a. Iterate over batches from train_loader.
       b. Each batch is a dict with "input_ids", "attention_mask", "labels".
       c. Forward pass: logits = model(input_ids). If the model returns
          (logits, loss), use the loss directly if labels are passed.
          Otherwise compute cross-entropy manually with label masking.
       d. Backward pass and optimizer step.
       e. Log training loss every config.log_interval steps.
    4. After each epoch, compute validation loss if val_loader is provided.
    5. Return a list of average training losses per epoch.

    Args:
        model: The language model to fine-tune.
        train_loader: DataLoader yielding batched dicts.
        val_loader: Optional validation DataLoader.
        config: SFT training configuration.

    Returns:
        List of average training losses, one per epoch.

    HINT: When computing loss manually with masked labels:
          loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)

    HINT: Shift logits and labels for next-token prediction:
          logits = logits[:, :-1, :].contiguous()
          labels = labels[:, 1:].contiguous()
    """
    # TODO: Implement
    # Step 1: model.to(config.device), model.train()
    # Step 2: optimizer = AdamW(model.parameters(), lr=config.lr)
    # Step 3: Training loop with loss logging
    # Step 4: Validation after each epoch
    # Step 5: Return epoch losses
    raise NotImplementedError("TODO: Implement train_sft")


def main() -> None:
    """
    End-to-end SFT pipeline.

    TODO: Implement this function

    Requirements:
    1. Create or load a language model.
    2. Create an InstructionDataset from sample_data/instructions.jsonl.
    3. Split into train and validation sets.
    4. Create DataLoaders.
    5. Train with train_sft.
    6. Save the fine-tuned model.

    HINT: Use torch.utils.data.random_split for train/val split.
    """
    # TODO: Implement
    raise NotImplementedError("TODO: Implement main")


if __name__ == "__main__":
    main()
