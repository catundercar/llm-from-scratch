"""
Sample tokenizer for Phase 8 testing.

This is PROVIDED -- students do not need to modify this file.

Provides a minimal tokenizer for testing instruction tuning without
requiring a real tokenizer like tiktoken or sentencepiece.
"""

from typing import Optional


class SampleTokenizer:
    """
    A simple character-level tokenizer for testing.

    In a real scenario, you would use tiktoken, sentencepiece, or
    a HuggingFace tokenizer. This provides the same interface for tests.
    """

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def encode(self, text: str, max_length: Optional[int] = None) -> list[int]:
        """Encode text to token IDs (character-level)."""
        ids = [min(ord(c), self.vocab_size - 1) for c in text]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        chars = []
        for i in ids:
            if i == self.pad_token_id:
                continue
            if i == self.eos_token_id:
                break
            if 32 <= i < 127:
                chars.append(chr(i))
            else:
                chars.append("?")
        return "".join(chars)

    def __call__(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> dict[str, list[int]]:
        """Tokenize with HuggingFace-like interface."""
        ids = self.encode(text)

        if truncation and max_length:
            ids = ids[:max_length]

        attention_mask = [1] * len(ids)

        if padding == "max_length" and max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        result = {"input_ids": ids, "attention_mask": attention_mask}

        if return_tensors == "pt":
            import torch
            result = {k: torch.tensor([v]) for k, v in result.items()}

        return result
