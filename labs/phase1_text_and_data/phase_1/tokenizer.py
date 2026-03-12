"""
Lab 1: Byte Pair Encoding (BPE) Tokenizer

Build a simplified BPE tokenizer from scratch. BPE starts with a character-level
vocabulary and iteratively merges the most frequent adjacent pair of tokens until
the desired vocabulary size is reached.

This is the same core algorithm used by GPT-2, GPT-3, and GPT-4 tokenizers
(with additional details like byte-level encoding and regex pre-splitting that
we omit here for clarity).

Students implement the TODO sections below.
"""

import json
from pathlib import Path
from collections import Counter

from phase_1.types import TokenID, TokenIDs, Vocabulary, MergeList, MergeRule


class BPETokenizer:
    """
    A simplified Byte Pair Encoding tokenizer.

    Attributes:
        vocab: Maps token ID -> string representation.
        merges: Ordered list of learned merge rules.
        token_to_id: Reverse mapping from string -> token ID.
    """

    def __init__(self) -> None:
        """Initialize an empty tokenizer. Call train() to build the vocabulary."""
        self.vocab: Vocabulary = {}
        self.merges: MergeList = []
        self.token_to_id: dict[str, TokenID] = {}

    # ------------------------------------------------------------------
    # HELPER (provided): rebuild the reverse mapping after any vocab change
    # ------------------------------------------------------------------
    def _rebuild_token_to_id(self) -> None:
        """Rebuild the token_to_id reverse mapping from self.vocab."""
        self.token_to_id = {s: i for i, s in self.vocab.items()}

    # ------------------------------------------------------------------
    # TODO 1: Build the initial character-level vocabulary
    # ------------------------------------------------------------------
    def _build_initial_vocab(self, text: str) -> TokenIDs:
        """
        Build the initial vocabulary from individual characters in the text,
        and convert the text into a list of token IDs.

        TODO: Implement this method

        Requirements:
        1. Find all unique characters in the text.
        2. Sort them to ensure deterministic ordering.
        3. Assign each character a unique integer ID starting from 0.
        4. Store the mapping in self.vocab (id -> character string).
        5. Rebuild self.token_to_id by calling self._rebuild_token_to_id().
        6. Return the text converted to a list of token IDs.

        HINT: sorted(set(text)) gives you a deterministic character list.

        Args:
            text: The training text.

        Returns:
            The text represented as a list of token IDs.
        """
        # TODO: Implement
        # Step 1: Get sorted unique characters
        # Step 2: Build self.vocab mapping id -> char
        # Step 3: Rebuild reverse mapping
        # Step 4: Convert text to token IDs and return
        raise NotImplementedError("TODO: Implement _build_initial_vocab")

    # ------------------------------------------------------------------
    # TODO 2: Count adjacent pair frequencies
    # ------------------------------------------------------------------
    def _get_pair_counts(self, token_ids: TokenIDs) -> Counter[MergeRule]:
        """
        Count the frequency of every adjacent pair of tokens.

        TODO: Implement this method

        Requirements:
        1. Iterate through the token_ids list.
        2. For each adjacent pair (token_ids[i], token_ids[i+1]), count it.
        3. Return a Counter mapping each pair tuple to its frequency.

        HINT: zip(token_ids, token_ids[1:]) gives you all adjacent pairs.

        Args:
            token_ids: Current list of token IDs.

        Returns:
            Counter mapping (id_a, id_b) -> frequency.
        """
        # TODO: Implement
        # Step 1: Create pairs from adjacent tokens
        # Step 2: Count each pair using Counter
        # Step 3: Return the Counter
        raise NotImplementedError("TODO: Implement _get_pair_counts")

    # ------------------------------------------------------------------
    # TODO 3: Merge the most frequent pair
    # ------------------------------------------------------------------
    def _merge_pair(self, token_ids: TokenIDs, pair: MergeRule, new_id: TokenID) -> TokenIDs:
        """
        Replace every occurrence of `pair` in token_ids with `new_id`.

        TODO: Implement this method

        Requirements:
        1. Scan through token_ids from left to right.
        2. Whenever you see pair[0] followed by pair[1], replace them with new_id.
        3. Return the new (shorter) list of token IDs.

        HINT: Use a while loop with an index. When you find the pair, append
              new_id and skip ahead by 2. Otherwise append the current token
              and advance by 1.

        Args:
            token_ids: Current list of token IDs.
            pair: The (left_id, right_id) pair to merge.
            new_id: The token ID for the merged token.

        Returns:
            A new list of token IDs with the pair merged.
        """
        # TODO: Implement
        # Step 1: Initialize empty result list and index i = 0
        # Step 2: While i < len(token_ids):
        #   - If token_ids[i] == pair[0] and i+1 < len and token_ids[i+1] == pair[1]:
        #       append new_id, i += 2
        #   - Else: append token_ids[i], i += 1
        # Step 3: Return result
        raise NotImplementedError("TODO: Implement _merge_pair")

    # ------------------------------------------------------------------
    # TODO 4: Train the tokenizer
    # ------------------------------------------------------------------
    def train(self, text: str, vocab_size: int) -> None:
        """
        Train the BPE tokenizer on the given text.

        TODO: Implement this method

        Requirements:
        1. Build the initial character-level vocabulary using _build_initial_vocab.
        2. Repeatedly (until vocab reaches vocab_size):
           a. Count pair frequencies with _get_pair_counts.
           b. Find the most frequent pair (if no pairs remain, stop).
           c. Create a new token by concatenating the string representations
              of the two tokens in the pair.
           d. Assign it the next available token ID.
           e. Add the new token to self.vocab.
           f. Record the merge rule in self.merges.
           g. Merge the pair in the token_ids list using _merge_pair.
        3. Rebuild self.token_to_id at the end.

        HINT: The new token string is self.vocab[pair[0]] + self.vocab[pair[1]].
        HINT: The new token ID is len(self.vocab) (before adding it).
        HINT: Use pair_counts.most_common(1)[0][0] to get the top pair.

        Args:
            text: Training text.
            vocab_size: Target vocabulary size (must be >= number of unique chars).
        """
        # TODO: Implement
        # Step 1: Build initial vocab and get token_ids
        # Step 2: Loop while len(self.vocab) < vocab_size:
        #   a. Get pair counts
        #   b. If no pairs, break
        #   c. Find most frequent pair
        #   d. Create new token string and ID
        #   e. Add to vocab
        #   f. Record merge
        #   g. Merge in token_ids
        # Step 3: Rebuild token_to_id
        raise NotImplementedError("TODO: Implement train")

    # ------------------------------------------------------------------
    # TODO 5: Encode text to token IDs
    # ------------------------------------------------------------------
    def encode(self, text: str) -> TokenIDs:
        """
        Encode a string into a list of token IDs using learned merges.

        TODO: Implement this method

        Requirements:
        1. Convert the text to character-level token IDs using self.token_to_id.
        2. Apply each merge rule (in order) to the token list.
        3. Return the final list of token IDs.

        HINT: Reuse self._merge_pair for each merge in self.merges.
        HINT: The order of merges matters -- apply them in the same order they
              were learned during training.

        Args:
            text: The string to encode.

        Returns:
            List of token IDs.

        Raises:
            KeyError: If a character is not in the vocabulary.
        """
        # TODO: Implement
        # Step 1: Convert each character to its token ID
        # Step 2: For each (pair, new_id) in self.merges, apply _merge_pair
        # Step 3: Return the result
        raise NotImplementedError("TODO: Implement encode")

    # ------------------------------------------------------------------
    # TODO 6: Decode token IDs to text
    # ------------------------------------------------------------------
    def decode(self, ids: TokenIDs) -> str:
        """
        Decode a list of token IDs back into a string.

        TODO: Implement this method

        Requirements:
        1. Look up each token ID in self.vocab to get its string.
        2. Concatenate all strings together.
        3. Return the result.

        HINT: This is a one-liner with a join and a list comprehension.

        Args:
            ids: List of token IDs.

        Returns:
            The decoded string.

        Raises:
            KeyError: If a token ID is not in the vocabulary.
        """
        # TODO: Implement
        # Step 1: Map each id to self.vocab[id]
        # Step 2: Join them into a single string
        raise NotImplementedError("TODO: Implement decode")

    # ------------------------------------------------------------------
    # PROVIDED: Save and load
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save the tokenizer vocabulary and merges to a JSON file."""
        path = Path(path)
        data = {
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "merges": [
                {"pair": [p[0], p[1]], "new_id": new_id}
                for (p, new_id) in self.merges
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def load(self, path: str | Path) -> None:
        """Load the tokenizer vocabulary and merges from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        self.vocab = {int(k): v for k, v in data["vocab"].items()}
        self.merges = [
            ((m["pair"][0], m["pair"][1]), m["new_id"])
            for m in data["merges"]
        ]
        self._rebuild_token_to_id()
