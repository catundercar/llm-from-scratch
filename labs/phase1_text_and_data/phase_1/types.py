"""
Type definitions for Phase 1.

These types are PROVIDED -- do not modify this file.
"""

from typing import TypeAlias

# A single token ID (integer index into the vocabulary)
TokenID: TypeAlias = int

# A sequence of token IDs
TokenIDs: TypeAlias = list[TokenID]

# A merge rule: maps a pair of token IDs to the merged token ID
MergeRule: TypeAlias = tuple[TokenID, TokenID]

# The vocabulary: maps token ID -> string representation
Vocabulary: TypeAlias = dict[TokenID, str]

# Merge list: ordered list of (pair, new_token_id)
MergeList: TypeAlias = list[tuple[MergeRule, TokenID]]
