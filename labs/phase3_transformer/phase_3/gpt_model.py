"""
Phase 3, Lab 2: GPT Model
==========================

Assemble all components into a complete GPT language model.

The GPT model consists of:
1. Token embedding + Positional embedding
2. Stack of TransformerBlocks
3. Final LayerNorm
4. Linear projection to vocabulary (lm_head)

Reference: Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2)
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase_3.transformer import LayerNorm, TransformerBlock


# ---------------------------------------------------------------------------
# TODO 1: GPT Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """
    Configuration for the GPT model.

    TODO: Define the following fields with default values

    Requirements:
    1. vocab_size: int = 50257    (GPT-2 BPE vocabulary size)
    2. block_size: int = 1024     (maximum sequence length)
    3. d_model: int = 768         (embedding dimension)
    4. n_heads: int = 12          (number of attention heads)
    5. n_layers: int = 12         (number of transformer blocks)
    6. dropout: float = 0.1       (dropout probability)
    """
    # TODO: Define fields
    # vocab_size: int = 50257
    # block_size: int = 1024
    # d_model: int = 768
    # n_heads: int = 12
    # n_layers: int = 12
    # dropout: float = 0.1
    pass


# ---------------------------------------------------------------------------
# TODO 2 & 3: GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    GPT Language Model.

    Architecture:
        Token IDs -> Token Embedding + Position Embedding -> Dropout
        -> TransformerBlock × n_layers -> Final LayerNorm -> lm_head -> Logits
    """

    def __init__(self, config: GPTConfig) -> None:
        """
        Initialize the GPT model.

        TODO: Implement this method

        Requirements:
        1. Store config.
        2. Create token embedding: nn.Embedding(vocab_size, d_model)
        3. Create position embedding: nn.Embedding(block_size, d_model)
        4. Create embedding dropout: nn.Dropout(dropout)
        5. Create a stack of n_layers TransformerBlocks using nn.ModuleList.
        6. Create final LayerNorm (ln_f).
        7. Create lm_head: nn.Linear(d_model, vocab_size, bias=False)
        8. (Optional) Weight tying: set lm_head.weight = token_embedding.weight

        HINT: Use nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])
              so PyTorch registers all blocks as submodules.

        HINT: Weight tying (step 8) reduces parameters and often improves
              performance. Do this by setting:
              self.lm_head.weight = self.token_embedding.weight

        Args:
            config: GPTConfig with model hyperparameters.
        """
        super().__init__()
        # TODO: Implement
        # Step 1: Store config
        # Step 2: Token embedding
        # Step 3: Position embedding
        # Step 4: Dropout
        # Step 5: Transformer blocks
        # Step 6: Final LayerNorm
        # Step 7: lm_head projection
        # Step 8: (Optional) Weight tying
        raise NotImplementedError("TODO: Implement GPT.__init__")

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the GPT model.

        TODO: Implement this method

        Requirements:
        1. Get batch size B and sequence length T from idx.shape.
        2. Assert T <= block_size.
        3. Compute token embeddings: token_embedding(idx)  -> (B, T, d_model)
        4. Compute position embeddings:
           positions = torch.arange(T, device=idx.device)
           pos_emb = position_embedding(positions)  -> (T, d_model)
        5. Add: x = token_emb + pos_emb (broadcasting handles batch dim)
        6. Apply dropout.
        7. Pass through each TransformerBlock.
        8. Apply final LayerNorm.
        9. Compute logits: lm_head(x)  -> (B, T, vocab_size)
        10. If targets is provided:
            - Compute cross-entropy loss:
              loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            - Return (logits, loss)
        11. If targets is None, return (logits, None).

        HINT: logits.view(-1, logits.size(-1)) reshapes to (B*T, vocab_size)
              targets.view(-1) reshapes to (B*T,)

        Args:
            idx: Input token IDs of shape (B, T).
            targets: Optional target token IDs of shape (B, T).

        Returns:
            Tuple of (logits, loss) where loss is None if targets not provided.
        """
        B, T = idx.shape
        # TODO: Implement
        # Step 1: Assert T <= block_size
        # Step 2: Token + Position embeddings
        # Step 3: Dropout
        # Step 4: Transformer blocks
        # Step 5: Final LayerNorm
        # Step 6: lm_head -> logits
        # Step 7: Compute loss if targets provided
        raise NotImplementedError("TODO: Implement GPT.forward")

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        TODO: Implement this method

        Requirements:
        1. Sum p.numel() for all parameters with requires_grad=True.
        2. Return the count.

        Returns:
            Total number of trainable parameters.
        """
        # TODO: Implement
        raise NotImplementedError("TODO: Implement GPT.count_parameters")
