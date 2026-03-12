"""
Phase 6 - GPT Classifier
==========================

This module wraps a pretrained GPT model with a classification head for
downstream tasks like spam detection. The approach:

1. Feed text through the GPT backbone to get hidden states.
2. Extract the hidden state of the **last token** in the sequence.
3. Pass it through a linear classification head to get class logits.

The backbone can be frozen (feature extraction) or unfrozen (full fine-tuning).

Dependencies:
- Phase 3: GPT model (used as the backbone)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# GPT Classifier
# ---------------------------------------------------------------------------

class GPTClassifier(nn.Module):
    """A classification model built on top of a pretrained GPT backbone.

    Architecture:
        input_ids -> GPT backbone -> last_token_hidden -> dropout -> Linear -> logits

    The GPT backbone produces hidden states of shape (B, T, n_embd).
    We take the hidden state at the last non-padding position and pass
    it through a linear layer to get class logits.

    TODO: Implement __init__ and forward

    Attributes:
        backbone: The pretrained GPT model.
        classifier_head: nn.Linear(n_embd, n_classes)
        dropout: nn.Dropout for regularization before the classification head.
    """

    def __init__(
        self,
        gpt_model: nn.Module,
        n_classes: int = 2,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize the classifier with a pretrained GPT backbone.

        TODO: Implement this method

        Requirements:
        1. Store the GPT model as self.backbone.
        2. Create a dropout layer: self.dropout = nn.Dropout(dropout)
        3. Create the classification head: self.classifier_head = nn.Linear(n_embd, n_classes)
           where n_embd can be obtained from gpt_model.config.n_embd (or a similar attribute).
        4. If freeze_backbone is True, freeze all GPT parameters by setting
           requires_grad = False for each parameter.

        HINT: To get n_embd, look for gpt_model.config.n_embd or count the
        output features of the last layer. The mock in tests uses a simple
        attribute: gpt_model.n_embd or gpt_model.config.n_embd.

        HINT: To freeze: for param in self.backbone.parameters(): param.requires_grad = False

        Args:
            gpt_model: A pretrained GPT model.
            n_classes: Number of output classes.
            freeze_backbone: Whether to freeze the GPT backbone weights.
            dropout: Dropout probability before the classification head.
        """
        super().__init__()
        # TODO: Implement __init__
        # Step 1: Store backbone
        # Step 2: Create dropout layer
        # Step 3: Determine n_embd from the GPT model
        # Step 4: Create linear classification head: Linear(n_embd, n_classes)
        # Step 5: If freeze_backbone, freeze all backbone parameters
        raise NotImplementedError("TODO: Implement GPTClassifier.__init__")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: GPT backbone -> last token -> classification head.

        TODO: Implement this method

        Requirements:
        1. Pass input_ids through the backbone WITHOUT targets:
           logits, _ = self.backbone(input_ids)
           This gives logits of shape (B, T, vocab_size), but we actually
           need the hidden states. See HINT below for how to get them.
        2. Extract the last-token hidden state for each sequence in the batch.
           If attention_mask is provided, use it to find the position of the
           last real (non-padding) token.
           If attention_mask is None, use position T-1 (last position).
        3. Apply dropout.
        4. Pass through self.classifier_head to get class logits.
        5. Return logits of shape (B, n_classes).

        HINT: The GPT model returns (logits, loss). To get hidden states
        instead of vocabulary logits, you need to access the hidden state
        before the final language model head. One approach:
        - Run the backbone up to (but not including) the LM head.
        - Or, if the backbone doesn't expose intermediate states, use a
          simpler approach: run the full backbone and use the output logits
          (shape B, T, vocab_size). This works for testing but is suboptimal.
          For a real implementation, you'd modify the backbone.

        For this lab, a pragmatic approach is to access backbone internals:
        1. Run token + position embeddings
        2. Run through transformer blocks
        3. Apply final layer norm
        4. Take the last-token hidden state
        5. Pass through YOUR classification head (not the LM head)

        If the mock model has a simpler structure (embedding -> head), just
        use the embedding output as the hidden state.

        HINT: To find the last non-padding position with an attention_mask:
        last_pos = attention_mask.sum(dim=1) - 1  # shape (B,)
        hidden = hidden_states[torch.arange(B), last_pos]  # shape (B, n_embd)

        Args:
            input_ids: Token IDs, shape (B, T).
            attention_mask: Optional mask, shape (B, T). 1 = real token, 0 = padding.

        Returns:
            Classification logits of shape (B, n_classes).
        """
        # TODO: Implement forward pass
        # Step 1: Get hidden states from backbone
        # Step 2: Find last non-padding position (or use T-1)
        # Step 3: Extract last-token hidden state: (B, n_embd)
        # Step 4: Apply dropout
        # Step 5: Pass through classifier_head -> (B, n_classes)
        # Step 6: Return class logits
        raise NotImplementedError("TODO: Implement GPTClassifier.forward")


# ---------------------------------------------------------------------------
# Backbone Freeze/Unfreeze Utilities
# ---------------------------------------------------------------------------

def freeze_backbone(model: GPTClassifier) -> None:
    """Freeze all parameters in the GPT backbone.

    After freezing, only the classification head will be updated during
    training. This is the standard "feature extraction" approach.

    TODO: Implement this function

    Requirements:
    1. Set requires_grad = False for all parameters in model.backbone.

    Args:
        model: A GPTClassifier instance.
    """
    # TODO: Implement backbone freezing
    # Step 1: Iterate over model.backbone.parameters()
    # Step 2: Set requires_grad = False for each parameter
    raise NotImplementedError("TODO: Implement freeze_backbone")


def unfreeze_backbone(model: GPTClassifier) -> None:
    """Unfreeze all parameters in the GPT backbone.

    After unfreezing, the entire model (backbone + head) will be updated
    during training. This is "full fine-tuning".

    TODO: Implement this function

    Requirements:
    1. Set requires_grad = True for all parameters in model.backbone.

    Args:
        model: A GPTClassifier instance.
    """
    # TODO: Implement backbone unfreezing
    # Step 1: Iterate over model.backbone.parameters()
    # Step 2: Set requires_grad = True for each parameter
    raise NotImplementedError("TODO: Implement unfreeze_backbone")
