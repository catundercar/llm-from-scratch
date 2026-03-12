"""
Phase 6 · Lab 2: GPT Classifier
=================================

Tests for:
- GPTClassifier output shape
- Backbone freezing and unfreezing
- Forward pass with and without attention mask

Difficulty: *** (Intermediate)
"""

import pytest
import torch

from phase6_classification.phase_6.classifier import (
    GPTClassifier,
    freeze_backbone,
    unfreeze_backbone,
)


# ===================================================================
# Tests: GPTClassifier
# ===================================================================


class TestGPTClassifier:
    """Tests for the GPT-based classifier."""

    def test_output_shape(self, mock_gpt):
        """Output should have shape (batch_size, n_classes)."""
        n_classes = 2
        classifier = GPTClassifier(mock_gpt, n_classes=n_classes, freeze_backbone=False)
        input_ids = torch.randint(0, 64, (4, 16))
        logits = classifier(input_ids)
        assert logits.shape == (4, n_classes)

    def test_output_shape_with_more_classes(self, mock_gpt):
        """Should work with different numbers of classes."""
        n_classes = 5
        classifier = GPTClassifier(mock_gpt, n_classes=n_classes, freeze_backbone=False)
        input_ids = torch.randint(0, 64, (2, 16))
        logits = classifier(input_ids)
        assert logits.shape == (2, n_classes)

    def test_forward_with_attention_mask(self, mock_gpt):
        """Forward should accept and use an attention mask."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=False)
        input_ids = torch.randint(0, 64, (3, 16))
        mask = torch.ones(3, 16, dtype=torch.long)
        mask[:, 10:] = 0  # last 6 positions are padding
        logits = classifier(input_ids, attention_mask=mask)
        assert logits.shape == (3, 2)

    def test_forward_without_attention_mask(self, mock_gpt):
        """Forward should work without an attention mask (use last position)."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=False)
        input_ids = torch.randint(0, 64, (3, 16))
        logits = classifier(input_ids)
        assert logits.shape == (3, 2)

    def test_output_is_differentiable(self, mock_gpt):
        """Output should have gradients when backbone is not frozen."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=False)
        input_ids = torch.randint(0, 64, (2, 16))
        logits = classifier(input_ids)
        loss = logits.sum()
        loss.backward()
        # Classification head should have gradients
        assert classifier.classifier_head.weight.grad is not None


# ===================================================================
# Tests: Backbone Freezing
# ===================================================================


class TestBackboneFreezing:
    """Tests for freeze/unfreeze backbone utilities."""

    def test_freeze_backbone_on_init(self, mock_gpt):
        """When freeze_backbone=True, backbone params should not require grad."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        for param in classifier.backbone.parameters():
            assert not param.requires_grad, \
                "Backbone params should be frozen when freeze_backbone=True"

    def test_classifier_head_always_trainable(self, mock_gpt):
        """The classification head should always require grad."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        assert classifier.classifier_head.weight.requires_grad

    def test_freeze_backbone_function(self, mock_gpt):
        """freeze_backbone() should set all backbone params to not require grad."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=False)
        # Initially all should require grad
        for param in classifier.backbone.parameters():
            assert param.requires_grad

        freeze_backbone(classifier)

        for param in classifier.backbone.parameters():
            assert not param.requires_grad

    def test_unfreeze_backbone_function(self, mock_gpt):
        """unfreeze_backbone() should set all backbone params to require grad."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        # Initially frozen
        for param in classifier.backbone.parameters():
            assert not param.requires_grad

        unfreeze_backbone(classifier)

        for param in classifier.backbone.parameters():
            assert param.requires_grad

    def test_frozen_backbone_no_grad_update(self, mock_gpt):
        """With frozen backbone, backward should not update backbone weights."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        input_ids = torch.randint(0, 64, (2, 16))
        logits = classifier(input_ids)
        loss = logits.sum()
        loss.backward()

        for param in classifier.backbone.parameters():
            assert param.grad is None or (param.grad == 0).all(), \
                "Frozen backbone should not accumulate gradients"
