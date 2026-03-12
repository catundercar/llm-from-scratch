"""
Phase 6 · Lab 3: Classification Training
==========================================

Tests for:
- train_epoch runs without error
- evaluate returns accuracy and loss
- train_classifier with early stopping

Difficulty: *** (Intermediate)
"""

import pytest
import torch

from phase6_classification.phase_6.classifier import GPTClassifier
from phase6_classification.phase_6.train_classifier import (
    evaluate,
    train_classifier,
    train_epoch,
)


# ===================================================================
# Tests: train_epoch
# ===================================================================


class TestTrainEpoch:
    """Tests for one epoch of classification training."""

    def test_runs_without_error(self, mock_gpt, classification_dataloader):
        """One training epoch should complete without exceptions."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, classifier.parameters()), lr=1e-3
        )
        loss = train_epoch(classifier, classification_dataloader, optimizer)
        assert isinstance(loss, float)

    def test_returns_positive_loss(self, mock_gpt, classification_dataloader):
        """Training loss should be positive."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, classifier.parameters()), lr=1e-3
        )
        loss = train_epoch(classifier, classification_dataloader, optimizer)
        assert loss > 0

    def test_updates_classifier_head(self, mock_gpt, classification_dataloader):
        """The classification head weights should change after training."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, classifier.parameters()), lr=1e-3
        )

        initial_weight = classifier.classifier_head.weight.clone()
        train_epoch(classifier, classification_dataloader, optimizer)

        assert not torch.allclose(
            classifier.classifier_head.weight, initial_weight
        ), "Classifier head should be updated after training"


# ===================================================================
# Tests: evaluate
# ===================================================================


class TestEvaluate:
    """Tests for evaluation function."""

    def test_returns_dict_with_loss_and_accuracy(self, mock_gpt, classification_dataloader):
        """evaluate() should return a dict with 'loss' and 'accuracy'."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        result = evaluate(classifier, classification_dataloader)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result

    def test_loss_is_positive(self, mock_gpt, classification_dataloader):
        """Evaluation loss should be positive."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        result = evaluate(classifier, classification_dataloader)
        assert result["loss"] > 0

    def test_accuracy_in_valid_range(self, mock_gpt, classification_dataloader):
        """Accuracy should be between 0 and 1."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        result = evaluate(classifier, classification_dataloader)
        assert 0 <= result["accuracy"] <= 1

    def test_accuracy_is_float(self, mock_gpt, classification_dataloader):
        """Accuracy should be a Python float."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        result = evaluate(classifier, classification_dataloader)
        assert isinstance(result["accuracy"], float)


# ===================================================================
# Tests: train_classifier
# ===================================================================


class TestTrainClassifier:
    """Tests for the full classification training loop."""

    def test_runs_without_error(self, mock_gpt, classification_dataloader):
        """Full training loop should complete without exceptions."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        result = train_classifier(
            classifier,
            train_loader=classification_dataloader,
            val_loader=classification_dataloader,
            epochs=2,
            lr=1e-3,
            patience=5,
        )
        assert isinstance(result, dict)

    def test_returns_expected_keys(self, mock_gpt, classification_dataloader):
        """Result should contain the expected keys."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        result = train_classifier(
            classifier,
            train_loader=classification_dataloader,
            val_loader=classification_dataloader,
            epochs=2,
            lr=1e-3,
            patience=5,
        )
        assert "best_val_accuracy" in result
        assert "epochs_trained" in result

    def test_epochs_trained_correct(self, mock_gpt, classification_dataloader):
        """epochs_trained should match the number of epochs actually run."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        result = train_classifier(
            classifier,
            train_loader=classification_dataloader,
            val_loader=classification_dataloader,
            epochs=3,
            lr=1e-3,
            patience=10,  # high patience to avoid early stopping
        )
        assert result["epochs_trained"] == 3

    def test_with_test_loader(self, mock_gpt, classification_dataloader):
        """Should evaluate on test set when test_loader is provided."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        result = train_classifier(
            classifier,
            train_loader=classification_dataloader,
            val_loader=classification_dataloader,
            test_loader=classification_dataloader,
            epochs=2,
            lr=1e-3,
            patience=5,
        )
        assert "final_test_accuracy" in result
        assert result["final_test_accuracy"] is not None

    def test_best_val_accuracy_in_range(self, mock_gpt, classification_dataloader):
        """Best validation accuracy should be between 0 and 1."""
        classifier = GPTClassifier(mock_gpt, n_classes=2, freeze_backbone=True)
        result = train_classifier(
            classifier,
            train_loader=classification_dataloader,
            val_loader=classification_dataloader,
            epochs=2,
            lr=1e-3,
            patience=5,
        )
        assert 0 <= result["best_val_accuracy"] <= 1
