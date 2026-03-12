"""
Tests for Phase 9, Lab 2: MoE-Enhanced Transformer

These tests verify:
    - MoETransformerBlock preserves dimensions and returns router probs
    - MoEGPT has correct architecture (interleaved MoE and dense blocks)
    - MoEGPT has more total params but similar active params to a dense model
    - Loss computation includes auxiliary load balancing loss

Tests are PROVIDED -- students do not modify this file.
"""

import pytest
import torch
from phase_9.moe_transformer import (
    MoETransformerBlock,
    DenseTransformerBlock,
    MoEGPT,
)
from phase_9.types import MoEConfig


class TestMoETransformerBlock:
    """Tests for the MoETransformerBlock."""

    def test_output_shape(
        self, small_moe_config: MoEConfig, dummy_hidden: torch.Tensor
    ) -> None:
        """Output shape should match input shape."""
        torch.manual_seed(0)
        block = MoETransformerBlock(small_moe_config)
        output, router_probs = block(dummy_hidden)
        assert output.shape == dummy_hidden.shape, (
            f"Expected {dummy_hidden.shape}, got {output.shape}"
        )

    def test_returns_router_probs(
        self, small_moe_config: MoEConfig, dummy_hidden: torch.Tensor
    ) -> None:
        """Should return router probabilities (not None)."""
        torch.manual_seed(0)
        block = MoETransformerBlock(small_moe_config)
        _, router_probs = block(dummy_hidden)
        assert router_probs is not None, (
            "MoE block should return router probs, not None"
        )

    def test_has_moe_layer(self, small_moe_config: MoEConfig) -> None:
        """Block should contain an MoE layer (not a dense FFN)."""
        from phase_9.moe import MoELayer
        block = MoETransformerBlock(small_moe_config)
        has_moe = any(isinstance(m, MoELayer) for m in block.modules())
        assert has_moe, "MoETransformerBlock should contain an MoELayer"

    def test_gradient_flows(
        self, small_moe_config: MoEConfig, dummy_hidden: torch.Tensor
    ) -> None:
        """Gradients should flow through the MoE transformer block."""
        torch.manual_seed(0)
        block = MoETransformerBlock(small_moe_config)
        x = dummy_hidden.clone().requires_grad_(True)
        output, _ = block(x)
        output.sum().backward()
        assert x.grad is not None, "Gradients should flow through the block"


class TestMoEGPT:
    """Tests for the MoEGPT model."""

    def test_logits_shape(
        self, small_moe_config: MoEConfig, dummy_input: torch.Tensor
    ) -> None:
        """Logits should have shape (batch, seq_len, vocab_size)."""
        torch.manual_seed(0)
        model = MoEGPT(small_moe_config)
        logits, _ = model(dummy_input)
        expected = (2, 16, small_moe_config.vocab_size)
        assert logits.shape == expected, (
            f"Expected logits shape {expected}, got {logits.shape}"
        )

    def test_loss_computed_with_targets(
        self, small_moe_config: MoEConfig, dummy_input: torch.Tensor
    ) -> None:
        """Loss should be computed when targets are provided."""
        torch.manual_seed(0)
        model = MoEGPT(small_moe_config)
        targets = torch.randint(0, small_moe_config.vocab_size, (2, 16))
        _, loss = model(dummy_input, targets=targets)
        assert loss is not None, "Loss should be computed when targets provided"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_no_loss_without_targets(
        self, small_moe_config: MoEConfig, dummy_input: torch.Tensor
    ) -> None:
        """Loss should be None when no targets are provided."""
        torch.manual_seed(0)
        model = MoEGPT(small_moe_config)
        _, loss = model(dummy_input)
        assert loss is None, "Loss should be None without targets"

    def test_has_interleaved_blocks(self, small_moe_config: MoEConfig) -> None:
        """Model should have both MoE and dense transformer blocks."""
        torch.manual_seed(0)
        model = MoEGPT(small_moe_config)

        has_moe_block = False
        has_dense_block = False
        for module in model.modules():
            if isinstance(module, MoETransformerBlock):
                has_moe_block = True
            if isinstance(module, DenseTransformerBlock):
                has_dense_block = True

        assert has_moe_block, "Model should contain MoE transformer blocks"
        assert has_dense_block, "Model should contain dense transformer blocks"

    def test_more_total_params_than_dense(
        self, small_moe_config: MoEConfig
    ) -> None:
        """MoE model should have more total params than a pure dense model."""
        torch.manual_seed(0)
        moe_model = MoEGPT(small_moe_config)

        # Create a dense-only config for comparison
        dense_config = MoEConfig(
            vocab_size=small_moe_config.vocab_size,
            n_layer=small_moe_config.n_layer,
            n_head=small_moe_config.n_head,
            n_embd=small_moe_config.n_embd,
            block_size=small_moe_config.block_size,
            dropout=small_moe_config.dropout,
            bias=small_moe_config.bias,
            n_experts=1,  # effectively dense
            top_k=1,
            moe_every_n_layers=small_moe_config.n_layer + 1,  # no MoE blocks
        )
        dense_model = MoEGPT(dense_config)

        moe_params = sum(p.numel() for p in moe_model.parameters())
        dense_params = sum(p.numel() for p in dense_model.parameters())

        assert moe_params > dense_params, (
            f"MoE model ({moe_params} params) should have more total params "
            f"than dense model ({dense_params} params)"
        )

    def test_forward_backward_works(
        self, small_moe_config: MoEConfig, dummy_input: torch.Tensor
    ) -> None:
        """Full forward-backward pass should work without errors."""
        torch.manual_seed(0)
        model = MoEGPT(small_moe_config)
        targets = torch.randint(0, small_moe_config.vocab_size, (2, 16))

        logits, loss = model(dummy_input, targets=targets)
        loss.backward()

        # Check that some gradients are non-zero
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Some parameters should have non-zero gradients"
