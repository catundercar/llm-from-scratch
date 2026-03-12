"""
Shared testing utilities for all lab phases.

Provides helper functions for common test patterns across the course.
"""

import torch
import sys
from typing import Any


def assert_tensor_shape(
    tensor: torch.Tensor, expected_shape: tuple[int, ...], name: str = "tensor"
) -> None:
    """Assert that a tensor has the expected shape, with a clear error message."""
    assert tensor.shape == expected_shape, (
        f"{name} has shape {tensor.shape}, expected {expected_shape}"
    )


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    name: str = "tensor",
) -> None:
    """Assert that two tensors are element-wise close."""
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{name} values differ beyond tolerance.\n"
        f"  Max absolute diff: {(actual - expected).abs().max().item():.6e}\n"
        f"  actual:   {actual}\n"
        f"  expected: {expected}"
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_deterministic(seed: int = 42) -> None:
    """Set random seeds for reproducibility in tests."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_grading(phase_name: str, test_modules: list[tuple[str, str]], project_dir: str) -> None:
    """
    Run pytest on each test module and print a grading report.

    Args:
        phase_name: Display name for the phase (e.g., "Phase 1 - Text & Data")
        test_modules: List of (lab_name, test_file_path) tuples
        project_dir: Root directory of the phase project
    """
    import subprocess
    import os

    bar_width = 20
    results = []

    for lab_name, test_file in test_modules:
        cmd = [
            sys.executable, "-m", "pytest", test_file,
            "-v", "--tb=no", "-q", "--no-header"
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=project_dir
        )

        # Parse pytest output for pass/fail counts
        output = proc.stdout + proc.stderr
        passed = 0
        failed = 0
        total = 0

        for line in output.strip().splitlines():
            line_stripped = line.strip()
            # Look for the summary line like "5 passed" or "3 passed, 2 failed"
            if "passed" in line_stripped or "failed" in line_stripped:
                parts = line_stripped.split()
                for i, part in enumerate(parts):
                    if part == "passed" or part == "passed,":
                        passed = int(parts[i - 1])
                    if part == "failed" or part == "failed,":
                        failed = int(parts[i - 1])

        total = passed + failed
        results.append((lab_name, passed, total))

    # Print the report
    total_passed = sum(r[1] for r in results)
    total_tests = sum(r[2] for r in results)

    print()
    print("=" * 50)
    print(f"  {phase_name} -- Grading Report")
    print("=" * 50)
    print()

    for lab_name, passed, total in results:
        if total == 0:
            pct = 0
        else:
            pct = int(100 * passed / total)
        filled = int(bar_width * pct / 100)
        empty = bar_width - filled
        bar = "\u2588" * filled + "\u2591" * empty
        print(f"  {lab_name}")
        print(f"  {bar} {pct:3d}%  ({passed}/{total} tests)")
        print()

    print("-" * 50)
    if total_tests == 0:
        overall_pct = 0
    else:
        overall_pct = int(100 * total_passed / total_tests)
    print(f"  Overall: {total_passed}/{total_tests} tests passed ({overall_pct}%)")
    print("=" * 50)
    print()
