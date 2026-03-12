#!/usr/bin/env python3
"""
Phase 5 · Auto-Grading Script
===============================

Runs all Phase 5 test suites and prints a visual progress report.

Usage:
    python scripts/grade.py
"""

import subprocess
import sys
import re
from pathlib import Path

PHASE_DIR = Path(__file__).parent.parent
LABS = [
    ("Lab 1: Text Generation Strategies", "tests/test_lab1_generate.py"),
]


def run_tests(test_path: str) -> tuple[int, int]:
    """Run a test file and return (passed, total)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-v", "--tb=no", "-q"],
        capture_output=True,
        text=True,
        cwd=str(PHASE_DIR),
    )
    output = result.stdout + result.stderr

    passed = 0
    failed = 0
    errors = 0

    match = re.search(r"(\d+) passed", output)
    if match:
        passed = int(match.group(1))

    match = re.search(r"(\d+) failed", output)
    if match:
        failed = int(match.group(1))

    match = re.search(r"(\d+) error", output)
    if match:
        errors = int(match.group(1))

    total = passed + failed + errors
    if total == 0:
        match = re.search(r"collected (\d+) item", output)
        if match:
            total = int(match.group(1))

    return passed, max(total, passed)


def progress_bar(passed: int, total: int, width: int = 20) -> str:
    """Create a visual progress bar."""
    if total == 0:
        return " " * width
    ratio = passed / total
    filled = int(width * ratio)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = f"{ratio * 100:.0f}%"
    return f"{bar} {pct:>4s}  ({passed}/{total} tests)"


def main():
    total_passed = 0
    total_tests = 0

    print()
    print("\u2550" * 50)
    print("  Phase 5 \u00b7 Text Generation \u00b7 Grading Report")
    print("\u2550" * 50)
    print()

    for lab_name, test_path in LABS:
        passed, total = run_tests(test_path)
        total_passed += passed
        total_tests += total
        bar = progress_bar(passed, total)
        print(f"  {lab_name}")
        print(f"  {bar}")
        print()

    print("\u2500" * 50)
    if total_tests > 0:
        overall_pct = total_passed / total_tests * 100
    else:
        overall_pct = 0
    print(f"  Overall: {total_passed}/{total_tests} tests passed ({overall_pct:.0f}%)")
    print()


if __name__ == "__main__":
    main()
