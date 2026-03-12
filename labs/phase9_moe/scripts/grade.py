#!/usr/bin/env python3
"""
Grading script for Phase 9: Mixture of Experts.

Run from the phase9_moe directory:
    python scripts/grade.py
"""

import sys
from pathlib import Path

project_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, project_dir)
sys.path.insert(0, str(Path(project_dir).parent / "shared"))

from testing_utils import run_grading

if __name__ == "__main__":
    test_modules = [
        ("Lab 1: MoE Layer", "tests/test_lab1_moe.py"),
        ("Lab 2: MoE Transformer", "tests/test_lab2_moe_transformer.py"),
    ]
    run_grading("Phase 9 -- Mixture of Experts", test_modules, project_dir)
