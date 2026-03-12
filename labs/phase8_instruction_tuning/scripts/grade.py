#!/usr/bin/env python3
"""
Grading script for Phase 8: Instruction Fine-Tuning.

Run from the phase8_instruction_tuning directory:
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
        ("Lab 1: Supervised Fine-Tuning", "tests/test_lab1_sft.py"),
        ("Lab 2: DPO", "tests/test_lab2_dpo.py"),
        ("Lab 3: Evaluation", "tests/test_lab3_evaluate.py"),
    ]
    run_grading("Phase 8 -- Instruction Fine-Tuning", test_modules, project_dir)
