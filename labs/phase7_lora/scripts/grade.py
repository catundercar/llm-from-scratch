#!/usr/bin/env python3
"""
Grading script for Phase 7: LoRA.

Run from the phase7_lora directory:
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
        ("Lab 1: LoRA Implementation", "tests/test_lab1_lora.py"),
        ("Lab 2: LoRA Training", "tests/test_lab2_train_lora.py"),
    ]
    run_grading("Phase 7 -- LoRA", test_modules, project_dir)
