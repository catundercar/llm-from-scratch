#!/usr/bin/env python3
"""Auto-grading script for Phase 1: Text & Data."""

import subprocess
import sys

def main():
    print("=" * 50)
    print("  Phase 1 Grading: Text & Data")
    print("=" * 50)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(__import__("pathlib").Path(__file__).parent.parent),
    )
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
