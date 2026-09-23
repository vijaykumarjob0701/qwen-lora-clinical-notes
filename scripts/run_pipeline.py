#!/usr/bin/env python3
"""Wrapper so `python scripts/run_pipeline.py` matches `python -m mlops.pipeline`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlops.pipeline import main

if __name__ == "__main__":
    main()
