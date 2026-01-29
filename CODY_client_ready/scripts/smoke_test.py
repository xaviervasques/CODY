"""Basic smoke test for the CODY repository.

Goal:
- Validate the Python environment (imports)
- Validate example datasets exist and are readable
- Do NOT run training/CV (keeps it fast)

Run:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)

def main() -> None:
    # 1) Core imports
    import sklearn  # noqa: F401
    import xgboost  # noqa: F401
    import lightgbm  # noqa: F401
    import torch  # noqa: F401
    import openpyxl  # noqa: F401

    # Optional imports used by some pipelines
    try:
        import imblearn  # noqa: F401
        import iterstrat  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing optional dependencies (imbalanced-learn / iterative-stratification). "
            "Install with: pip install -r requirements.txt"
        ) from e

    # 2) Example dataset presence
    ds_lc = REPO_ROOT / "datasets" / "dataset_lc"
    _assert(ds_lc.exists(), f"Missing dataset folder: {ds_lc}")
    xlsx = sorted(ds_lc.glob("*.xlsx"))
    _assert(len(xlsx) > 0, f"No .xlsx files found in: {ds_lc}")

    # 3) Read one file and validate expected columns minimally
    df = pd.read_excel(xlsx[0])
    required_cols = {"From", "To"}
    _assert(required_cols.issubset(df.columns), f"Missing required columns {required_cols} in {xlsx[0].name}")

    # distance columns heuristic
    dist_cols = [c for c in df.columns if c.endswith("_distance")]
    _assert(len(dist_cols) > 0, f"No *_distance columns detected in {xlsx[0].name}")

    print("✅ Smoke test passed.")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Example file: {xlsx[0].name}  | rows={len(df)}  | distance_cols={len(dist_cols)}")

if __name__ == "__main__":
    main()
