"""Shared paths and helpers for training / evaluation (cwd = ai/)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AI_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
IMG_SIZE = 224


def _default_num_workers() -> int:
    import os
    import sys

    # Windows spawn + Jupyter is flaky; Colab/Linux can use more workers.
    if sys.platform.startswith("win"):
        return 0
    return min(4, max(2, (os.cpu_count() or 2) // 2))


NUM_WORKERS = _default_num_workers()


def ensure_ai_cwd() -> Path:
    import os

    os.chdir(AI_ROOT)
    return AI_ROOT


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_manifest(path: str | Path = "metadata/manifest.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["held_out"] = df["held_out"].astype(bool)
    return df


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows allowed in train/val/test for the main detector (excludes JPEG)."""
    return df[~df["held_out"]].copy()


def freeze_snapshot(out_path: str | Path = "metadata/data_freeze.json") -> dict:
    """Record counts so the split is treated as frozen from this point."""
    df = load_manifest()
    snap = {
        "seed": SEED,
        "manifest": "metadata/manifest.csv",
        "total_rows": int(len(df)),
        "held_out_rows": int(df["held_out"].sum()),
        "by_split_label": {
            f"{s}/{lab}": int(n)
            for (s, lab), n in usable(df).groupby(["split", "label"]).size().items()
        },
        "attacks_in_train": sorted(
            usable(df).loc[usable(df)["split"] == "train", "attack"].unique().tolist()
        ),
        "held_out_attacks": sorted(df.loc[df["held_out"], "attack"].unique().tolist()),
        "note": "Do not regenerate attacks or change the split after this freeze.",
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    return snap


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
