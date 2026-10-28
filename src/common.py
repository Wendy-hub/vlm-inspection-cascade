"""Shared paths, category info, and the calibration/evaluation split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJ / "datasets" / "visa"
STAGE1 = PROJ / "outputs" / "stage1"
CASCADE = PROJ / "outputs" / "cascade"

CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]

# Plain-language product descriptions given to the VLM reviewer.
DESCRIPTIONS = {
    "candle": "a tray holding four white candles",
    "capsules": "a group of green capsules",
    "cashew": "a single cashew nut",
    "chewinggum": "a single piece of chewing gum",
    "fryum": "a single wheel-shaped fryum snack",
    "macaroni1": "four macaroni pieces on a plain background",
    "macaroni2": "four macaroni pieces on a textured background",
    "pcb1": "a printed circuit board (PCB) with components",
    "pcb2": "a printed circuit board (PCB) with components",
    "pcb3": "a printed circuit board (PCB) with components",
    "pcb4": "a printed circuit board (PCB) with components",
    "pipe_fryum": "a single pipe-shaped fryum snack",
}

SEED = 0


def load_stage1(category: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (scores dataframe, anomaly maps) for one category, in matching order."""
    df = pd.read_csv(STAGE1 / category / "scores.csv")
    maps = np.load(STAGE1 / category / "maps.npy", mmap_mode="r")
    df["category"] = category
    df["idx"] = np.arange(len(df))
    return df, maps


def assign_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Stratified 50/50 split of the test set into 'calib' (thresholds) and 'eval' (reporting)."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(df), dtype=object)
    for label in (0, 1):
        idx = np.flatnonzero(df["label"].to_numpy() == label)
        rng.shuffle(idx)
        half = len(idx) // 2
        split[idx[:half]] = "calib"
        split[idx[half:]] = "eval"
    out = df.copy()
    out["split"] = split
    return out
