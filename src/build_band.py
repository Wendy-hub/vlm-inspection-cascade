"""Three-way decision from stage-1 scores: auto-pass / auto-reject / send to review.

Thresholds are fit on the 'calib' half of each category's test set and applied to both halves;
all reported numbers use the 'eval' half only.

  s_lo : highest threshold that keeps recall >= --pass-recall on calib  (score < s_lo -> auto-pass)
  s_hi : lowest threshold with false-alarm rate <= --reject-fpr on calib (score > s_hi -> auto-reject)
  tau  : single-threshold baseline at recall >= --target-recall on calib

Output: outputs/cascade/band.csv and outputs/cascade/thresholds.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from common import CASCADE, CATEGORIES, assign_split, load_stage1


def recall_threshold(bad_scores: np.ndarray, recall: float) -> float:
    """Largest t such that at least `recall` of bad scores are >= t."""
    s = np.sort(bad_scores)
    k = int(math.floor((1 - recall) * len(s) + 1e-9))
    return float(s[min(k, len(s) - 1)])


def fpr_threshold(good_scores: np.ndarray, fpr: float) -> float:
    """Smallest t such that at most `fpr` of good scores are > t."""
    s = np.sort(good_scores)[::-1]
    k = int(math.floor(fpr * len(s) + 1e-9))
    return float(s[min(k, len(s) - 1)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--pass-recall", type=float, default=0.99)
    ap.add_argument("--reject-fpr", type=float, default=0.01)
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = ap.parse_args()

    CASCADE.mkdir(parents=True, exist_ok=True)
    rows, thresholds = [], {}
    for cat in args.categories:
        df, _ = load_stage1(cat)
        df = assign_split(df)
        calib = df[df.split == "calib"]
        bad = calib.loc[calib.label == 1, "score"].to_numpy()
        good = calib.loc[calib.label == 0, "score"].to_numpy()

        tau = recall_threshold(bad, args.target_recall)
        s_lo = recall_threshold(bad, args.pass_recall)
        s_hi = fpr_threshold(good, args.reject_fpr)

        zone = np.where(df.score < s_lo, "pass", np.where(df.score > s_hi, "reject", "review"))
        if s_lo > s_hi:  # perfectly separated on calib: nothing to review
            zone = np.where(df.score >= tau, "reject", "pass")
        df["zone"] = zone
        df["stage1_pred"] = (df.score >= tau).astype(int)
        thresholds[cat] = {"tau": tau, "s_lo": s_lo, "s_hi": s_hi}
        rows.append(df)

    band = pd.concat(rows, ignore_index=True)
    band.to_csv(CASCADE / "band.csv", index=False)
    (CASCADE / "thresholds.json").write_text(json.dumps({"args": vars(args), "thresholds": thresholds}, indent=2))

    summary = band.groupby(["category", "split"]).zone.value_counts().unstack(fill_value=0)
    print(summary.to_string())
    n_review = int((band.zone == "review").sum())
    print(f"\nreview zone: {n_review} of {len(band)} images ({n_review / len(band):.1%}) -> VLM calls per variant")


if __name__ == "__main__":
    main()
