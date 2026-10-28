"""README figure: one random example of each decision the hybrid policy makes on the review band.

Outcomes on the eval half's review band, for one reviewer variant and its hybrid thresholds
(t_ok, t_def from results.json):
  caught       defect, P(defect) >= t_def                       -> auto-reject
  cleared      normal, P(defect) <= t_ok                        -> auto-pass
  to_human     defect the VLM answered "normal" for, but P(defect) > t_ok -> a human
  false_alarm  normal, P(defect) >= t_def                       -> auto-reject
Each panel shows the part with the true defect outlined (VisA mask) and the two regions the VLM
zoomed into, one of those zooms, P(defect) and the VLM's own one-sentence reason.

Usage: python src/example_figure.py [--variant zoom_ref_heat] [--seed 0]
Output: outputs/cascade/fig_examples.jpg
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from common import CASCADE, DATA_ROOT, load_stage1
from evaluate import load_vlm
from vlm_review import CROP_SIDE, crop_box, find_peaks

INK, INK2, RULE, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"  # status colours, always with an icon + label
TRUTH = "#e0218a"  # true-defect outline: magenta, deliberately not a status colour
FULL_SIDE = 560    # display size of the full part


def truth_mask(image_path: str) -> np.ndarray | None:
    p = Path(image_path)  # visa_pytorch/<cat>/test/bad/000.JPG -> visa_pytorch/<cat>/ground_truth/bad/000.png
    mask = DATA_ROOT / p.parent.parent.parent / "ground_truth" / p.parent.name / f"{p.stem}.png"
    return np.asarray(Image.open(mask).convert("L")) > 0 if mask.exists() else None


def outline(ax, mask: np.ndarray | None) -> None:
    if mask is None or not mask.any():
        return
    lims = ax.get_xlim(), ax.get_ylim()
    for color, lw in (("white", 3.0), (TRUTH, 1.5)):  # white halo keeps the line visible on any part
        ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=lw)
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])


def tag(ax, x: float, y: float, text: str) -> None:
    ax.text(x, y, text, color="white", fontsize=6.5, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="square,pad=0.18", fc="black", ec="none", alpha=0.75))


def picture(ax, img: Image.Image) -> None:
    ax.imshow(np.asarray(img))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(RULE)
        s.set_linewidth(0.6)


def dashed_box(ax, xy: tuple[float, float], w: float, h: float, transform=None) -> None:
    kw = {"transform": transform} if transform is not None else {}
    ax.add_patch(Rectangle(xy, w, h, fill=False, ec="black", lw=2.4, alpha=0.75, **kw))
    ax.add_patch(Rectangle(xy, w, h, fill=False, ec="white", lw=1.2, ls=(0, (3, 2)), **kw))


def panel(fig, gs, col: int, r, head: tuple[str, str, str], sub: str, count: str) -> None:
    icon, title, color = head

    ax = fig.add_subplot(gs[0, col])
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0.9), 1, 0.08, transform=ax.transAxes, color=color, clip_on=False))
    ax.text(0, 0.62, icon, transform=ax.transAxes, color=color, fontsize=10.5, fontweight="bold", va="center")
    ax.text(0.085, 0.62, title, transform=ax.transAxes, color=INK, fontsize=10, fontweight="bold", va="center")
    ax.text(0, 0.33, sub, transform=ax.transAxes, color=INK2, fontsize=8.2, va="center")
    ax.text(0, 0.07, count, transform=ax.transAxes, color=INK2, fontsize=8.2, va="center")

    full = Image.open(DATA_ROOT / r.image_path).convert("RGB")
    _, maps = load_stage1(r.category)
    amap = np.asarray(maps[int(r.idx)], np.float32)
    boxes = [crop_box(full.size, pk, amap.shape) for pk in find_peaks(amap)]  # the VLM's two zooms
    mask = truth_mask(r.image_path) if r.label == 1 else None
    if mask is not None and mask.shape != (full.height, full.width):
        mask = np.asarray(Image.fromarray(mask).resize(full.size, Image.NEAREST))

    # Full part, with the true defect and the zoomed regions.
    s = FULL_SIDE / max(full.size)
    small = full.resize((round(full.width * s), round(full.height * s)), Image.BICUBIC)
    ax = fig.add_subplot(gs[1, col])
    picture(ax, small)
    if mask is not None:
        outline(ax, np.asarray(Image.fromarray(mask).resize(small.size, Image.NEAREST)))
    for k, (x0, y0, side) in enumerate(boxes, 1):
        dashed_box(ax, (x0 * s - 0.5, y0 * s - 0.5), side * s, side * s)
        tag(ax, x0 * s + 4, y0 * s + 4, str(k))

    # One zoom: the one that overlaps the true defect most (defects), otherwise the top peak.
    k = 0
    if mask is not None:
        k = int(np.argmax([mask[y0:y0 + side, x0:x0 + side].sum() for x0, y0, side in boxes]))
    x0, y0, side = boxes[k]
    ax = fig.add_subplot(gs[2, col])
    picture(ax, full.crop((x0, y0, x0 + side, y0 + side)).resize((CROP_SIDE, CROP_SIDE), Image.BICUBIC))
    if mask is not None:
        crop_mask = Image.fromarray(mask[y0:y0 + side, x0:x0 + side]).resize((CROP_SIDE, CROP_SIDE), Image.NEAREST)
        outline(ax, np.asarray(crop_mask))
    tag(ax, 8, 8, f"zoom {k + 1}")

    # P(defect) and the VLM's own words.
    ax = fig.add_subplot(gs[3, col])
    ax.axis("off")
    answer = "defect" if r.defect else "normal"
    kind = f" ({r.type})" if r.defect and isinstance(r.type, str) and r.type.lower() != "none" else ""
    ax.text(0, 0.98, f"{r.category} · p = {r.p_defect:.3f}", transform=ax.transAxes,
            color=INK, fontsize=8.4, fontweight="bold", va="top")
    answer_line = textwrap.shorten(f"VLM answer: {answer}{kind}", width=36, placeholder="…)")
    ax.text(0, 0.81, answer_line, transform=ax.transAxes, color=INK, fontsize=8.2, va="top")
    lines = textwrap.wrap(f"“{r.reason}”", width=35)
    if len(lines) > 4:
        lines = lines[:4]
        lines[-1] = lines[-1][:32].rstrip() + "…”"
    ax.text(0, 0.64, "\n".join(lines), transform=ax.transAxes, color=INK2, fontsize=7.9, fontstyle="italic",
            va="top", linespacing=1.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="zoom_ref_heat")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pol = json.loads((CASCADE / "results.json").read_text())["policies"][f"hybrid[{args.variant}]"]
    t_ok, t_def = pol["t_ok"], pol["t_def"]

    band = pd.read_csv(CASCADE / "band.csv")
    vlm = load_vlm(args.variant)[["category", "image_path", "defect", "p_defect", "type", "location", "reason"]]
    df = band.merge(vlm, on=["category", "image_path"], how="left")
    df = df[(df.zone == "review") & (df.split == "eval") & df.p_defect.notna()]
    p, bad, good = df.p_defect, df.label == 1, df.label == 0
    n_bad, n_good = int(bad.sum()), int(good.sum())

    said_normal = bad & (df.defect == False)  # noqa: E712  (defects the VLM answered "normal" for)
    to_human = said_normal & (p > t_ok) & (p < t_def)
    n_said, n_human = int(said_normal.sum()), int(to_human.sum())
    outcomes = {  # key: (rows, header, rule line, count line)
        "caught": (bad & (p >= t_def), ("✓", "Caught", GOOD),
                   f"defect, p ≥ {t_def:g} → auto-reject", f"{int((bad & (p >= t_def)).sum())} of {n_bad} defects"),
        "cleared": (good & (p <= t_ok), ("✓", "Cleared", GOOD),
                    f"normal, p ≤ {t_ok:g} → auto-pass", f"{int((good & (p <= t_ok)).sum())} of {n_good} normal parts"),
        "to_human": (to_human, ("⚠", "Sent to a human", WARNING), "defect the VLM called normal",
                     (f"all {n_said}" if n_human == n_said else f"{n_human} of {n_said}")
                     + f" had {t_ok:g} < p < {t_def:g}"),
        "false_alarm": (good & (p >= t_def), ("✗", "False alarm", CRITICAL),
                        f"normal, p ≥ {t_def:g} → auto-reject", f"{int((good & (p >= t_def)).sum())} of {n_good} normal parts"),
    }
    n_missed = int((bad & (p <= t_ok)).sum())
    n_auto = int(((p >= t_def) | (p <= t_ok)).sum())

    # Random example per outcome; rarest outcome first so that each panel shows a different product.
    picks, used = {}, set()
    for key in sorted(outcomes, key=lambda k: outcomes[k][0].sum()):
        g = df[outcomes[key][0]].sample(frac=1, random_state=args.seed)
        if g.empty:
            continue
        fresh = g[~g.category.isin(used)]
        picks[key] = (fresh if len(fresh) else g).iloc[0]
        used.add(picks[key].category)
    keys = [k for k in outcomes if k in picks]

    fig = plt.figure(figsize=(8.8, 5.3), facecolor=SURFACE)
    gs = fig.add_gridspec(4, len(keys), height_ratios=[0.6, 1.37, 1.37, 0.9], left=0.014, right=0.986,
                          top=0.865, bottom=0.07, wspace=0.09, hspace=0.07)
    for col, key in enumerate(keys):
        rows, head, rule, count = outcomes[key]
        panel(fig, gs, col, picks[key], head, rule, count)
        r = picks[key]
        print(f"{key:12s} n={int(rows.sum()):3d}  {r.category:11s} {r.image_path}  p={r.p_defect:.3f}  "
              f"answer={'defect' if r.defect else 'normal'}  type={r.type}  reason={r.reason}")

    fig.text(0.014, 0.985, "What the VLM reviewer decides on uncertain parts", fontsize=12, fontweight="bold",
             color=INK, va="top")
    missed = "auto-passes no defect" if n_missed == 0 else f"auto-passes {n_missed} defects"
    sub = (f"Held-out review band: {len(df)} parts ({n_bad} defective, {n_good} normal). p = the VLM's P(defect). "
           f"With t_ok = {t_ok:g} and t_def = {t_def:g}, reviewer {args.variant} decides {n_auto} parts without a "
           f"human and {missed}. One random example per outcome (seed {args.seed}).")
    fig.text(0.014, 0.93, textwrap.fill(sub, width=130), fontsize=8.4, color=INK2, va="top", linespacing=1.35)

    leg = fig.add_axes([0.014, 0.012, 0.972, 0.042])
    leg.axis("off")
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    leg.plot([0, 0.03], [0.5, 0.5], color="white", lw=3.0, solid_capstyle="butt")
    leg.plot([0, 0.03], [0.5, 0.5], color=TRUTH, lw=1.5, solid_capstyle="butt")
    leg.text(0.037, 0.5, "true defect (VisA mask)", fontsize=8, color=INK2, va="center")
    dashed_box(leg, (0.235, 0.12), 0.022, 0.76)
    leg.text(0.264, 0.5, "the two regions the VLM zoomed into (detector heatmap peaks); one zoom is shown below "
             "each part", fontsize=8, color=INK2, va="center")

    out = CASCADE / "fig_examples.jpg"
    fig.savefig(out, dpi=160, facecolor=SURFACE, pil_kwargs={"quality": 92})
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
