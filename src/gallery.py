"""Build a small HTML gallery of reviewer decisions on the eval half (caught / cleared / missed / false alarm).

Usage: python src/gallery.py --variant zoom_ref --per-group 4
Output: outputs/cascade/gallery_<variant>.html (self-contained, images inlined)
"""

from __future__ import annotations

import argparse
import html

import numpy as np
import pandas as pd
from PIL import Image

from common import CASCADE, DATA_ROOT, load_stage1
from vlm_review import CROP_SIDE, crop_at, find_peaks, heat_overlay, resize_max, to_data_url

GROUPS = [
    ("caught", "defect, VLM said defect", 1, True),
    ("cleared", "normal, VLM said normal (false alarm removed)", 0, False),
    ("missed", "defect, VLM said normal (dangerous)", 1, False),
    ("false_alarm", "normal, VLM said defect", 0, True),
]


def card(r, vmax: float) -> str:
    full = Image.open(DATA_ROOT / r.image_path).convert("RGB")
    _, maps = load_stage1(r.category)
    amap = np.asarray(maps[int(r.idx)], np.float32)
    small = resize_max(full, 360)
    crop = crop_at(full, find_peaks(amap, k=1)[0], amap.shape).resize((CROP_SIDE // 2, CROP_SIDE // 2))
    imgs = "".join(f'<img src="{to_data_url(x)}">' for x in (small, crop, heat_overlay(small, amap, vmax)))
    conf = "" if pd.isna(r.confidence) else f"{r.confidence:.0f}"
    return (f'<div class="card"><div class="imgs">{imgs}</div>'
            f"<p><b>{html.escape(r.category)}</b> · score {r.score:.2f} · confidence {conf}</p>"
            f"<p>{html.escape(str(r.type))} @ {html.escape(str(r.location))}</p>"
            f"<p class='why'>{html.escape(str(r.reason))}</p></div>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="zoom_ref")
    ap.add_argument("--per-group", type=int, default=4)
    args = ap.parse_args()

    band = pd.read_csv(CASCADE / "band.csv")
    vlm = pd.read_json(CASCADE / f"vlm_{args.variant}.jsonl", lines=True)
    df = vlm.merge(band[["category", "image_path", "idx"]], on=["category", "image_path"])
    df = df[(df.split == "eval") & df.defect.notna()]
    import json
    vmax = {c: v["s_hi"] for c, v in json.loads((CASCADE / "thresholds.json").read_text())["thresholds"].items()}

    sections = []
    for key, title, label, said in GROUPS:
        g = df[(df.label == label) & (df.defect == said)]
        g = g.sample(min(args.per_group, len(g)), random_state=0) if len(g) else g
        cards = "".join(card(r, vmax[r.category]) for r in g.itertuples())
        sections.append(f"<h2>{title} ({len(df[(df.label == label) & (df.defect == said)])} images)</h2>"
                        f'<div class="grid">{cards or "<p>none</p>"}</div>')

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Reviewer gallery: {args.variant}</title>
<style>body{{font-family:Arial,sans-serif;margin:16px;background:#fff;color:#14181f}}
.grid{{display:flex;flex-wrap:wrap;gap:12px}} .card{{border:1px solid #d3d9e2;padding:8px;width:min(100%,600px)}}
.imgs{{display:flex;gap:4px;flex-wrap:wrap}} .imgs img{{max-height:180px}} p{{margin:4px 0;font-size:13px}}
.why{{color:#5b6472}}</style></head><body>
<h1>VLM reviewer decisions on uncertain parts ({args.variant}, eval half)</h1>
<p>Each card: full part, crop at the detector's top heatmap peak, heatmap overlay.</p>
{''.join(sections)}</body></html>"""
    out = CASCADE / f"gallery_{args.variant}.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
