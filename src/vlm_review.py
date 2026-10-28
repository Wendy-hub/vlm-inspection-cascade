"""Stage 2: a VLM reviewer re-judges the images that stage 1 sent to review.

Reviewer variants (each adds one tool, for an ablation):
  img            full image only
  zoom           + 2 crops at the stage-1 heatmap peaks, cut from the full-resolution image
  zoom_ref       + 2 nearest defect-free reference images of the same product
  zoom_ref_heat  + the stage-1 heatmap overlay

Talks to a vLLM OpenAI-compatible server. Results go to outputs/cascade/vlm_<variant><suffix>.jsonl
and the script resumes where it stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib import colormaps
from openai import AsyncOpenAI
from PIL import Image

from common import CASCADE, DATA_ROOT, DESCRIPTIONS, load_stage1

VARIANTS = ["img", "zoom", "zoom_ref", "zoom_ref_heat"]
FULL_SIDE = 768
CROP_SIDE = 448
THUMB = 48

SYSTEM = ("You are an experienced visual quality inspector on a production line. "
          "You decide whether the part shown has a manufacturing defect.")
INSTRUCTIONS = (
    "Decide whether the inspected part has a defect. Defects can be small: scratches, cracks, chips, "
    "holes, dents, stains or spots, burns, deformation, missing or extra pieces, bent or damaged "
    "components, wrong color. Normal variation in position, rotation, lighting, background and natural "
    "texture is NOT a defect.\n"
    'Reply with JSON only: {"defect": true or false, "confidence": integer 0-100, '
    '"type": short defect name or "none", "location": short description or "none", '
    '"reason": one sentence}'
)


# ---------- image tools ----------

def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def resize_max(img: Image.Image, side: int) -> Image.Image:
    w, h = img.size
    s = side / max(w, h)
    return img.resize((round(w * s), round(h * s)), Image.BICUBIC) if s < 1 else img


def find_peaks(amap: np.ndarray, k: int = 2, radius: int = 40) -> list[tuple[int, int]]:
    a = amap.astype(np.float32).copy()
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    peaks = []
    for _ in range(k):
        y, x = np.unravel_index(int(np.argmax(a)), a.shape)
        peaks.append((int(y), int(x)))
        a[(yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2] = -np.inf
    return peaks


def crop_box(size: tuple[int, int], peak: tuple[int, int], grid: tuple[int, int],
             frac: float = 0.3) -> tuple[int, int, int]:
    """(x0, y0, side) of a square crop (side = frac * short side) centred on a heatmap peak, kept inside the image."""
    w, h = size
    side = int(frac * min(w, h))
    cx, cy = peak[1] * w / grid[1], peak[0] * h / grid[0]
    x0 = int(min(max(cx - side / 2, 0), w - side))
    y0 = int(min(max(cy - side / 2, 0), h - side))
    return x0, y0, side


def crop_at(img: Image.Image, peak: tuple[int, int], grid: tuple[int, int], frac: float = 0.3) -> Image.Image:
    x0, y0, side = crop_box(img.size, peak, grid, frac)
    return img.crop((x0, y0, x0 + side, y0 + side)).resize((CROP_SIDE, CROP_SIDE), Image.BICUBIC)


def heat_overlay(img_small: Image.Image, amap: np.ndarray, vmax: float) -> Image.Image:
    m = np.clip(amap.astype(np.float32) / max(vmax, 1e-6), 0, 1)
    heat = Image.fromarray((colormaps["jet"](m)[..., :3] * 255).astype(np.uint8)).resize(img_small.size, Image.BILINEAR)
    return Image.blend(img_small.convert("RGB"), heat, 0.45)


def thumb(path) -> np.ndarray:
    im = Image.open(path)
    im.draft("L", (THUMB * 4, THUMB * 4))  # fast JPEG downscale while decoding
    v = np.asarray(im.convert("L").resize((THUMB, THUMB), Image.BILINEAR), np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


@lru_cache(maxsize=None)
def train_thumbs(category: str) -> tuple[list, np.ndarray]:
    cache = CASCADE / "thumbs" / f"{category}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return list(d["paths"]), d["vecs"]
    paths = sorted((DATA_ROOT / "visa_pytorch" / category / "train" / "good").glob("*"))
    vecs = np.stack([thumb(p) for p in paths])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, paths=np.array([str(p) for p in paths], dtype=object), vecs=vecs)
    return [str(p) for p in paths], vecs


@lru_cache(maxsize=None)
def stage1_maps(category: str) -> np.ndarray:
    return load_stage1(category)[1]


# ---------- prompt ----------

def build_messages(row, variant: str, vmax: float) -> list[dict]:
    desc = DESCRIPTIONS[row.category]
    full = Image.open(DATA_ROOT / row.image_path).convert("RGB")
    amap = np.asarray(stage1_maps(row.category)[row.idx], np.float32)
    parts: list[dict] = []

    def add(img: Image.Image, caption: str) -> None:
        n = sum(p["type"] == "image_url" for p in parts) + 1
        parts.append({"type": "text", "text": f"Image {n}: {caption}"})
        parts.append({"type": "image_url", "image_url": {"url": to_data_url(img)}})

    small = resize_max(full, FULL_SIDE)
    add(small, f"{desc} under inspection (full view).")
    if variant.startswith("zoom"):
        for i, pk in enumerate(find_peaks(amap)):
            add(crop_at(full, pk, amap.shape),
                f"zoomed-in view of suspicious region {i + 1} flagged by an automatic detector "
                "(the detector is often wrong, so this region may be perfectly normal).")
    if "ref" in variant:
        paths, vecs = train_thumbs(row.category)
        sims = vecs @ thumb(DATA_ROOT / row.image_path)
        for j in np.argsort(-sims)[:2]:
            add(resize_max(Image.open(paths[j]).convert("RGB"), FULL_SIDE),
                f"a defect-free reference example of {desc}.")
    if "heat" in variant:
        add(heat_overlay(small, amap, vmax),
            "the detector's anomaly heatmap over the part (red = unusual, blue = normal); use it only as a hint.")
    parts.append({"type": "text", "text": INSTRUCTIONS})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": parts}]


def p_defect_from_logprobs(logprobs) -> float | None:
    """P(defect) read from the token that answers "defect": true/false.

    The model's self-reported confidence is nearly constant (often 80), so the token probability is
    used as a calibrated score instead.
    """
    if not logprobs or not logprobs.content:
        return None
    for tok in logprobs.content:
        if tok.token.strip().lower() in ("true", "false"):
            p = {"true": 0.0, "false": 0.0}
            for alt in tok.top_logprobs or []:
                key = alt.token.strip().lower()
                if key in p:
                    p[key] += float(np.exp(alt.logprob))
            total = p["true"] + p["false"]
            return p["true"] / total if total > 0 else None
    return None


def parse_reply(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    defect = d.get("defect")
    if isinstance(defect, str):
        defect = defect.strip().lower() in ("true", "yes", "1")
    try:
        conf = float(d.get("confidence"))
    except (TypeError, ValueError):
        conf = None
    return {"defect": defect if isinstance(defect, bool) else None, "confidence": conf,
            "type": d.get("type"), "location": d.get("location"), "reason": d.get("reason")}


# ---------- runner ----------

async def run_variant(rows: list, variant: str, args, client: AsyncOpenAI, vmax: dict, pool) -> None:
    out_path = CASCADE / f"vlm_{variant}{args.suffix}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["category"], r["image_path"]))
    todo = [r for r in rows if (r.category, r.image_path) not in done]
    print(f"[{variant}] {len(todo)} to review ({len(done)} already done)", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    loop = asyncio.get_running_loop()
    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as f:

        async def one(row) -> None:
            async with sem:
                messages = await loop.run_in_executor(pool, build_messages, row, variant, vmax[row.category])
                t0 = time.time()
                try:
                    resp = await client.chat.completions.create(
                        model=args.model, messages=messages, temperature=0, max_tokens=256,
                        logprobs=True, top_logprobs=5)
                    text, usage = resp.choices[0].message.content, resp.usage
                    ptok, ctok = usage.prompt_tokens, usage.completion_tokens
                    p_def = p_defect_from_logprobs(resp.choices[0].logprobs)
                except Exception as e:  # keep going; failed calls count as abstentions
                    text, ptok, ctok, p_def = f"ERROR: {e}", None, None, None
                rec = {"category": row.category, "image_path": row.image_path, "label": int(row.label),
                       "split": row.split, "score": float(row.score), "variant": variant,
                       **parse_reply(text), "p_defect": p_def,
                       "raw": text, "latency_s": round(time.time() - t0, 3),
                       "prompt_tokens": ptok, "completion_tokens": ctok}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

        await asyncio.gather(*(one(r) for r in todo))
    if todo:
        dt = time.time() - t_start
        print(f"[{variant}] done in {dt:.0f}s ({len(todo) / dt:.2f} images/s)", flush=True)


async def main_async(args) -> None:
    band = pd.read_csv(CASCADE / "band.csv")
    rows = band[band.zone == "review"]
    if args.split != "all":
        rows = rows[rows.split == args.split]
    if args.limit:
        rows = rows.groupby("category").head(max(1, args.limit // 12))
    rows = list(rows.itertuples(index=False))
    th = json.loads((CASCADE / "thresholds.json").read_text())["thresholds"]
    vmax = {c: v["s_hi"] for c, v in th.items()}

    # Build reference thumbnails up front, one category at a time (not from worker threads).
    for c in sorted({r.category for r in rows}):
        stage1_maps(c)
        if any("ref" in v for v in args.variants):
            train_thumbs(c)

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)
    with ThreadPoolExecutor(max_workers=4) as pool:  # image prep; shared machine, keep it small
        for v in args.variants:
            await run_variant(rows, v, args, client, vmax, pool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--base-url", default="http://127.0.0.1:8901/v1")
    ap.add_argument("--model", default="qwen2.5-vl-7b")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--split", default="all", choices=["all", "calib", "eval"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test: about this many images in total")
    ap.add_argument("--suffix", default="", help="output file suffix, e.g. _smoke or _latency")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
