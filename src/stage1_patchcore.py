"""Stage 1: train PatchCore on each VisA category and save raw image scores and anomaly maps.

Outputs per category (in outputs/stage1/<category>/):
  scores.csv   image_path, label (0 good / 1 bad), score (raw kNN distance, higher = more anomalous)
  maps.npy     float16 anomaly maps, shape [N, 256, 256], same order as scores.csv
  meta.json    image AUROC, train time, per-image inference time, memory bank size

Usage:
  python src/stage1_patchcore.py --prepare-only            # extract VisA + build split once
  python src/stage1_patchcore.py --categories pcb1 pcb2    # train + predict
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from anomalib.data import Visa
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.post_processing import PostProcessor
from sklearn.metrics import roc_auc_score

CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
PROJ = Path(__file__).resolve().parents[1]


def run_category(category: str, root: Path, out_root: Path, coreset: float, workers: int,
                 weights: Path) -> dict:
    out_dir = out_root / category
    out_dir.mkdir(parents=True, exist_ok=True)

    datamodule = Visa(root=root, category=category, train_batch_size=32,
                      eval_batch_size=32, num_workers=workers)
    # Raw kNN distances: no min-max normalization, so scores keep their full ranking.
    # Backbone weights: torchvision's ImageNet WideResNet-50 (as in the PatchCore paper), loaded from a
    # local file because the HF mirror is rate-limited from this server.
    model = Patchcore(backbone="wide_resnet50_2", pre_trained=False, coreset_sampling_ratio=coreset,
                      post_processor=PostProcessor(enable_normalization=False),
                      evaluator=False, visualizer=False)
    backbone = model.model.feature_extractor.feature_extractor
    missing, _ = backbone.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
    assert not missing, f"backbone weights missing keys: {missing[:5]}"
    engine = Engine(default_root_dir=str(out_dir / "anomalib"), accelerator="gpu",
                    devices=1, logger=False)

    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    train_s = time.time() - t0

    t0 = time.time()
    batches = engine.predict(model=model, datamodule=datamodule, return_predictions=True)
    infer_s = time.time() - t0

    def rel(p: str) -> str:
        try:
            return str(Path(p).resolve().relative_to(root.resolve()))
        except ValueError:
            return str(p)

    paths, labels, scores, maps = [], [], [], []
    for b in batches:
        paths += [rel(p) for p in b.image_path]
        labels += b.gt_label.int().cpu().tolist()
        scores += b.pred_score.float().cpu().reshape(-1).tolist()
        amap = b.anomaly_map.float().cpu()
        if amap.ndim == 4:
            amap = amap[:, 0]
        maps.append(amap.numpy().astype(np.float16))

    with open(out_dir / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "score"])
        w.writerows(zip(paths, labels, scores))
    np.save(out_dir / "maps.npy", np.concatenate(maps))

    meta = {
        "category": category,
        "n_test": len(labels),
        "n_bad": int(sum(labels)),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "train_seconds": round(train_s, 1),
        "infer_ms_per_image": round(1000 * infer_s / len(labels), 1),
        "memory_bank_size": int(model.model.memory_bank.shape[0]),
        "coreset_sampling_ratio": coreset,
        "backbone_weights": weights.name,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)

    del model, engine
    torch.cuda.empty_cache()
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=PROJ / "datasets" / "visa")
    ap.add_argument("--out", type=Path, default=PROJ / "outputs" / "stage1")
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)  # shared machine: keep CPU use modest
    ap.add_argument("--weights", type=Path,
                    default=Path(os.environ.get("WRN50_WEIGHTS",
                                                PROJ / "models" / "torchvision" / "wide_resnet50_2-95faca4d.pth")))
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(4)
    if args.prepare_only:
        # Extracting and splitting touches every category, so do it once before parallel runs.
        Visa(root=args.root, category=args.categories[0]).prepare_data()
        print("prepared:", sorted(p.name for p in (args.root / "visa_pytorch").iterdir()))
        return

    for c in args.categories:
        if (args.out / c / "meta.json").exists():
            print(f"skip {c} (done)")
            continue
        run_category(c, args.root, args.out, args.coreset, args.workers, args.weights)


if __name__ == "__main__":
    main()
