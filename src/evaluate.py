"""Score every policy on the held-out 'eval' half and write tables + figures.

Policies
  stage1_tau     single PatchCore threshold (recall >= 95% on calib), no review
  3way_human     auto-pass / auto-reject / a human re-judges the review zone (human = ground truth)
  3way_vlm[v]    the VLM's yes/no verdict decides the whole review zone, no human
  3way_vlmP[v]   the VLM's P(defect) decides the review zone (p >= t), t picked on calib for recall
  hybrid[v]      review zone: p >= t_def -> reject, p <= t_ok -> pass, else a human.
                 (t_ok, t_def) picked on calib: minimise human work subject to recall >= --min-recall
                 and false alarms <= 3way_human's + --max-fa-increase.
P(defect) is the probability of the "true" token in the reply, not the model's self-reported confidence.
Outputs: outputs/cascade/results.json, results.md, fig_*.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import CASCADE, CATEGORIES, STAGE1

VARIANTS = ["img", "zoom", "zoom_ref", "zoom_ref_heat"]
T_OK = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]  # 0.0 = never pass on VLM say-so
T_DEF = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]         # 1.01 = never reject on VLM say-so
T_AUTO = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def rates(label: np.ndarray, pred: np.ndarray, human: np.ndarray) -> dict:
    bad, good = label == 1, label == 0
    return {
        "recall": float(pred[bad].mean()),
        "miss_rate": float(1 - pred[bad].mean()),
        "false_alarm_rate": float(pred[good].mean()),
        "human_review_rate": float(human.mean()),
        "n": int(len(label)),
    }


def base_pred(df: pd.DataFrame) -> np.ndarray:
    return df.zone.map({"pass": 0, "reject": 1, "review": -1}).to_numpy().copy()


def human_pred(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred, rev = base_pred(df), df.zone.to_numpy() == "review"
    pred[rev] = df.label.to_numpy()[rev]
    return pred, rev


def verdict_pred(df: pd.DataFrame) -> np.ndarray:
    pred, rev = base_pred(df), df.zone.to_numpy() == "review"
    pred[rev] = df.vlm_defect.map({True: 1, False: 0}).fillna(1).to_numpy()[rev]  # abstain -> defect
    return pred


def p_auto_pred(df: pd.DataFrame, t: float) -> np.ndarray:
    pred, rev = base_pred(df), df.zone.to_numpy() == "review"
    p = df.p_defect.to_numpy(dtype=float)
    pred[rev] = np.where(np.isnan(p[rev]), 1, (p[rev] >= t).astype(int))
    return pred


def hybrid_pred(df: pd.DataFrame, t_ok: float, t_def: float) -> tuple[np.ndarray, np.ndarray]:
    pred, rev = base_pred(df), df.zone.to_numpy() == "review"
    p = df.p_defect.to_numpy(dtype=float)
    known = ~np.isnan(p)
    rej = rev & known & (p >= t_def)
    ok = rev & known & (p <= t_ok) & ~rej
    hum = rev & ~rej & ~ok
    pred[rej], pred[ok] = 1, 0
    pred[hum] = df.label.to_numpy()[hum]
    return pred, hum


def load_vlm(variant: str) -> pd.DataFrame | None:
    p = CASCADE / f"vlm_{variant}.jsonl"
    if not p.exists():
        return None
    df = pd.read_json(p, lines=True)
    return df.drop_duplicates(["category", "image_path"], keep="last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-recall", type=float, default=0.98)
    ap.add_argument("--max-fa-increase", type=float, default=0.02)
    args = ap.parse_args()

    band = pd.read_csv(CASCADE / "band.csv")
    ca, ev = band[band.split == "calib"], band[band.split == "eval"]
    results: dict = {"settings": vars(args), "policies": {}, "vlm_in_review_zone": {},
                     "per_category": {}, "hybrid_curves": {}}

    for cat in CATEGORIES:
        meta = json.loads((STAGE1 / cat / "meta.json").read_text())
        e = ev[ev.category == cat]
        results["per_category"][cat] = {
            "image_auroc": meta["image_auroc"],
            "infer_ms_per_image": meta["infer_ms_per_image"],
            "eval_review_rate": float((e.zone == "review").mean()),
            "eval_stage1_recall": float(e[e.label == 1].stage1_pred.mean()),
            "eval_stage1_false_alarm": float(e[e.label == 0].stage1_pred.mean()),
        }

    results["policies"]["stage1_tau"] = rates(ev.label.to_numpy(), ev.stage1_pred.to_numpy(),
                                              np.zeros(len(ev), bool))
    results["policies"]["3way_human"] = rates(ev.label.to_numpy(), *human_pred(ev))
    human_fa_calib = rates(ca.label.to_numpy(), *human_pred(ca))["false_alarm_rate"]

    for v in VARIANTS:
        vlm = load_vlm(v)
        if vlm is None:
            continue
        cols = ["category", "image_path", "defect", "confidence", "p_defect", "latency_s",
                "prompt_tokens", "completion_tokens"]
        m = band.merge(vlm[[c for c in cols if c in vlm.columns]].rename(
            columns={"defect": "vlm_defect", "confidence": "vlm_conf"}), on=["category", "image_path"], how="left")
        if "p_defect" not in m:
            m["p_defect"] = np.nan
        m_ca, m_ev = m[m.split == "calib"], m[m.split == "eval"]

        # How good is the reviewer on the uncertain images themselves (eval half)?
        rz = m[m.zone == "review"]
        rz_ev = rz[rz.split == "eval"]
        yl = rz_ev.label.to_numpy()
        vd = rz_ev.vlm_defect.map({True: 1, False: 0}).fillna(1).to_numpy().astype(int)
        results["vlm_in_review_zone"][v] = {
            "n_eval": int(len(rz_ev)),
            "accuracy": float((vd == yl).mean()),
            "defect_recall": float(vd[yl == 1].mean()) if (yl == 1).any() else None,
            "normal_clear_rate": float(1 - vd[yl == 0].mean()) if (yl == 0).any() else None,
            "p_defect_auroc": (float(roc_auc_score(yl, rz_ev.p_defect.fillna(0.5)))
                               if rz_ev.p_defect.notna().any() and len(set(yl)) == 2 else None),
            "stage1_score_auroc": float(roc_auc_score(yl, rz_ev.score)) if len(set(yl)) == 2 else None,
            "self_conf_distinct_values": int(rz.vlm_conf.nunique()),
            "parse_fail_rate": float(1 - rz.vlm_defect.notna().mean()),
            "mean_latency_s": float(rz.latency_s.mean()),
            "mean_prompt_tokens": float(rz.prompt_tokens.mean()),
        }

        results["policies"][f"3way_vlm[{v}]"] = rates(m_ev.label.to_numpy(), verdict_pred(m_ev),
                                                      np.zeros(len(m_ev), bool))

        # Fully automatic on P(defect): largest t that keeps calib recall >= min_recall.
        t_auto = next((t for t in reversed(T_AUTO)
                       if rates(m_ca.label.to_numpy(), p_auto_pred(m_ca, t), np.zeros(len(m_ca), bool))["recall"]
                       >= args.min_recall), T_AUTO[0])
        results["policies"][f"3way_vlmP[{v}]"] = {
            **rates(m_ev.label.to_numpy(), p_auto_pred(m_ev, t_auto), np.zeros(len(m_ev), bool)), "t": t_auto}

        # Hybrid: grid search on calib.
        best = None
        for t_def in T_DEF:
            for t_ok in T_OK:
                if t_ok >= t_def:
                    continue
                r = rates(m_ca.label.to_numpy(), *hybrid_pred(m_ca, t_ok, t_def))
                if r["recall"] < args.min_recall or r["false_alarm_rate"] > human_fa_calib + args.max_fa_increase:
                    continue
                key = (r["human_review_rate"], r["false_alarm_rate"])
                if best is None or key < best[0]:
                    best = (key, t_ok, t_def)
        t_ok, t_def = (best[1], best[2]) if best else (0.0, 1.01)  # fallback = all review to humans
        results["policies"][f"hybrid[{v}]"] = {
            **rates(m_ev.label.to_numpy(), *hybrid_pred(m_ev, t_ok, t_def)), "t_ok": t_ok, "t_def": t_def}
        results["hybrid_curves"][v] = [
            {"t_ok": t, "eval": rates(m_ev.label.to_numpy(), *hybrid_pred(m_ev, t, t_def))}
            for t in T_OK if t < t_def]

    (CASCADE / "results.json").write_text(json.dumps(results, indent=2))
    write_markdown(results)
    make_figures(results)
    print((CASCADE / "results.md").read_text(encoding="utf-8"))


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def write_markdown(r: dict) -> None:
    lines = ["## Policies (eval half, all categories pooled)", "",
             "| policy | recall | miss rate | false-alarm rate | human review rate | thresholds |",
             "|---|---|---|---|---|---|"]
    for k, v in r["policies"].items():
        th = ", ".join(f"{t}={v[t]}" for t in ("t", "t_ok", "t_def") if t in v)
        lines.append(f"| {k} | {pct(v['recall'])} | {pct(v['miss_rate'])} | "
                     f"{pct(v['false_alarm_rate'])} | {pct(v['human_review_rate'])} | {th} |")
    lines += ["", "## VLM reviewer inside the review zone (eval half)", "",
              "| variant | n | verdict accuracy | defect recall | normal clear rate | P(defect) AUROC | "
              "stage-1 score AUROC | distinct self-confidence values | parse fails | latency (s, under load) | prompt tokens |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in r["vlm_in_review_zone"].items():
        f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
        lines.append(f"| {k} | {v['n_eval']} | {pct(v['accuracy'])} | {pct(v['defect_recall'])} | "
                     f"{pct(v['normal_clear_rate'])} | {f3(v['p_defect_auroc'])} | {f3(v['stage1_score_auroc'])} | "
                     f"{v['self_conf_distinct_values']} | {pct(v['parse_fail_rate'])} | "
                     f"{v['mean_latency_s']:.2f} | {v['mean_prompt_tokens']:.0f} |")
    lines += ["", "## Per category (stage 1, eval half)", "",
              "| category | image AUROC | ms/image | review rate | stage-1 recall | stage-1 false alarms |",
              "|---|---|---|---|---|---|"]
    for k, v in r["per_category"].items():
        lines.append(f"| {k} | {v['image_auroc']:.3f} | {v['infer_ms_per_image']:.0f} | {pct(v['eval_review_rate'])} | "
                     f"{pct(v['eval_stage1_recall'])} | {pct(v['eval_stage1_false_alarm'])} |")
    (CASCADE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(r: dict) -> None:
    if r["vlm_in_review_zone"]:
        vs = list(r["vlm_in_review_zone"])
        keys = [("p_defect_auroc", "P(defect) AUROC"), ("stage1_score_auroc", "stage-1 score AUROC"),
                ("accuracy", "verdict accuracy")]
        x = np.arange(len(vs))
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for i, (k, lab) in enumerate(keys):
            ax.bar(x + (i - 1) * 0.25, [r["vlm_in_review_zone"][v][k] or 0 for v in vs], 0.25, label=lab)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_xticks(x, vs)
        ax.set_ylim(0, 1)
        ax.set_title("Separating defects from normals inside the review zone (eval half)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(CASCADE / "fig_reviewer_ablation.png", dpi=160)
        plt.close(fig)

    if r["hybrid_curves"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for v, curve in r["hybrid_curves"].items():
            fa = r["policies"][f"hybrid[{v}]"]["false_alarm_rate"]
            ax.plot([c["eval"]["human_review_rate"] for c in curve], [c["eval"]["recall"] for c in curve],
                    marker="o", ms=3, label=f"hybrid[{v}] (false alarms {fa:.1%} at chosen point)")
        for k, mk in (("stage1_tau", "s"), ("3way_human", "*")):
            p = r["policies"][k]
            ax.scatter(p["human_review_rate"], p["recall"], marker=mk, s=80, zorder=5,
                       label=f"{k} (false alarms {p['false_alarm_rate']:.1%})")
        ax.set_xlabel("human review rate (share of all parts)")
        ax.set_ylabel("recall (defects caught)")
        ax.set_title("Human workload vs recall (eval half)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(CASCADE / "fig_workload_vs_recall.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
