"""Sina plot of per-example end-to-end FVE for the tinyNLA eval records.

Reads eval_records_{sft,rl}.jsonl + eval_records_gold_{rl|sft}.jsonl produced by
nla.train_tiny_nla --phase eval and renders one sina per condition:
  sft / sft-shuffled / rl / rl-shuffled / AR-on-gold.

Usage: python -m nla.tiny_plot --save-dir /workspace-vast/jbauer/nla_tiny/tn1
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def sina(ax, x_center, vals, color, seed=0):
    vals = np.asarray(vals, dtype=float)
    ymin, ymax = vals.min(), vals.max()
    axis_range = max(ymax - ymin, 1e-9)
    bw = 0.05 * max(axis_range, 1.0)  # 0.05 in 0-1 axis units, scaled proportionally
    kde = gaussian_kde(vals, bw_method=lambda k: bw / max(k.dataset.std(), 1e-6))
    grid = np.linspace(ymin, ymax, 200)
    dens = kde(grid)
    dens_n = dens / dens.max() * 0.38  # half-width envelope
    # faint violin outline
    ax.fill_betweenx(grid, x_center - dens_n, x_center + dens_n, color=color, alpha=0.15, lw=0)
    # per-example dots, deterministic jitter scaled by local density
    rng = np.random.default_rng(seed)
    local = np.interp(vals, grid, dens_n)
    jit = rng.uniform(-1, 1, size=len(vals)) * local
    ax.scatter(x_center + jit, vals, s=8, color=color, alpha=0.7, zorder=3, lw=0)
    # EXTENT box: min->max, median line, quartile ticks
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    w = 0.42
    ax.add_patch(plt.Rectangle((x_center - w, ymin), 2 * w, ymax - ymin,
                               fill=False, edgecolor="black", lw=1.0, zorder=4))
    ax.hlines(med, x_center - w, x_center + w, color="black", lw=1.5, zorder=5)
    for q in (q1, q3):
        ax.hlines(q, x_center - w * 0.35, x_center + w * 0.35, color="black", lw=1.0, zorder=5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", required=True)
    p.add_argument("--backbone-label", default="gemma-3-270m")
    args = p.parse_args()
    save = Path(args.save_dir)
    results = json.loads((save / "eval_results.json").read_text())

    conditions = []  # (label, per-example FVE list, color)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ci = 0
    for tag in ("sft", "rl"):
        f = save / f"eval_records_{tag}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        b = results[tag]["baseline_mse"]
        for key, suffix in (("reward", ""), ("reward_shuffled", " shuffled")):
            vals = [1 - (-r[key]) / b for r in rows if r[key] is not None]
            conditions.append((f"{tag}{suffix}\n(n={len(vals)})", vals, colors[ci % len(colors)]))
            ci += 1
        g = save / f"eval_records_gold_{tag}.jsonl"
        if tag == "rl" or not (save / "eval_records_gold_rl.jsonl").exists():
            grows = [json.loads(l) for l in g.read_text().splitlines()]
            gb = results[tag]["gold_baseline_mse"]
            gvals = [1 - (-r["reward"]) / gb for r in grows if r["reward"] is not None]
            conditions.append((f"AR-on-gold ({tag})\n(n={len(gvals)})", gvals, colors[ci % len(colors)]))
            ci += 1

    fig, ax = plt.subplots(figsize=(2 + 1.6 * len(conditions), 5))
    for i, (label, vals, color) in enumerate(conditions):
        sina(ax, i, vals, color, seed=i)
        ax.text(i, ax.get_ylim()[0], "", ha="center")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(range(len(conditions)), [c[0] for c in conditions])
    ax.set_ylabel("per-example FVE (1 - mse / predict-mean baseline)")
    ax.set_title(f"tinyNLA ({args.backbone_label} AV/AR on Qwen3-8B L24): held-out end-to-end FVE")
    out = save / "tiny_nla_eval_sina.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
