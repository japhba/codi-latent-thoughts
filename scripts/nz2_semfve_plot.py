"""nz2: semantic metrics vs TRUE reconstruction FVE, step as colormap.
Top-level LEFT-RIGHT split: h24 (left) vs z (right). Successor of
nz1_fve_metrics_plot.py, but sourced from the workbench-persisted
eval_scores.jsonl (wandb history sync died on pod relaunches).

x = eval FVE% of the source at that eval step, computed from the per-row
    critic rewards (mse = -reward, valid extractions only) against the
    trainer's own pooled predict-the-mean baselines (z 0.550 / h24 0.501,
    deterministic recomputation printed at every startup).
y = unconditional semantic means (failed extraction scores 0), exactly the
    monitor's definition: chain recall (gold intermediates in text), answer
    hit, on-topic trueness (gemini-3-flash judge).
-> reports/nz2/nz2_sem_vs_fve.png"""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCORES = "/workspace-vast/jbauer/bt1/nz2/eval_scores.jsonl"
OUT = "/workspace-vast/jbauer/activation_oracles_dev/reports/nz2/nz2_sem_vs_fve.png"
BASELINE = {"z": 0.550, "h24": 0.501}   # pooled predict-the-mean mse (trainer startup print)

rows = {}
for l in open(SCORES):
    r = json.loads(l)
    rows[(r["step"], r["idx"])] = r          # dedup: last write wins
rows = list(rows.values())
print(f"{len(rows)} deduped eval rows")

agg = defaultdict(lambda: defaultdict(list))  # (src, step) -> field -> vals
for r in rows:
    src = r["stratum"].split("_p")[0]
    a = agg[(src, r["step"])]
    if r["extracted"]:
        a["mse"].append(-r["reward"])
        a["chain"].append(r["chain_recall"] if r["chain_recall"] is not None else 0.0)
        a["ans"].append(float(r["answer_hit"]))
        if r.get("ontopic_trueness") is not None:
            a["topic"].append(r["ontopic_trueness"])
    else:
        a["chain"].append(0.0); a["ans"].append(0.0); a["topic"].append(0.0)

series = {}
for (src, step), a in agg.items():
    if not a["mse"]:
        continue
    fve = (1.0 - np.mean(a["mse"]) / BASELINE[src]) * 100.0
    series.setdefault(src, []).append(
        (step, fve, np.mean(a["chain"]), np.mean(a["ans"]),
         np.mean(a["topic"]) if a["topic"] else np.nan))

MET = [("chain recall", 2, "o"), ("answer hit", 3, "s"), ("on-topic trueness", 4, "^")]
fig, axes = plt.subplots(1, 2, sharey=True)
steps_all = sorted({s for v in series.values() for s, *_ in v})
norm = plt.Normalize(min(steps_all), max(steps_all))
cmap = plt.get_cmap("viridis")
for ax, src in zip(axes, ("h24", "z")):
    # drop warmup evals with FVE < 0 (first steps, critic still settling) —
    # they compress the whole axis into a sliver
    pts = [p for p in sorted(series[src]) if p[1] > 0]
    st = np.array([p[0] for p in pts]); fv = np.array([p[1] for p in pts])
    for lab, col, mk in MET:
        y = np.array([p[col] for p in pts])
        ax.plot(fv, y, color="0.85", zorder=1)
        ax.scatter(fv, y, c=st, cmap=cmap, norm=norm, marker=mk, label=lab, zorder=2)
    ax.set_title(src)
    ax.set_xlabel("eval FVE %")
axes[0].set_ylabel("score (0–1, unconditional)")
axes[0].legend()
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="training step")
fig.suptitle("nz2 — semantic content vs reconstruction fidelity (h24 | z)")
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(OUT)

# ---- aligned AO-vs-NLA numbers (last 5 eval steps per src) ----
for src in ("h24", "z"):
    pts = sorted(series[src])[-5:]
    print(f"{src}: FVE {np.mean([p[1] for p in pts]):.1f}% | chain "
          f"{np.mean([p[2] for p in pts]):.3f} | ans {np.mean([p[3] for p in pts]):.3f} "
          f"| topic {np.mean([p[4] for p in pts]):.3f} (steps {pts[0][0]}–{pts[-1][0]})")
