"""Latent-trace ablations on the LOAD-BEARING rows (post: Performance / Ablations, and
Appendix F). Load-bearing = CODI needs its latent thoughts: the six-thought decode is
correct AND the zero-thought decode is wrong (n=104).

All decodes come from the masked rerun of the companion package (full padding masks on
every cached step): artifacts/rv6/controls (recurrent_controls) plus
codi_repro.extra_ablations (shuffled, the next test problem's thoughts, Gaussian thoughts).

Two outputs:
  figs/ablations_lb.png             main text: fraction of load-bearing problems still
                                    answered correctly; after an intervention all later
                                    thoughts are generated autoregressively from the
                                    corrupted state (whole-trace replacements leave nothing
                                    to regenerate). Single-position rows average over the
                                    six positions. 95% bootstrap intervals over problems.
  figs/ablations_reuse_regen.png    Appendix F: one thought replaced (Gaussian noise, the
                                    dataset-mean thought, another problem's thought, zeros)
                                    at each position, reusing the recorded later thoughts
                                    vs regenerating them.

Inputs (env-overridable): RV6_CONTROLS (baseline.json, predictions.json), EXTRA_PREDICTIONS
Output dir: env FIG_DIR (default figs)
"""
import json, math, os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RV6 = os.environ.get("RV6_CONTROLS", os.path.join(HERE, "..", "artifacts", "rv6", "controls"))
EXTRA_F = os.environ.get("EXTRA_PREDICTIONS", os.path.join(HERE, "..", "artifacts", "extra_predictions.json"))
FIG_DIR = os.environ.get("FIG_DIR", os.path.join(HERE, "..", "figs"))
C_REGEN, C_REUSE = "C0", "0.6"


def ok(pred, gold):
    return pred is not None and math.isclose(float(pred), float(gold), rel_tol=1e-6, abs_tol=1e-6)


base = json.load(open(os.path.join(RV6, "baseline.json")))["rows"]
preds = {c: v["predictions"] for c, v in json.load(open(os.path.join(RV6, "predictions.json"))).items()}
preds.update({c: v["predictions"] for c, v in json.load(open(EXTRA_F)).items()})
lb = [k for k, r in enumerate(base) if ok(r["six_pred"], r["gold"]) and not ok(r["zero_pred"], r["gold"])]
print(f"load-bearing n={len(lb)}")
draws = np.random.default_rng(11).integers(0, len(lb), (5000, len(lb)))


def kept(conds):
    """per-problem fraction of the listed conditions that keep the correct answer"""
    return np.array([np.mean([ok(preds[c][k], base[k]["gold"]) for c in conds]) for k in lb])


def interval(v):
    return np.percentile(v[draws].mean(1), [2.5, 97.5])


os.makedirs(FIG_DIR, exist_ok=True)

# ---- main text: one mode, later thoughts always regenerated ----
MAIN = [("only the first 3 thoughts", ["trunc_3_regenerated"]),
        ("one thought replaced by noise", [f"noise_p{j}_regenerated" for j in range(1, 7)]),
        ("one thought from another problem", [f"donor_p{j}_regenerated" for j in range(1, 7)]),
        ("thoughts shuffled", ["shuffle"]),
        ("all thoughts from another problem", ["swap_neighbor"]),
        ("all thoughts replaced by noise", ["gauss"])]
fig, ax = plt.subplots(figsize=(6.5, 2.9))
ys = np.arange(len(MAIN))[::-1]
for y, (label, conds) in zip(ys, MAIN):
    v = kept(conds)
    lo, hi = interval(v)
    ax.barh(y, v.mean(), height=0.62, color=C_REGEN)
    ax.errorbar(v.mean(), y, xerr=[[v.mean() - lo], [hi - v.mean()]], ls="none", color="k", capsize=3)
    print(f"{label:34s} {v.mean():.2f} [{lo:.2f}, {hi:.2f}]")
ax.set_yticks(ys)
ax.set_yticklabels([m[0] for m in MAIN])
ax.set_xlim(0, 1)
ax.set_xlabel("load-bearing problems still answered correctly")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = os.path.join(FIG_DIR, "ablations_lb.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")

# ---- Appendix F: reuse vs regenerate the later thoughts ----
KINDS = [("noise", "Gaussian noise"), ("mean", "mean thought"), ("donor", "another problem's"), ("zero", "zeros")]
pos = np.arange(1, 7)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
for ax, (kind, title) in zip(axes, KINDS):
    for mode, col, lab in (("frozen", C_REUSE, "later thoughts reused"), ("regenerated", C_REGEN, "later thoughts regenerated")):
        vs = [kept([f"{kind}_p{j}_{mode}"]) for j in pos]
        m = np.array([v.mean() for v in vs])
        ci = np.array([interval(v) for v in vs])
        ax.errorbar(pos, m, yerr=[m - ci[:, 0], ci[:, 1] - m], color=col, marker="o", capsize=3, label=lab)
    ax.set_title(title)
    ax.set_xticks(pos)
    ax.set_xlabel("replaced thought")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("load-bearing problems\nstill answered correctly")
fig.suptitle("one thought replaced by")
fig.tight_layout()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, frameon=False)
out = os.path.join(FIG_DIR, "ablations_reuse_regen.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
