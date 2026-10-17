"""Figure: NLA training and evaluation (post: Natural Language Autoencoders).

Left (train): double-log residual variance 1 - FVE vs training step for the two readers
(nz4 runs; the trainer's eval/fve_pct, a directional FVE over accepted descriptions),
centered Gaussian smooth of the log residual. Right (eval): value recovery of the
all-thought rerun (companion package, rv6): at each of the nine recoverable reader
checkpoints (warmstart = adapter disabled, then steps 400..3200) every one of the six
thought positions is verbalized independently and the six descriptions are scored
jointly. Values are LLM-extracted from the descriptions without access to the question
or gold labels, then matched against the reference values. SOLID = gold final answer
mentioned, DASHED = any reference intermediate other than the answer mentioned. Nine
checkpoints only, so the eval panel uses a linear step axis. No markers on the eval
curves; the full variant draws a grid on every panel.

TWO OUTPUTS: the main-text variant (train + eval) and the full appendix variant, which
adds a middle double-log 1 - recovery panel (shared limits with the train panel, the
warmstart drawn at step 1) and the unrelated-donor control (descriptions of the
thoughts of a question seven positions further on, intermediates, dotted).

Inputs (env-overridable):
  NLA_FVE   default repo/artifacts/nz4_fve_curves.json
  AV_SCORES default artifacts/rv6/scores/all_thought_scores.json
Output: figs/nla_training_eval.png, figs/nla_training_eval_full.png
"""
import json, os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FVE_F = os.environ.get("NLA_FVE", os.path.join(HERE, "..", "artifacts", "nz4_fve_curves.json"))
SCORES_F = os.environ.get("AV_SCORES", os.path.join(HERE, "..", "artifacts",
                                                    "rv6", "scores", "all_thought_scores.json"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs", "nla_training_eval.png"))
OUT_FULL = os.environ.get("FIG_OUT_FULL",
                          os.path.join(HERE, "..", "figs", "nla_training_eval_full.png"))
SRCS = ("h24", "z")
COL = {"z": "#c14e1f", "h24": "#1f5cab"}
LAB = {"z": r"$\mathbf{z}$ reader", "h24": r"$\mathbf{h}^{(24)}$ reader"}
CRIT = [("answer", "final answer", "-"), ("intermediate", "any intermediate", "--")]


def gsmooth_log(vals, sigma=3.0):
    """centered Gaussian smooth of the LOG residual over the point index."""
    lv = np.log(np.asarray(vals, float))
    idx = np.arange(len(lv))
    w = np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / sigma) ** 2)
    return np.exp((w * lv[None, :]).sum(1) / w.sum(1))


curves = json.load(open(FVE_F))
TRAIN = {}
for src in SRCS:
    c = curves[src]
    pts = [(s, 1 - v / 100) for s, v in zip(c["steps"], c["fve_pct"]) if s > 0]
    TRAIN[src] = ([s for s, _ in pts], gsmooth_log([r for _, r in pts]))

# lens baselines at each reader's own layer (h24 / the fed-back z), on the same questions with the same
# criterion: top-5 tokens pooled over the six thoughts, a problem counted if LogitLens or J-lens finds the value
LENS_F = os.environ.get("READER_LENS", os.path.join(HERE, "..", "artifacts", "reader_layer_lens_verdicts.json"))
LENS = {}
for r in json.load(open(LENS_F)):
    if r["src"] in SRCS:
        LENS.setdefault((r["src"], r["kind"].split(":")[1]), {}).setdefault(r["uid"], []).append(r["matched"])
LENS = {k: float(np.mean([any(v) for v in d.values()])) for k, d in LENS.items()}
print("lens baselines at the reader layers:", {f"{a}/{b}": round(v, 3) for (a, b), v in LENS.items()})

EVAL = {}     # (src, kind, crit) -> (steps, rates)
for r in json.load(open(SCORES_F))["scores"]:
    for crit, _, _ in CRIT:
        EVAL.setdefault((r["src"], r["kind"], crit), []).append((r["step"], r[crit]["correct"] / r[crit]["n"]))
EVAL = {k: tuple(map(list, zip(*sorted(v)))) for k, v in EVAL.items()}
for src in SRCS:
    for crit, _, _ in CRIT:
        steps, ys = EVAL[(src, "own", crit)]
        print(f"{src:>3} {crit:12s} " + " ".join(f"{s}:{y:.2f}" for s, y in zip(steps, ys)))


def render(out, with_mid):
    if with_mid:
        fig = plt.figure(figsize=(13.5, 4.2))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.14, 1, 1])
        axr, axe, axa = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])
    else:
        fig = plt.figure(figsize=(9.5, 4.2))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.14, 1])
        axr, axa, axe = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), None

    for src in SRCS:
        xs, sm = TRAIN[src]
        axr.plot(xs, sm, color=COL[src], label=LAB[src])
    axr.set_xscale("log")
    axr.set_yscale("log")
    axr.set_xlabel("training step (log)")
    axr.set_ylabel("residual variance 1 − FVE (log)")
    axr.legend(loc="lower left")

    for src in SRCS:
        for crit, _, ls in CRIT:
            steps, ys = EVAL[(src, "own", crit)]
            axa.plot(steps, ys, color=COL[src], ls=ls)
            axa.axhline(LENS[(src, crit)], color=COL[src], ls=ls, alpha=0.45)
            if axe is not None:
                axe.plot([max(s, 1) for s in steps], [1 - y for y in ys], color=COL[src], ls=ls)
        if with_mid:
            steps, ys = EVAL[(src, "donor", "intermediate")]
            axa.plot(steps, ys, color=COL[src], ls=":")
    axa.set_xlabel("training step")
    axa.set_ylabel("fraction of problems recovered")
    axa.set_ylim(0, 1)
    axa.set_xlim(-100, 3300)
    axa.set_xticks([0, 1000, 2000, 3000])
    handles = [plt.Line2D([], [], color="0.2", ls=ls, label=lab) for _, lab, ls in CRIT]
    handles.append(plt.Line2D([], [], color="0.2", alpha=0.45, label="lenses at the reader's layer"))
    if with_mid:
        handles.append(plt.Line2D([], [], color="0.2", ls=":", label="unrelated problem's thoughts"))
    axa.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

    if axe is not None:
        axe.set_xscale("log")
        axe.set_yscale("log")
        axe.set_xlabel("training step (log)")
        axe.set_ylabel("1 − fraction recovered (log)")
        xl = (min(axr.get_xlim()[0], axe.get_xlim()[0]), max(axr.get_xlim()[1], axe.get_xlim()[1]))
        yl = (min(axr.get_ylim()[0], axe.get_ylim()[0]), max(axr.get_ylim()[1], axe.get_ylim()[1]))
        for a in (axr, axe):
            a.set_xlim(xl)
            a.set_ylim(yl)
    panels_all = [a for a in (axr, axe, axa) if a is not None]
    for a in panels_all:
        a.spines[["top", "right"]].set_visible(False)
    if with_mid:    # appendix scaling variant: grid on every panel (minor lines too, for the log axes)
        for a in panels_all:
            a.grid(which="major", alpha=0.4)
            a.grid(which="minor", alpha=0.15)
    fig.tight_layout()
    br, ba = axr.get_position(), axa.get_position()
    fig.text((br.x0 + br.x1) / 2, br.y1 + 0.03, "train", ha="center", va="bottom", fontsize="large", fontweight="bold")
    e0 = axe.get_position().x0 if axe is not None else ba.x0
    fig.text((e0 + ba.x1) / 2, ba.y1 + 0.03, "eval", ha="center", va="bottom", fontsize="large", fontweight="bold")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


os.makedirs(os.path.dirname(OUT), exist_ok=True)
render(OUT, with_mid=False)
render(OUT_FULL, with_mid=True)
