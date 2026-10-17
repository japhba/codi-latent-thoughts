"""Figure: fd1 activation-oracle training and evaluation (post: Appendix D), in the
visual style of the NLA figure (fig_12): a train panel and eval panel(s) separated by
a whitespace gap under "train" / "eval" titles, despined axes, one color per read and
one linestyle per criterion, lens baselines as faint horizontal lines in the read's
color, the legend outside the right of the eval panel.

TWO OUTPUTS FROM ONE DATA PASS: the post variant (Appendix D, FIG_OUT) shows train +
decode score; the full variant (Appendix E, FIG_OUT_FULL) adds a middle double-log
val CE / chance panel (identical limits with the train panel).

Left (train): online train CE vs step, double log, centered rolling mean (half-width
up to 25 log points = 250 steps, shrinking at the edges), from step 10. The trainer
logs "step N loss L" every 10 steps; restart logs are stitched by keeping only
step >= the log's "resumed from step R", last write wins per step.
Eval (decode score, linear y, log step axis since the checkpoints are log-spaced, no markers):
SOLID = first extracted number == gold answer, DASHED = any gold non-answer chain value
semantically expressed (judged), per read: full read (h at L9/18/27/35/36 + z,
thoughts 1-6), z-only read (thoughts 1-6), fair question-only control (all five h
layers + prj(h^(36)_q) over question tokens). The untrained init (step 0) is drawn at
step 1. Faint horizontal lines: LogitLens / J-lens on the same streams and questions
with the same criterion, a problem counted if either lens finds the value (as fig_12).
The full variant draws a grid on every panel.
Middle (full variant): held-out val CE over chance, SOLID = answer label, DASHED =
all-intermediate-values label; chance = per-token NLL under a unigram fit on the
label's own targets.

Inputs (env-overridable):
  AO_CURVES    default artifacts/ao_curves.json
  READER_LENS  default artifacts/reader_layer_lens_verdicts.json
Output: figs/fd1_performance.png, figs/fd1_performance_full.png
"""
import json, os, re

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.join(HERE, "..", "artifacts")
CURVES_F = os.environ.get("AO_CURVES", os.path.join(REPRO, "ao_curves.json"))
LENS_F = os.environ.get("READER_LENS", os.path.join(REPRO, "reader_layer_lens_verdicts.json"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs", "fd1_performance.png"))
OUT_FULL = os.environ.get("FIG_OUT_FULL", os.path.join(HERE, "..", "figs", "fd1_performance_full.png"))
FEEDS = ("ALL", "Z", "qonly6")
COL = {"ALL": "#1f5cab", "Z": "#c14e1f", "qonly6": "#3a8f5c"}
LAB = {"ALL": r"full read: $\mathbf{h}^{(\ell)}_t, \mathbf{z}_t$",
       "Z": r"$\mathbf{z}$-only read: $\mathbf{z}_t$",
       "qonly6": "question tokens only"}
LENS_SRC = {"ALL": "ao_ALL", "Z": "ao_Z"}
CRIT = [("answer", "final answer", "-"), ("intermediate", "any intermediate", "--")]
VAL_LS = {"answer": "-", "allvals": "--"}

data = json.load(open(CURVES_F))

train = {}
for _, text in sorted(data["train_logs"].items()):
    resumes = [int(s) for s in re.findall(r"resumed from step (\d+)", text)]
    r0 = max(resumes) if resumes else 0
    for mt in re.finditer(r"^step (\d+) loss ([\d.]+)", text, re.M):
        if int(mt.group(1)) >= r0:
            train[int(mt.group(1))] = float(mt.group(2))
tsteps = sorted(train)
assert (tsteps[0], tsteps[-1]) == (0, 66490), (tsteps[0], tsteps[-1])
smooth = []
for i in range(len(tsteps)):
    h = min(25, i, len(tsteps) - 1 - i)
    smooth.append(np.mean([train[s] for s in tsteps[i - h:i + h + 1]]))
TRAIN = [(s, v) for s, v in zip(tsteps, smooth) if s >= 10]

answer = {r["step"]: r for r in data["answer_rows"]}
judged = {}
for r in data["intermediate_verdicts"]:
    judged.setdefault((r["src"], r["step"]), []).append(float(r["matched"]))
EVAL = {}     # (feed, crit) -> (steps, rates), init drawn at step 1
for feed in FEEDS:
    st = sorted(answer)
    EVAL[(feed, "answer")] = ([max(s, 1) for s in st], [answer[s][f"{feed}|answer"] for s in st])
    st = sorted(s for f, s in judged if f == feed)
    EVAL[(feed, "intermediate")] = ([max(s, 1) for s in st], [np.mean(judged[(feed, s)]) for s in st])
    for crit, _, _ in CRIT:
        ys = EVAL[(feed, crit)][1]
        print(f"{feed:>6} {crit:12s} init {ys[0]:.2f} -> final {ys[-1]:.2f}")

val = {}
for r in data["val_rows"]:
    for k, v in r.items():
        if "|" in k:
            val.setdefault(k, {})[r["step"]] = v
qval = {r["step"]: r for r in data["question_val_rows"]}
VAL = {}      # (feed, label) -> (steps, CE / chance)
for feed in FEEDS:
    for lab in VAL_LS:
        rows = {s: r[f"qonly6|{lab}"] for s, r in qval.items()} if feed == "qonly6" else val[f"{feed}|{lab}"]
        st = sorted(rows)
        VAL[(feed, lab)] = ([max(s, 1) for s in st], [rows[s] / data["chance"][lab] for s in st])

LENS = {}
for r in json.load(open(LENS_F)):
    if r["src"] in LENS_SRC.values():
        LENS.setdefault((r["src"], r["kind"].split(":")[1]), {}).setdefault(r["uid"], []).append(r["matched"])
LENS = {k: float(np.mean([any(v) for v in d.values()])) for k, d in LENS.items()}
print("lens baselines on the AO streams:", {f"{a}/{b}": round(v, 3) for (a, b), v in LENS.items()})


def render(out, with_mid):
    if with_mid:
        fig = plt.figure(figsize=(13.5, 4.2))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.14, 1, 1])
        axr, axv, axa = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])
    else:
        fig = plt.figure(figsize=(9.5, 4.2))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.14, 1])
        axr, axa, axv = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), None

    axr.plot(*zip(*TRAIN), color="0.25")
    axr.set_xscale("log")
    axr.set_yscale("log")
    axr.set_xlabel("training step (log)")
    axr.set_ylabel("train cross-entropy (log)")

    for feed in FEEDS:
        for crit, _, ls in CRIT:
            axa.plot(*EVAL[(feed, crit)], color=COL[feed], ls=ls)
            if feed in LENS_SRC:
                axa.axhline(LENS[(LENS_SRC[feed], crit)], color=COL[feed], ls=ls, alpha=0.45)
        if axv is not None:
            for lab, ls in VAL_LS.items():
                axv.plot(*VAL[(feed, lab)], color=COL[feed], ls=ls)
    axa.set_xscale("log")
    axa.set_xlabel("training step (log)")
    axa.set_ylabel("fraction of problems recovered")
    axa.set_ylim(0, 1)
    handles = [plt.Line2D([], [], color=COL[f], label=LAB[f]) for f in FEEDS]
    handles += [plt.Line2D([], [], color="0.2", ls=ls, label=lab) for _, lab, ls in CRIT]
    handles.append(plt.Line2D([], [], color="0.2", alpha=0.45, label="lenses on the same streams"))
    axa.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

    panels = [a for a in (axr, axv) if a is not None]
    if axv is not None:
        axv.set_xscale("log")
        axv.set_yscale("log")
        axv.set_xlabel("training step (log)")
        axv.set_ylabel("val CE / chance (log)")
        axv.legend(handles=[plt.Line2D([], [], color="0.2", ls=ls, label=f"{lab} label")
                            for lab, ls in (("answer", "-"), ("all-values", "--"))], loc="lower left", frameon=False)
        yl = (min(a.get_ylim()[0] for a in panels), max(a.get_ylim()[1] for a in panels))
        for a in panels:
            a.set_ylim(yl)
    xl = (min(a.get_xlim()[0] for a in panels + [axa]), max(a.get_xlim()[1] for a in panels + [axa]))
    panels_all = panels + [axa]
    for a in panels_all:
        a.set_xlim(xl)
        a.spines[["top", "right"]].set_visible(False)
    if with_mid:    # appendix scaling variant: grid on every panel (minor lines too, for the log axes)
        for a in panels_all:
            a.grid(which="major", alpha=0.4)
            a.grid(which="minor", alpha=0.15)
    fig.tight_layout()
    br, ba = axr.get_position(), axa.get_position()
    fig.text((br.x0 + br.x1) / 2, br.y1 + 0.03, "train", ha="center", va="bottom", fontsize="large", fontweight="bold")
    e0 = axv.get_position().x0 if axv is not None else ba.x0
    fig.text((e0 + ba.x1) / 2, ba.y1 + 0.03, "eval", ha="center", va="bottom", fontsize="large", fontweight="bold")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


os.makedirs(os.path.dirname(OUT), exist_ok=True)
render(OUT, with_mid=False)
render(OUT_FULL, with_mid=True)
