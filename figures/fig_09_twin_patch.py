"""Twin-donor cross-patch figure (post: Interventions / Transplant-patching z and h),
two panels side by side: left, the original problem's thoughts z_i patched into the
TWIN question's rollout; right, FULL-COLUMN patches (the original's fed thought z_i
together with all 36 block outputs replace the twin rollout's at the marked thought
positions, downstream thoughts regenerate).
Reads artifacts/lbtwin_patch.json (env LBTWIN_PATCH overrides), produced by
codi_twin_patch_lb.py: LOAD-BEARING pairs, mined value-edit twins of the
trace-necessary GSM8k test rows (LLM-verified, noCoT decode fails on both sides,
50 pairs / 35 problems). Per panel: x = all-six-positions plus each single position,
filled bars = adoption rate of the transplanted (the original's) answer with Wilson
95% CIs (left axis, shared). Hollow bars (right axis, shared) show the
ANY-INTERMEDIATE decode accuracy of the final NLA reader of that panel's stream (z
reader left, h24 reader right) AT THE FED POSITION, for eyeballing whether the
causal adoption tracks the decodable fraction (the intermediates criterion because
the answer may not be represented yet at the patched thought positions): binary per
example, does the readout semantically express at least one gold non-answer chain
value, judged by gemini-3-flash (per-position scans artifacts/nz4_posgen_3200.jsonl
and nz4_posgen_h24_3200.jsonl judged via scripts/codi_posjudge.py ->
artifacts/nz4_posjudge_3200.jsonl), the RAW own rate (the donor coincidence floors,
~0.04-0.16 per position, are printed to the console but not subtracted, per
editorial call); the 1-6 tick shows the mean over the six single-position reads.
KEY per-position finding (2026-09-04): z_3 decodes COMPUTED intermediates far above
the floor (judged 0.67 vs 0.10 donor; mechanical computed-only 0.72 vs 0.07, no
question-echo excess there), so the first-thought scans (fig_12, fig_05) understate
mid-position z readability. The AO baselines were dropped per editorial call (NLA
only). The artifact also holds the forward direction, the controls (identity, own
repatch, matched noise) and the layer-24-row conditions. The superseded val-set
campaigns (single-pass-solvable pairs) stay in artifacts/zpatch_twin.json and
artifacts/hpatch_twin.json. Until 2026-10-14 the two panels were separate figures
(fig_09_zpatch_twin.py -> zpatch_twin.png, fig_11_hpatch_twin.py -> hpatch_twin.png).
Output: figs/twin_patch.png (env FIG_OUT).
"""
import json, math, os
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("LBTWIN_PATCH", os.path.join(HERE, "..", "artifacts", "lbtwin_patch.json"))
POSJ_F = os.environ.get("NZ4_POSJUDGE", os.path.join(HERE, "..", "artifacts", "nz4_posjudge_3200.jsonl"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs", "twin_patch.png"))
S = json.load(open(SRC))["summary"]
pj = [json.loads(l) for l in open(POSJ_F)]

TICKS = ["1–6"] + [str(k + 1) for k in range(6)]
KEYS = ["all6"] + [f"p{k + 1}" for k in range(6)]
PANELS = [("revz", "z", r"$\mathbf{z}_i$ patched"),
          ("revcol", "h24", r"column $\{\mathbf{z}_i\} \cup \{\mathbf{h}^{(\ell)}_i\}_{\ell=1}^{36}$ patched")]


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def pos_rate(src, t, kind):
    sub = [r["matched"] for r in pj if (r["src"], r["t"], r["kind"]) == (src, t, kind)]
    return sum(sub) / len(sub)


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
x = np.arange(len(KEYS))
twins = []
for ax, (cond, src, title) in zip(axes, PANELS):
    # per-position any-intermediate decode baseline: tick i = judged rate of the read
    # of ITS position (the NLA reads one vector at a time); 1-6 = mean over positions
    nla_pos = [pos_rate(src, t, "own") for t in range(6)]
    nla_bars = [sum(nla_pos) / 6] + nla_pos
    print(f"NLA {src} per-position (raw own rate):", [f"{v:.3f}" for v in nla_bars],
          "(first = mean over positions; donor floors:", [f"{pos_rate(src, t, 'donor'):.3f}" for t in range(6)], ")")
    conds = [f"{cond}_{k}" for k in KEYS]
    adopts = np.array([S[c]["adopt"] for c in conds])
    cis = [wilson(round(S[c]["adopt"] * S[c]["n"]), S[c]["n"]) for c in conds]
    yerr = [[p - lo for p, (lo, _) in zip(adopts, cis)], [hi - p for p, (_, hi) in zip(adopts, cis)]]
    ax.bar(x - 0.20, adopts, width=0.34, label="adoption of the transplanted answer (left axis)")
    ax.errorbar(x - 0.20, adopts, yerr=yerr, ls="none", color="k", capsize=3, lw=1)
    ax2 = ax.twinx()
    ax2.bar(x + 0.20, nla_bars, width=0.25, facecolor="none", edgecolor="0.25", lw=1.2, hatch="////",
            label="NLA read of the transplanted position matches a gold intermediate (right axis)")
    twins.append(ax2)
    ax.set_xticks(x)
    ax.set_xticklabels(TICKS)
    ax.set_xlabel("thought position $i$ of the twin's rollout")
    ax.set_title(title)
    for a in (ax, ax2):
        a.set_ylim(0, 1)
        a.spines["top"].set_visible(False)
twins[0].sharey(twins[1])
twins[0].tick_params(labelright=False)          # both right axes are 0..1: label the outer one only
axes[0].set_ylabel("adoption rate of the\ntransplanted answer")
twins[1].set_ylabel("any-intermediate match\naccuracy of the final reader")
h = [a.get_legend_handles_labels() for a in (axes[0], twins[0])]
fig.tight_layout()
fig.legend([h[0][0][0], h[1][0][0]], [h[0][1][0], h[1][1][0]], loc="lower center", bbox_to_anchor=(0.5, 1.0),
           ncol=2, frameon=False)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(OUT, {c: (round(S[c]["keep"], 3), round(S[c]["adopt"], 3), S[c]["n"]) for c in S})
