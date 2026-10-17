"""NLA round-trip intervention figure (post: Interventions / via the NLA).

The language-mediated loop with the final step-3200 verbalizer (AV) and its matching
reconstructor (AR): the AV describes each of the six thought vectors of the receiver
(the twin problem), numeric intermediate values in the descriptions are replaced by the
donor's corresponding values (v -> v'), the AR reconstructs vectors from the (edited)
descriptions, and all six are patched back jointly with downstream regeneration.
Conditions, on the 59 twin pairs where both problems are six-only successes:
  recon roundtrip   describe -> reconstruct -> patch, no text edit (fidelity control)
  value edit v->v'  the intervention
  matched random    own vector + random direction at the edit's angular displacement
  donor's text      reconstruct from the donor's own descriptions
One panel per source (z, h24). Grouped bars per condition: keeps the receiver's answer
(gray) and adopts the donor's (counterfactual) answer (C0), 95% bootstrap CIs clustered
by original problem (companion package, rv6 twin_summary.json).
Output: figs/nla_intervention_z.png (env FIG_OUT). CPU, matplotlib defaults.
"""
import json, os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.environ.get("TWIN_SUMMARY", os.path.join(HERE, "..", "results",
                                                      "twin_summary.json"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs", "nla_intervention_z.png"))

cond = json.load(open(SUMMARY))["conditions"]
BARS = [("own", "recon roundtrip, no edit"), ("edited", r"value edit $v \to v'$"),
        ("sham", "matched random displacement"), ("donor", "donor's description")]
SRCS = [("z", r"$\mathbf{z}$"), ("h24", r"$\mathbf{h}^{(24)}$")]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
ys = range(len(BARS))[::-1]
for ax, (src, title) in zip(axes, SRCS):
    for y, (key, _) in zip(ys, BARS):
        stats = cond[f"nla_{src}_{key}_all6"]["both_six_only"]
        for metric, off, color, lab in (("keep", 0.19, "0.75", "keeps its own answer"),
                                        ("adopt", -0.19, "C0", "adopts the counterfactual answer")):
            s = stats[metric]
            ax.barh(y + off, s["rate"], height=0.38, color=color, label=lab if y == ys[0] else None)
            ax.errorbar(s["rate"], y + off, xerr=[[s["rate"] - s["ci95"][0]], [s["ci95"][1] - s["rate"]]],
                        ls="none", color="k", capsize=3)
            print(f"{src:>3} {key:7s} {metric:5s} {s['rate']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] n={s['n']}")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of load-bearing twin pairs")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_yticks(list(ys))
axes[0].set_yticklabels([b[1] for b in BARS])
fig.tight_layout()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, frameon=False)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(OUT)
