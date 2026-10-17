"""Figure: pooled PCA spectra of the two NLA read objects (post: Appendix A).

Single panel, x = number of principal components (log), y = cumulative variance
explained, one curve per read object: z (fed-back thought vector) and h24 (residual
stream, layer 24), pooled over thought positions and CODI-correct problems on
unit-normalized vectors (the NLA reward geometry). Dotted guides at 90 percent with
the component counts. Colors match the NLA training figure.

Inputs (env-overridable):
  PCA_SPECTRA  default repo/artifacts/pca_spectra.json (from scripts/codi_pca_spectra.py)
Output: figs/pca_dims.png (env FIG_OUT)
"""
import json, os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("PCA_SPECTRA", os.path.join(HERE, "..", "artifacts", "pca_spectra.json"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs", "pca_dims.png"))
COL = {"z": "#c23b3b", "h24": "#2a78d6"}
LAB = {"z": r"thought vector $\mathbf{z}$",
       "h24": r"residual stream $\mathbf{h}^{(24)}$"}

spec = json.load(open(SRC))
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for src in ("h24", "z"):
    s = spec[src]
    cum = np.cumsum(s["evr"])
    ax.plot(np.arange(1, len(cum) + 1), cum, color=COL[src], label=LAB[src])
    k90 = s["pcs_90"]
    ax.plot([k90], [cum[k90 - 1]], marker="o", ms=6, mfc="none", mec=COL[src])
    ax.annotate(f"{k90} PCs → 90%", (k90, cum[k90 - 1]), textcoords="offset points",
                xytext=(6, -12), ha="left", color=COL[src])
    print(f"{src}: PR={s['pr']} PCs 50/90/95%: {s['pcs_50']}/{s['pcs_90']}/{s['pcs_95']}")
ax.axhline(0.9, color="0.7", lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_xlabel("number of principal components (log)")
ax.set_ylabel("cumulative variance explained")
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
