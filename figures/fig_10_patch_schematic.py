"""Schematic of the intervention geometry (post: Interventions intro): the transformer
grid over CODI's rollout with the patched objects highlighted. Pure illustration, no
data. x = positions (question tokens, bocot, thoughts z1..z6, answer tokens),
y = layers with layer 1 at the bottom (project convention). Shows:
  blue row segment   h24 patch: residual stream at layer 24, thought positions
  orange column(s)   column patch: the fed thought z_t plus ALL layer outputs
                     at selected thought positions (outline spans the input row)
  red input squares  z swap: the fed-back thought vector entering the next column
  feedback arrows    h36 at thought t -> prj -> input of thought t+1
Output: figs/patch_schematic.png (env FIG_OUT).
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs", "patch_schematic.png"))

NLAY = 12                  # drawn layers (schematic stand-in for 36)
NQ = 4                     # question slots (one is an ellipsis: the question is longer)
THT = 6
NA = 3                     # answer slots (one is an ellipsis: the answer is longer)
L24 = 8                    # drawn row standing in for layer 24
COL_PATCH = [2, 4]         # thought positions with a full-column patch (1-indexed: t3, t5)

xs_q = [0, 1, 3]           # drawn question cells; x=2 is the omission column
x_bot = NQ
xs_t = [NQ + 1 + t for t in range(THT)]
xs_a = [NQ + 1 + THT + 1 + a for a in (0, 2)]   # drawn answer cells; middle is omitted
ell_x = [2, NQ + 1 + THT + 2]                   # omission columns (question, answer)
all_x = xs_q + [x_bot] + xs_t + xs_a

C_GRID, C_H24, C_COL, C_Z = "0.82", "#2a78d6", "#eb6834", "#c23b3b"

fig, ax = plt.subplots(figsize=(9, 4.6))
for x in all_x:
    for y in range(NLAY):
        ax.add_patch(Rectangle((x - 0.38, y - 0.38), 0.76, 0.76, facecolor="white",
                               edgecolor=C_GRID, lw=0.8))
# input (embedding) row below layer 0
for x in all_x:
    ax.add_patch(Rectangle((x - 0.38, -1.38), 0.76, 0.76, facecolor="0.95",
                           edgecolor=C_GRID, lw=0.8))
# omission columns: question and answer are longer than drawn
for x in ell_x:
    ax.text(x, (NLAY - 1) / 2, "⋯", ha="center", va="center", color="0.45")
    ax.text(x, -1.0, "⋯", ha="center", va="center", color="0.45")

# highlight: h24 row patch at thought positions
for x in xs_t:
    ax.add_patch(Rectangle((x - 0.38, L24 - 0.38), 0.76, 0.76, facecolor=C_H24,
                           edgecolor="white", lw=0.8, alpha=0.9))
# highlight: full-column patch at selected thought positions
for t in COL_PATCH:
    x = xs_t[t]
    ax.add_patch(Rectangle((x - 0.44, -1.44), 0.88, NLAY + 0.88, fill=False,
                           edgecolor=C_COL, lw=2.2))
    for y in range(NLAY):
        ax.add_patch(Rectangle((x - 0.38, y - 0.38), 0.76, 0.76, facecolor=C_COL,
                               edgecolor="white", lw=0.8, alpha=0.35))
    ax.add_patch(Rectangle((x - 0.38, L24 - 0.38), 0.76, 0.76, facecolor=C_H24,
                           edgecolor="white", lw=0.8, alpha=0.9))
# highlight: fed z vectors (input row of thought positions)
for x in xs_t:
    ax.add_patch(Rectangle((x - 0.38, -1.38), 0.76, 0.76, facecolor=C_Z,
                           edgecolor="white", lw=0.8, alpha=0.75))

ax.annotate(r"$\mathbf{h}^{(24)}$ patch at" "\nthought positions", (xs_a[-1] + 1.4, L24),
            ha="left", va="center", color=C_H24)
ax.annotate(r"column patch: $\{\mathbf{z}\} \cup \{\mathbf{h}^{(\ell)}\}_{\ell=1}^{36}$"
            "\nat selected positions", (xs_a[-1] + 1.4, NLAY - 2.0),
            ha="left", va="center", color=C_COL)
ax.annotate(r"$\mathbf{z}$ swap: the fed vectors", (xs_a[-1] + 1.4, -1.0), ha="left", va="center",
            color=C_Z)
ax.plot([xs_t[-1] + 0.44, xs_a[-1] + 1.2], [L24, L24], color=C_H24, lw=0.7, ls=":")

ax.set_xticks([NQ / 2 - 0.5, x_bot] + xs_t + [sum(xs_a) / len(xs_a)])
ax.set_xticklabels(["question", "bocot"] + [rf"$\mathbf{{z}}_{t + 1}$" for t in range(THT)] + ["answer"])
ax.set_yticks([0, L24, NLAY - 1])
ax.set_yticklabels(["1", "24", "36"])
ax.set_xlabel("position")
ax.set_ylabel("layer")
ax.set_xlim(-0.8, xs_a[-1] + 5.2)
ax.set_ylim(-2.0, NLAY + 1.3)
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(OUT)
