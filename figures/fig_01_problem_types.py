"""Figure: problem-type decomposition of the 1319 GSM8k test problems.

One full-width stacked bar over ALL test problems, isomorphic to the nested-donut
view: the thin top bar shows the base-Qwen role (hue), the main bar beneath it
subdivides each role by CODI outcome (shading of the role hue):
  CODI-necessary (dark) = six-thought decode correct AND zero-thought decode wrong
  CODI single-passed (mid) = six- AND zero-thought decodes both correct
  CODI wrong (light) = six-thought decode wrong
Red boxes mark the LOAD-BEARING problems (CODI-necessary segments across all base
roles, n=104): the rows where CODI needs its latent thoughts. CODI outcomes come from
the masked rv6 decodes (artifacts/rv6/controls/baseline.json: full padding
masks on every cached step, so answers no longer depend on batch composition).

Inputs (env-overridable):
  CODI_BASELINE   default artifacts/rv6/controls/baseline.json
  BASE_ROLES      default artifacts/base_qwen_test.json
Output: figs/problem_types.png (or env FIG_OUT)
"""
import json, math, os

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
INST_F = os.environ.get("CODI_BASELINE", os.path.join(HERE, "..", "artifacts",
                                                      "rv6", "controls", "baseline.json"))
BASE_F = os.environ.get("BASE_ROLES", os.path.join(HERE, "..", "artifacts", "base_qwen_test.json"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs", "problem_types.png"))

inst = json.load(open(INST_F))["rows"]
base = json.load(open(BASE_F))["rows"]

def ok(rec, mode):
    p = rec[f"{mode}_pred"]
    return p is not None and math.isclose(float(p), float(rec["gold"]), rel_tol=1e-6, abs_tol=1e-6)

counts = {}
for rec in inst:
    role = base[str(rec["i"])]["role"]
    six, zero = ok(rec, "six"), ok(rec, "zero")
    codi = "wrong" if not six else ("necessary" if not zero else "singlepass")
    counts[(role, codi)] = counts.get((role, codi), 0) + 1
n = sum(counts.values())

ROLES = [("load_bearing", "base needs verbal CoT", "#2a78d6"),
         ("other", "base unreliable even with CoT", "#eb6834"),
         ("easy", "base solves without CoT", "#1baf7a")]
CODI = [("necessary", "CODI-necessary", "dark"),
        ("singlepass", "CODI single-passed", "mid"),
        ("wrong", "CODI wrong", "light")]

def shade(hex_color, level):
    r, g, b = mcolors.to_rgb(hex_color)
    if level == "dark":
        return tuple(0.72 * c for c in (r, g, b))
    t = {"mid": 0.55, "light": 0.25}[level]
    return tuple(1 - t * (1 - c) for c in (r, g, b))

fig, ax = plt.subplots(figsize=(9.5, 2.9))
Y_ROLE, Y_OUT = 1.0, 0.0                      # thin role bar above, outcome bar below
left = 0
for role, rlabel, hue in ROLES:
    tot = sum(counts.get((role, c), 0) for c, _, _ in CODI)
    ax.barh(Y_ROLE, tot, left=left, height=0.32, color=hue, edgecolor="white")
    left += tot
left = 0
lb_segs = []                                  # (left, width) of every dark segment
for role, rlabel, hue in ROLES:
    for codi, clabel, strength in CODI:
        c = counts.get((role, codi), 0)
        if c == 0:
            continue
        ax.barh(Y_OUT, c, left=left, height=0.62, color=shade(hue, strength),
                edgecolor="white")
        if codi == "necessary":
            lb_segs.append((left, c))
        left += c
# red boxes around the load-bearing rows (all CODI-necessary segments)
for x0, w in lb_segs:
    ax.add_patch(Rectangle((x0, Y_OUT - 0.31), w, 0.62, fill=False,
                           edgecolor="#c81e1e", lw=1.8, zorder=5))

ax.set_yticks([Y_ROLE, Y_OUT])
ax.set_yticklabels(["base model", "CODI"])
ax.set_xlim(0, n)
ax.set_ylim(-0.55, 1.45)
ax.set_xlabel("GSM8k test problems")
legend = ([Patch(facecolor=h, label=l) for _, l, h in ROLES]
          + [Patch(facecolor=shade("#777777", s), label=l) for _, l, s in CODI])
ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"wrote {OUT}  counts={counts}")
