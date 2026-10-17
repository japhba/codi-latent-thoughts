"""Two-set area-proportional Venn over the CoT-need categories (post: Appendix C):
  A base needs CoT      (role load_bearing in base_qwen_test.json: the base model
                         fails the single pass but solves with verbal CoT)
  B CODI needs some CoT (battery trunc_0, the latent decode with the trace removed,
                         fails, but the latent mode (battery identity) or the verbal
                         mode (heldout harness) solves it, greedy throughout)
Circle areas scale with the set sizes and the center distance is solved by bisection
so the lens area matches the intersection count. Region sizes drawn as percentages
of the 1319 test problems, counts printed.

Inputs (env-overridable): CODI_INSTANCES, BASE_ROLES, HELDOUT_INSTANCES.
Output: figs/category_venn.png (env FIG_OUT).
"""
import json, math, os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
INST_F = os.environ.get("CODI_INSTANCES", os.path.join(ART, "controls_instances.json"))
BASE_F = os.environ.get("BASE_ROLES", os.path.join(ART, "base_qwen_test.json"))
HELD_F = os.environ.get("HELDOUT_INSTANCES", os.path.join(ART, "heldout_instances.json"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs", "category_venn.png"))

inst = json.load(open(INST_F))
base = json.load(open(BASE_F))["rows"]
held = {r["i"]: r for r in json.load(open(HELD_F))["gsm8k"]}


def close(a, b):
    return a is not None and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)


cnt = {}
for r in inst:
    h = held[r["i"]]
    A = base[str(r["i"])]["role"] == "load_bearing"
    nocot = close(r["pred"]["trunc_0"], r["gold"])
    latent = close(r["pred"]["identity"], r["gold"])
    verbal = close(h["pred"]["verbal_greedy"], h["gold"])
    B = (not nocot) and (latent or verbal)
    cnt[(A, B)] = cnt.get((A, B), 0) + 1
N = sum(cnt.values())
for c in sorted(cnt, reverse=True):
    print("".join("AB"[i] if c[i] else "." for i in range(2)), cnt[c])

nA = cnt[(True, False)] + cnt[(True, True)]
nB = cnt[(False, True)] + cnt[(True, True)]
nAB = cnt[(True, True)]


def lens(d, r1, r2):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a1 = math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    tri = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return r1 * r1 * a1 + r2 * r2 * a2 - tri


rA = 0.28
rB = rA * math.sqrt(nB / nA)
target = math.pi * rA * rA * nAB / nA
lo, hi = abs(rA - rB), rA + rB
for _ in range(80):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if lens(mid, rA, rB) > target else (lo, mid)
d = (lo + hi) / 2
print(f"rA={rA:.3f} rB={rB:.3f} d={d:.3f}")

cxA, cy = 0.40, 0.48
cxB = cxA + d
COLORS = ["C0", "C3"]


def pct(code):
    return f"{100 * cnt.get(code, 0) / N:.1f}%"


fig, ax = plt.subplots(figsize=(6.0, 3.8))
for (cx, r), col in zip([(cxA, rA), (cxB, rB)], COLORS):
    ax.add_patch(Circle((cx, cy), r, facecolor=col, alpha=0.14, edgecolor=col, lw=1.8))
ax.text(cxA, cy + rA + 0.05, f"base needs CoT ({100 * nA / N:.1f}%)",
        ha="center", va="center", color=COLORS[0])
ax.text(cxB + 0.04, cy - rB - 0.09, f"CODI needs some CoT ({100 * nB / N:.1f}%)",
        ha="center", va="center", color=COLORS[1])
ax.text((cxA - rA + cxB - rB) / 2, cy, pct((True, False)), ha="center", va="center")
ax.text((cxB - rB + cxA + rA) / 2, cy, pct((True, True)), ha="center", va="center")
ax.annotate(pct((False, True)), xy=((cxA + rA + cxB + rB) / 2, cy),
            xytext=(cxB + rB + 0.10, cy + 0.14), ha="center", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.8, color="0.4"))
ax.text(0.99, 0.985, f"in neither: {pct((False, False))}", ha="right", va="top",
        color="0.35", transform=ax.transAxes)
ax.set_xlim(0.05, 0.98)
ax.set_ylim(0.10, 0.90)
ax.set_aspect(1)
ax.axis("off")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
