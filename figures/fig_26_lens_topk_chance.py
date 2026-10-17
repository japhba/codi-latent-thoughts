"""Chance-controlled LogitLens / J-lens readability at the thought positions (Appendix B).

For each of the 156 held-out reader evaluation questions, layer (0 = the fed-back thought z, 1..36 = block
outputs) and lens, the top-k tokens of the six thought positions are pooled and mapped to the numbers they
express (the companion package's reviewed gold-blind token dictionary where it covers a token, otherwise a
deterministic parser for numerals, English number words and single Chinese numerals). A problem counts if
the pool contains its gold final answer, or any intermediate result of its reference solution. The chance
baseline scores the same pool against the gold values of every other question and averages, so it gives the
hit rate expected from the pool's numbers alone. Rows: k = 5 and k = 100.

Inputs: artifacts/lens_topk.npz and its decoded token strings artifacts/lens_topk_tokens.json (both from
codi_repro.lens_topk), artifacts/
Output: figs/lens_topk_chance.png (env FIG_OUT), summary results/lens_topk_chance_summary.json
"""
import json, os, re, sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.join(HERE, "..")
sys.path.insert(0, REPRO)
from codi_repro.all_layer_analysis import dictionary, literal  # noqa: E402
from codi_repro.analysis import canonical_number, read  # noqa: E402
from codi_repro.curve_lenses import intermediates  # noqa: E402

TOPK_F = os.environ.get("LENS_TOPK", os.path.join(REPRO, "artifacts", "lens_topk.npz"))
TOKENS_F = os.environ.get("LENS_TOPK_TOKENS", os.path.join(REPRO, "artifacts", "lens_topk_tokens.json"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs", "lens_topk_chance.png"))
COL = {"LogitLens": "0.15", "J-lens": "0.6"}
KS = (5, 100)

WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
              "hundred": 100, "thousand": 1000, "million": 1000000})
HANZI = dict(zip("零一二两三四五六七八九十百千万", [0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000]))

TOKENS = json.load(open(TOKENS_F))
semantics = dictionary()


def token_values(token_id):
    s = TOKENS[str(int(token_id))].strip()
    if s in semantics:
        return set(semantics[s])
    if (v := literal(s)) is not None:
        return {v}
    w = re.sub(r"[^a-z]", "", s.lower())
    if w in WORDS and len(w) == len(re.sub(r"[^A-Za-z]", "", s)):
        return {canonical_number(str(WORDS[w]))}
    if s in HANZI:
        return {canonical_number(str(HANZI[s]))}
    return set()


d = np.load(TOPK_F)
top, uids, methods = d["top_ids"], d["uids"].tolist(), d["methods"].tolist()     # top: [method, 37, n, 6, K]
problems = {p["uid"]: p for p in read("artifacts/problems.json") + read("artifacts/ao_curves.json")["problems"]}
values_of = {t: token_values(t) for t in np.unique(top)}
gold = {"answer": [{canonical_number(problems[u]["answer"])} for u in uids],
        "intermediate": [intermediates(problems[u]) for u in uids]}

summary = {}
fig, axes = plt.subplots(len(KS), 2, figsize=(11, 6.4), sharex=True, sharey=True)
for row, k in enumerate(KS):
    for col, criterion in enumerate(("answer", "intermediate")):
        ax = axes[row, col]
        G = gold[criterion]
        eligible = [i for i, g in enumerate(G) if g]
        for m, method in enumerate(methods):
            own, chance = [], []
            for hs in range(37):
                pools = [set().union(*(values_of[t] for t in top[m, hs, i, :, :k].ravel())) for i in eligible]
                own.append(np.mean([bool(p & G[i]) for p, i in zip(pools, eligible)]))
                chance.append(np.mean([np.mean([bool(p & G[j]) for j in eligible if j != i]) for p, i in zip(pools, eligible)]))
            summary[f"{method}/{criterion}/k{k}"] = {"own": own, "chance": chance, "n": len(eligible)}
            ax.plot(range(37), own, color=COL[method], label=f"{method}")
            ax.plot(range(37), chance, color=COL[method], ls=":", label=f"{method}, chance")
            best = int(np.argmax(np.array(own) - np.array(chance)))
            print(f"k={k:3d} {criterion:12s} {method:9s} z: {own[0]:.2f} (chance {chance[0]:.2f})  h24: {own[24]:.2f} ({chance[24]:.2f})  "
                  f"largest excess at {best}: {own[best]:.2f} vs {chance[best]:.2f}")
        ax.set_title(f"{'final answer' if criterion == 'answer' else 'any intermediate result'}, top {k}")
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
        ax.set_xticklabels([r"$\mathbf{z}$", "6", "12", "18", "24", "30", "36"])
        ax.spines[["top", "right"]].set_visible(False)
for ax in axes[-1]:
    ax.set_xlabel("layer read at the thought positions")
for ax in axes[:, 0]:
    ax.set_ylabel("fraction of problems")
fig.tight_layout()
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4, frameon=False)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
os.makedirs(os.path.join(REPRO, "results"), exist_ok=True)
json.dump(summary, open(os.path.join(REPRO, "results", "lens_topk_chance_summary.json"), "w"))
print(OUT)
