"""Does the ORDER in which the nx1 reader emits gold intermediates track the problem's chain order?

For each viz gen: find first-occurrence emission rank of each recovered gold value,
compute pairwise concordance (emission order vs gold chain order). Permutation null
within row. Magnitude control: concordance restricted to pairs where magnitude order
DISAGREES with chain order (kills the "say small numbers first" confound).
"""
import json, re, random
from collections import defaultdict

random.seed(0)

def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"

rows = [json.loads(l) for l in open("/workspace-vast/jbauer/bt1/nx1/viz_gens.jsonl")]
by_ckpt = defaultdict(list)
for r in rows: by_ckpt[r["ckpt"]].append(r)
print("ckpts:", {k: len(v) for k, v in by_ckpt.items()})

def emission_ranks(gen, gold):
    """first-occurrence emission rank (index among all number tokens) per distinct gold value"""
    ems = []
    for m in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", gen):
        try: ems.append(canon(m.group()))
        except ValueError: pass
    first = {}
    for rank, v in enumerate(ems):
        if v not in first: first[v] = rank
    # gold chain index = first position of the value in the ordered chain
    gidx = {}
    for i, v in enumerate(gold):
        c = canon(v)
        if c not in gidx: gidx[c] = i
    return {c: (gidx[c], first[c]) for c in gidx if c in first}  # value -> (chain_idx, emission_rank)

def analyze(ck):
    pairs_c = pairs_t = 0          # all pairs
    mag_c = mag_t = 0              # magnitude-discordant pairs only
    perrow_taus, contrib = [], 0
    perm_stats = []
    for r in by_ckpt[ck]:
        rec = emission_ranks(r["gen"], r["gold"])
        if len(rec) < 2: continue
        contrib += 1
        items = sorted(rec.values())  # by chain idx
        c = t = 0
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (ci, ei), (cj, ej) = items[a], items[b]
                if ei == ej: continue
                t += 1; conc = ei < ej
                c += conc
                vi = float(canon([v for v, x in rec.items() if x == items[a]][0]))
                vj = float(canon([v for v, x in rec.items() if x == items[b]][0]))
                if vi > vj:  # magnitude order disagrees with chain order
                    mag_t += 1; mag_c += conc
        pairs_c += c; pairs_t += t
        if t: perrow_taus.append(c / t)
        # permutation: shuffle emission ranks among recovered values
        ranks = [e for _, e in items]
        ps = []
        for _ in range(200):
            random.shuffle(ranks)
            pc = sum(ranks[a] < ranks[b] for a in range(len(ranks)) for b in range(a + 1, len(ranks)) if ranks[a] != ranks[b])
            pt = sum(1 for a in range(len(ranks)) for b in range(a + 1, len(ranks)) if ranks[a] != ranks[b])
            ps.append(pc / pt if pt else 0.5)
        perm_stats.append(ps)
    if not pairs_t:
        print(f"{ck:16s} no usable rows"); return
    obs = pairs_c / pairs_t
    # pooled permutation p (two-sided): mean over rows of shuffled concordance, 200 draws
    nperm = 200
    pooled = []
    for k in range(nperm):
        num = den = 0
        for i, r in enumerate(perm_stats):
            num += r[k]; den += 1
        pooled.append(num / den)
    mrow = sum(perrow_taus) / len(perrow_taus)
    lo = sum(1 for p in pooled if abs(p - 0.5) >= abs(mrow - 0.5)) / nperm
    mag = f"{mag_c/mag_t:.3f} (n={mag_t})" if mag_t else "n/a"
    print(f"{ck:16s} rows≥2vals {contrib:3d}  pooled-concordance {obs:.3f}  row-mean {mrow:.3f}  perm-p {lo:.3f}  mag-discordant-pairs {mag}")

for ck in by_ckpt: analyze(ck)
