"""nx2 per-position ordering: WHICH chain step surfaces at WHICH thought position?

Input: posgen.jsonl (row, t, kind own/donor, gen, gold ordered chain).
Outputs: P(step i recovered | position t) matrix, mean normalized step fraction per
position, Spearman(position, recovered step index), donor as chance floor,
and chain-length dependence of the z3 content.
"""
import json, re, sys
from collections import defaultdict
import numpy as np

def spearmanr(x, y, nperm=2000):
    """Spearman rho + permutation p (two-sided). Returns object with .statistic/.pvalue"""
    rng = np.random.default_rng(0)
    def rank(a):
        a = np.asarray(a, float); order = a.argsort(kind="stable"); r = np.empty(len(a))
        r[order] = np.arange(len(a))
        # average ties
        for v in np.unique(a):
            m = a == v; r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    rho = float((rx * ry).sum() / denom) if denom else 0.0
    cnt = 0
    for _ in range(nperm):
        p = rng.permutation(ry)
        r = float((rx * p).sum() / denom) if denom else 0.0
        cnt += abs(r) >= abs(rho)
    class R: statistic, pvalue = rho, cnt / nperm
    return R

def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "/workspace-vast/jbauer/bt1/nx2/posgen.jsonl")]
print(f"{len(rows)} gens")

def recovered_steps(r):
    ems = set()
    for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", r["gen"]):
        try: ems.add(canon(m))
        except ValueError: pass
    gidx = {}
    for i, v in enumerate(r["gold"]):
        c = canon(v)
        if c not in gidx: gidx[c] = i
    return [(i, len(r["gold"])) for c, i in gidx.items() if c in ems]

for kind in ("own", "donor"):
    sub = [r for r in rows if r["kind"] == kind]
    # matrix: P(step i recovered | position t), i as raw index (cap 4+) and as fraction
    mat = defaultdict(lambda: defaultdict(int)); denom = defaultdict(lambda: defaultdict(int))
    pairs_t, pairs_f = [], []          # (position, step index), (position, step fraction)
    for r in sub:
        L = len(r["gold"])
        rec = recovered_steps(r)
        for i in range(L):
            denom[r["t"]][min(i, 3)] += 1
        for i, _ in rec:
            mat[r["t"]][min(i, 3)] += 1
            pairs_t.append((r["t"], i))
            pairs_f.append((r["t"], (i + 0.5) / L))
    print(f"\n=== {kind} ===  P(step i recovered | position t):")
    print("        step1   step2   step3   step4+")
    for t in range(6):
        cells = " ".join(f"{mat[t][i]/max(denom[t][i],1):7.3f}" for i in range(4))
        print(f"  z{t+1}   {cells}   (n_rec={sum(mat[t].values())})")
    if len(pairs_t) > 5:
        rho_t = spearmanr([p for p, _ in pairs_t], [s for _, s in pairs_t])
        rho_f = spearmanr([p for p, _ in pairs_f], [s for _, s in pairs_f])
        print(f"  Spearman(position, step idx)      rho={rho_t.statistic:+.3f} p={rho_t.pvalue:.3g} (n={len(pairs_t)})")
        print(f"  Spearman(position, step fraction) rho={rho_f.statistic:+.3f} p={rho_f.pvalue:.3g}")
    # mean step fraction per position
    frac = defaultdict(list)
    for p, fr in pairs_f: frac[p].append(fr)
    print("  mean step-fraction per position:", {f"z{t+1}": f"{np.mean(frac[t]):.2f}(n={len(frac[t])})" for t in sorted(frac)})

# chain-length dependence at the dominant position (z3, t=2), own only
print("\n=== z3 content vs chain length (own) ===")
by_len = defaultdict(lambda: defaultdict(int)); nrows = defaultdict(int)
for r in rows:
    if r["kind"] != "own" or r["t"] != 2: continue
    L = len(r["gold"]); nrows[L] += 1
    for i, _ in recovered_steps(r): by_len[L][i] += 1
for L in sorted(by_len):
    tot = sum(by_len[L].values())
    dist = " ".join(f"s{i+1}:{by_len[L][i]}" for i in sorted(by_len[L]))
    print(f"  len={L} (rows={nrows[L]}, recoveries={tot}): {dist}")
