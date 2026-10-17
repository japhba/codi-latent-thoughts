"""Pooled PCA spectra of the two NLA read objects (post: Appendix B).

Computes the eigen-spectrum of z (the fed-back thought vector) and h24 (residual
stream, layer 24) POOLED over thought positions 1..6 and CODI-correct problems, on
the same vectors the NLA trains on and with the same geometry as its reward: each
vector unit-normalized (direction-normalized, the recon reward is a direction MSE),
then centered PCA. Successor of scripts/bt1/nz2_pca_analysis.py with the by-position
strata dropped per the writeup figure.

Inputs (env FD1_DATA, default /workspace-vast/jbauer/bt1/fd1/data):
  gsm_meta.jsonl + gsm_z.npy      z vectors [n, >=6, 4096]
  gsml24s*_{meta.jsonl,h.npy}     h24 captures, thought positions at [:, 2:8, 0]
Output: reports/codi_writeup/repo/artifacts/pca_spectra.json
  {src: {"evr": [4096 eigenvalue fractions, descending], "n_vectors": int,
         "pr": participation ratio, "pcs_50/90/95": ints}}
CPU-only, ~2 min.
"""
import glob, json, os

import numpy as np

DATA = os.environ.get("FD1_DATA", "/workspace-vast/jbauer/bt1/fd1/data")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "artifacts", "pca_spectra.json")
N_PROB = 6000

meta = [json.loads(l) for l in open(f"{DATA}/gsm_meta.jsonl")]


def load_source(which):
    if which == "z":
        keep = [i for i, m in enumerate(meta) if m["correct"]][:N_PROB]
        Z = np.load(f"{DATA}/gsm_z.npy", mmap_mode="r")
        X = np.stack([np.asarray(Z[i, :6], dtype=np.float32) for i in keep])
    else:
        X_l = []
        for shard in sorted(glob.glob(f"{DATA}/gsml24s*_meta.jsonl")):
            name = os.path.basename(shard).replace("_meta.jsonl", "")
            sm = [json.loads(l) for l in open(shard)]
            H = np.load(f"{DATA}/{name}_h.npy", mmap_mode="r")
            for i, m in enumerate(sm):
                if not m["correct"]:
                    continue
                X_l.append(np.asarray(H[i, 2:8, 0], dtype=np.float32))
                if len(X_l) >= N_PROB:
                    break
            if len(X_l) >= N_PROB:
                break
        X = np.stack(X_l)
    X = X.reshape(-1, X.shape[-1])                       # pool positions
    X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
    return X


out = {}
for src in ("z", "h24"):
    X = load_source(src)
    X = X - X.mean(0)
    cov = (X.T @ X).astype(np.float64) / (len(X) - 1)
    ev = np.linalg.eigvalsh(cov)[::-1]
    ev = np.clip(ev, 0, None)
    evr = ev / ev.sum()
    cum = np.cumsum(evr)
    pcs = {p: int(np.searchsorted(cum, p / 100) + 1) for p in (50, 90, 95)}
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    out[src] = dict(evr=[float(v) for v in evr], n_vectors=int(len(X)),
                    pr=round(pr, 1), **{f"pcs_{p}": v for p, v in pcs.items()})
    print(f"{src}: n={len(X)} PR={pr:.1f} PCs for 50/90/95%: "
          f"{pcs[50]}/{pcs[90]}/{pcs[95]}")
json.dump(out, open(OUT, "w"))
print(f"wrote {os.path.abspath(OUT)}")
