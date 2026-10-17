"""Activation patching stratified by CHAIN LENGTH (hops): does the causal latent locus
stay pinned at L1, or spread to later thoughts as a problem needs more reasoning steps?

The broad patching (codi_paper1_patching.py) found L1 (the first latent) is the sole
causal answer-bottleneck; L2-L6 are inert. That was pooled over all problems (~76%
need no latents). Here we ask the decisive follow-up: stratify by #hops and see whether
later positions become causally load-bearing on longer chains (= genuine sequential
latent computation) or L1 keeps doing everything regardless of difficulty (= the model
crams the computation into the first latent no matter what).

Cheaper position-level intervention: for each thought position we restore the ENTIRE
clean resid column (all 36 layers at once) into the corrupted run, so it's 7 patches per
problem instead of the 9x63 grid -- enough budget for a decent count per hop-bucket.
Reuses CodiRunner from codi_paper1_patching.

  srun -p general --qos=high --gres=gpu:1 --mem=128G --time=1:30:00 \
    --job-name=codi_pbh_$USER bash -lc 'set -a; source .env; set +a; \
    HF_HOME=/workspace-vast/$USER/hf_cache \
    /var/tmp/jbauer/venvs/activation_oracles_dev/bin/python \
    scripts/codi_patching_by_hops.py --n 2500 --per-bucket 50'
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "third_party", "CODI"))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
os.environ.setdefault("HF_HOME", f"/workspace-vast/{os.environ.get('USER', 'jbauer')}/hf_cache")
print = functools.partial(print, flush=True)

from codi_paper1_patching import CodiRunner, parse_numbers, CKPT, BASE, NUM_LATENT, POS  # noqa: E402

OUT_DIR = os.path.join(_REPO, "reports", "codi_paper_faithful")
BUCKET_NAMES = ["2", "3", "4", "5", "6+"]


def bucket_of(hops):
    if hops < 2:
        return None
    return "6+" if hops >= 6 else str(hops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--per-bucket", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    from datasets import load_dataset
    from src.model import CODI
    model = CODI.from_pretrained(checkpoint_path=CKPT, model_name_or_path=BASE, lora_r=128, lora_alpha=32,
                                 num_latent=NUM_LATENT, use_prj=True, prj_dim=4096, prj_no_ln=False,
                                 prj_dropout=0.0, dtype="bfloat16")
    model.eval()
    tok = model.tokenizer
    R = CodiRunner(model, tok)
    n_layers = len(R.layers)
    digit_tok = {d: tok(d, add_special_tokens=False).input_ids[0] for d in "0123456789"
                 if len(tok(d, add_special_tokens=False).input_ids) == 1}
    digit_ids = set(digit_tok.values())

    ds = load_dataset("zen-E/GSM8k-Aug-NL", split="train")
    lift_sum = {b: np.zeros(len(POS)) for b in BUCKET_NAMES}
    cnt = {b: 0 for b in BUCKET_NAMES}
    for di in range(args.n):
        if all(cnt[b] >= args.per_bucket for b in BUCKET_NAMES):
            break
        ex = ds[di]
        gold = str(ex["answer"]).strip()
        chain = [x for x in parse_numbers(ex["cot"]) if x != gold]
        b = bucket_of(len(chain))
        if b is None or cnt[b] >= args.per_bucket:
            continue
        q = ex["question"].strip().replace("  ", " ") + " Output only the answer and nothing else."
        ids = tok(q, return_tensors="pt").input_ids.to("cuda")
        toks = ids[0].tolist()
        cand = [i for i in range(len(toks) - 8) if toks[i] in digit_ids]
        if not cand:
            continue
        # screen cheaply (no capture) before paying for the all-layer capture
        clean_logits, _ = R.run(ids, capture=False)
        clean_tok = int(clean_logits.argmax())
        clean_pred = tok.decode([clean_tok]).strip()
        if clean_pred == "" or clean_pred != gold[: len(clean_pred)]:
            continue
        cpos = cand[len(cand) // 2]
        alt = next((t for d, t in digit_tok.items() if t != toks[cpos]), None)
        if alt is None:
            continue
        cids = ids.clone(); cids[0, cpos] = alt
        corr_logits, _ = R.run(cids, capture=False)

        def metric(lg):
            return float(lg.softmax(-1)[clean_tok])
        c_clean, c_corr = metric(clean_logits), metric(corr_logits)
        if c_clean - c_corr < 0.05:
            continue
        _, clean_cap = R.run(ids, capture=True)   # kept only: capture all (layer, position)
        lift = np.zeros(len(POS))
        for pi in range(len(POS)):
            patch = {(L, pi): clean_cap[(L, pi)] for L in range(n_layers)}
            patched, _ = R.run(cids, capture=False, patch=patch)
            lift[pi] = (metric(patched) - c_corr) / (c_clean - c_corr)
        lift_sum[b] += lift
        cnt[b] += 1
        if sum(cnt.values()) % 25 == 0:
            print(f"  kept {sum(cnt.values())}  counts={cnt}  last(hops={len(chain)},b={b}) "
                  f"lift={np.round(lift, 2)}")

    means = {b: (lift_sum[b] / cnt[b]).tolist() if cnt[b] else None for b in BUCKET_NAMES}
    json.dump(dict(positions=POS, counts=cnt, mean_lift=means),
              open(os.path.join(OUT_DIR, "patching_by_hops.json"), "w"))
    print(f"\n[patching-by-hops] counts={cnt}")
    print("mean position recovery lift by hop bucket:")
    print("  hops |  " + "  ".join(f"{p:>5}" for p in POS))
    for b in BUCKET_NAMES:
        if cnt[b]:
            print(f"  {b:>4} | " + "  ".join(f"{x:5.2f}" for x in means[b]))
    _plot(means, cnt)
    print("PBH-DONE")


def _plot(means, cnt):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for b in BUCKET_NAMES:
        if means[b] is not None:
            ax[0].plot(range(len(POS)), means[b], "o-", label=f"{b} hops (n={cnt[b]})")
    ax[0].set_xticks(range(len(POS))); ax[0].set_xticklabels(POS)
    ax[0].set_xlabel("patched thought position (full resid column)")
    ax[0].set_ylabel("mean answer-recovery lift")
    ax[0].set_title("Causal locus by chain length: does it spread past L1?")
    ax[0].grid(alpha=0.3); ax[0].legend(); ax[0].axhline(0, color="k", lw=0.5)
    M = np.array([means[b] if means[b] is not None else [np.nan] * len(POS) for b in BUCKET_NAMES])
    im = ax[1].imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="mean recovery lift")
    ax[1].set_xticks(range(len(POS))); ax[1].set_xticklabels(POS)
    ax[1].set_yticks(range(len(BUCKET_NAMES)))
    ax[1].set_yticklabels([f"{b} ({cnt[b]})" for b in BUCKET_NAMES])
    ax[1].set_xlabel("thought position"); ax[1].set_ylabel("# hops")
    ax[1].set_title("position x hop-bucket recovery lift")
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "p1_patching_by_hops.png")
    plt.savefig(png, dpi=150, bbox_inches="tight"); print(f"PNG: {png}")


if __name__ == "__main__":
    main()
