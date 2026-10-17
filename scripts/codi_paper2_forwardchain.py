"""Faithful Paper 2 (Dilgren & Wiegreffe 2604.04902) FORWARD-CHAINING trace decode on
our CODI Qwen3-8B. (Their code is unreleased; reimplemented from the paper's description.)

Their method (3 steps):
  (1) candidate steps: assume a latent's top logit-lens integer token is the RESULT of an
      arithmetic step; find operator/operand combos (over question numbers + prior results)
      that produce it.
  (2) verify: counterfactually change an operand in the input and check the latent's
      vocabulary projection changes as predicted.
  (3) assemble: chain verified steps backward from the final answer.
They report a verified trace is recoverable 54-93% for CORRECT predictions, far less for
incorrect ones (decodability tracks correctness).

We measure three nested rates, for correct- vs incorrect-prediction problems:
  - arith_reachable : answer reachable by arithmetic over QUESTION numbers alone
                      (no latents) -- the substrate baseline.
  - read_decodable  : the same, but every intermediate result must also appear in some
                      latent's logit-lens top-k (Paper 2 step 1+3, "read from latents").
  - verified_decodable : read_decodable AND each path step survives the counterfactual
                      operand-swap check (Paper 2 step 2).

  srun -p general --qos=high --gres=gpu:1 --mem=128G --time=1:30:00 \
    --job-name=codi_p2fc_$USER bash -lc 'set -a; source .env; set +a; \
    HF_HOME=/workspace-vast/$USER/hf_cache \
    /var/tmp/jbauer/venvs/activation_oracles_dev/bin/python \
    scripts/codi_paper2_forwardchain.py --n 400 --verify 120'
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import os
import re
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "third_party", "CODI"))
os.environ.setdefault("HF_HOME", f"/workspace-vast/{os.environ.get('USER', 'jbauer')}/hf_cache")
print = functools.partial(print, flush=True)

CKPT = "/workspace-vast/jbauer/codi_eval_ckpt/checkpoint-7580"
BASE = "Qwen/Qwen3-8B"
NUM_LATENT = 6
TOPK = 10
OUT_DIR = os.path.join(_REPO, "reports", "codi_paper_faithful")


def parse_nums(text):
    out = []
    for s in re.findall(r"-?\d+(?:\.\d+)?", text):
        out.append(float(s) if "." in s else int(s))
    return out


def as_int(tok_str):
    s = tok_str.strip()
    return int(s) if re.fullmatch(r"\d{1,7}", s) else None


def reachable(target, operands, readable=None, max_depth=4, max_results=4000):
    """Forward-chain: can `target` be built from operands via +,-,*,/ where every
    derived result is in `readable` (if given)? Returns (bool, path). Bounded: results
    are pruned to plausible magnitudes and capped (the unbounded readable=None baseline
    would otherwise explode combinatorially)."""
    target = round(float(target), 3)
    results = {round(float(o), 3): [] for o in operands}
    if target in results:
        return True, []
    for _ in range(max_depth):
        new = {}
        keys = list(results)
        for a, b in itertools.permutations(keys, 2):
            for op, f in (("+", a + b), ("-", a - b), ("*", a * b), ("/", a / b if b else None)):
                if f is None:
                    continue
                r = round(float(f), 3)
                if r < 0 or abs(r) > 1e6 or (r != int(r) and abs(r * 100 - round(r * 100)) > 1e-6):
                    continue
                if r in results or r in new:
                    continue
                if readable is not None and r != target and r not in readable:
                    continue
                new[r] = results[a] + [(a, op, b, r)]
                if r == target:
                    return True, new[r]
        if not new:
            break
        # prune to keep the search bounded (smallest-magnitude first)
        for r in sorted(new, key=abs)[:max_results]:
            results[r] = new[r]
        if len(results) > max_results * 2:
            results = {k: results[k] for k in sorted(results, key=abs)[:max_results]}
    return (target in results), results.get(target, [])


@torch.no_grad()
def latent_reads(model, tok, q, bocot, eocot, head, norm, return_latents=False):
    ids = tok(q, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(input_ids=ids, tokenizer=tok, max_new_tokens=4, num_latent_iterations=NUM_LATENT,
                         greedy=True, output_hidden_states=True, return_latent_vectors=True,
                         sot_token=bocot, eot_token=eocot)
    last = torch.stack([v.squeeze(0).squeeze(0) for v in out["latent_vectors_pre_prj"]])  # [7,4096]
    logits = head(norm(last.to(head.weight.dtype)))
    topk = logits.topk(TOPK, dim=-1).indices.cpu().tolist()
    reads = []                                            # per position: set of int values read
    for row in topk:
        vals = {as_int(tok.decode([t])) for t in row}
        reads.append({v for v in vals if v is not None})
    gen = tok.decode(out["sequences"][0], skip_special_tokens=True).strip()
    pred = parse_nums(gen)
    return reads, (pred[0] if pred else None), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--offset", type=int, default=6000)
    ap.add_argument("--verify", type=int, default=120, help="max problems to run counterfactual verification on")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    from datasets import load_dataset
    from src.model import CODI
    model = CODI.from_pretrained(checkpoint_path=CKPT, model_name_or_path=BASE, lora_r=128, lora_alpha=32,
                                 num_latent=NUM_LATENT, use_prj=True, prj_dim=4096, prj_no_ln=False,
                                 prj_dropout=0.0, dtype="bfloat16")
    model.eval()
    tok = model.tokenizer
    bocot, eocot = tok.convert_tokens_to_ids("<|bocot|>"), tok.convert_tokens_to_ids("<|eocot|>")
    head, norm = model.codi.get_output_embeddings(), model.codi.model.model.norm
    digit_tok = {d: tok(d, add_special_tokens=False).input_ids[0] for d in "0123456789"
                 if len(tok(d, add_special_tokens=False).input_ids) == 1}
    digit_ids = set(digit_tok.values())

    ds = load_dataset("zen-E/GSM8k-Aug-NL", split="train")
    rows = []
    n_verified = 0
    for di in range(args.offset, args.offset + args.n):
        ex = ds[di]
        gold = str(ex["answer"]).strip()
        q = ex["question"].strip().replace("  ", " ") + " Output only the answer and nothing else."
        qnums = parse_nums(ex["question"])
        if not qnums:
            continue
        reads, pred, ids = latent_reads(model, tok, q, bocot, eocot, head, norm)
        try:
            goldv = float(gold)
        except ValueError:
            continue
        correct = pred is not None and abs(float(pred) - goldv) < 1e-6
        all_reads = set().union(*reads) if reads else set()
        arith_ok, _ = reachable(goldv, qnums)
        read_ok, path = reachable(goldv, qnums, readable=all_reads)

        # step 2: counterfactual verification of the read path (subset of problems)
        verified = None
        if read_ok and path and n_verified < args.verify:
            verified = _verify_path(model, tok, ids, ex["question"], q, path, reads, qnums,
                                    bocot, eocot, head, norm, digit_ids, digit_tok)
            n_verified += 1
        rows.append(dict(ds_index=di, gold=gold, pred=pred, correct=correct,
                         arith=arith_ok, read=read_ok, verified=verified,
                         path_len=len(path)))
        if len(rows) <= 5:
            print(f"  di={di} gold={gold} pred={pred} correct={correct} arith={arith_ok} "
                  f"read={read_ok} verified={verified} reads={[sorted(r)[:5] for r in reads]}")

    _report(rows)
    print("P2-FC-DONE")


@torch.no_grad()
def _verify_path(model, tok, ids, question, q, path, reads, qnums, bocot, eocot, head, norm, digit_ids, digit_tok):
    """For each step (a,op,b,r) read at some latent, change an operand digit in the
    question and check that the result read by a latent shifts away from r (Paper 2 step 2)."""
    toks = ids[0].tolist()
    qpos = [i for i in range(len(toks) - 8) if toks[i] in digit_ids]
    if not qpos:
        return False
    cpos = qpos[len(qpos) // 2]
    cur = toks[cpos]
    alt = next((t for d, t in digit_tok.items() if t != cur), None)
    if alt is None:
        return False
    cids = ids.clone(); cids[0, cpos] = alt
    out = model.generate(input_ids=cids, tokenizer=tok, max_new_tokens=4, num_latent_iterations=NUM_LATENT,
                         greedy=True, output_hidden_states=True, return_latent_vectors=True,
                         sot_token=bocot, eot_token=eocot)
    last = torch.stack([v.squeeze(0).squeeze(0) for v in out["latent_vectors_pre_prj"]])
    logits = head(norm(last.to(head.weight.dtype)))
    topk = logits.topk(TOPK, dim=-1).indices.cpu().tolist()
    reads2 = [{as_int(tok.decode([t])) for t in row} for row in topk]
    reads2 = [{v for v in s if v is not None} for s in reads2]
    # a step's read is "causal" if at least one path result that was readable before is
    # NO LONGER read after the operand is corrupted
    path_results = {round(float(r), 3) for (_, _, _, r) in path}
    before = set().union(*reads) if reads else set()
    after = set().union(*reads2) if reads2 else set()
    moved = {r for r in path_results if (int(r) in before) and (int(r) not in after)}
    return len(moved) > 0


def _report(rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def rate(sub, key):
        s = [r for r in rows if r[key] is not None and r["correct"] == sub]
        return (np.mean([bool(r[key]) for r in s]) if s else float("nan")), len(s)
    print(f"\n[forward-chain] n={len(rows)}  (correct preds: {sum(r['correct'] for r in rows)})")
    out = {}
    for sub, name in [(True, "correct"), (False, "incorrect")]:
        line = {}
        for key in ("arith", "read", "verified"):
            r, n = rate(sub, key)
            line[key] = dict(rate=None if np.isnan(r) else round(float(r), 3), n=n)
        out[name] = line
        print(f"  {name:9s}: arith_reachable={line['arith']['rate']} (n={line['arith']['n']})  "
              f"read_decodable={line['read']['rate']} (n={line['read']['n']})  "
              f"verified={line['verified']['rate']} (n={line['verified']['n']})")
    json.dump(dict(summary=out, rows=rows), open(os.path.join(OUT_DIR, "forward_chain.json"), "w"))

    cats = ["arith\nreachable", "read\ndecodable", "verified\ndecodable"]
    cor = [out["correct"][k]["rate"] or 0 for k in ("arith", "read", "verified")]
    inc = [out["incorrect"][k]["rate"] or 0 for k in ("arith", "read", "verified")]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, cor, w, label="correct predictions")
    ax.bar(x + w / 2, inc, w, label="incorrect predictions")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("fraction of problems"); ax.set_ylim(0, 1)
    ax.set_title("Paper-2 forward-chaining trace decodability (CODI Qwen3-8B)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "p2_forward_chain.png")
    plt.savefig(png, dpi=150, bbox_inches="tight"); print(f"PNG: {png}")


if __name__ == "__main__":
    main()
