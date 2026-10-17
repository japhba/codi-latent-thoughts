"""Extended latent-CoT controls battery on the CODI Qwen3-8B organism (ckpt-7580), GSM8k test.

Extends third_party/CODI/experiments/{ablate_latents,swap_latents}.py (the model-card
battery: truncation / single-drop / shuffle / neighbor-transplant, aggregates only in
codi_eval_ckpt/eval_logs/{ablation,swap}_7580.json) with
  - dropfirst_k : remove the FIRST k thoughts, keep the last 6-k in order (the card only
                  had truncation = remove the LAST 6-k)
  - reverse, repeat_z1, mean_lat (per-position dataset mean), gauss (per-position
    mean+std-matched Gaussian) — content-free reference traces
  - swap_hopmatch : donor = a different problem from the SAME hop bucket (cyclic
    derangement within bucket) — "problem-instance patching within task family"
  - propagated corruptions noise1_regen / noise3_regen / donor3_regen : replace z_k and
    REGENERATE the downstream latents before decoding (the card ablations froze the
    trace; these ask whether the latent chain error-corrects)
and per-instance records (question, gold, hops, per-latent final-layer logit-lens top-5,
prediction under every condition, donor ids) for the interactive report viewer.

The decode path is kept byte-faithful to experiments/ablate_latents.py (bocot appended
directly after the question WITHOUT eos; greedy; logits[:, :vocab-1]; LAST-number
extraction; float compare) so `identity` should reproduce the card's 52.01%.

Hops for GSM8k test = number of <<...>> calculator annotations in the official solution.

srun -p general --qos=high --gres=gpu:1 --mem=128G --time=4:00:00 --job-name=codi_wr_$USER \
  bash -lc 'set -a; source .env; set +a; HF_HOME=/workspace-vast/$USER/hf_cache \
  /var/tmp/jbauer/venvs/activation_oracles_dev/bin/python scripts/codi_controls_battery.py'
"""
from __future__ import annotations

import argparse
import functools
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
NL = 6
SEED = 11
DEV = "cuda"
OUT_DIR = os.path.join(_REPO, "reports", "codi_writeup")
BUCKETS = ["<2", "2", "3", "4", "5", "6+"]


def bucket_of(h):
    if h < 2: return "<2"
    return "6+" if h >= 6 else str(h)


def extract_number(sentence):
    sentence = sentence.replace(",", "")
    pred = re.findall(r"-?\d+\.?\d*", sentence)
    return float(pred[-1]) if pred else float("inf")


@torch.no_grad()
def record_latents(model, tok, input_ids, attention_mask, topk=5):
    """Mirror of experiments/ablate_latents.record_latents, additionally returning the
    final-layer logit-lens top-k of the hidden state each latent is projected FROM
    (= the forward pass's own logits at that position; zero extra compute)."""
    bsz = input_ids.size(0)
    bot = torch.full((bsz, 1), tok.bot_id, dtype=torch.long, device=DEV)
    ids_bot = torch.cat((input_ids, bot), dim=1)
    mask_bot = torch.cat((attention_mask, torch.ones_like(bot)), dim=1)
    out = model.codi(input_ids=ids_bot, use_cache=True, output_hidden_states=True,
                     attention_mask=mask_bot)
    past = out.past_key_values
    hidden = out.hidden_states[-1][:, -1:, :]
    latents, lens_ids, lens_ps = [], [], []
    for _ in range(NL):
        p = out.logits[:, -1, : model.codi.config.vocab_size - 1].float().softmax(-1)
        tp, ti = p.topk(topk, dim=-1)
        lens_ids.append(ti); lens_ps.append(tp)
        l_in = model.prj(hidden).to(model.codi.dtype) if model.use_prj else hidden
        latents.append(l_in)
        out = model.codi(inputs_embeds=l_in, use_cache=True, output_hidden_states=True,
                         past_key_values=past)
        past = out.past_key_values
        hidden = out.hidden_states[-1][:, -1:, :]
    return (ids_bot, mask_bot, torch.cat(latents, dim=1),
            torch.stack(lens_ids, 1).cpu(), torch.stack(lens_ps, 1).cpu())


@torch.no_grad()
def decode_with_latents(model, tok, ids_bot, mask_bot, latents, max_new_tokens=64):
    """Verbatim experiments/ablate_latents.decode_with_latents."""
    embed = model.get_embd(model.codi, model.model_name)
    bsz = ids_bot.size(0)
    q_embeds = embed(ids_bot)
    if latents is not None and latents.size(1) > 0:
        inputs = torch.cat([q_embeds, latents], dim=1)
        mask = torch.cat([mask_bot, torch.ones(bsz, latents.size(1), dtype=mask_bot.dtype,
                                               device=DEV)], dim=1)
    else:
        inputs, mask = q_embeds, mask_bot
    out = model.codi(inputs_embeds=inputs, attention_mask=mask, use_cache=True)
    past = out.past_key_values
    return _greedy_tail(model, tok, past, bsz, max_new_tokens)


@torch.no_grad()
def regen_decode(model, tok, ids_bot, mask_bot, latents, rpos, replacement, max_new_tokens=64):
    """Feed latents[:, :rpos] as recorded, `replacement` at rpos, then REGENERATE the
    remaining NL-rpos-1 latents autoregressively before decoding the answer."""
    embed = model.get_embd(model.codi, model.model_name)
    bsz = ids_bot.size(0)
    out = model.codi(inputs_embeds=embed(ids_bot), attention_mask=mask_bot, use_cache=True,
                     output_hidden_states=True)
    past = out.past_key_values
    for i in range(NL):
        if i < rpos:
            l_in = latents[:, i:i + 1]
        elif i == rpos:
            l_in = replacement
        else:
            hidden = out.hidden_states[-1][:, -1:, :]
            l_in = model.prj(hidden).to(model.codi.dtype) if model.use_prj else hidden
        out = model.codi(inputs_embeds=l_in, use_cache=True, output_hidden_states=True,
                         past_key_values=past)
        past = out.past_key_values
    return _greedy_tail(model, tok, past, bsz, max_new_tokens)


@torch.no_grad()
def _greedy_tail(model, tok, past, bsz, max_new_tokens):
    embed = model.get_embd(model.codi, model.model_name)
    eot = torch.full((bsz, 1), tok.eot_id, dtype=torch.long, device=DEV)
    cur = embed(eot)
    finished = torch.zeros(bsz, dtype=torch.bool, device=DEV)
    toks = [[] for _ in range(bsz)]
    for _ in range(max_new_tokens):
        out = model.codi(inputs_embeds=cur, use_cache=True, past_key_values=past)
        past = out.past_key_values
        logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]
        nxt = torch.argmax(logits, dim=-1)
        for b in range(bsz):
            if not finished[b]:
                t = nxt[b].item()
                if t == tok.eos_token_id:
                    finished[b] = True
                else:
                    toks[b].append(t)
        if finished.all():
            break
        cur = embed(nxt.unsqueeze(1))
    return [tok.decode(t, skip_special_tokens=False) for t in toks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = full GSM8k test (1319)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    from datasets import load_dataset
    from src.model import CODI
    model = CODI.from_pretrained(checkpoint_path=CKPT, model_name_or_path=BASE, lora_r=128,
                                 lora_alpha=32, num_latent=NL, use_prj=True, prj_dim=4096,
                                 prj_no_ln=False, prj_dropout=0.0, dtype="bfloat16")
    model.eval()
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")
    tok.bot_id = tok.convert_tokens_to_ids("<|bocot|>")
    tok.eot_id = tok.convert_tokens_to_ids("<|eocot|>")

    ds = load_dataset("gsm8k", "main")["test"]
    n = min(args.limit, len(ds)) if args.limit else len(ds)
    questions = [f"{ds[i]['question'].strip().replace('  ', ' ')} Output only the answer "
                 f"and nothing else." for i in range(n)]
    raw_q = [ds[i]["question"] for i in range(n)]
    golds = [float(ds[i]["answer"].split("####")[-1].replace(",", "")) for i in range(n)]
    hops = [len(re.findall(r"<<", ds[i]["answer"])) for i in range(n)]
    print(f"[battery] n={n}  hop buckets: " +
          " ".join(f"{b}:{sum(bucket_of(h) == b for h in hops)}" for b in BUCKETS))

    # donor maps (global indices)
    donor_nb = [(i + 1) % n for i in range(n)]
    donor_hm = list(range(n))
    by_bucket = {}
    for i, h in enumerate(hops):
        by_bucket.setdefault(bucket_of(h), []).append(i)
    for idxs in by_bucket.values():
        if len(idxs) < 2:
            for i in idxs:
                donor_hm[i] = donor_nb[i]
        else:
            for a, b in zip(idxs, idxs[1:] + idxs[:1]):
                donor_hm[a] = b

    D = model.codi.config.hidden_size
    lat_all = torch.empty(n, NL, D, dtype=torch.bfloat16)
    lens_top = [None] * n
    n_batches = (n + args.batch_size - 1) // args.batch_size
    batches = [list(range(s * args.batch_size, min((s + 1) * args.batch_size, n)))
               for s in range(n_batches)]

    from tqdm.auto import tqdm
    for idx in tqdm(batches, desc="phase1 record"):
        batch = tok([questions[i] for i in idx], return_tensors="pt", padding="longest",
                    padding_side="left")
        _, _, lat, li, lp = record_latents(model, tok, batch["input_ids"].to(DEV),
                                           batch["attention_mask"].to(DEV))
        lat_all[idx] = lat.cpu()
        for j, gi in enumerate(idx):
            lens_top[gi] = [[(tok.decode([int(li[j, k, r])]), round(float(lp[j, k, r]), 4))
                             for r in range(li.shape[2])] for k in range(NL)]

    mean_lat = lat_all.float().mean(0)                     # (NL, D)
    std_lat = lat_all.float().std(0)

    gen = torch.Generator().manual_seed(SEED)
    def gauss_like(rows):
        z = torch.randn(len(rows), NL, D, generator=gen)
        return (mean_lat + std_lat * z).to(torch.bfloat16)

    shuffles = {}
    for i in range(n):
        rng = np.random.RandomState(SEED * 100003 + i)
        p = rng.permutation(NL)
        while (p == np.arange(NL)).all():
            p = rng.permutation(NL)
        shuffles[i] = p

    CONDS = (["identity"]
             + [f"trunc_{k}" for k in range(NL)]
             + [f"drop_{k + 1}" for k in range(NL)]
             + [f"dropfirst_{k}" for k in range(1, NL)]
             + ["shuffle", "reverse", "swap_neighbor", "swap_hopmatch",
                "gauss", "mean_lat", "repeat_z1",
                "noise1_regen", "noise3_regen", "donor3_regen"])
    preds = {c: [None] * n for c in CONDS}

    for idx in tqdm(batches, desc="phase2 decode"):
        batch = tok([questions[i] for i in idx], return_tensors="pt", padding="longest",
                    padding_side="left")
        input_ids = batch["input_ids"].to(DEV)
        attention_mask = batch["attention_mask"].to(DEV)
        bsz = len(idx)
        bot = torch.full((bsz, 1), tok.bot_id, dtype=torch.long, device=DEV)
        ids_bot = torch.cat((input_ids, bot), dim=1)
        mask_bot = torch.cat((attention_mask, torch.ones_like(bot)), dim=1)
        lat = lat_all[idx].to(DEV)
        gauss_b = gauss_like(idx).to(DEV)

        for cond in CONDS:
            if cond == "identity":
                cl = lat
            elif cond.startswith("trunc_"):
                cl = lat[:, : int(cond.split("_")[1])]
            elif cond.startswith("dropfirst_"):
                cl = lat[:, int(cond.split("_")[1]):]
            elif cond.startswith("drop_"):
                k = int(cond.split("_")[1]) - 1
                cl = lat[:, [j for j in range(NL) if j != k]]
            elif cond == "shuffle":
                perm = torch.tensor(np.stack([shuffles[i] for i in idx]), device=DEV)
                cl = lat[torch.arange(bsz, device=DEV)[:, None], perm]
            elif cond == "reverse":
                cl = lat.flip(1)
            elif cond == "swap_neighbor":
                cl = lat_all[[donor_nb[i] for i in idx]].to(DEV)
            elif cond == "swap_hopmatch":
                cl = lat_all[[donor_hm[i] for i in idx]].to(DEV)
            elif cond == "gauss":
                cl = gauss_b
            elif cond == "mean_lat":
                cl = mean_lat.to(torch.bfloat16).to(DEV).unsqueeze(0).expand(bsz, -1, -1)
            elif cond == "repeat_z1":
                cl = lat[:, :1].expand(-1, NL, -1)
            elif cond.endswith("_regen"):
                if cond.startswith("noise"):
                    rpos = int(cond[5]) - 1
                    rep = gauss_b[:, rpos:rpos + 1]
                else:                                     # donor3_regen
                    rpos = 2
                    rep = lat_all[[donor_hm[i] for i in idx]][:, 2:3].to(DEV)
                out = regen_decode(model, tok, ids_bot, mask_bot, lat, rpos, rep,
                                   args.max_new)
                for j, gi in enumerate(idx):
                    preds[cond][gi] = out[j]
                continue
            out = decode_with_latents(model, tok, ids_bot, mask_bot, cl, args.max_new)
            for j, gi in enumerate(idx):
                preds[cond][gi] = out[j]

        done = idx[-1] + 1
        key = ["identity", "trunc_0", "shuffle", "swap_hopmatch", "gauss", "noise1_regen"]
        acc = {c: np.mean([extract_number(preds[c][i]) == golds[i] for i in range(done)])
               for c in key}
        print("  running acc: " + "  ".join(f"{c}={100 * acc[c]:.1f}" for c in key))

    # ---- aggregate + per-instance dumps ----
    nums = {c: [extract_number(p) for p in preds[c]] for c in CONDS}
    own = {c: [nums[c][i] == golds[i] for i in range(n)] for c in CONDS}
    donor_gold = {"swap_neighbor": [golds[donor_nb[i]] for i in range(n)],
                  "swap_hopmatch": [golds[donor_hm[i]] for i in range(n)],
                  "donor3_regen": [golds[donor_hm[i]] for i in range(n)]}
    summary = {"n": n, "ckpt": CKPT, "seed": SEED, "conditions": CONDS,
               "accuracy": {c: float(np.mean(own[c])) for c in CONDS},
               "accuracy_by_bucket": {c: {b: float(np.mean([own[c][i] for i in by_bucket[b]]))
                                          for b in BUCKETS if b in by_bucket}
                                      for c in CONDS},
               "bucket_counts": {b: len(v) for b, v in by_bucket.items()},
               "donor_rate": {c: float(np.mean([nums[c][i] == dg[i] for i in range(n)]))
                              for c, dg in donor_gold.items()},
               "donor_rate_identity": {c: float(np.mean([nums["identity"][i] == dg[i]
                                                         for i in range(n)]))
                                       for c, dg in donor_gold.items()}}
    with open(os.path.join(OUT_DIR, "controls_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    def fmt_num(x):
        return None if x == float("inf") else (int(x) if float(x).is_integer() else x)

    instances = []
    for i in range(n):
        instances.append({
            "i": i, "question": raw_q[i], "gold": fmt_num(golds[i]), "hops": hops[i],
            "bucket": bucket_of(hops[i]), "donor_nb": donor_nb[i], "donor_hm": donor_hm[i],
            "lens": lens_top[i],
            "pred": {c: fmt_num(nums[c][i]) for c in CONDS},
            "raw": {c: preds[c][i][:60] for c in CONDS},
        })
    with open(os.path.join(OUT_DIR, "controls_instances.json"), "w") as f:
        json.dump(instances, f)

    print(f"\n[battery] n={n} accuracies:")
    for c in CONDS:
        d = summary["donor_rate"].get(c)
        print(f"  {c:>14}: {100 * summary['accuracy'][c]:5.2f}%"
              + (f"  donor={100 * d:.2f}%" if d is not None else ""))
    print("BATTERY-DONE")


if __name__ == "__main__":
    main()
