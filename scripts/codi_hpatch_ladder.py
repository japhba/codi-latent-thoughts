"""h-stream cross-patch ladder on the robust doubly-load-bearing core (n=44) plus
hop-matched correct non-LB controls (n=44), GSM8k test, CODI ckpt-7580.

Analog of the z-level swap battery at the residual level: during the latent rollout,
replace h^24 (hidden_states[24] = output of decoder block 23, the nz1/fd1 capture
convention) at thought positions with a target vector, then let the chain REGENERATE
downstream (a patched h24 changes z_t and everything after it). Conditions:
  identity   : plain regen rollout, no hook (gate: must reproduce battery identity preds)
  repatch    : patch OWN recorded h24 at all 6 positions (harness gate)
  donor      : hop-matched donor row's h24 at all 6 positions (the cross-patch)
  noise      : own h24 + random direction scaled per position to that row's donor
               displacement (distribution-kick control at matched magnitude)
  selfpos    : own h24s rotated by one position (phase violation)
  donor_p{k} : donor h24 at position k only, k=1..6
Scores per condition: hit-gold, keep (== identity pred), follows-donor-pred,
follows-donor-gold.
-> reports/codi_writeup/hpatch_ladder.json

srun -p general --qos=high --gres=gpu:1 --mem=128G --time=2:00:00 --job-name=codi_hpatch_jb \
  bash -lc 'set -a; source .env; set +a; HF_HOME=/workspace-vast/$USER/hf_cache \
  /var/tmp/jbauer/venvs/activation_oracles_dev/bin/python scripts/codi_hpatch_ladder.py'
"""
from __future__ import annotations

import functools
import json
import math
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
H_IDX = 24            # hidden_states index (0 = embeddings) -> block 23 output
SEED = 11
DEV = "cuda"
OUT_DIR = os.path.join(_REPO, "reports", "codi_writeup")
BS = 44


def bucket_of(h):
    if h < 2: return "<2"
    return "6+" if h >= 6 else str(h)


def extract_number(sentence):
    sentence = sentence.replace(",", "")
    pred = re.findall(r"-?\d+\.?\d*", sentence)
    return float(pred[-1]) if pred else float("inf")


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


@torch.no_grad()
def record(model, tok, input_ids, attention_mask):
    """Latent rollout recording the fed latents z_1..z_6 AND h^24 at thought
    positions t1..t6 (the forward where z_k is the input)."""
    bsz = input_ids.size(0)
    bot = torch.full((bsz, 1), tok.bot_id, dtype=torch.long, device=DEV)
    ids_bot = torch.cat((input_ids, bot), dim=1)
    mask_bot = torch.cat((attention_mask, torch.ones_like(bot)), dim=1)
    out = model.codi(input_ids=ids_bot, use_cache=True, output_hidden_states=True,
                     attention_mask=mask_bot)
    past = out.past_key_values
    hidden = out.hidden_states[-1][:, -1:, :]
    latents, h24s = [], []
    for _ in range(NL):
        l_in = model.prj(hidden).to(model.codi.dtype) if model.use_prj else hidden
        latents.append(l_in)
        out = model.codi(inputs_embeds=l_in, use_cache=True, output_hidden_states=True,
                         past_key_values=past)
        past = out.past_key_values
        h24s.append(out.hidden_states[H_IDX][:, -1])
        hidden = out.hidden_states[-1][:, -1:, :]
    return torch.cat(latents, dim=1), torch.stack(h24s, dim=1)   # (B,NL,D), (B,NL,D)


STATE: dict = {}


def _hook(mod, inp, out):
    if "vec" in STATE:
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1] = STATE["vec"].to(h.dtype)
    return out


@torch.no_grad()
def patched_rollout(model, tok, input_ids, attention_mask, targets, max_new_tokens=64):
    """Full latent rollout + greedy answer decode. targets: (B,NL,D) tensor or None per
    position via a (B,NL,D) tensor with a NL-length bool mask `active`; here targets is
    either None (identity) or a tuple (tgt, active) with tgt (B,NL,D)."""
    bsz = input_ids.size(0)
    bot = torch.full((bsz, 1), tok.bot_id, dtype=torch.long, device=DEV)
    ids_bot = torch.cat((input_ids, bot), dim=1)
    mask_bot = torch.cat((attention_mask, torch.ones_like(bot)), dim=1)
    out = model.codi(input_ids=ids_bot, use_cache=True, output_hidden_states=True,
                     attention_mask=mask_bot)
    past = out.past_key_values
    hidden = out.hidden_states[-1][:, -1:, :]
    for t in range(NL):
        l_in = model.prj(hidden).to(model.codi.dtype) if model.use_prj else hidden
        if targets is not None and targets[1][t]:
            STATE["vec"] = targets[0][:, t]
        out = model.codi(inputs_embeds=l_in, use_cache=True, output_hidden_states=True,
                         past_key_values=past)
        STATE.pop("vec", None)
        past = out.past_key_values
        hidden = out.hidden_states[-1][:, -1:, :]
    # greedy answer tail
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
                tk = nxt[b].item()
                if tk == tok.eos_token_id:
                    finished[b] = True
                else:
                    toks[b].append(tk)
        if finished.all():
            break
        cur = embed(nxt.unsqueeze(1))
    return [tok.decode(t, skip_special_tokens=False) for t in toks]


def main():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)

    inst = json.load(open(os.path.join(OUT_DIR, "controls_instances.json")))
    core = json.load(open(os.path.join(OUT_DIR, "robust_core_idx.json")))
    core = sorted(core)

    def correct(rec, cond):
        p = rec["pred"][cond]
        return p is not None and close(float(p), float(rec["gold"]))

    # hop-matched controls: identity-correct AND trunc_0-correct (not CODI-necessary)
    core_set = set(core)
    cands = {}
    for rec in inst:
        if rec["i"] in core_set: continue
        if correct(rec, "identity") and correct(rec, "trunc_0"):
            cands.setdefault(rec["bucket"], []).append(rec["i"])
    ctl = []
    for b in [inst[i]["bucket"] for i in core]:
        pool = [c for c in cands.get(b, []) if c not in ctl]
        if not pool:
            pool = [c for cs in cands.values() for c in cs if c not in ctl]
        ctl.append(pool[rng.randint(len(pool))])
    rows = core + ctl
    group = ["lb"] * len(core) + ["ctl"] * len(ctl)
    n = len(rows)
    print(f"[hpatch] core={len(core)} ctl={len(ctl)} buckets core: " +
          " ".join(f"{b}:{sum(inst[i]['bucket'] == b for i in core)}"
                   for b in ['<2', '2', '3', '4', '5', '6+']))

    # donor map: hop-bucket cyclic derangement over the combined set
    donor = list(range(n))
    by_bucket = {}
    for j, i in enumerate(rows):
        by_bucket.setdefault(inst[i]["bucket"], []).append(j)
    for idxs in by_bucket.values():
        if len(idxs) < 2:
            for j in idxs: donor[j] = (j + 1) % n
        else:
            for a, b in zip(idxs, idxs[1:] + idxs[:1]): donor[a] = b

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

    codi_inner = model.codi.get_base_model() if hasattr(model.codi, "get_base_model") else model.codi
    handle = codi_inner.model.layers[H_IDX - 1].register_forward_hook(_hook)

    ds = load_dataset("gsm8k", "main")["test"]
    questions = [f"{ds[i]['question'].strip().replace('  ', ' ')} Output only the answer "
                 f"and nothing else." for i in rows]
    golds = [float(ds[i]["answer"].split("####")[-1].replace(",", "")) for i in rows]

    D = codi_inner.config.hidden_size
    h24_all = torch.empty(n, NL, D, dtype=torch.bfloat16)
    batches = [list(range(s, min(s + BS, n))) for s in range(0, n, BS)]
    for idx in batches:
        batch = tok([questions[j] for j in idx], return_tensors="pt", padding="longest",
                    padding_side="left")
        _, h24 = record(model, tok, batch["input_ids"].to(DEV),
                        batch["attention_mask"].to(DEV))
        h24_all[idx] = h24.cpu()

    disp = (h24_all[donor].float() - h24_all.float()).norm(dim=-1)   # (n, NL)
    noise_dir = torch.randn(n, NL, D, generator=torch.Generator().manual_seed(SEED))
    noise_dir = noise_dir / noise_dir.norm(dim=-1, keepdim=True)
    noise_tgt = (h24_all.float() + disp[..., None] * noise_dir).to(torch.bfloat16)
    selfpos_tgt = h24_all[:, [(t + 1) % NL for t in range(NL)]]

    ALL = [True] * NL
    CONDS = {"identity": None, "repatch": (h24_all, ALL), "donor": (h24_all[donor], ALL),
             "noise": (noise_tgt, ALL), "selfpos": (selfpos_tgt, ALL)}
    for k in range(NL):
        CONDS[f"donor_p{k + 1}"] = (h24_all[donor], [t == k for t in range(NL)])

    preds, texts = {c: [None] * n for c in CONDS}, {c: [None] * n for c in CONDS}
    from tqdm.auto import tqdm
    for cond, tgt in tqdm(CONDS.items(), desc="conditions"):
        for idx in batches:
            batch = tok([questions[j] for j in idx], return_tensors="pt", padding="longest",
                        padding_side="left")
            targets = None if tgt is None else (tgt[0][idx].to(DEV), tgt[1])
            outs = patched_rollout(model, tok, batch["input_ids"].to(DEV),
                                   batch["attention_mask"].to(DEV), targets)
            for j, txt in zip(idx, outs):
                texts[cond][j] = txt
                preds[cond][j] = extract_number(txt)
    handle.remove()

    # gate: repatch must reproduce identity (allow a few bf16 flips)
    flips = sum(not close(preds["repatch"][j], preds["identity"][j]) for j in range(n))
    print(f"[gate] repatch vs identity flips: {flips}/{n}")
    assert flips <= max(2, n // 20), "repatch harness gate failed"

    records, summary = [], {}
    for j in range(n):
        rec = dict(i=rows[j], group=group[j], hops=inst[rows[j]]["hops"], gold=golds[j],
                   donor_i=rows[donor[j]], donor_gold=golds[donor[j]],
                   donor_identity_pred=preds["identity"][donor[j]],
                   conds={c: dict(pred=preds[c][j], text=texts[c][j]) for c in CONDS})
        records.append(rec)
    for g in ("lb", "ctl"):
        js = [j for j in range(n) if group[j] == g]
        for c in CONDS:
            summary[f"{g}.{c}"] = dict(
                n=len(js),
                gold=float(np.mean([close(preds[c][j], golds[j]) for j in js])),
                keep=float(np.mean([close(preds[c][j], preds["identity"][j]) for j in js])),
                donor_pred=float(np.mean([close(preds[c][j], preds["identity"][donor[j]])
                                          for j in js])),
                donor_gold=float(np.mean([close(preds[c][j], golds[donor[j]]) for j in js])))
            print(f"{g:3s} {c:10s} " + " ".join(f"{k}={v:.3f}" for k, v in
                                                summary[f'{g}.{c}'].items() if k != 'n'))
    json.dump(dict(summary=summary, mean_disp=float(disp.mean()),
                   records=records), open(os.path.join(OUT_DIR, "hpatch_ladder.json"), "w"),
              indent=1)
    print(f"wrote {os.path.join(OUT_DIR, 'hpatch_ladder.json')}")


if __name__ == "__main__":
    main()
