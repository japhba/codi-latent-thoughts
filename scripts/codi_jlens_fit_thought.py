"""Fit per-layer J-Lens transports ON CODI ROLLOUTS, including the fed-vector row.

Same estimator as third_party/jacobian-lens (the paper reduction): per output dim a
one-hot cotangent at every valid target position of h_tgt = block-35 output (pre final
norm), one backward, mean gradient over the valid source positions, averaged over
problems. Differences vs the WikiText qwen8b_lens_shard* fit:
  model     : the CODI checkpoint (LoRA r=128 active in-graph), not base Qwen3-8B
  graph     : teacher-forced replay of the CODI rollout — token embeddings of
              [question, eos, bocot] plus the 6 recorded fed vectors z as one
              inputs_embeds leaf (feedback z=prj(h36) is NOT in the graph, exactly
              as teacher-forced text replay holds later tokens fixed)
  positions : valid = the 7 displayed figure columns (Enc = bocot position, t1..t6)
  sources   : block outputs 0..34 (key l maps hidden_states[l+1], the shard
              convention) PLUS the inputs_embeds leaf: J_emb ~= E[dh36/dz], the same
              estimator applied one row down to the fed vector, averaged over the 6
              thought-input positions only (z is only fed there)
Fit problems: GSM8k-Aug-NL train ds_index >= 5200 (the figure substrate extraction
covers 4000..5199, so the fit set is disjoint), nonempty chain, CODI-correct rollout.
Per-problem sanity: teacher-forced final state must match the cached rollout.

Output: concept_probes/out/saeprobes/jlens/codi_thought_lens.pt
  {"J": {l: [4096,4096] fp32}, "J_emb": [4096,4096] fp32, "meta": {...}}

srun -p general --qos=high --gres=gpu:1 --mem=128G --time=4:00:00 --job-name=codi_jfit_jb \
  bash scripts/slurm/run_codi_jfit.sh
"""
from __future__ import annotations

import functools
import json
import math
import os
import re
import sys
import time

import torch
from tqdm.auto import tqdm

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "third_party", "CODI"))
os.environ.setdefault("HF_HOME", f"/workspace-vast/{os.environ.get('USER', 'jbauer')}/hf_cache")
print = functools.partial(print, flush=True)

CKPT = "/workspace-vast/jbauer/codi_eval_ckpt/checkpoint-7580"
BASE = "Qwen/Qwen3-8B"
NL, D, NBLK = 6, 4096, 36
DEV = "cuda"
N_FIT = int(os.environ.get("JFIT_N", 96))
OFFSET = int(os.environ.get("JFIT_OFFSET", 5200))
DIM_BATCH = int(os.environ.get("JFIT_DIM_BATCH", 16))
OUT_DIR = os.path.join(_REPO, "concept_probes", "out", "saeprobes", "jlens")
OUT = os.path.join(OUT_DIR, "codi_thought_lens.pt")
FIT_CKPT = os.path.join(OUT_DIR, "codi_thought_fit_ckpt.pt")


def parse_numbers(text):
    out, seen = [], set()
    for s in re.findall(r"-?\d+(?:\.\d+)?", text):
        v = (s.rstrip("0").rstrip(".") or "0") if "." in s else s
        if v not in seen:
            seen.add(v); out.append(v)
    return out


def atomic_save(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def roll(model, tok, embed, ids_bot, eocot, max_new_tokens=8):
    """Cached rollout: fed z's [NL, D], greedy answer text, final post-norm state."""
    out = model.codi(input_ids=ids_bot, use_cache=True, output_hidden_states=True)
    past = out.past_key_values
    hidden = out.hidden_states[-1][:, -1:, :]
    zs = []
    for _ in range(NL):
        z = model.prj(hidden).to(model.codi.dtype)
        zs.append(z[:, 0])
        out = model.codi(inputs_embeds=z, use_cache=True, output_hidden_states=True,
                         past_key_values=past)
        past = out.past_key_values
        hidden = out.hidden_states[-1][:, -1:, :]
    final_postnorm = hidden[0, 0].clone()
    cur = embed(torch.tensor([[eocot]], device=DEV))
    toks = []
    for _ in range(max_new_tokens):
        out = model.codi(inputs_embeds=cur, use_cache=True, past_key_values=past)
        past = out.past_key_values
        nxt = int(out.logits[0, -1, : model.codi.config.vocab_size - 1].argmax())
        if nxt == tok.eos_token_id:
            break
        toks.append(nxt)
        cur = embed(torch.tensor([[nxt]], device=DEV))
    return torch.cat(zs, 0), tok.decode(toks, skip_special_tokens=True), final_postnorm


def accumulate_problem(inner, layers, embed, ids_bot, zs, Jsum, Jsum_emb):
    """One teacher-forced forward (replicated DIM_BATCH times) + ceil(D/DIM_BATCH)
    backwards; adds this problem's per-layer Jacobian rows into the running sums."""
    P0 = ids_bot.shape[1]
    valid = torch.arange(P0 - 1, P0 + NL, device=DEV)      # Enc(bocot) + t1..t6
    tpos = torch.arange(P0, P0 + NL, device=DEV)           # thought inputs only
    with torch.no_grad():
        base = torch.cat([embed(ids_bot)[0], zs.to(inner.dtype)], 0)   # [P0+NL, D]
    leaf = base.unsqueeze(0).repeat(DIM_BATCH, 1, 1).requires_grad_(True)
    acts = {}
    hooks = [blk.register_forward_hook(
                 lambda m, i, o, l=l: acts.__setitem__(l, o[0] if isinstance(o, tuple) else o))
             for l, blk in enumerate(layers)]
    try:
        with torch.enable_grad():
            inner.model(inputs_embeds=leaf, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    target = acts[NBLK - 1]
    sources = [leaf] + [acts[l] for l in range(NBLK - 1)]
    brange = torch.arange(DIM_BATCH, device=DEV)
    cot = torch.zeros_like(target)
    n_passes = math.ceil(D / DIM_BATCH)
    for pi, d0 in enumerate(range(0, D, DIM_BATCH)):
        nd = min(DIM_BATCH, D - d0)
        cot.zero_()
        cot[brange[:nd, None], valid[None, :], d0 + brange[:nd, None]] = 1.0
        grads = torch.autograd.grad(target, sources, grad_outputs=cot,
                                    retain_graph=pi < n_passes - 1)
        Jsum_emb[d0:d0 + nd] += grads[0][:nd][:, tpos].float().mean(1).cpu()
        for l in range(NBLK - 1):
            Jsum[l][d0:d0 + nd] += grads[1 + l][:nd][:, valid].float().mean(1).cpu()
    return target[0].detach()


def main():
    from datasets import load_dataset
    from src.model import CODI
    model = CODI.from_pretrained(checkpoint_path=CKPT, model_name_or_path=BASE, lora_r=128,
                                 lora_alpha=32, num_latent=NL, use_prj=True, prj_dim=4096,
                                 prj_no_ln=False, prj_dropout=0.0, dtype="bfloat16")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = model.tokenizer
    bocot = tok.convert_tokens_to_ids("<|bocot|>")
    eocot = tok.convert_tokens_to_ids("<|eocot|>")
    inner = model.codi.get_base_model() if hasattr(model.codi, "get_base_model") else model.codi
    layers = inner.model.layers
    embed = model.get_embd(model.codi, model.model_name)
    assert len(layers) == NBLK and inner.config.hidden_size == D
    ds = load_dataset("zen-E/GSM8k-Aug-NL", split="train")

    Jsum = {l: torch.zeros(D, D) for l in range(NBLK - 1)}
    Jsum_emb = torch.zeros(D, D)
    count, next_di, skipped = 0, OFFSET, 0
    if os.path.exists(FIT_CKPT):
        st = torch.load(FIT_CKPT, map_location="cpu", weights_only=True)
        assert st["offset"] == OFFSET and st["dim_batch"] == DIM_BATCH
        Jsum, Jsum_emb = st["Jsum"], st["Jsum_emb"]
        count, next_di, skipped = st["count"], st["next_di"], st["skipped"]
        print(f"[jfit] resumed at ds_index {next_di}, count {count}")

    pbar = tqdm(total=N_FIT, initial=count, desc="fit problems")
    di = next_di
    while count < N_FIT:
        ex = ds[di]
        di += 1
        gold = str(ex["answer"]).strip()
        chain = [x for x in parse_numbers(ex["cot"]) if x != gold]
        if not chain:
            skipped += 1
            continue
        q = ex["question"].strip().replace("  ", " ") + " Output only the answer and nothing else."
        ids = tok(q, return_tensors="pt").input_ids.to(DEV)
        ids_bot = torch.cat([ids, torch.tensor([[tok.eos_token_id, bocot]], device=DEV)], dim=1)
        zs, gen, final_postnorm = roll(model, tok, embed, ids_bot, eocot)
        pred = parse_numbers(gen)
        if not pred or pred[0] != gold:
            skipped += 1
            continue
        t0 = time.time()
        tgt_row = accumulate_problem(inner, layers, embed, ids_bot, zs, Jsum, Jsum_emb)
        cos = torch.nn.functional.cosine_similarity(
            inner.model.norm(tgt_row[-1:]).float(), final_postnorm[None].float()).item()
        assert cos > 0.98, f"replay mismatch at ds {di - 1}: cos={cos:.4f}"
        count += 1
        pbar.update(1)
        print(f"[jfit] ds {di - 1} seq={ids_bot.shape[1] + NL} replay-cos={cos:.5f} "
              f"{time.time() - t0:.0f}s (count {count}/{N_FIT}, skipped {skipped})")
        if count % 8 == 0 or count == N_FIT:
            atomic_save({"Jsum": Jsum, "Jsum_emb": Jsum_emb, "count": count, "next_di": di,
                         "skipped": skipped, "offset": OFFSET, "dim_batch": DIM_BATCH}, FIT_CKPT)
    pbar.close()

    J = {l: Jsum[l] / count for l in range(NBLK - 1)}
    J_emb = Jsum_emb / count
    meta = dict(ckpt=CKPT, base=BASE, estimator="paper reduction, target=block-35 output",
                valid_positions="Enc(bocot)+t1..t6; J_emb over t1..t6 only",
                ds="zen-E/GSM8k-Aug-NL train", offset=OFFSET, n_problems=count,
                skipped=skipped, dim_batch=DIM_BATCH, gate="CODI-correct, nonempty chain")
    atomic_save({"J": J, "J_emb": J_emb, "meta": meta}, OUT)
    sq = math.sqrt(D)
    print(f"[jfit] wrote {OUT}; ||J_emb||/sqrt(d)={J_emb.norm().item() / sq:.3f}")

    # diagnostic: how far the CODI-thought fit moved from the WikiText base-model fit
    import glob
    shards = sorted(glob.glob(os.path.join(OUT_DIR, "qwen8b_lens_shard*.pt")))
    if shards:
        acc = {}
        for pth in shards:
            sd = torch.load(pth, map_location="cpu", weights_only=False)
            for l, Jl in sd["J"].items():
                acc[l] = acc.get(l, 0) + Jl.float() / len(shards)
        for l in (0, 8, 16, 23, 26, 32):
            c = torch.nn.functional.cosine_similarity(
                J[l].flatten(), acc[l].flatten(), dim=0).item()
            print(f"[jfit] L{l}: cos(new, wikitext)={c:.3f} "
                  f"||new||/sqrt(d)={J[l].norm().item() / sq:.3f} "
                  f"||wt||/sqrt(d)={acc[l].norm().item() / sq:.3f}")
    print("CODI-JFIT-DONE")


if __name__ == "__main__":
    main()
