"""NLA-mediated twin cross-patch, CODI ckpt-7580: the language-bottleneck version of
codi_zpatch_twin.py, same pairs and per-position grid. The original problem's thought
z_i is verbalized by the NLA reader (existing zlb_gens.jsonl reads), the co-trained
encoder (nz3_plain critic_latest) reconstructs a vector from that text alone, and the
reconstruction (direction x the true ||z_i||, the zlb convention) is patched into the
TWIN question's rollout at the marked positions, downstream regenerating. If the
verbalization carried the thought's causal content, the twin should adopt the
original's answer like the raw z patch does.

Conditions:
  identity        : twin rollout, no patch (sanity: answers cf_twin)
  reconz_own_p3   : ORIGINAL rollout with its own reconstruction at position 3
                    (round-trip fidelity gate, should keep gold)
  nlaz_all6       : twin rollout, reconstructions of the original z at all 6 positions
  nlaz_p{1..6}    : twin rollout, one position
Scores: keep (twin rollout answers cf_twin / gate answers gold), adopt (the
transplanted original answer gold).
-> reports/codi_writeup/repo/artifacts/nlapatch_twin.json

srun -p general --qos=high --gres=gpu:1 --mem=128G --time=1:00:00 --job-name=codi_ntw_jb \
  bash scripts/slurm/run_codi_nlatwin.sh
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
sys.path.insert(0, os.path.join(_REPO, "third_party", "EasyNLA"))
os.environ.setdefault("HF_HOME", f"/workspace-vast/{os.environ.get('USER', 'jbauer')}/hf_cache")
print = functools.partial(print, flush=True)

CKPT = "/workspace-vast/jbauer/codi_eval_ckpt/checkpoint-7580"
BASE = "Qwen/Qwen3-8B"
NZ2 = "/workspace-vast/jbauer/bt1/nz2"
NZ1 = "/workspace-vast/jbauer/bt1/nz1"
CRITIC = f"{NZ2}/runs_mirror/nz3_plain/critic_latest"
NL = 6
SEED = 11
DEV = "cuda"
OUT = os.path.join(_REPO, "reports", "codi_writeup", "repo", "artifacts", "nlapatch_twin.json")


def extract_number(sentence):
    sentence = sentence.replace(",", "")
    pred = re.findall(r"-?\d+\.?\d*", sentence)
    return float(pred[-1]) if pred else float("inf")


def close(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    except (ValueError, OverflowError):
        return False


@torch.no_grad()
def rollout(model, tok, questions, record=False, tgt=None, max_new_tokens=64):
    """Batched latent rollout + greedy decode (verbatim from codi_zpatch_twin.py)."""
    batch = tok(questions, return_tensors="pt", padding="longest", padding_side="left")
    input_ids, attention_mask = batch["input_ids"].to(DEV), batch["attention_mask"].to(DEV)
    bsz = input_ids.size(0)
    bot = torch.full((bsz, 1), tok.bot_id, dtype=torch.long, device=DEV)
    ids_bot = torch.cat((input_ids, bot), dim=1)
    mask_bot = torch.cat((attention_mask, torch.ones_like(bot)), dim=1)
    out = model.codi(input_ids=ids_bot, use_cache=True, output_hidden_states=True,
                     attention_mask=mask_bot)
    past = out.past_key_values
    hidden = out.hidden_states[-1][:, -1:, :]
    zs = []
    for t in range(NL):
        l_in = model.prj(hidden).to(model.codi.dtype) if model.use_prj else hidden
        if tgt is not None and tgt["active"][t]:
            l_in = tgt["vec"][:, t][:, None].to(DEV).to(model.codi.dtype)
        if record:
            zs.append(l_in[:, 0])
        out = model.codi(inputs_embeds=l_in, use_cache=True, output_hidden_states=True,
                         past_key_values=past)
        past = out.past_key_values
        hidden = out.hidden_states[-1][:, -1:, :]
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
    texts = [tok.decode(t, skip_special_tokens=False) for t in toks]
    if record:
        return texts, torch.stack(zs, 1).cpu()        # [B, NL, D]
    return texts


def load_bodies():
    """First parseable <explanation> body per (doc, z position) from zlb_gens."""
    bodies = {}
    for l in open(f"{NZ2}/zlb_gens.jsonl"):
        g = json.loads(l)
        if not g["stratum"].startswith("z_p"):
            continue
        key = (g["doc"], int(g["stratum"].split("z_p")[1]))
        if key in bodies:
            continue
        m = re.search(r"<explanation>(.*?)</explanation>", g["text"], re.S)
        if m and m.group(1).strip():
            bodies[key] = m.group(1).strip()
    return bodies


def main():
    torch.manual_seed(SEED)
    pairs, seen = [], set()
    for fn in ("zlb_twins.jsonl", "zlb_twins_multi.jsonl"):
        for l in open(f"{NZ2}/{fn}"):
            r = json.loads(l)
            key = (r["doc"], r["twin_question"])
            if key in seen: continue
            seen.add(key)
            pairs.append(r)
    print(f"[nla-twin] candidate pairs: {len(pairs)}")
    bodies = load_bodies()
    docs = sorted({p["doc"] for p in pairs})
    assert all((d, k + 1) in bodies for d in docs for k in range(NL)), "missing reads"

    # ---- stage 1: AR-encode the original verbalizations (directions only) ----
    from transformers import AutoTokenizer
    from nla.config import load_nla_config
    from nla.models import NLACriticModel
    from nla.utils.critic import critic_predict

    ntok = AutoTokenizer.from_pretrained(f"{NZ1}/warmstart/av")
    cfg = load_nla_config(f"{NZ1}/data/mix_rl.parquet", ntok)
    print(f"[nla-twin] mse_scale={cfg.mse_scale}")
    critic = NLACriticModel.from_pretrained(CRITIC, torch_dtype=torch.bfloat16).eval().to(DEV)

    def render(template, explanation):
        if "{ctx}" in template:
            return template.format(explanation=explanation, ctx="")
        return template.format(explanation=explanation)

    keys = [(d, k + 1) for d in docs for k in range(NL)]
    dirs = {}
    with torch.no_grad():
        for s0 in range(0, len(keys), 32):
            chunk = keys[s0:s0 + 32]
            texts = [render(cfg.critic_prompt_template, bodies[k]) for k in chunk]
            enc = ntok(texts, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(DEV)
            p = critic_predict(critic, enc["input_ids"], enc["attention_mask"], cfg.mse_scale)
            for k, v in zip(chunk, p.float().cpu().numpy()):
                dirs[k] = v / np.linalg.norm(v)
    del critic
    torch.cuda.empty_cache()
    print(f"[nla-twin] encoded {len(dirs)} reconstructions")

    # ---- stage 2: CODI rollouts, gates, splices ----
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

    SUF = " Output only the answer and nothing else."
    q_orig = [p["question"].strip() + SUF for p in pairs]
    q_twin = [p["twin_question"].strip() + SUF for p in pairs]

    texts_o, z_o = rollout(model, tok, q_orig, record=True)
    texts_t, z_t = rollout(model, tok, q_twin, record=True)
    keep = [i for i, p in enumerate(pairs)
            if close(extract_number(texts_o[i]), p["gold"])
            and close(extract_number(texts_t[i]), p["cf_twin"])]
    print(f"[nla-twin] both-correct pairs: {len(keep)}/{len(pairs)}")
    for _ in range(3):
        pairs = [pairs[i] for i in keep]
        q_orig = [q_orig[i] for i in keep]
        q_twin = [q_twin[i] for i in keep]
        texts_o, z_o = rollout(model, tok, q_orig, record=True)
        texts_t, z_t = rollout(model, tok, q_twin, record=True)
        keep = [j for j, p in enumerate(pairs)
                if close(extract_number(texts_o[j]), p["gold"])
                and close(extract_number(texts_t[j]), p["cf_twin"])]
        if len(keep) == len(pairs):
            break
        print(f"[nla-twin] re-record dropped {len(pairs) - len(keep)} borderline rows")
    else:
        raise RuntimeError("re-record did not stabilize in 3 rounds")
    n = len(pairs)

    # reconstruction of the ORIGINAL's z: direction from text x the true per-pair norm
    zn = z_o.float().norm(dim=-1)                                # [n, NL]
    zhat = torch.stack([torch.stack([
        torch.from_numpy(dirs[(pairs[j]["doc"], t + 1)]) * zn[j, t] for t in range(NL)])
        for j in range(n)])                                      # [n, NL, D]
    cos = torch.nn.functional.cosine_similarity(zhat.float(), z_o.float(), dim=-1)
    print(f"[nla-twin] recon-vs-true z cosine: mean {cos.mean():.3f} min {cos.min():.3f}")

    ALL = [True] * NL
    CONDS = {"identity": (q_twin, None),
             "reconz_own_p3": (q_orig, dict(vec=zhat, active=[t == 2 for t in range(NL)])),
             "nlaz_all6": (q_twin, dict(vec=zhat, active=ALL))}
    for k in range(NL):
        CONDS[f"nlaz_p{k + 1}"] = (q_twin, dict(vec=zhat, active=[t == k for t in range(NL)]))

    preds, texts = {}, {}
    from tqdm.auto import tqdm
    for cond, (qs, tgt) in tqdm(CONDS.items(), desc="conditions"):
        outs = rollout(model, tok, qs, tgt=tgt)
        texts[cond] = outs
        preds[cond] = [extract_number(x) for x in outs]

    gate = float(np.mean([close(preds["reconz_own_p3"][j], pairs[j]["gold"]) for j in range(n)]))
    print(f"[gate] reconz_own_p3 keeps gold: {gate:.3f}")

    summary, records = {}, []
    for cond in CONDS:
        own, tr = ("gold", "cf_twin") if cond == "reconz_own_p3" else ("cf_twin", "gold")
        summary[cond] = dict(
            n=n,
            keep=float(np.mean([close(preds[cond][j], pairs[j][own]) for j in range(n)])),
            adopt=float(np.mean([close(preds[cond][j], pairs[j][tr]) for j in range(n)])))
        print(f"{cond:14s} keep={summary[cond]['keep']:.3f} adopt={summary[cond]['adopt']:.3f}")
    for j, p in enumerate(pairs):
        records.append(dict(doc=p["doc"], gold=p["gold"], cf_twin=p["cf_twin"],
                            recon_cos=[round(float(c), 4) for c in cos[j]],
                            conds={c: dict(pred=preds[c][j], text=texts[c][j]) for c in CONDS}))
    json.dump(dict(summary=summary, records=records), open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    print("CODI-NLATWIN-DONE")


if __name__ == "__main__":
    main()
