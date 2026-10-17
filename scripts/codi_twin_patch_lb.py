"""Twin cross-patch on LOAD-BEARING pairs: mined value-edit twins of trace-necessary
GSM8k test rows (codi_lb_twins_mine.py -> LLM-verified lb_twins.jsonl), CODI
ckpt-7580. Unlike the val-set campaign, every kept pair here has BOTH sides
unsolvable without the latent thoughts (noCoT decode fails), so the patched
thoughts are load-bearing for the receiving rollout.

Stage 1 (chunks of 48): gate candidates on orig pred == gold, twin pred == cf_twin,
and noCoT WRONG on both sides; then keep <=2 per doc, cap 64 pairs.
Stage 2 (one batch, re-record loop for RoPE/batch consistency): record h+z on both
sides, run conditions on the TWIN question's rollout:
  identity          : no patch
  repatchz_p3       : twin's own z at position 3 (z harness gate)
  repatchcol_p3     : twin's own column+z at position 3 (column gate)
  revz_all6/p{1..6} : original z_o patched at the marked positions
  revcol_all6/p{..} : original column {z_i} u {h^(l)_i}_(l=1..36) patched
Scores: keep (pred == cf_twin, the twin's own answer), adopt (pred == gold, the
transplanted original's answer).
-> reports/codi_writeup/repo/artifacts/lbtwin_patch.json

srun -p general --qos=high --gres=gpu:1 --mem=128G --time=2:30:00 --job-name=codi_lbt_jb \
  bash scripts/slurm/run_codi_lbtwin.sh
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
PAIRS_F = os.path.join(_REPO, "reports", "codi_writeup", "lb_twins.jsonl")
NL = 6
NH = 37
SEED = 11
DEV = "cuda"
CHUNK = 48
MAX_PER_DOC = 2
CAP = 64
OUT = os.path.join(_REPO, "reports", "codi_writeup", "repo", "artifacts", "lbtwin_patch.json")


def extract_number(sentence):
    sentence = sentence.replace(",", "")
    pred = re.findall(r"-?\d+\.?\d*", sentence)
    return float(pred[-1]) if pred else float("inf")


def close(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    except (ValueError, OverflowError):
        return False


STATE: dict = {}


def make_hook(l):
    def hook(mod, inp, out):
        if "col" in STATE:
            h = out[0] if isinstance(out, tuple) else out
            h[:, -1] = STATE["col"][:, l + 1].to(h.dtype)
        return out
    return hook


@torch.no_grad()
def rollout(model, tok, questions, record=False, tgt=None, nocot=False, max_new_tokens=64):
    """Batched latent rollout + greedy decode. record -> also return h[B,NL,NH,D]
    and the fed thoughts z[B,NL,D]. tgt = dict(kind='col'|'z', vec, active,
    zvec=None): col replaces the full column (vec [B,NL,NH,D]) and, via zvec, the
    fed thought; kind 'z' replaces only the fed thought (vec [B,NL,D]).
    nocot -> skip the latent loop entirely (single-pass decode)."""
    assert not (record and nocot)
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
    hs, zs = [], []
    for t in range(0 if nocot else NL):
        l_in = model.prj(hidden).to(model.codi.dtype) if model.use_prj else hidden
        if tgt is not None and tgt["active"][t]:
            if tgt["kind"] == "z":
                l_in = tgt["vec"][:, t][:, None].to(DEV).to(model.codi.dtype)
            else:
                l_in = tgt["zvec"][:, t][:, None].to(DEV).to(model.codi.dtype)
                STATE["col"] = tgt["vec"][:, t].to(DEV)
        out = model.codi(inputs_embeds=l_in, use_cache=True, output_hidden_states=True,
                         past_key_values=past)
        STATE.clear()
        past = out.past_key_values
        if record:
            hs.append(torch.stack([out.hidden_states[k][:, -1] for k in range(NH)], 1))
            zs.append(l_in[:, 0])
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
        return texts, torch.stack(hs, 1).cpu(), torch.stack(zs, 1).cpu()
    return texts


SUF = " Output only the answer and nothing else."


def norm_q(q):
    return q.strip().replace("  ", " ") + SUF


def main():
    torch.manual_seed(SEED)
    cands = [json.loads(l) for l in open(PAIRS_F)]
    print(f"[lbtwin] verified candidates: {len(cands)}")

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
    handles = [blk.register_forward_hook(make_hook(l))
               for l, blk in enumerate(codi_inner.model.layers)]

    # ---- stage 1: chunked gating ----
    from tqdm.auto import tqdm
    ok = []
    stats = dict(orig=0, twin=0, nocot_o_fail=0, nocot_t_fail=0)
    for s0 in tqdm(range(0, len(cands), CHUNK), desc="gate chunks"):
        ch = cands[s0:s0 + CHUNK]
        qo = [norm_q(c["question"]) for c in ch]
        qt = [norm_q(c["twin_question"]) for c in ch]
        po = [extract_number(x) for x in rollout(model, tok, qo)]
        pt = [extract_number(x) for x in rollout(model, tok, qt)]
        no = [extract_number(x) for x in rollout(model, tok, qo, nocot=True)]
        nt = [extract_number(x) for x in rollout(model, tok, qt, nocot=True)]
        for j, c in enumerate(ch):
            a = close(po[j], c["gold"])
            b = close(pt[j], c["cf_twin"])
            fo = not close(no[j], c["gold"])
            ft = not close(nt[j], c["cf_twin"])
            stats["orig"] += a; stats["twin"] += b
            stats["nocot_o_fail"] += fo; stats["nocot_t_fail"] += ft
            if a and b and fo and ft:
                ok.append(c)
    print(f"[gate1] of {len(cands)}: {stats}; all four criteria: {len(ok)}")

    per_doc = {}
    pairs = []
    for c in ok:
        if per_doc.get(c["doc"], 0) >= MAX_PER_DOC or len(pairs) >= CAP:
            continue
        per_doc[c["doc"]] = per_doc.get(c["doc"], 0) + 1
        pairs.append(c)
    print(f"[gate1] selected {len(pairs)} pairs across {len(per_doc)} docs")
    assert len(pairs) >= 8, "too few load-bearing pairs survived gating"

    # ---- stage 2: single-batch re-record loop ----
    for rnd in range(4):
        q_orig = [norm_q(p["question"]) for p in pairs]
        q_twin = [norm_q(p["twin_question"]) for p in pairs]
        texts_o, h_o, z_o = rollout(model, tok, q_orig, record=True)
        texts_t, h_t, z_t = rollout(model, tok, q_twin, record=True)
        no = [extract_number(x) for x in rollout(model, tok, q_orig, nocot=True)]
        nt = [extract_number(x) for x in rollout(model, tok, q_twin, nocot=True)]
        keep = [j for j, p in enumerate(pairs)
                if close(extract_number(texts_o[j]), p["gold"])
                and close(extract_number(texts_t[j]), p["cf_twin"])
                and not close(no[j], p["gold"]) and not close(nt[j], p["cf_twin"])]
        if len(keep) == len(pairs):
            break
        print(f"[lbtwin] re-record dropped {len(pairs) - len(keep)} borderline rows")
        pairs = [pairs[j] for j in keep]
    else:
        raise RuntimeError("re-record did not stabilize in 4 rounds")
    n = len(pairs)
    print(f"[lbtwin] final pairs: {n} across {len({p['doc'] for p in pairs})} docs")

    ALL = [True] * NL
    P3 = [t == 2 for t in range(NL)]
    CONDS = {"identity": None,
             "repatchz_p3": dict(kind="z", vec=z_t, active=P3),
             "repatchcol_p3": dict(kind="col", vec=h_t, zvec=z_t, active=P3),
             "revz_all6": dict(kind="z", vec=z_o, active=ALL),
             "revcol_all6": dict(kind="col", vec=h_o, zvec=z_o, active=ALL)}
    for k in range(NL):
        act = [t == k for t in range(NL)]
        CONDS[f"revz_p{k + 1}"] = dict(kind="z", vec=z_o, active=act)
        CONDS[f"revcol_p{k + 1}"] = dict(kind="col", vec=h_o, zvec=z_o, active=act)

    preds, texts = {}, {}
    for cond, tgt in tqdm(CONDS.items(), desc="conditions"):
        outs = rollout(model, tok, q_twin, tgt=tgt)
        texts[cond] = outs
        preds[cond] = [extract_number(x) for x in outs]
    for h in handles:
        h.remove()

    for gate in ("repatchz_p3", "repatchcol_p3"):
        flips = sum(not close(preds[gate][j], preds["identity"][j]) for j in range(n))
        print(f"[gate] {gate} vs identity flips: {flips}/{n}")
        assert flips <= max(2, n // 10), f"{gate} harness gate failed"

    summary, records = {}, []
    for cond in CONDS:
        summary[cond] = dict(
            n=n,
            keep=float(np.mean([close(preds[cond][j], pairs[j]["cf_twin"]) for j in range(n)])),
            adopt=float(np.mean([close(preds[cond][j], pairs[j]["gold"]) for j in range(n)])))
        print(f"{cond:14s} keep={summary[cond]['keep']:.3f} adopt={summary[cond]['adopt']:.3f}")
    for j, p in enumerate(pairs):
        records.append(dict(doc=p["doc"], i=p["i"], gold=p["gold"], cf_twin=p["cf_twin"],
                            g=p["g"], g_twin=p["g_twin"],
                            conds={c: dict(pred=preds[c][j], text=texts[c][j])
                                   for c in CONDS}))
    json.dump(dict(summary=summary, records=records), open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    print("CODI-LBTWIN-DONE")


if __name__ == "__main__":
    main()
