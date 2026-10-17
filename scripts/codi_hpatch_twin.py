"""Twin-donor h cross-patch on the zlb twin campaign docs (GSM8k-Aug-NL val rows),
CODI ckpt-7580. Donors are value-only TWIN problems (one given edited, CoT
repropagated), so the counterfactual target cf_twin is exact and the donor is the
closest possible on-manifold foreign computation.

Pairs: zlb_twins.jsonl (17) + zlb_twins_multi.jsonl (30 variants), deduped, gated on
subject correctness both ways (original pred == gold AND twin pred == cf_twin).

FULL column = the fed thought z_i (the l_in input at that latent step) replaced
together with all 36 block outputs, so the position's entire state is the donor's.
Row conditions (h24) leave the fed thought untouched.

Conditions (patch INTO the original question's rollout, downstream thoughts regenerate):
  identity        : no patch
  repatch24       : own h24 at all 6 thought positions (harness gate)
  repatchcol_p3   : own FULL column at position 3 (column gate)
  twin24_all6     : twin h24 at all 6 positions
  twin24_p{1..6}  : twin h24 at one position
  twincol_all6    : twin FULL column at all 6 positions
  twincol_p{1..6} : twin FULL column at one position
  noise24_all6    : own h24 + random direction at the twin displacement norm
Scores: keep (pred == gold), adopt (pred == cf_twin). A noCoT decode (latent loop
skipped) flags per pair whether CODI solves each side without its thoughts;
summary_lb restricts every condition to pairs whose RECEIVING rollout needs them.
-> reports/codi_writeup/repo/artifacts/hpatch_twin.json

srun -p general --qos=high --gres=gpu:1 --mem=128G --time=1:30:00 --job-name=codi_htw_jb \
  bash scripts/slurm/run_codi_htwin.sh
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
NZ2 = "/workspace-vast/jbauer/bt1/nz2"
NL = 6
NH = 37                     # hidden_states entries (0 = embeddings)
SEED = 11
DEV = "cuda"
OUT = os.path.join(_REPO, "reports", "codi_writeup", "repo", "artifacts", "hpatch_twin.json")


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
        elif "row24" in STATE and l == 23:
            h = out[0] if isinstance(out, tuple) else out
            h[:, -1] = STATE["row24"].to(h.dtype)
        return out
    return hook


@torch.no_grad()
def rollout(model, tok, questions, record=False, tgt=None, nocot=False, max_new_tokens=64):
    """Batched latent rollout + greedy decode. record -> also return h[B,NL,NH,D]
    and the fed thoughts z[B,NL,D] (the l_in inputs). tgt = dict(kind='row24'|'col',
    vec=[B,NL,D] or [B,NL,NH,D], active=[NL] bools, zvec=[B,NL,D] or None: replace
    the fed thought at the active positions too). nocot -> skip the latent loop
    entirely (single-pass decode)."""
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
            if tgt.get("zvec") is not None:
                l_in = tgt["zvec"][:, t][:, None].to(DEV).to(model.codi.dtype)
            STATE[tgt["kind"]] = tgt["vec"][:, t].to(DEV)
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
        return texts, torch.stack(hs, 1).cpu(), torch.stack(zs, 1).cpu()   # h, z
    return texts


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
    print(f"[twin-h] candidate pairs: {len(pairs)}")

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

    SUF = " Output only the answer and nothing else."
    q_orig = [p["question"].strip() + SUF for p in pairs]
    q_twin = [p["twin_question"].strip() + SUF for p in pairs]

    texts_o, h_o, z_o = rollout(model, tok, q_orig, record=True)
    texts_t, h_t, z_t = rollout(model, tok, q_twin, record=True)
    keep = [i for i, p in enumerate(pairs)
            if close(extract_number(texts_o[i]), p["gold"])
            and close(extract_number(texts_t[i]), p["cf_twin"])]
    print(f"[twin-h] both-correct pairs: {len(keep)}/{len(pairs)} "
          f"(orig-correct {sum(close(extract_number(texts_o[i]), p['gold']) for i, p in enumerate(pairs))}, "
          f"twin-correct {sum(close(extract_number(texts_t[i]), p['cf_twin']) for i, p in enumerate(pairs))})")
    # RE-RECORD on the kept subset: left padding shifts absolute positions with the
    # batch max length, so recorded h must come from the same batch the patched
    # rollouts will use (else the repatch gates fail on RoPE phase alone). Rows that
    # flip correctness under the new batch composition (borderline bf16) are dropped.
    for _ in range(3):
        pairs = [pairs[i] for i in keep]
        q_orig = [q_orig[i] for i in keep]
        q_twin = [q_twin[i] for i in keep]
        texts_o, h_o, z_o = rollout(model, tok, q_orig, record=True)
        texts_t, h_t, z_t = rollout(model, tok, q_twin, record=True)
        keep = [j for j, p in enumerate(pairs)
                if close(extract_number(texts_o[j]), p["gold"])
                and close(extract_number(texts_t[j]), p["cf_twin"])]
        if len(keep) == len(pairs):
            break
        print(f"[twin-h] re-record dropped {len(pairs) - len(keep)} borderline rows")
    else:
        raise RuntimeError("re-record did not stabilize in 3 rounds")
    n = len(pairs)

    nocot_o = rollout(model, tok, q_orig, nocot=True)
    nocot_t = rollout(model, tok, q_twin, nocot=True)
    nocot_ok_o = [close(extract_number(nocot_o[j]), pairs[j]["gold"]) for j in range(n)]
    nocot_ok_t = [close(extract_number(nocot_t[j]), pairs[j]["cf_twin"]) for j in range(n)]
    print(f"[nocot] solvable without thoughts: orig {sum(nocot_ok_o)}/{n}, "
          f"twin {sum(nocot_ok_t)}/{n}")

    disp24 = (h_t[:, :, 24].float() - h_o[:, :, 24].float()).norm(dim=-1)   # [n, NL]
    nd = torch.randn(*h_o[:, :, 24].shape, generator=torch.Generator().manual_seed(SEED))
    nd = nd / nd.norm(dim=-1, keepdim=True)
    noise24 = (h_o[:, :, 24].float() + disp24[..., None] * nd).to(torch.bfloat16)

    ALL = [True] * NL
    CONDS = {"identity": None,
             "repatch24": dict(kind="row24", vec=h_o[:, :, 24], active=ALL),
             "repatchcol_p3": dict(kind="col", vec=h_o, zvec=z_o, active=[t == 2 for t in range(NL)]),
             "twin24_all6": dict(kind="row24", vec=h_t[:, :, 24], active=ALL),
             "twincol_all6": dict(kind="col", vec=h_t, zvec=z_t, active=ALL),
             "noise24_all6": dict(kind="row24", vec=noise24, active=ALL)}
    for k in range(NL):
        act = [t == k for t in range(NL)]
        CONDS[f"twin24_p{k + 1}"] = dict(kind="row24", vec=h_t[:, :, 24], active=act)
        CONDS[f"twincol_p{k + 1}"] = dict(kind="col", vec=h_t, zvec=z_t, active=act)
    # REVERSE direction: the original's columns into the TWIN question's rollout
    # (keep = the twin question's own answer cf_twin, adopt = the transplanted gold)
    REV = {"revcol_all6": dict(kind="col", vec=h_o, zvec=z_o, active=ALL)}
    for k in range(NL):
        REV[f"revcol_p{k + 1}"] = dict(kind="col", vec=h_o, zvec=z_o,
                                       active=[t == k for t in range(NL)])

    preds, texts = {}, {}
    from tqdm.auto import tqdm
    for cond, tgt in tqdm(CONDS.items(), desc="fwd conditions"):
        outs = rollout(model, tok, q_orig, tgt=tgt)
        texts[cond] = outs
        preds[cond] = [extract_number(x) for x in outs]
    for cond, tgt in tqdm(REV.items(), desc="rev conditions"):
        outs = rollout(model, tok, q_twin, tgt=tgt)
        texts[cond] = outs
        preds[cond] = [extract_number(x) for x in outs]
    for h in handles:
        h.remove()

    for gate, ref in (("repatch24", "identity"), ("repatchcol_p3", "identity")):
        flips = sum(not close(preds[gate][j], preds[ref][j]) for j in range(n))
        print(f"[gate] {gate} vs identity flips: {flips}/{n}")
        assert flips <= max(2, n // 10), f"{gate} harness gate failed"

    summary, summary_lb, records = {}, {}, []
    for cond in list(CONDS) + list(REV):
        own, tr = ("cf_twin", "gold") if cond in REV else ("gold", "cf_twin")
        lb = [j for j in range(n)
              if not (nocot_ok_t[j] if cond in REV else nocot_ok_o[j])]
        summary[cond] = dict(
            n=n,
            keep=float(np.mean([close(preds[cond][j], pairs[j][own]) for j in range(n)])),
            adopt=float(np.mean([close(preds[cond][j], pairs[j][tr]) for j in range(n)])))
        summary_lb[cond] = dict(
            n=len(lb),
            keep=float(np.mean([close(preds[cond][j], pairs[j][own]) for j in lb])),
            adopt=float(np.mean([close(preds[cond][j], pairs[j][tr]) for j in lb])))
        print(f"{cond:14s} keep={summary[cond]['keep']:.3f} adopt={summary[cond]['adopt']:.3f}"
              f"  | lb n={len(lb)} keep={summary_lb[cond]['keep']:.3f}"
              f" adopt={summary_lb[cond]['adopt']:.3f}")
    for j, p in enumerate(pairs):
        records.append(dict(doc=p["doc"], gold=p["gold"], cf_twin=p["cf_twin"],
                            g=p.get("g"), g_twin=p.get("g_twin"),
                            nocot_orig_ok=bool(nocot_ok_o[j]), nocot_twin_ok=bool(nocot_ok_t[j]),
                            conds={c: dict(pred=preds[c][j], text=texts[c][j])
                                   for c in list(CONDS) + list(REV)}))
    json.dump(dict(summary=summary, summary_lb=summary_lb,
                   mean_disp24=float(disp24.mean()), records=records),
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
