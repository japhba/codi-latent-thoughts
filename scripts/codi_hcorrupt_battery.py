"""h-column corruption battery on the doubly load-bearing core, CODI ckpt-7580:
the h-level counterpart of the z ablation battery's content corruptions, so the two
can be compared row-paired on the same problems and prompt format.

Rows: the core idx from reports/codi_writeup/lb_core_idx.json (base needs verbal
CoT AND trace-necessary), GSM8k test, battery prompt format verbatim.

Phase 1 records each row's own full columns (all 37 hidden-state entries at the 6
thought positions) in ONE batch; phase 2 decodes with the columns replaced at every
thought position (hook patches the current last position at each latent step,
downstream regenerates within the corrupted sequence):
  identity          : no patch
  repatchcol_p3     : own column at position 3 (harness gate)
  shufflecol        : own columns, positions permuted (per-row seeded permutation)
  reversecol        : own columns, positions reversed
  swapcol_neighbor  : the next core row's columns (cyclic donor)
  gausscol          : elementwise mean+std-matched Gaussian columns (over core rows)
Scores per condition: accuracy vs gold on the rows that are identity-correct under
this harness; per-row preds stored for paired deltas against the z battery.
-> reports/codi_writeup/repo/artifacts/hcorrupt_battery.json

srun -p general --qos=high --gres=gpu:1 --mem=128G --time=1:00:00 --job-name=codi_hcb_jb \
  bash scripts/slurm/run_codi_hcorrupt.sh
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
CORE_F = os.path.join(_REPO, "reports", "codi_writeup", "lb_core_idx.json")
NL = 6
NH = 37
SEED = 11
DEV = "cuda"
OUT = os.path.join(_REPO, "reports", "codi_writeup", "repo", "artifacts", "hcorrupt_battery.json")


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
def rollout(model, tok, questions, record=False, cols=None, active=None, max_new_tokens=64):
    """Batched latent rollout + greedy decode. record -> also return h [B,NL,NH,D].
    cols [B,NL,NH,D] + active [NL]: replace the full column at the active positions."""
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
    hs = []
    for t in range(NL):
        l_in = model.prj(hidden).to(model.codi.dtype) if model.use_prj else hidden
        if cols is not None and active[t]:
            STATE["col"] = cols[:, t].to(DEV)
        out = model.codi(inputs_embeds=l_in, use_cache=True, output_hidden_states=True,
                         past_key_values=past)
        STATE.clear()
        past = out.past_key_values
        if record:
            hs.append(torch.stack([out.hidden_states[k][:, -1] for k in range(NH)], 1))
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
        return texts, torch.stack(hs, 1).cpu()        # [B, NL, NH, D]
    return texts


def main():
    torch.manual_seed(SEED)
    from datasets import load_dataset
    core = sorted(json.load(open(CORE_F))["core_idx"])
    ds = load_dataset("gsm8k", "main")["test"]
    questions = [f"{ds[i]['question'].strip().replace('  ', ' ')} Output only the answer "
                 f"and nothing else." for i in core]
    golds = [float(ds[i]["answer"].split("####")[-1].replace(",", "")) for i in core]
    n = len(core)
    print(f"[hcorrupt] core rows: {n}")

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

    # ONE batch throughout so RoPE phases match between record and patched decodes
    texts_id, h_o = rollout(model, tok, questions, record=True)
    id_ok = [close(extract_number(texts_id[j]), golds[j]) for j in range(n)]
    print(f"[hcorrupt] identity-correct: {sum(id_ok)}/{n}")

    rng = np.random.default_rng(SEED)
    perms = np.stack([rng.permutation(NL) for _ in range(n)])
    shuf = torch.stack([h_o[j, perms[j]] for j in range(n)])
    rev = h_o.flip(1)
    donor = h_o.roll(-1, dims=0)                       # next core row, cyclic
    mu = h_o.float().mean(0, keepdim=True)
    sd = h_o.float().std(0, keepdim=True)
    g = torch.Generator().manual_seed(SEED)
    gauss = (mu + sd * torch.randn(*h_o.shape, generator=g)).to(h_o.dtype)

    ALL = [True] * NL
    CONDS = {"identity": (None, None),
             "repatchcol_p3": (h_o, [t == 2 for t in range(NL)]),
             "shufflecol": (shuf, ALL),
             "reversecol": (rev, ALL),
             "swapcol_neighbor": (donor, ALL),
             "gausscol": (gauss, ALL)}

    preds, texts = {}, {}
    from tqdm.auto import tqdm
    for cond, (cols, active) in tqdm(CONDS.items(), desc="conditions"):
        outs = rollout(model, tok, questions, cols=cols, active=active) \
            if cols is not None else texts_id
        texts[cond] = outs
        preds[cond] = [extract_number(x) for x in outs]
    for h in handles:
        h.remove()

    flips = sum(not close(preds["repatchcol_p3"][j], preds["identity"][j]) for j in range(n))
    print(f"[gate] repatchcol_p3 vs identity flips: {flips}/{n}")
    assert flips <= max(2, n // 10), "repatchcol_p3 harness gate failed"

    kept = [j for j in range(n) if id_ok[j]]
    summary = {}
    for cond in CONDS:
        acc = float(np.mean([close(preds[cond][j], golds[j]) for j in kept]))
        summary[cond] = dict(n=len(kept), acc=acc)
        print(f"{cond:16s} acc={acc:.3f}  (delta {100 * (acc - 1):+.1f}pp)")
    records = [dict(i=int(core[j]), gold=golds[j], identity_ok=bool(id_ok[j]),
                    preds={c: preds[c][j] for c in CONDS}) for j in range(n)]
    json.dump(dict(summary=summary, records=records), open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    print("CODI-HCORRUPT-DONE")


if __name__ == "__main__":
    main()
