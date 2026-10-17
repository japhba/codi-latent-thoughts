"""Sequential per-thought AO readout on the load-bearing core, ALL layers injected.
--part capture: CODI latent rollout on core+control test rows, hooks on all 36 blocks;
    store [6 thoughts, 37, 4096] fp16 (entry 0 = fed z, 1..36 = block outputs)
    -> /workspace-vast/jbauer/lz1_local/test_allL.npz
--part eval: stock ceselder/qwen3-8b-ao-v3-best; for each thought i (1..6) inject the
    full 36-layer stack (layers 0..35, one position each, layer-major) and generate for
    framed_open + bare_1sent, own + donor (same thought index, shifted problem).
    Metrics per (set, thought, prompt, cond): f1_chain, f1_q, num_chain, num_drv,
    argmax-sentence histogram. Full generations saved.
    -> /workspace-vast/jbauer/lz1_local/seq_ao.json (copy to reports/codi_writeup/)
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import re
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
print = functools.partial(print, flush=True)

DIR = "/workspace-vast/jbauer/lz1_local"
CKPT = "/workspace-vast/jbauer/codi_eval_ckpt/checkpoint-7580"
BASE = "Qwen/Qwen3-8B"
DEV = "cuda"


def clean(s):
    return re.sub(r"<<[^>]*>>", "", s).strip()


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def f1(a, b):
    A, B = toks(a), toks(b)
    if not A or not B:
        return 0.0
    c = sum(min(A.count(t), B.count(t)) for t in set(A))
    return 0.0 if c == 0 else 2 * (c / len(A)) * (c / len(B)) / ((c / len(A)) + (c / len(B)))


def nums(s):
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def eval_rows():
    sets = json.load(open(f"{DIR}/lz_eval_sets.json"))
    meta = json.load(open(f"{DIR}/test_shard000.json"))
    byidx = {m["ds_index"]: m for m in meta}
    rows = []
    for sname, idxs in sets.items():
        for i in idxs:
            m = byidx.get(i)
            if m is None:
                continue
            sents = [clean(s) for s in m["sents"] if clean(s)]
            if len(sents) < 3:
                continue
            rows.append(dict(set=sname, i=i, q=m["q"], sents=sents))
    return rows


@torch.no_grad()
def capture():
    sys.path.insert(0, f"{_REPO}/third_party/CODI")
    from src.model import CODI
    model = CODI.from_pretrained(checkpoint_path=CKPT, model_name_or_path=BASE,
                                 lora_r=128, lora_alpha=32, num_latent=6, use_prj=True,
                                 prj_dim=4096, prj_no_ln=False, prj_dropout=0.0,
                                 dtype="bfloat16")
    model.eval().to(DEV)
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")
    bocot = tok.convert_tokens_to_ids("<|bocot|>")
    blocks = model.codi.model.model.layers
    cap = {}
    hooks = [blocks[L].register_forward_hook(
        (lambda LL: lambda _m, _i, out: cap.__setitem__(
            LL, (out[0] if isinstance(out, tuple) else out).detach()))(L))
        for L in range(len(blocks))]
    rows = eval_rows()
    print(f"{len(rows)} rows")
    ARR, DSI = [], []
    from tqdm.auto import tqdm
    BS = 16
    try:
        for s0 in tqdm(range(0, len(rows), BS)):
            chunk = rows[s0:s0 + BS]
            qs = [r["q"] + " Output only the answer and nothing else." for r in chunk]
            batch = tok(qs, return_tensors="pt", padding="longest", padding_side="left")
            ids = torch.cat([batch["input_ids"],
                             torch.tensor([[tok.eos_token_id, bocot]]).expand(len(chunk), 2)], 1).to(DEV)
            att = torch.cat([batch["attention_mask"],
                             torch.ones(len(chunk), 2, dtype=torch.long)], 1).to(DEV)
            out = model.codi(input_ids=ids, attention_mask=att, use_cache=True,
                             output_hidden_states=True)
            pkv = out.past_key_values
            latent = out.hidden_states[-1][:, -1:, :]
            per = [[] for _ in chunk]
            for _ in range(6):
                z = model.prj(latent).to(model.codi.dtype) if model.use_prj else latent
                out = model.codi(inputs_embeds=z, use_cache=True, past_key_values=pkv,
                                 output_hidden_states=True)
                pkv = out.past_key_values
                step = torch.stack([z[:, 0]] + [cap[L][:, -1] for L in range(len(blocks))], 1)
                for bi in range(len(chunk)):
                    per[bi].append(step[bi].to(torch.float16).cpu())
                latent = out.hidden_states[-1][:, -1:, :]   # post-final-norm, CODI's real feedback
            for bi, r in enumerate(chunk):
                ARR.append(torch.stack(per[bi]))         # [6, 37, 4096]
                DSI.append(r["i"])
    finally:
        for h in hooks:
            h.remove()
    np.savez(f"{DIR}/test_allL.npz", arr=torch.stack(ARR).numpy(), ds_index=np.array(DSI))
    print(f"saved {len(ARR)} rows [6,37,4096]")
    print("SEQCAP-DONE")


PROMPTS = [
    ("framed_open", "These are internal reasoning states of a model solving "
                    "a problem. What is the model thinking about?"),
    ("bare_1sent", "What is the model thinking about? Answer in a single sentence only."),
]


@torch.no_grad()
def evaluate():
    from nl_probes.utils.activation_utils import get_hf_submodule
    from nl_probes.utils.dataset_utils import construct_batch, create_training_datapoint
    from nl_probes.utils.steering_hooks import get_hf_activation_steering_hook
    import scripts.pw1_sweep_eval as sw
    sw.ADAPTER = "ceselder/qwen3-8b-ao-v3-best"
    from scripts.pw1_sweep_eval import DTYPE, HOOK_LAYER, load_ao
    model, tok = load_ao()
    submodule = get_hf_submodule(model, HOOK_LAYER)

    d = np.load(f"{DIR}/test_allL.npz")
    arr, byds = d["arr"], {int(x): k for k, x in enumerate(d["ds_index"])}
    rows = [r for r in eval_rows() if r["i"] in byds]
    BAND = [21, 22, 23, 24, 25]
    inject = os.environ.get("SEQ_INJECT", "all36")     # all36 | band
    records = []
    from tqdm.auto import tqdm
    for r_i, r in enumerate(tqdm(rows)):
        k = byds[r["i"]]
        kd = byds[rows[(r_i + 37) % len(rows)]["i"]]
        dps, keys = [], []
        for cond, kk in (("own", k), ("donor", kd)):
            for t in range(6):
                if inject == "band":                   # native 5-layer band, single thought
                    acts = torch.tensor(arr[kk, t, [1 + L for L in BAND]]
                                        .astype(np.float32)).to(DTYPE)      # [5,4096]
                    layers, npos = BAND, 1
                else:
                    acts = torch.tensor(arr[kk, t, 1:37].astype(np.float32)).to(DTYPE)
                    layers, npos = list(range(36)), 1
                for pname, prm in PROMPTS:
                    dps.append(create_training_datapoint(
                        datapoint_type="seq_ao", prompt=prm, correct_response="x",
                        layers=layers, num_positions=npos, tokenizer=tok,
                        acts_BD=acts, feature_idx=-1, meta_info={}))
                    keys.append((cond, t, pname, "single"))
            if inject == "band":                       # all-6-thoughts arm, layer-major
                win = torch.tensor(arr[kk, :, [1 + L for L in BAND]]        # [5,6,4096] layer-major
                                   .reshape(30, 4096).astype(np.float32)).to(DTYPE)
                for pname, prm in PROMPTS:
                    dps.append(create_training_datapoint(
                        datapoint_type="seq_ao", prompt=prm, correct_response="x",
                        layers=BAND, num_positions=6, tokenizer=tok,
                        acts_BD=win, feature_idx=-1, meta_info={}))
                    keys.append((cond, -1, pname, "all6"))
        batch = construct_batch(dps, tok, torch.device(DEV))
        gen_in = []
        for bi in range(len(dps)):
            n_lab = int((batch.labels[bi] != -100).sum())
            gen_in.append(batch.input_ids[bi][: batch.input_ids.shape[1] - n_lab])
        Lm = max(len(x) for x in gen_in)
        ids = torch.full((len(gen_in), Lm), tok.pad_token_id or tok.eos_token_id,
                         device=DEV, dtype=torch.long)
        att = torch.zeros_like(ids)
        inj = []
        for bi, x in enumerate(gen_in):
            off = Lm - len(x)
            ids[bi, off:] = x
            att[bi, off:] = 1
            inj.append([p + off - (batch.input_ids.shape[1] - len(x))
                        for p in batch.injection_indices[bi]])
        hook = get_hf_activation_steering_hook(
            batch.context_activations, inj, 1.0, torch.device(DEV), DTYPE)
        h = submodule.register_forward_hook(hook)
        try:
            g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=50,
                               do_sample=False, pad_token_id=tok.eos_token_id)
        finally:
            h.remove()
        chain = " ".join(r["sents"])
        nq, nc = nums(r["q"]), nums(chain)
        drv = nc - nq
        for bi, (cond, t, pname, arm) in enumerate(keys):
            txt = tok.decode(g[bi, Lm:], skip_special_tokens=True).strip()
            ng = nums(txt)
            f1s = [f1(txt, s) for s in r["sents"]]
            records.append(dict(set=r["set"], i=r["i"], cond=cond, thought=t + 1,
                                arm=arm, prompt=pname, gen=txt,
                                f1_chain=f1(txt, chain), f1_q=f1(txt, r["q"]),
                                num_chain=len(ng & nc) / max(1, len(nc)),
                                num_drv=len(ng & drv) / max(1, len(drv)),
                                best_sent=int(np.argmax(f1s)) + 1))
    out = {"records": records}
    json.dump(out, open(f"{DIR}/seq_ao_{inject}.json", "w"))
    import collections
    print(f"\n{'cond':>6} {'prompt':>12} {'t':>2} {'f1_ch':>6} {'f1_q':>6} {'num_ch':>6} {'num_drv':>7}")
    for cond in ("own", "donor"):
        for pname, _ in PROMPTS:
            for t in ([0] if inject == "band" else []) + list(range(1, 7)):
                v = [x for x in records if x["cond"] == cond and x["prompt"] == pname
                     and (x["thought"] == t if t else x["arm"] == "all6") and x["set"] == "core"
                     and (x["arm"] == "single" if t else True)]
                if not v:
                    continue
                a = np.mean([[x["f1_chain"], x["f1_q"], x["num_chain"], x["num_drv"]] for x in v], 0)
                lbl = "all6" if t == 0 else str(t)
                print(f"{cond:>6} {pname:>12} {lbl:>4} {a[0]:6.3f} {a[1]:6.3f} {a[2]:6.3f} {a[3]:7.3f}")
    print("SEQAO-DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["capture", "eval"], required=True)
    a = ap.parse_args()
    (capture if a.part == "capture" else evaluate)()
