"""AO version of the thought-ordering analysis (post: appendix AO section): does the
fd1c reader, fed a SINGLE thought position, recover chain steps in forward order like
the single-position NLA reader (nx2) does?

For each held row (same 100 rows as the fd1 val cells) and each thought position
t = 1..6, inject the full six-stream column {h^(9,18,27,35,36)_t, z_t} at that single
position and greedy-generate the allvals label ("List every intermediate result
computed (any order)."). Conditions: own (row's own acts) and donor (the next held
row's acts, scored against THIS row's chain = the chance floor, mirroring
nx2_pos_order). Analysis mirrors fig_08: every generated number that canon-matches the
gold ordered chain contributes y = (step_index + 0.5)/chain_length at x = t, Spearman
with permutation p, own vs donor.

Reader: the final fd1c checkpoint (step 66000 LoRA).
-> /workspace-vast/jbauer/bt1/fd1/dur/fd1_pos_order.jsonl (resumable by (i,t,cond))

srun -p general --qos=high --gres=gpu:1 --mem=96G --time=1:30:00 --job-name=fd1po_jb \
  bash scripts/slurm/run_fd1_pos_order.sh
"""
from __future__ import annotations

import functools
import json
import os
import sys
import zlib

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "scripts", "bt1"))
os.environ.setdefault("FD1_DATA", "/workspace-vast/jbauer/bt1/fd1/data")
os.environ.setdefault("HF_HOME", f"/workspace-vast/{os.environ.get('USER', 'jbauer')}/hf_cache")
print = functools.partial(print, flush=True)

BASE = "Qwen/Qwen3-8B"
MARKER_ID, INJ_SCALE = 149705, 150.0
DUR = "/workspace-vast/jbauer/bt1/fd1/dur"
LORA = f"{DUR}/fd1c_step66000_lora.pt"
OUT = f"{DUR}/fd1_pos_order.jsonl"
SRC6 = ["L9", "L18", "L27", "L35", "LF", "Z"]
MAX_NEW = 64
DEV = "cuda"


def main():
    import random as _rnd
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fd1_data import Pool, build_sample

    tok = AutoTokenizer.from_pretrained(BASE)
    assert tok.decode([MARKER_ID]) == "㈎"
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV)
    model = get_peft_model(model, LoraConfig(
        r=128, lora_alpha=256, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    sd = torch.load(LORA, map_location="cpu", weights_only=True)
    missing = model.load_state_dict(sd, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys[:3]
    model.eval()
    embed = model.get_input_embeddings()

    gsm = Pool("gsm")
    held = gsm.ok[-200:][:100]
    print(f"pool ok {len(gsm.ok)}; held {len(held)}; lora {LORA}")

    jobs = []
    for j, i in enumerate(held):
        donor = held[(j + 1) % len(held)]
        for t in range(6):
            jobs.append((i, i, t, "own"))
            jobs.append((i, donor, t, "donor"))

    done = set()
    if os.path.exists(OUT):
        done = {(r["i"], r["t"], r["cond"]) for r in map(json.loads, open(OUT))}
        print(f"resume: {len(done)} already done")
    jobs = [jb for jb in jobs if (jb[0], jb[2], jb[3]) not in done]
    print(f"{len(jobs)} generations to run")

    def encode(prompt):
        txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, enable_thinking=False, tokenize=False)
        return tok(txt, add_special_tokens=False).input_ids

    @torch.no_grad()
    def gen_batch(batch):
        prompts, vecs_list = [], []
        for (i, src_i, t, cond) in batch:
            crng = _rnd.Random(zlib.crc32(f"posorder|{i}|{t}|{cond}".encode()))
            s = build_sample(gsm, crng, mode="act", i=src_i, kind="allvals",
                             spec=(SRC6, "thoughts", [t]))
            prompts.append(encode(s["prompt"]))
            vecs_list.append(s["vecs"])
        B, L = len(prompts), max(len(p) for p in prompts)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((B, L), pad, dtype=torch.long)
        att = torch.zeros(B, L, dtype=torch.long)
        for b, p in enumerate(prompts):                       # left-pad for generate
            ids[b, L - len(p):] = torch.tensor(p); att[b, L - len(p):] = 1
        ids_dev, att_dev = ids.to(DEV), att.to(DEV)
        E = embed(ids_dev)
        mask = ids_dev == MARKER_ID
        assert mask.sum(1).tolist() == [len(v) for v in vecs_list]
        V = torch.from_numpy(np.stack([v for vs in vecs_list for v in vs])).to(DEV)
        E[mask] = (V / V.norm(dim=1, keepdim=True) * INJ_SCALE).to(E.dtype)
        out = model.generate(inputs_embeds=E, attention_mask=att_dev,
                             max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=pad)
        return [tok.decode(seq, skip_special_tokens=True) for seq in out]

    for s0 in range(0, len(jobs), 16):
        batch = jobs[s0:s0 + 16]
        texts = gen_batch(batch)
        with open(OUT, "a") as f:
            for (i, src_i, t, cond), g in zip(batch, texts):
                f.write(json.dumps(dict(i=i, donor=src_i if cond == "donor" else None,
                                        t=t + 1, cond=cond, gen=g)) + "\n")
        if (s0 // 16) % 10 == 0:
            print(f"{s0 + len(batch)}/{len(jobs)}  e.g. [{batch[0][3]} t{batch[0][2] + 1}] {texts[0][:80]!r}")

    # ---- inline analysis (mirrors fig_08 recovered-step logic) ----
    def canon(s):
        f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
        return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"

    import re as _re
    chains = {i: [c["val"] for c in gsm.meta[i]["chain"]] for i in held}
    xs, ys = {"own": [], "donor": []}, {"own": [], "donor": []}
    for r in map(json.loads, open(OUT)):
        gold = chains[r["i"]]
        if len(gold) < 2:
            continue
        ems = set()
        for m in _re.findall(r"-?\d[\d,]*(?:\.\d+)?", r["gen"]):
            try: ems.add(canon(m))
            except ValueError: pass
        gidx = {}
        for k, v in enumerate(gold):
            if v not in gidx: gidx[v] = k
        for c, k in gidx.items():
            if c in ems:
                xs[r["cond"]].append(r["t"]); ys[r["cond"]].append((k + 0.5) / len(gold))
    from scipy.stats import spearmanr
    for cond in ("own", "donor"):
        rho, p = spearmanr(xs[cond], ys[cond])
        means = [np.mean([y for x, y in zip(xs[cond], ys[cond]) if x == t]) for t in range(1, 7)]
        print(f"{cond}: n={len(xs[cond])} rho={rho:+.3f} p={p:.5f} "
              f"per-pos means {[f'{m:.2f}' for m in means]}")
    print("FD1-POSORDER-DONE")


if __name__ == "__main__":
    main()
