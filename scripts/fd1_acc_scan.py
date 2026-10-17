"""Generative accuracy scan for the fd1 AO figure's raw-accuracy panel: greedy
readouts at the saved fd1c checkpoints for the three eval feeds of the CE panel
(full six-stream thought read, z-only thought read, fair six-stream question-only
bound with the pseudo-z prj(h^(36)_q) group, exactly as in fd1_qonly6_scan.py) on
the same 48 held-out problems, scored per label:
  answer  -> exact match: first number extracted from the reply == canon gold answer
  allvals -> chain-values F1 (the trainer's held_* metric: extracted-number set vs
             gold chain values)

Cells: {ALL, Z, qonly6} x {answer, allvals}, questions verbatim from make_qa.
-> /workspace-vast/jbauer/bt1/fd1/dur/fd1_acc_scan.jsonl (append, resumable by step)

srun -p general --qos=high --gres=gpu:1 --mem=96G --time=4:00:00 --job-name=fd1acc_jb \
  bash scripts/slurm/run_fd1_acc_scan.sh
"""
from __future__ import annotations

import functools
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "scripts", "bt1"))
os.environ.setdefault("FD1_DATA", "/workspace-vast/jbauer/bt1/fd1/data")
os.environ.setdefault("HF_HOME", f"/workspace-vast/{os.environ.get('USER', 'jbauer')}/hf_cache")
print = functools.partial(print, flush=True)

BASE = "Qwen/Qwen3-8B"
CODI_CKPT = "/workspace-vast/jbauer/codi_eval_ckpt/checkpoint-7580/pytorch_model.bin"
MARKER_ID, INJ_SCALE = 149705, 150.0
MARKER = "㈎"
DUR = "/workspace-vast/jbauer/bt1/fd1/dur"
OUT = f"{DUR}/fd1_acc_scan.jsonl"
VN = 48
STEPS = [0, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 66000]
ALLSRC = ["L9", "L18", "L27", "L35", "LF", "Z"]
SRC5 = ["L9", "L18", "L27", "L35", "LF"]
ANSQ = "What is the final answer of the computation?"
VALQ = "List every intermediate result computed (any order)."
DEV = "cuda"
LF_STREAM = 4


def load_prj():
    sd = torch.load(CODI_CKPT, map_location="cpu", weights_only=True)
    prj = nn.Sequential()
    prj.add_module("0", nn.Dropout(0.0))
    prj.add_module("1", nn.Linear(4096, 4096))
    prj.add_module("2", nn.GELU())
    prj.add_module("3", nn.Linear(4096, 4096))
    prj.add_module("ln", nn.LayerNorm(4096))
    prj.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("prj.")})
    return prj.to(torch.bfloat16).eval()


def main():
    from peft import LoraConfig, get_peft_model
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fd1_data import PREAMBLE, Pool, canon, render_acts

    tok = AutoTokenizer.from_pretrained(BASE)
    assert tok.decode([MARKER_ID]) == MARKER
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV)
    model = get_peft_model(model, LoraConfig(
        r=128, lora_alpha=256, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.eval()
    embed = model.get_input_embeddings()

    gsm = Pool("gsm")
    held_rows = gsm.ok[-200:][:100]
    rows = held_rows[:VN]
    print(f"pool gsm ok {len(gsm.ok)}; scoring {len(rows)} held rows")

    # pseudo-z at question tokens, validated as in fd1_qonly6_scan.py
    prj = load_prj()
    with torch.no_grad():
        cos = []
        for i in rows[:16]:
            h_bocot_lf = torch.from_numpy(np.array(gsm.h[i, 1, LF_STREAM], dtype=np.float32))
            z1_hat = prj(h_bocot_lf.to(torch.bfloat16)).float()
            z1 = torch.from_numpy(np.array(gsm.z[i, 0], dtype=np.float32))
            cos.append(float(F.cosine_similarity(z1_hat, z1, dim=0)))
        print(f"prj parity vs stored z1: cos min={min(cos):.5f} mean={np.mean(cos):.5f}")
        assert min(cos) > 0.99, cos
        zq = {}
        for i in rows:
            L = int(gsm.qlen[i])
            hq = torch.from_numpy(np.array(gsm.qh[i, :L, LF_STREAM], dtype=np.float32))
            zq[i] = [v.astype(np.float32) for v in prj(hq.to(torch.bfloat16)).float().numpy()]

    def feed(i, name):
        if name == "qonly6":
            acts, vecs = render_acts(gsm, i, SRC5, "q_only", [])
            return acts + f"\n[Q Z] {MARKER * len(zq[i])}", list(vecs) + zq[i]
        return render_acts(gsm, i, ALLSRC if name == "ALL" else ["Z"], "thoughts", list(range(6)))

    def nums_in(text):
        out = []
        for mt in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
            try:
                out.append(canon(mt))
            except ValueError:
                pass
        return out

    def f1_of(text, gold):
        E, G = set(nums_in(text)), set(gold)
        if not E or not G:
            return 0.0
        hit = len(E & G)
        p, r = hit / len(E), hit / len(G)
        return 2 * p * r / (p + r) if hit else 0.0

    @torch.no_grad()
    def gen(prompt, vecs, max_new):
        txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = torch.tensor([tok(txt, add_special_tokens=False).input_ids], device=DEV)
        E = embed(ids)
        mask = ids == MARKER_ID
        assert int(mask.sum()) == len(vecs)
        V = torch.from_numpy(np.stack(list(vecs))).to(DEV)
        E[mask] = (V / V.norm(dim=1, keepdim=True) * INJ_SCALE).to(E.dtype)
        out = model.generate(inputs_embeds=E, attention_mask=torch.ones_like(ids),
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0], skip_special_tokens=True)

    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["step"] for l in open(OUT)}
        print(f"resume: steps already done {sorted(done)}")

    for step in tqdm(STEPS, desc="steps"):
        if step in done:
            continue
        if step > 0:
            sd = torch.load(f"{DUR}/fd1c_step{step}_lora.pt", map_location="cpu", weights_only=True)
            missing = model.load_state_dict(sd, strict=False)
            assert not missing.unexpected_keys, missing.unexpected_keys[:3]
        row = {"step": step, "n": len(rows)}
        for fname in ("ALL", "Z", "qonly6"):
            accs, f1s = [], []
            for i in tqdm(rows, desc=f"s{step} {fname}", leave=False):
                acts, vecs = feed(i, fname)
                m = gsm.meta[i]
                a_txt = gen(PREAMBLE + acts + f"\n\nQuestion: {ANSQ}", vecs, 16)
                v_txt = gen(PREAMBLE + acts + f"\n\nQuestion: {VALQ}", vecs, 48)
                a_nums = nums_in(a_txt)
                accs.append(1.0 if a_nums and a_nums[0] == canon(m["answer"]) else 0.0)
                f1s.append(f1_of(v_txt, [c["val"] for c in m["chain"]]))
            row[f"{fname}|answer"] = float(np.mean(accs))
            row[f"{fname}|allvals"] = float(np.mean(f1s))
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"step {step}: " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()
                                           if k not in ("step", "n")))
    print("FD1-ACCSCAN-DONE")


if __name__ == "__main__":
    main()
