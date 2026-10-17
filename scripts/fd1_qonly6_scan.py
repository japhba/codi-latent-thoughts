"""Six-stream question-only bound for the fd1 AO figure: the thought reads feed six
streams per position ({h^(9,18,27,35,36)} + z) while the qonly5 bound feeds only the
five h layers, a stream-count confound. This scan adds a pseudo-z at every question
token, prj(h^(36)_q) with CODI ckpt-7580's own projection module, so the question
control has the identical per-position feed structure (5 residuals + 1 projected).

Mirrors fd1_qonly5_scan.py exactly (same held rows, VN=48, and the SAME cell rng key
"qonly5|{label}" so the QA pieces are identical and only the feed differs): the
qonly5 sample is built, then a "[Q Z]" marker group with the pseudo-z vectors is
spliced in before the question. Injection normalizes every vector to unit * 150, so
prj output scale is irrelevant.

prj is loaded from the CODI checkpoint (keys prj.1/prj.3/prj.ln, Dropout-Linear-GELU-
Linear-LayerNorm, run in bf16 like the capture) and validated exactly: prj of the
stored bocot h^(36) must reproduce the stored z_1 per row (cos > 0.99).

Cells: {qonly6} x {answer, allvals}.
-> /workspace-vast/jbauer/bt1/fd1/dur/fd1_qonly6.jsonl (append, resumable by step)

srun -p general --qos=high --gres=gpu:1 --mem=96G --time=2:30:00 --job-name=fd1q6_jb \
  bash scripts/slurm/run_fd1_qonly6.sh
"""
from __future__ import annotations

import functools
import json
import os
import sys
import zlib

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
OUT = f"{DUR}/fd1_qonly6.jsonl"
VN = 48
STEPS = [0, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 66000]
SPEC5 = (["L9", "L18", "L27", "L35", "LF"], "q_only", [])
LABELS = ["answer", "allvals"]
DEV = "cuda"
LF_STREAM = 4                                  # stream axis index of LF in *_qh.npy


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
    import random as _rnd
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fd1_data import Pool, build_sample

    tok = AutoTokenizer.from_pretrained(BASE)
    assert tok.decode([MARKER_ID]) == MARKER
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV)
    model = get_peft_model(model, LoraConfig(
        r=128, lora_alpha=256, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.eval()
    embed = model.get_input_embeddings()
    base_lm = model.get_base_model()
    body, lm_head = base_lm.model, base_lm.lm_head

    gsm = Pool("gsm")
    held_rows = gsm.ok[-200:][:100]
    print(f"pool gsm ok {len(gsm.ok)}; held {len(held_rows)}")

    # ---- pseudo-z at question tokens: prj(h^(36)_q), validated against stored z1 ----
    prj = load_prj()
    with torch.no_grad():
        cos = []
        for i in held_rows[:16]:
            h_bocot_lf = torch.from_numpy(np.array(gsm.h[i, 1, LF_STREAM], dtype=np.float32))
            z1_hat = prj(h_bocot_lf.to(torch.bfloat16)).float()
            z1 = torch.from_numpy(np.array(gsm.z[i, 0], dtype=np.float32))
            cos.append(float(F.cosine_similarity(z1_hat, z1, dim=0)))
        print(f"prj parity vs stored z1: cos min={min(cos):.5f} mean={np.mean(cos):.5f}")
        assert min(cos) > 0.99, cos
        zq = {}
        for i in held_rows[:VN]:
            L = int(gsm.qlen[i])
            hq = torch.from_numpy(np.array(gsm.qh[i, :L, LF_STREAM], dtype=np.float32))
            zq[i] = [v.astype(np.float32) for v in prj(hq.to(torch.bfloat16)).float().numpy()]
    print(f"pseudo-z built for {len(zq)} rows, e.g. qlen={len(zq[held_rows[0]])}")

    def encode(prompt, target):
        txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, enable_thinking=False, tokenize=False)
        pids = tok(txt, add_special_tokens=False).input_ids
        tids = tok(target, add_special_tokens=False).input_ids + [tok.eos_token_id]
        return pids + tids, [-100] * len(pids) + tids

    cells = {}
    for lab in LABELS:
        crng = _rnd.Random(zlib.crc32(f"qonly5|{lab}".encode()))   # same QA draws as qonly5
        items = []
        for j in range(VN):
            i = held_rows[j % len(held_rows)]
            s = build_sample(gsm, crng, mode="act", i=i, kind=lab, spec=SPEC5)
            assert "\n\nQuestion:" in s["prompt"]
            s["prompt"] = s["prompt"].replace(
                "\n\nQuestion:", f"\n[Q Z] {MARKER * len(zq[i])}\n\nQuestion:")
            s["vecs"] = list(s["vecs"]) + zq[i]
            items.append((encode(s["prompt"], s["target"]), s["vecs"]))
        cells[f"qonly6|{lab}"] = items
    print(f"cells built: { {k: len(v) for k, v in cells.items()} }")

    @torch.no_grad()
    def cell_ce(items):
        tot, ntok = 0.0, 0
        for s0 in range(0, len(items), 16):
            ch = items[s0:s0 + 16]
            seqs = [e[0][0] for e in ch]
            B, L = len(seqs), max(len(s) for s in seqs)
            ids = torch.full((B, L), tok.pad_token_id or tok.eos_token_id, dtype=torch.long)
            att = torch.zeros(B, L, dtype=torch.long)
            labels = torch.full((B, L), -100, dtype=torch.long)
            for b, ((s, lab), _) in enumerate(ch):
                ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
                labels[b, :len(lab)] = torch.tensor(lab)
            ids_dev = ids.to(DEV)
            E = embed(ids_dev)
            vecs_list = [v for (_, v) in ch]
            lens = [len(v) for v in vecs_list]
            mask = ids_dev == MARKER_ID
            assert mask.sum(1).tolist() == lens
            V = torch.from_numpy(np.stack([v for vs in vecs_list for v in vs])).to(DEV)
            E[mask] = (V / V.norm(dim=1, keepdim=True) * INJ_SCALE).to(E.dtype)
            h = body(inputs_embeds=E, attention_mask=att.to(DEV), use_cache=False).last_hidden_state
            lb = labels.to(DEV)[:, 1:]
            sel = lb != -100
            ce = F.cross_entropy(lm_head(h[:, :-1][sel]).float(), lb[sel], reduction="none")
            tot += float(ce.sum()); ntok += int(ce.numel())
        return tot / ntok

    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["step"] for l in open(OUT)}
        print(f"resume: steps already done {sorted(done)}")

    for step in STEPS:
        if step in done:
            continue
        if step > 0:
            sd = torch.load(f"{DUR}/fd1c_step{step}_lora.pt", map_location="cpu", weights_only=True)
            missing = model.load_state_dict(sd, strict=False)
            assert not missing.unexpected_keys, missing.unexpected_keys[:3]
        row = {"step": step}
        for cname, items in cells.items():
            row[cname] = cell_ce(items)
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"step {step}: " + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "step"))
    print("FD1-QONLY6-DONE")


if __name__ == "__main__":
    main()
