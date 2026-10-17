"""Fair question-only bound for the fd1 AO figure: recompute the q_only val cells
with ALL FIVE h layers (L9,L18,L27,L35,LF) over the question tokens, across the
saved per-1000-step LoRA checkpoints. The trainer's stored qonly cell fed only
L27+LF, which under-feeds the question side relative to the 6-source thought reads.

Mirrors fd1_sft.py's val_matrix path exactly: same held rows (gsm.ok[-200:][:100]),
same VN=48 with crc32(f"{iname}|{lab}") cell rng, same chat-template encode, marker
embedding injection (unit-norm * 150 at MARKER_ID), token-level CE on target
positions. Also recomputes the original 2-layer qonly cell at each checkpoint as a
parity check against fd1_valloss_c.jsonl (expected to match up to bf16/attn noise).
Step 0 = fresh PEFT init (LoRA B=0, i.e. the base model): the before-training bound.

Cells: {qonly5, qonly} x {answer, allvals}.
-> /workspace-vast/jbauer/bt1/fd1/dur/fd1_qonly5.jsonl (append, resumable by step)

srun -p general --qos=high --gres=gpu:1 --mem=96G --time=2:30:00 --job-name=fd1q5_jb \
  bash scripts/slurm/run_fd1_qonly5.sh
"""
from __future__ import annotations

import functools
import json
import os
import sys
import zlib

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "scripts", "bt1"))
os.environ.setdefault("FD1_DATA", "/workspace-vast/jbauer/bt1/fd1/data")
os.environ.setdefault("HF_HOME", f"/workspace-vast/{os.environ.get('USER', 'jbauer')}/hf_cache")
print = functools.partial(print, flush=True)

BASE = "Qwen/Qwen3-8B"
MARKER_ID, INJ_SCALE = 149705, 150.0
DUR = "/workspace-vast/jbauer/bt1/fd1/dur"
OUT = f"{DUR}/fd1_qonly5.jsonl"
VN = 48
STEPS = [0, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 66000]
CELLS = [("qonly5", (["L9", "L18", "L27", "L35", "LF"], "q_only", [])),
         ("qonly", (["L27", "LF"], "q_only", []))]
LABELS = ["answer", "allvals"]
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
    model.eval()
    embed = model.get_input_embeddings()
    base_lm = model.get_base_model()
    body, lm_head = base_lm.model, base_lm.lm_head

    gsm = Pool("gsm")
    held_rows = gsm.ok[-200:][:100]
    print(f"pool gsm ok {len(gsm.ok)}; held {len(held_rows)}")

    def encode(prompt, target):
        txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, enable_thinking=False, tokenize=False)
        pids = tok(txt, add_special_tokens=False).input_ids
        tids = tok(target, add_special_tokens=False).input_ids + [tok.eos_token_id]
        return pids + tids, [-100] * len(pids) + tids

    cells = {}
    for iname, spec in CELLS:
        for lab in LABELS:
            crng = _rnd.Random(zlib.crc32(f"{iname}|{lab}".encode()))
            samples = [build_sample(gsm, crng, mode="act", i=held_rows[j % len(held_rows)],
                                    kind=lab, spec=spec) for j in range(VN)]
            cells[f"{iname}|{lab}"] = [(encode(s["prompt"], s["target"]), s["vecs"]) for s in samples]
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
    print("FD1-QONLY5-DONE")


if __name__ == "__main__":
    main()
