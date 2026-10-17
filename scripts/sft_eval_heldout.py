"""Held-out eval of the no-CoT SFT baseline (sq1) on the same battery as
codi_heldout_tasks.py: gsm8k / svamp / multi-arith / gsm-hard / commonsense, greedy,
identical question formatting and answer extraction. One mode (the model only knows
direct answering — that is the point of the baseline).

-> reports/codi_writeup/sft_baseline.json  {tasks: {task: {n, accuracy}}, instances}

srun -p general --qos=high --gres=gpu:1 --mem=96G --time=2:00:00 --job-name=sq1_$USER \
  bash -lc 'set -a; source .env; set +a; bash scripts/slurm/aod_venv_run.sh \
  scripts/sft_eval_heldout.py'
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
os.environ.setdefault("HF_HOME", f"/workspace-vast/{os.environ.get('USER', 'jbauer')}/hf_cache")
print = functools.partial(print, flush=True)

from codi_heldout_tasks import TASKS, extract_pred, load_task  # noqa: E402

ADAPTER = "/workspace-vast/jbauer/exp/training/sq1/final"
BASE = "Qwen/Qwen3-8B"
OUT = os.path.join(_REPO, "reports", "codi_writeup", "sft_baseline.json")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--cot", action="store_true",
                    help="CoT-SFT adapter (sq2): bare question, long generation")
    args = ap.parse_args()
    out_path = OUT.replace("sft_baseline", "sftcot_baseline") if args.cot else OUT
    if args.cot and args.adapter == ADAPTER:
        args.adapter = "/workspace-vast/jbauer/exp/training/sq2/final"

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa",
                                                 device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    from tqdm.auto import tqdm
    summary, instances = {}, {}
    for task in TASKS:
        questions, golds = load_task(task)
        if args.limit:
            questions, golds = questions[: args.limit], golds[: args.limit]
        # CSQA trains WITHOUT the answer_only suffix in the CODI pipeline — mirror that;
        # the CoT-SFT variant trains on BARE questions everywhere
        if task == "commonsense":
            questions = [q.rsplit(" Output only the answer", 1)[0].strip() + "\n"
                         for q in questions]
        elif args.cot:
            questions = [q.rsplit(" Output only the answer", 1)[0] for q in questions]
        max_new = 256 if args.cot else 12
        preds = []
        for s in tqdm(range(0, len(questions), args.batch_size), desc=task, leave=False):
            batch = tok(questions[s:s + args.batch_size], return_tensors="pt",
                        padding=True, padding_side="left").to(model.device)
            g = model.generate(**batch, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            for row in g[:, batch["input_ids"].shape[1]:]:
                preds.append(tok.decode(row, skip_special_tokens=True))
        ok = [extract_pred(p, task) == g for p, g in zip(preds, golds)]
        summary[task] = {"n": len(questions), "accuracy": float(np.mean(ok))}
        instances[task] = [{"i": i, "pred": preds[i].strip()[:400 if args.cot else 40],
                            "ok": bool(ok[i])} for i in range(len(questions))]
        print(f"[{task}] n={len(questions)} acc {100 * np.mean(ok):.2f}%")

    with open(out_path, "w") as f:
        json.dump({"adapter": args.adapter, "base": BASE, "cot": args.cot,
                   "tasks": summary, "instances": instances}, f)
    print(f"[out] {out_path}")
    print("SFT-EVAL-DONE")


if __name__ == "__main__":
    main()
