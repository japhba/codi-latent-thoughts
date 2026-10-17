"""Per-task held-out eval of the CODI Qwen3-8B organism (ckpt-7580): latent (6 thoughts)
vs direct (skip_thinking) vs verbalized CoT, on the CODI paper's eval battery.

Training tasks were GSM8k-Aug-NL + CommonsenseQA; everything here is a *test* set the
model never saw, and GSM-Hard / SVAMP / MultiArith are additionally *task-level* held-out
(different distribution from the training task). The question this answers: aggregate
latent ~ verbal on GSM8k, but do SOME tasks show a real latent-over-direct gain — as the
LessWrong post reports for Llama-1B (GSM8k +4.88 / CSQA -0.25 / MultiArith +15.55 /
SVAMP +2.73)?

Modes (all single-sample):
  latent        CODI.generate(num_latent_iterations=6, greedy=True)
  direct        CODI.generate(skip_thinking=True, greedy=True)   [eos, <|eocot|>] path
  verbal_greedy CODI.generate(verbalize_cot=True, greedy=True)   known to be degraded
  verbal_t01    CODI.generate(verbalize_cot=True, temperature=0.1, top_k=40, top_p=0.95)

Datasets + fields verbatim from third_party/CODI/test.py; answer extraction = the fork's
(LAST number, float compare; CSQA = last character letter). Per-instance records incl.
the verbal CoT text go to reports/codi_writeup/heldout_instances.json for the viewer.

srun -p general --qos=high --gres=gpu:1 --mem=128G --time=6:00:00 --job-name=codi_wr_$USER \
  bash -lc 'set -a; source .env; set +a; HF_HOME=/workspace-vast/$USER/hf_cache \
  /var/tmp/jbauer/venvs/activation_oracles_dev/bin/python scripts/codi_heldout_tasks.py'
"""
from __future__ import annotations

import argparse
import functools
import json
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
NL = 6
DEV = "cuda"
OUT_DIR = os.path.join(_REPO, "reports", "codi_writeup")
TASKS = ["gsm8k", "svamp", "multi-arith", "gsm-hard", "commonsense"]
MODES = ["latent", "direct", "verbal_greedy", "verbal_t01"]


def load_task(name):
    from datasets import concatenate_datasets, load_dataset
    qn, an = "question", "answer"
    if name == "gsm-hard":
        test_set = load_dataset("juyoung-trl/gsm-hard")["train"]
        qn, an = "instruction", "response"
    elif name == "multi-arith":
        test_set = load_dataset("ChilleD/MultiArith")["test"]
        an = "final_ans"
    elif name == "svamp":
        d = load_dataset("ChilleD/SVAMP")
        test_set = concatenate_datasets([d["train"], d["test"]])
        qn, an = "question_concat", "Answer"
    elif name == "commonsense":
        test_set = load_dataset("zen-E/CommonsenseQA-GPT4omini")["validation"]
    elif name == "gsm8k":
        test_set = load_dataset("gsm8k", "main")["test"]
    questions, golds = [], []
    for ex in test_set:
        questions.append(f"{ex[qn].strip().replace('  ', ' ')} Output only the answer "
                         f"and nothing else.")
        a = ex[an]
        if isinstance(a, str) and a in "ABCDE":
            golds.append(a)
        else:
            if isinstance(a, str) and "####" in a:
                a = a.split("####")[-1]
            golds.append(float(str(a).replace(",", "")))
    return questions, golds


def extract_pred(sentence, task):
    if task == "commonsense":
        s = sentence.strip()
        return s[-1] if s else ""
    sentence = sentence.replace(",", "")
    pred = re.findall(r"-?\d+\.?\d*", sentence)
    return float(pred[-1]) if pred else float("inf")


@torch.no_grad()
def run_mode(model, tok, questions, mode, batch_size, max_new):
    outs = []
    n_batches = (len(questions) + batch_size - 1) // batch_size
    from tqdm.auto import tqdm
    for s in tqdm(range(n_batches), desc=mode, leave=False):
        bq = questions[s * batch_size:(s + 1) * batch_size]
        batch = tok(bq, return_tensors="pt", padding="longest", padding_side="left")
        kw = dict(input_ids=batch["input_ids"].to(DEV),
                  attention_mask=batch["attention_mask"].to(DEV), tokenizer=tok,
                  max_new_tokens=max_new, num_latent_iterations=NL,
                  return_latent_vectors=False, sot_token=tok.bot_id, eot_token=tok.eot_id,
                  greedy=True, temperature=0.1, top_k=40, top_p=0.95)
        if mode == "direct":
            kw.update(skip_thinking=True)
        elif mode == "verbal_greedy":
            kw.update(verbalize_cot=True)
        elif mode == "verbal_t01":
            kw.update(verbalize_cot=True, greedy=False)
        out = model.generate(**kw)
        for seq in out["sequences"]:
            t = [x for x in seq.tolist() if x != tok.pad_token_id]
            if tok.eos_token_id in t:
                t = t[: t.index(tok.eos_token_id)]
            outs.append(tok.decode(t, skip_special_tokens=False))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tasks", default=",".join(TASKS))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(11)

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

    summary, instances = {}, {}
    for task in args.tasks.split(","):
        questions, golds = load_task(task)
        if args.limit:
            questions, golds = questions[: args.limit], golds[: args.limit]
        print(f"\n[{task}] n={len(questions)}")
        texts = {}
        for mode in MODES:
            max_new = 256 if mode.startswith("verbal") else 64
            texts[mode] = run_mode(model, tok, questions, mode, args.batch_size, max_new)
            acc = np.mean([extract_pred(t, task) == g for t, g in zip(texts[mode], golds)])
            print(f"  {mode:>14}: {100 * acc:.2f}%")
        summary[task] = {"n": len(questions),
                         "accuracy": {m: float(np.mean([extract_pred(t, task) == g
                                                        for t, g in zip(texts[m], golds)]))
                                      for m in MODES}}
        instances[task] = [
            {"i": i, "question": questions[i].rsplit(" Output only", 1)[0],
             "gold": golds[i] if isinstance(golds[i], str) else
                     (int(golds[i]) if float(golds[i]).is_integer() else golds[i]),
             "pred": {m: (lambda v: v if isinstance(v, str) else
                          (None if v == float("inf") else
                           (int(v) if float(v).is_integer() else v)))(
                          extract_pred(texts[m][i], task)) for m in MODES},
             "verbal_text": texts["verbal_t01"][i][:600]}
            for i in range(len(questions))]
        with open(os.path.join(OUT_DIR, "heldout_summary.json"), "w") as f:
            json.dump({"ckpt": CKPT, "modes": MODES, "tasks": summary}, f, indent=1)
        with open(os.path.join(OUT_DIR, "heldout_instances.json"), "w") as f:
            json.dump(instances, f)

    print("\n[heldout] latent - direct deltas (pp):")
    for task, s in summary.items():
        d = 100 * (s["accuracy"]["latent"] - s["accuracy"]["direct"])
        print(f"  {task:>12}: {d:+.2f}  (latent {100 * s['accuracy']['latent']:.2f} / "
              f"direct {100 * s['accuracy']['direct']:.2f}, n={s['n']})")
    print("HELDOUT-DONE")


if __name__ == "__main__":
    main()
