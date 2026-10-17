"""Base Qwen3-8B (no CODI) necessity labels for every GSM8k TEST instance — the
"default Qwen" filter axis for the codi_writeup controls viewer.

Protocol lifted from scripts/jlens_loadbearing_select.py (terse non-thinking CoT;
thinking mode truncates ~40% and depresses the CoT arm), run on ALL 1319 test rows:
  A: greedy no-CoT ("<q> Output only the answer and nothing else.", 12 tok)
  B: greedy CoT    (chat template, thinking OFF, terse step-by-step, <=320 tok)
  C: K=4 resamples (T=0.7) of both arms on ALL rows
  role: load_bearing (no-CoT greedy wrong ∧ res<=1/4 ∧ CoT greedy right ∧ res>=3/4)
        easy         (no-CoT greedy right ∧ res>=3/4 ∧ CoT greedy right)
        other        (everything else)

-> reports/codi_writeup/base_qwen_test.json  {meta, rows: {i: {...}}} keyed by test index,
   aligned with controls_instances.json's `i`.

srun -p general --qos=high --gres=gpu:1 --mem=96G --time=1:30:00 --job-name=codi_wr_$USER \
  bash -lc 'set -a; source .env; set +a; bash scripts/slurm/aod_venv_run.sh scripts/qwen_base_gsm8k_test.py'
"""
from __future__ import annotations

import functools
import json
import os
import re

import numpy as np

print = functools.partial(print, flush=True)
os.environ.setdefault("HF_HOME", f"/workspace-vast/{os.environ.get('USER', 'jbauer')}/hf_cache")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_REPO, "reports", "codi_writeup", "base_qwen_test.json")
BASE = "Qwen/Qwen3-8B"
COT_MAX_NEW = 320
K_RES = 4
T_RES = 0.7
COT_INSTR = ("\nSolve it step by step. Write each step as one short sentence that shows the "
             "arithmetic, e.g. 'She sold 48/2 = 24 clips in May.' Then give the final numeric "
             "answer on its own last line as '#### <number>'.")


def first_int(s):
    m = re.search(r"-?\d[\d,]*", s)
    return m.group(0).replace(",", "") if m else ""


def last_int(s):
    m = re.findall(r"-?\d[\d,]*", s)
    return m[-1].replace(",", "") if m else ""


def cot_pred(text):
    after = text.split("####")[-1] if "####" in text else text
    return first_int(after) or last_int(text)


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_dataset("gsm8k", "main")["test"]
    golds = [ex["answer"].split("####")[-1].strip().replace(",", "") for ex in ds]
    questions = [ex["question"].strip().replace("  ", " ") for ex in ds]
    n = len(ds)

    tok = AutoTokenizer.from_pretrained(BASE)
    llm = LLM(model=BASE, max_model_len=800, gpu_memory_utilization=0.5,
              dtype="bfloat16", enforce_eager=True)
    nocot_prompts = [q + " Output only the answer and nothing else." for q in questions]
    cot_prompts = [tok.apply_chat_template(
        [{"role": "user", "content": q + COT_INSTR}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in questions]

    rows = {i: {} for i in range(n)}
    outs = llm.generate(nocot_prompts, SamplingParams(temperature=0.0, max_tokens=12))
    for i, o in enumerate(outs):
        rows[i]["nocot_ok"] = first_int(o.outputs[0].text) == golds[i]
    outs = llm.generate(cot_prompts, SamplingParams(temperature=0.0, max_tokens=COT_MAX_NEW))
    for i, o in enumerate(outs):
        g = o.outputs[0]
        rows[i]["cot_ok"] = g.finish_reason == "stop" and cot_pred(g.text) == golds[i]
    trunc = np.mean([o.outputs[0].finish_reason != "stop" for o in outs])
    print(f"[greedy] no-CoT {sum(r['nocot_ok'] for r in rows.values())}/{n}  "
          f"CoT {sum(r['cot_ok'] for r in rows.values())}/{n}  truncated {trunc:.3f}")
    assert trunc < 0.05, f"CoT truncation {trunc:.3f} too high"

    outs = llm.generate(nocot_prompts, SamplingParams(temperature=T_RES, n=K_RES, max_tokens=12))
    for i, o in enumerate(outs):
        rows[i]["nocot_res_acc"] = float(np.mean([first_int(g.text) == golds[i] for g in o.outputs]))
    outs = llm.generate(cot_prompts, SamplingParams(temperature=T_RES, n=K_RES, max_tokens=COT_MAX_NEW))
    for i, o in enumerate(outs):
        rows[i]["cot_res_acc"] = float(np.mean(
            [(g.finish_reason == "stop") and cot_pred(g.text) == golds[i] for g in o.outputs]))

    for i, r in rows.items():
        if (not r["nocot_ok"] and r["nocot_res_acc"] <= 1 / K_RES
                and r["cot_ok"] and r["cot_res_acc"] >= 3 / K_RES):
            r["role"] = "load_bearing"
        elif r["nocot_ok"] and r["nocot_res_acc"] >= 3 / K_RES and r["cot_ok"]:
            r["role"] = "easy"
        else:
            r["role"] = "other"

    counts = {}
    for r in rows.values():
        counts[r["role"]] = counts.get(r["role"], 0) + 1
    with open(OUT, "w") as f:
        json.dump(dict(model=BASE, n=n, k_res=K_RES, t_res=T_RES, cot_max_new=COT_MAX_NEW,
                       role_counts=counts, rows=rows), f)
    print(f"[out] {OUT}  roles={counts}")
    print("BASEQ-DONE")


if __name__ == "__main__":
    main()
