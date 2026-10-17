"""No-CoT SFT baseline: plain Qwen3-8B LoRA finetune on EXACTLY the CODI training data
(zen-E/GSM8k-Aug-NL + zen-E/CommonsenseQA-GPT4omini, answer_only format) — no latent
machinery, no CoT distillation, no special tokens. This isolates what CODI's 6 thoughts
add over "just SFT the answers on the same data" (the CODI paper's No-CoT-SFT baseline,
never trained for our 8B organism).

Data formatting mirrors third_party/CODI/src/datasets.py (answer_only=True):
  GSM8k-Aug-NL: q + " Output only the answer and nothing else."  ->  last word of answer
                (skip if answer starts non-digit — their negative-sign filter — or
                 q+cot+answer > 512 tok)
  CSQA:         q.strip() + "\\n"  ->  answer letter   (their branch OVERWRITES the
                answer_only suffix for commonsense — mirrored deliberately)
Loss on answer tokens (+eos) only. Recipe matched to CODI: LoRA r128/a32/dropout .1 on
q,k,v,o,gate,up,down; lr 2e-4 cosine, warmup .03, wd .1, clip 2.0, eff batch 256, 5 ep.

Evals DURING training (rank 0, every --eval-steps): GSM8k test[:250] greedy first-int
acc, logged to wandb; halt on stationarity (<0.3pp improvement over the trailing eval
window after >=1 epoch); fail-early if <5% after 2000 steps. Resumable (save_steps +
auto resume_from_checkpoint).

  srun -p general --qos=high --gres=gpu:4 --cpus-per-task=48 --mem=400G --time=12:00:00 \
    --job-name=sq1_$USER bash -lc 'set -a; source .env; set +a; \
    export NCCL_NVLS_ENABLE=0 NCCL_SOCKET_IFNAME=vxlan0 HF_HOME=/workspace-vast/$USER/hf_cache; \
    /var/tmp/jbauer/venvs/activation_oracles_dev/bin/torchrun --nproc-per-node 4 \
    scripts/sft_qwen8b_codi_data.py'
"""
from __future__ import annotations

import argparse
import functools
import os
import re

import numpy as np
import torch

os.environ.setdefault("HF_HOME", f"/workspace-vast/{os.environ.get('USER', 'jbauer')}/hf_cache")
os.environ.setdefault("WANDB_PROJECT", "codi")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print = functools.partial(print, flush=True)

BASE = "Qwen/Qwen3-8B"
OUT_DIR = "/workspace-vast/jbauer/exp/training/sq1"
MAX_TOK = 512
EVAL_N = 250


def build_examples(tok, limit=0, cot=False):
    """cot=False -> No-CoT SFT targets (answer only, answer_only prompt suffix).
    cot=True  -> CoT-SFT targets (write the chain, then 'The answer is: X'; bare
    question, mirroring the CODI paper's SFT-CoT / datasets.py non-answer_only)."""
    from datasets import load_dataset
    rows = []
    ds = load_dataset("zen-E/GSM8k-Aug-NL", split="train")
    for ex in ds:
        if ex["answer"] is None:
            continue
        answer = ex["answer"].split(" ")[-1].replace("####", "")
        if not answer or not answer[0].isdigit():
            continue
        if len(tok.encode(ex["question"] + ex["cot"] + ex["answer"])) > MAX_TOK:
            continue
        if cot:
            rows.append((ex["question"], f"{ex['cot'].strip()}\nThe answer is: {answer}"))
        else:
            rows.append((f"{ex['question']} Output only the answer and nothing else.", answer))
        if limit and len(rows) >= limit // 2:
            break
    n_gsm = len(rows)
    ds = load_dataset("zen-E/CommonsenseQA-GPT4omini", split="train")
    for ex in ds:
        q, c, ans = ex["question"].strip() + "\n", ex["cot"].strip(), str(ex["answer"]).strip()
        if len(tok.encode(q + " " + c + " " + ans)) > MAX_TOK:
            continue
        rows.append((q, f"{c}\nThe answer is: {ans}" if cot else ans))
        if limit and len(rows) >= limit:
            break
    print(f"[data] {n_gsm} gsm8k-aug-nl + {len(rows) - n_gsm} csqa = {len(rows)} (cot={cot})")
    return rows


class SFTData(torch.utils.data.Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        q, a = self.rows[i]
        qi = self.tok.encode(q, add_special_tokens=False)
        ai = self.tok.encode(" " + a, add_special_tokens=False) + [self.tok.eos_token_id]
        ids = (qi + ai)[:MAX_TOK]
        labels = ([-100] * len(qi) + ai)[:MAX_TOK]
        return dict(input_ids=ids, labels=labels)


def collate(feats, pad_id):
    L = max(len(f["input_ids"]) for f in feats)
    ids = torch.full((len(feats), L), pad_id, dtype=torch.long)
    lab = torch.full((len(feats), L), -100, dtype=torch.long)
    att = torch.zeros((len(feats), L), dtype=torch.long)
    for j, f in enumerate(feats):
        n = len(f["input_ids"])
        ids[j, :n] = torch.tensor(f["input_ids"]); lab[j, :n] = torch.tensor(f["labels"])
        att[j, :n] = 1
    return dict(input_ids=ids, labels=lab, attention_mask=att)


def first_int(s):
    m = re.search(r"-?\d[\d,]*", s)
    return m.group(0).replace(",", "") if m else ""


def last_int(s):
    m = re.findall(r"-?\d[\d,]*", s)
    return m[-1].replace(",", "") if m else ""


class GenEvalCallback:
    """Rank-0 greedy generation eval on GSM8k test[:EVAL_N]; stationarity halt +
    fail-early. Other ranks idle at the next NCCL sync while rank 0 generates."""

    def __init__(self, tok, model, cot=False):
        from datasets import load_dataset
        from transformers import TrainerCallback
        ds = load_dataset("gsm8k", "main")["test"]
        self.prompts = [ds[i]["question"].strip() if cot else
                        f"{ds[i]['question'].strip()} Output only the answer and nothing "
                        f"else." for i in range(EVAL_N)]
        self.golds = [ds[i]["answer"].split("####")[-1].strip().replace(",", "")
                      for i in range(EVAL_N)]
        self.max_new = 192 if cot else 8
        self.tok, self.model, self.hist = tok, model, []
        cb = self

        class _CB(TrainerCallback):
            def on_step_end(self, args, state, control, **kw):
                if state.global_step % args.eval_steps or not state.is_world_process_zero:
                    return control
                acc = cb.run(state.global_step)
                cb.hist.append((state.global_step, acc))
                if state.global_step >= 2000 and max(a for _, a in cb.hist) < 0.05:
                    raise RuntimeError(f"FAIL-EARLY: eval acc {acc:.3f} still <5% at "
                                       f"step {state.global_step} — redesign, don't rerun")
                if len(cb.hist) >= 8 and state.epoch and state.epoch >= 1.0:
                    recent = max(a for _, a in cb.hist[-3:])
                    before = max(a for _, a in cb.hist[:-3])
                    if recent - before < 0.003:
                        print(f"[halt] stationary: last3 max {recent:.4f} vs prior max "
                              f"{before:.4f} — stopping at step {state.global_step}")
                        control.should_training_stop = True
                return control
        self.cb = _CB()

    @torch.no_grad()
    def run(self, step):
        self.model.eval()
        ok = 0
        for s in range(0, EVAL_N, 64):
            batch = self.tok(self.prompts[s:s + 64], return_tensors="pt", padding=True,
                             padding_side="left").to(self.model.device)
            g = self.model.generate(**batch, max_new_tokens=self.max_new, do_sample=False,
                                    pad_token_id=self.tok.eos_token_id)
            for j, row in enumerate(g[:, batch["input_ids"].shape[1]:]):
                ok += last_int(self.tok.decode(row, skip_special_tokens=True)) == \
                    self.golds[s + j]
        acc = ok / EVAL_N
        print(f"[eval] step {step}: gsm8k-test[:{EVAL_N}] greedy acc {acc:.4f}")
        try:
            import wandb
            if wandb.run:
                wandb.log({"eval/gsm8k_test250_acc": acc}, step=step)
        except Exception:
            pass
        self.model.train()
        return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--cot", action="store_true", help="CoT-SFT variant (sq2)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or ("/workspace-vast/jbauer/exp/training/sq2" if args.cot else OUT_DIR)
    tag = "sq2" if args.cot else "sq1"

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from transformers.trainer_utils import get_last_checkpoint

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows = build_examples(tok, args.limit, cot=args.cot)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(
        r=128, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                        "down_proj"]))
    model.print_trainable_parameters()

    world = int(os.environ.get("WORLD_SIZE", 1))
    per_dev = 16
    accum = max(1, 256 // (per_dev * world))
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=per_dev, gradient_accumulation_steps=accum,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.1, max_grad_norm=2.0, bf16=True, logging_steps=20,
        eval_steps=args.eval_steps, save_steps=250, save_total_limit=2,
        report_to="wandb", run_name=f"{tag}_{'cot' if args.cot else 'nocot'}_sft", seed=11,
        dataloader_num_workers=4, remove_unused_columns=False)
    os.environ.setdefault("WANDB_RUN_GROUP", tag)

    ev = GenEvalCallback(tok, model, cot=args.cot)
    trainer = Trainer(model=model, args=targs, train_dataset=SFTData(rows, tok),
                      data_collator=lambda f: collate(f, tok.pad_token_id
                                                      or tok.eos_token_id),
                      callbacks=[ev.cb])
    last = get_last_checkpoint(args.out) if os.path.isdir(args.out) else None
    print(f"[train] world={world} per_dev={per_dev} accum={accum} resume={last}")
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(os.path.join(args.out, "final"))
    if trainer.is_world_process_zero():
        tok.save_pretrained(os.path.join(args.out, "final"))
    # no rank-0-only final eval here: the halt-step in-training eval already measured
    # it, and a long solo generation overruns the other ranks' NCCL watchdog (600s)
    # at the exit barrier (sq2 lesson; sq1's SIGABRT was the barrier-less variant)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    print("SFT-DONE")


if __name__ == "__main__":
    main()
