"""Faithful Paper 1 (Liang & Pan 2602.00449) ACTIVATION PATCHING on our CODI Qwen3-8B.

Their `systematic_latent_patching` (model_interp.py): corrupt an input number, then for
each (layer resid_post x phase/position) restore the CLEAN activation and measure how
much the correct answer recovers — lift = (patched - corrupted)/(clean - corrupted).
Phases = encoder tokens, BoT, latent_0..5, EoT/ANS.

Ported to our HF CODI via a manual latent forward loop (mirrors CODI.generate's thinking
phase) with forward hooks on the decoder layers, so we can both CAPTURE resid_post at
each (layer, thought position) on the clean run and OVERWRITE it on the corrupted run.

Corruption = swap one single-digit token in the QUESTION for a different digit (positions
stay aligned, so the latent positions match between clean & corrupted). We keep only
problems where the clean answer is correct AND the corruption changes the model's answer.
Metric = P(clean-answer first token) at the answer readout.

  srun -p general --qos=high --gres=gpu:1 --mem=128G --time=1:30:00 \
    --job-name=codi_p1pat_$USER bash -lc 'set -a; source .env; set +a; \
    HF_HOME=/workspace-vast/$USER/hf_cache \
    /var/tmp/jbauer/venvs/activation_oracles_dev/bin/python \
    scripts/codi_paper1_patching.py --n 600 --keep 80'
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
NUM_LATENT = 6
POS = ["Enc"] + [f"L{i + 1}" for i in range(NUM_LATENT)]
LAYER_SUBSET = [1, 5, 9, 14, 18, 23, 27, 31, 35]  # Qwen3-8B has 36 decoder layers (0..35)
OUT_DIR = os.path.join(_REPO, "reports", "codi_paper_faithful")


def parse_numbers(text):
    out, seen = [], set()
    for s in re.findall(r"-?\d+(?:\.\d+)?", text):
        v = (s.rstrip("0").rstrip(".") or "0") if "." in s else s
        if v not in seen:
            seen.add(v); out.append(v)
    return out


class CodiRunner:
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.codi = model.codi
        self.layers = model.codi.model.model.layers
        self.bocot = tok.convert_tokens_to_ids("<|bocot|>")
        self.eocot = tok.convert_tokens_to_ids("<|eocot|>")
        self.vocab = model.codi.config.vocab_size
        self._state = {"pos": None}
        self._cap = None          # dict (L,p)->vec when capturing
        self._patch = None        # (L, p, vec) when patching
        self._handles = []

    def _hook(self, L):
        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            p = self._state["pos"]
            if p is not None:
                if self._cap is not None:
                    self._cap[(L, p)] = h[:, -1, :].detach().clone()   # keep on GPU; no per-layer sync
                if self._patch is not None and (L, p) in self._patch:
                    h[:, -1, :] = torch.as_tensor(self._patch[(L, p)], device=h.device, dtype=h.dtype)
            return out
        return fn

    @torch.no_grad()
    def run(self, input_ids, capture=False, patch=None):
        self._cap = {} if capture else None
        self._patch = patch
        self._handles = [ly.register_forward_hook(self._hook(L)) for L, ly in enumerate(self.layers)]
        try:
            dev = input_ids.device
            bot = torch.tensor([[self.tok.eos_token_id, self.bocot]], device=dev)
            ids_bot = torch.cat([input_ids, bot], dim=1)
            self._state["pos"] = 0
            o = self.codi(input_ids=ids_bot, use_cache=True, output_hidden_states=True)
            pkv = o.past_key_values
            last = o.hidden_states[-1][:, -1, :].unsqueeze(1)
            latent = self.model.prj(last).to(self.codi.dtype) if self.model.use_prj else last
            for i in range(NUM_LATENT):
                self._state["pos"] = i + 1
                o = self.codi(inputs_embeds=latent, use_cache=True, output_hidden_states=True, past_key_values=pkv)
                pkv = o.past_key_values
                last = o.hidden_states[-1][:, -1, :].unsqueeze(1)
                latent = self.model.prj(last).to(self.codi.dtype) if self.model.use_prj else last
            self._state["pos"] = None
            eot_emb = self.codi.get_input_embeddings()(torch.tensor([[self.eocot]], device=dev))
            o = self.codi(inputs_embeds=eot_emb, use_cache=True, past_key_values=pkv)
            ans_logits = o.logits[:, -1, : self.vocab].float().cpu()[0]
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []
        return ans_logits, self._cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--keep", type=int, default=80)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    from datasets import load_dataset
    from src.model import CODI
    model = CODI.from_pretrained(checkpoint_path=CKPT, model_name_or_path=BASE, lora_r=128, lora_alpha=32,
                                 num_latent=NUM_LATENT, use_prj=True, prj_dim=4096, prj_no_ln=False,
                                 prj_dropout=0.0, dtype="bfloat16")
    model.eval()
    tok = model.tokenizer
    R = CodiRunner(model, tok)

    # single-char digit token ids (for in-place corruption that preserves positions)
    digit_tok = {}
    for d in "0123456789":
        ids = tok(d, add_special_tokens=False).input_ids
        if len(ids) == 1:
            digit_tok[d] = ids[0]
    digit_ids = set(digit_tok.values())

    ds = load_dataset("zen-E/GSM8k-Aug-NL", split="train")
    lift_sum = np.zeros((len(LAYER_SUBSET), len(POS)))
    n_used = 0
    for di in range(args.n):
        if n_used >= args.keep:
            break
        ex = ds[di]
        gold = str(ex["answer"]).strip()
        q = ex["question"].strip().replace("  ", " ") + " Output only the answer and nothing else."
        ids = tok(q, return_tensors="pt").input_ids.to("cuda")
        toks = ids[0].tolist()
        # candidate question digit positions (exclude the fixed suffix region: last ~8 tokens)
        cand = [i for i in range(len(toks) - 8) if toks[i] in digit_ids]
        if not cand:
            continue
        clean_logits, clean_cap = R.run(ids, capture=True)
        clean_tok = int(clean_logits.argmax())
        clean_pred = tok.decode([clean_tok]).strip()
        if clean_pred != gold[: len(clean_pred)] or clean_pred == "":
            continue   # keep only clean-correct (first-token of gold)
        # corrupt: change one question digit to a different digit
        cpos = cand[len(cand) // 2]
        cur = toks[cpos]
        alt = next((t for d, t in digit_tok.items() if t != cur), None)
        if alt is None:
            continue
        cids = ids.clone(); cids[0, cpos] = alt
        corr_logits, _ = R.run(cids, capture=False)

        def metric(lg):
            return float(lg.softmax(-1)[clean_tok])
        c_clean, c_corr = metric(clean_logits), metric(corr_logits)
        if c_clean - c_corr < 0.05:
            continue   # corruption didn't move the answer enough -> uninformative
        # patch each (layer, position) clean->corrupted
        lift = np.zeros((len(LAYER_SUBSET), len(POS)))
        for li, L in enumerate(LAYER_SUBSET):
            for pi in range(len(POS)):
                vec = clean_cap[(L, pi)]
                patched, _ = R.run(cids, capture=False, patch={(L, pi): vec})
                lift[li, pi] = (metric(patched) - c_corr) / (c_clean - c_corr)
        lift_sum += lift
        n_used += 1
        if n_used <= 3 or n_used % 20 == 0:
            print(f"[{n_used}] di={di} clean={clean_pred} corr_drop={c_clean-c_corr:.2f} "
                  f"max_lift={lift.max():.2f}@{POS[lift.max(0).argmax()]}")

    lift_avg = (lift_sum / max(1, n_used))
    json.dump(dict(layers=LAYER_SUBSET, positions=POS, n=n_used, lift=lift_avg.tolist()),
              open(os.path.join(OUT_DIR, "patching.json"), "w"))
    print(f"\n[patching] n_used={n_used}")
    print("mean recovery lift (layer x position):")
    print("        " + "  ".join(f"{p:>5}" for p in POS))
    for li, L in enumerate(LAYER_SUBSET):
        print(f"  L{L:2d}: " + "  ".join(f"{lift_avg[li,pi]:5.2f}" for pi in range(len(POS))))
    _plot(lift_avg)
    print("P1-PATCH-DONE")


def _plot(lift):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(lift, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean recovery lift")
    ax.set_xticks(range(len(POS))); ax.set_xticklabels(POS)
    ax.set_yticks(range(len(LAYER_SUBSET))); ax.set_yticklabels([f"L{L}" for L in LAYER_SUBSET])
    ax.set_xlabel("patched thought position"); ax.set_ylabel("patched resid_post layer")
    ax.set_title("Activation patching: clean->corrupted answer recovery")
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "p1_patching.png")
    plt.savefig(png, dpi=150, bbox_inches="tight"); print(f"PNG: {png}")


if __name__ == "__main__":
    main()
