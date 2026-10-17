"""Faithful Paper 1 (Liang & Pan 2602.00449) interp on our CODI Qwen3-8B: EXACT
logit-lens layer-sweep + LINEAR PROBES.

Their notebook (third_party/codi_interp/latent-cot-thinking/notebook/
activation_patching_logit_lens.ipynb):
  * logit-lens = for each layer l: ln_final(resid_post[l, pos]) -> unembed -> softmax,
    then read the *probability of the GT intermediate token* at each position; build an
    (intermediate x position) heatmap over CORRECT examples, plus prob-by-layer curves.
  * probes = a single linear layer (no bias, no nonlinearity) on frozen activations,
    64/16/20 split, predicting the intermediate / input states.

Ported faithfully to our HF CODI (full layer sweep over all 37 hidden states, at the 7
"thought" positions Enc + L1..L6). GSM8k intermediates are multi-token, so the per-token
target is the GT value's FIRST token (the single-token analog of their mod-50 states).

Targets (each gives a layer x position matrix): final ANSWER, FIRST intermediate, LAST
intermediate. logit-lens reports mean P(target first-token); the probe reports held-out
classification accuracy of the target first-token (vs a majority-class baseline) — the
learned-decodability complement to the model's own unembedding direction.

  srun -p general --qos=high --gres=gpu:1 --mem=128G --time=1:00:00 \
    --job-name=codi_p1pl_$USER bash -lc 'set -a; source .env; set +a; \
    HF_HOME=/workspace-vast/$USER/hf_cache \
    /var/tmp/jbauer/venvs/activation_oracles_dev/bin/python \
    scripts/codi_paper1_probes_logitlens.py --n 1200 --offset 4000'
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
from collections import Counter

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
OUT_DIR = os.path.join(_REPO, "reports", "codi_paper_faithful")
ACTS_NPY = os.path.join(_REPO, "data", "codi_faithful_acts.npy")
META_JSON = os.path.join(_REPO, "data", "codi_faithful_meta.json")


def parse_numbers(text: str) -> list[str]:
    out, seen = [], set()
    for s in re.findall(r"-?\d+(?:\.\d+)?", text):
        v = (s.rstrip("0").rstrip(".") or "0") if "." in s else s
        if v not in seen:
            seen.add(v); out.append(v)
    return out


@torch.no_grad()
def extract(args):
    """Run CODI on N problems; cache resid at all layers x the 7 thought positions."""
    from datasets import load_dataset
    from src.model import CODI
    model = CODI.from_pretrained(checkpoint_path=CKPT, model_name_or_path=BASE, lora_r=128,
                                 lora_alpha=32, num_latent=NUM_LATENT, use_prj=True, prj_dim=4096,
                                 prj_no_ln=False, prj_dropout=0.0, dtype="bfloat16")
    model.eval()
    tok = model.tokenizer
    bocot, eocot = tok.convert_tokens_to_ids("<|bocot|>"), tok.convert_tokens_to_ids("<|eocot|>")
    ds = load_dataset("zen-E/GSM8k-Aug-NL", split="train")

    acts, meta = [], []
    for di in range(args.offset, min(args.offset + args.n, len(ds))):
        ex = ds[di]
        gold = str(ex["answer"]).strip()
        chain = [x for x in parse_numbers(ex["cot"]) if x != gold]
        if not chain:
            continue
        q = ex["question"].strip().replace("  ", " ") + " Output only the answer and nothing else."
        ids = tok(q, return_tensors="pt").input_ids.to("cuda")
        out = model.generate(input_ids=ids, tokenizer=tok, max_new_tokens=4, num_latent_iterations=NUM_LATENT,
                             greedy=True, output_hidden_states=True, return_latent_vectors=False,
                             sot_token=bocot, eot_token=eocot)
        hs = out["hidden_states"]                         # [seq, n_layers, hidden]
        prompt_len = ids.shape[1] + 2
        pos_idx = [prompt_len - 1] + list(range(prompt_len, prompt_len + NUM_LATENT))
        a = hs[pos_idx, :, :].float().cpu().numpy().astype(np.float16)   # [7, n_layers, hidden]
        gen = tok.decode(out["sequences"][0], skip_special_tokens=True).strip()
        pred = parse_numbers(gen)
        acts.append(a)
        meta.append(dict(ds_index=di, gold=gold, chain=chain, hops=len(chain),
                         pred=pred[0] if pred else "", correct=bool(pred) and pred[0] == gold))
    A = np.stack(acts)                                    # [N, 7, n_layers, hidden]
    os.makedirs(os.path.dirname(ACTS_NPY), exist_ok=True)
    np.save(ACTS_NPY, A)
    json.dump(meta, open(META_JSON, "w"))
    n_layers = A.shape[2]
    acc = np.mean([m["correct"] for m in meta])
    print(f"[extract] saved acts {A.shape} (n_layers={n_layers}); latent acc={acc:.3f}; "
          f"hop dist={dict(Counter(m['hops'] for m in meta))}")
    return A, meta, model, tok


def first_digit_tok(tok, v: str) -> int:
    """Token id of the number's leading DIGIT char. GSM8k values are multi-token and
    Qwen prefixes a shared whitespace token, so 'first token of the string' is
    degenerate; the model decodes latents to bare single-digit tokens, so we target the
    leading digit (10-way) — the single-token analog of Paper 1's mod-50 states."""
    d = re.search(r"\d", v)
    return tok(d.group(), add_special_tokens=False).input_ids[0]


def target_token(tok, m, which: str):
    if which == "answer":
        return first_digit_tok(tok, m["gold"])
    if which == "first_int":
        return first_digit_tok(tok, m["chain"][0])
    if which == "last_int":
        return first_digit_tok(tok, m["chain"][-1])


@torch.no_grad()
def logit_lens(A, meta, model, tok):
    """Mean P(target first-token) at each (layer, position), over CORRECT examples."""
    norm = model.codi.model.model.norm
    head = model.codi.get_output_embeddings()
    N, P, Lc, H = A.shape
    correct = np.array([m["correct"] for m in meta])
    res = {}
    dev = head.weight.device
    for which in ("answer", "first_int", "last_int"):
        tgt = np.array([target_token(tok, m, which) for m in meta])
        mat = np.zeros((Lc, P))
        for li in range(Lc):
            for pi in range(P):
                x = torch.from_numpy(A[:, pi, li, :].astype(np.float32)).to(dev, head.weight.dtype)
                logits = head(norm(x))                    # [N, vocab]
                p = logits.float().softmax(-1)
                pt = p[torch.arange(N), torch.from_numpy(tgt).to(p.device)]
                mat[li, pi] = float(pt.cpu().numpy()[correct].mean())
        res[which] = mat.tolist()
    json.dump(res, open(os.path.join(OUT_DIR, "logit_lens_sweep.json"), "w"))
    return res


def train_probe(X, y, n_classes, dev, epochs=300):
    """Single linear layer, no bias/nonlinearity (Paper 1). Returns held-out accuracy."""
    n = len(y); idx = np.random.RandomState(0).permutation(n)
    ntr, nval = int(0.64 * n), int(0.16 * n)
    tr, va, te = idx[:ntr], idx[ntr:ntr + nval], idx[ntr + nval:]
    Xt = torch.from_numpy(X).float().to(dev)
    yt = torch.from_numpy(y).long().to(dev)
    W = torch.nn.Linear(X.shape[1], n_classes, bias=False).to(dev)
    opt = torch.optim.Adam(W.parameters(), lr=1e-2, weight_decay=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    best_va, best_te = -1, 0.0
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(W(Xt[tr]), yt[tr]); loss.backward(); opt.step()
        if _ % 20 == 0:
            with torch.no_grad():
                va_acc = (W(Xt[va]).argmax(-1) == yt[va]).float().mean().item()
                if va_acc > best_va:
                    best_va = va_acc
                    best_te = (W(Xt[te]).argmax(-1) == yt[te]).float().mean().item()
    return best_te


def probes(A, meta, tok):
    dev = "cuda"
    N, P, Lc, H = A.shape
    res, base = {}, {}
    for which in ("answer", "first_int", "last_int"):
        raw = np.array([target_token(tok, m, which) for m in meta])
        # restrict to first-tokens with >=20 support; map to dense labels
        cnt = Counter(raw.tolist())
        keep = {t for t, c in cnt.items() if c >= 20}
        mask = np.array([t in keep for t in raw])
        labs = sorted(keep); lab2i = {t: i for i, t in enumerate(labs)}
        y = np.array([lab2i[t] for t in raw[mask]])
        maj = max(Counter(y.tolist()).values()) / len(y)
        base[which] = maj
        mat = np.zeros((Lc, P))
        for li in range(Lc):
            for pi in range(P):
                X = A[mask, pi, li, :].astype(np.float32)
                mat[li, pi] = train_probe(X, y, len(labs), dev)
        res[which] = mat.tolist()
        print(f"[probe] {which}: n={mask.sum()} classes={len(labs)} majority-baseline={maj:.3f} "
              f"best cell acc={np.max(mat):.3f}")
    json.dump(dict(acc=res, baseline=base), open(os.path.join(OUT_DIR, "probes.json"), "w"))
    return res, base


def plot(ll, pr, base):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for which in ("answer", "first_int", "last_int"):
        L = np.array(ll[which]); Pm = np.array(pr[which])
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        im0 = ax[0].imshow(L, aspect="auto", cmap="YlOrRd", vmin=0, origin="lower")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, label="mean P(target first-token)")
        ax[0].set_title(f"Exact logit-lens — {which}")
        im1 = ax[1].imshow(Pm, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower")
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="probe test accuracy")
        ax[1].set_title(f"Linear probe — {which} (majority base={base[which]:.2f})")
        for a in ax:
            a.set_xticks(range(len(POS))); a.set_xticklabels(POS); a.set_xlabel("thought position")
            a.set_ylabel("model layer (0=embed .. top)")
        plt.tight_layout()
        png = os.path.join(OUT_DIR, f"p1_faithful_{which}.png")
        plt.savefig(png, dpi=150, bbox_inches="tight"); print(f"PNG: {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--offset", type=int, default=4000)
    ap.add_argument("--reuse-acts", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.reuse_acts and os.path.exists(ACTS_NPY):
        A = np.load(ACTS_NPY); meta = json.load(open(META_JSON))
        from src.model import CODI
        model = CODI.from_pretrained(checkpoint_path=CKPT, model_name_or_path=BASE, lora_r=128,
                                     lora_alpha=32, num_latent=NUM_LATENT, use_prj=True, prj_dim=4096,
                                     prj_no_ln=False, prj_dropout=0.0, dtype="bfloat16")
        model.eval(); tok = model.tokenizer
        print(f"[reuse] acts {A.shape}")
    else:
        A, meta, model, tok = extract(args)
    ll = logit_lens(A, meta, model, tok)
    del model; torch.cuda.empty_cache()
    pr, base = probes(A, meta, tok)
    plot(ll, pr, base)
    print("P1-FAITHFUL-DONE")


if __name__ == "__main__":
    main()
