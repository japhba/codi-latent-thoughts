"""tinyNLA: AV + AR for Qwen3-8B L24 activations on a tiny pretrained backbone.

The shipped recipe (docs/qwen3_8b_run.md) warm-starts AV/AR as fine-tuned copies
of the 8B subject model. The from-scratch line (zero_nla/fromscratch_nla.md) showed
tabula-rasa tiny AV/AR encode but never *describe* — its closing diagnosis was
that faithful verbalization needs a warm-start and/or an out-of-distribution
subject. tinyNLA is exactly that combination at minimum cost:

  subject  = Qwen3-8B layer 24 (d=4096, frozen, raw vectors from datagen)
  AV       = tiny pretrained LM (default google/gemma-3-270m, d=640), full-FT,
             + trainable Linear(4096 -> d_tiny) ingestion projection feeding a
             Karvonen norm-matched ADD injection at the residual after layer 1.
  AR       = same tiny backbone (lm_head dropped), full-FT,
             + Linear(d_tiny -> 4096) regression head read at the last token of
             the critic prompt. Direction-MSE loss (normalize to sqrt(4096)).

The projection in and regression head out are the only new layers — they are
what lets a d=640 model ingest/emit d=4096 subject activations.

Phases (run sequentially, one GPU):
  av_sft  CE on the Claude explanation given the injected activation
          (av_sft_shuf.parquet: response + activation_vector).
  ar_sft  dir-MSE on (critic prompt, activation) (ar_sft_shuf_clean.parquet).
  rl      GRPO on the AV (reward = -dirMSE from the AR) + supervised AR
          co-training on the rollout explanations. Reference policy = frozen
          copy of the SFT AV (full-FT, so no LoRA-disable trick).
  eval    held-out end-to-end FVE + shuffled-v control + AR-on-gold ceiling.

Run from the nanoNLA repo root:
  python -m nla.train_tiny_nla --phase av_sft --data-dir <dir> --save-dir <dir>
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from nla.injection import karvonen_inject_in_residual
from nla.schema import compute_predict_mean_baselines, extract_explanation, normalize_activation

D_SUBJECT = 4096
MSE_SCALE = math.sqrt(D_SUBJECT)  # 64.0 — same convention as the 8B run
# Marker must be a SINGLE token in the backbone tokenizer; first hit wins.
# ㊗ is single-token for gemma (249568); █ for GPT-NeoX/pythia (24177).
MARKER_CANDIDATES = ["㊗", "█", "¤", "★", "§"]
AV_PROMPT_TMPL = "Activation: <concept>{m}</concept>\nDescription:\n"
# Module-level defaults are the gemma scheme (kept for external importers,
# e.g. scripts/avbench_tiny_nla_rollouts.py); main() re-resolves per tokenizer.
INJ_CHAR = "㊗"
AV_PROMPT = AV_PROMPT_TMPL.format(m=INJ_CHAR)


def resolve_marker(tokenizer) -> str:
    for ch in MARKER_CANDIDATES:
        if len(tokenizer.encode(ch, add_special_tokens=False)) == 1:
            return ch
    raise AssertionError(f"no single-token marker among {MARKER_CANDIDATES}")


def _decoder_stack(backbone):
    """(inner_model, layers) across archs: gemma/llama/qwen .model.layers,
    GPT-NeoX .gpt_neox.layers, GPT-2 .transformer.h."""
    cur = backbone
    for _ in range(4):
        if hasattr(cur, "layers"):
            return cur, cur.layers
        if hasattr(cur, "h"):
            return cur, cur.h
        for attr in ("model", "gpt_neox", "transformer"):
            nxt = getattr(cur, attr, None)
            if nxt is not None:
                cur = nxt
                break
        else:
            break
    raise AssertionError(f"no decoder layers under {type(backbone).__name__}")
CRITIC_PREFIX = "Summary of the following text: <text>"
CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
MAX_LEN = 320  # tokens, AV full sequence and AR prompt; overlong rows skipped
VAL_DOC_MOD = 50  # hash(doc_id) % 50 == 0 -> val (~2%), DOC-level split


# ----------------------------------------------------------------------- models
class TinyAV(nn.Module):
    """Tiny verbalizer: backbone + ingestion projection + Karvonen layer-1 hook.

    Set .vectors_ref[0] to a [N, 4096] float32 tensor before any forward whose
    input contains N injection markers; the hook projects to d_tiny and
    ADD-norm-matches at each marker position. Set back to None after.
    """

    def __init__(self, backbone_name: str, inj_id: int, left_id: int, right_id: int):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_name, torch_dtype=torch.float32, attn_implementation="sdpa",
        )
        d_tiny = self.backbone.config.hidden_size
        self.proj = nn.Linear(D_SUBJECT, d_tiny, bias=False)
        self.vectors_ref = [None]
        self._register_hook(inj_id, left_id, right_id)

    def _register_hook(self, inj_id, left_id, right_id, layer_idx=1):
        state = {"input_ids": None}

        def embed_hook(module, args, kwargs, output):
            ids = kwargs.get("input") if kwargs else None
            if ids is None and args:
                ids = args[0]
            state["input_ids"] = ids
            return output

        def layer_hook(module, args, output):
            resid, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
            input_ids = state["input_ids"]
            if input_ids is None or resid.shape[1] < 2:  # decode step: no marker
                return output
            v = self.vectors_ref[0]
            if v is None:
                return output
            v_n = normalize_activation(v.to(self.proj.weight.device), MSE_SCALE)
            u = self.proj(v_n)  # [N, d_tiny]; grad flows into proj
            injected = karvonen_inject_in_residual(input_ids, resid, u, inj_id, left_id, right_id)
            return injected if rest is None else (injected, *rest)

        self.backbone.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
        _, layers = _decoder_stack(self.backbone)
        layers[layer_idx].register_forward_hook(layer_hook)

    def forward(self, **kwargs):
        return self.backbone(**kwargs)

    def generate(self, **kwargs):
        return self.backbone.generate(**kwargs)


class TinyAR(nn.Module):
    """Tiny reconstructor: backbone encoder (final RMSNorm kept) + d_tiny->4096 head."""

    def __init__(self, backbone_name: str):
        super().__init__()
        full = AutoModelForCausalLM.from_pretrained(
            backbone_name, torch_dtype=torch.float32, attn_implementation="sdpa",
        )
        self.encoder = _decoder_stack(full)[0]  # inner model — keeps embeddings + final norm
        self.head = nn.Linear(full.config.hidden_size, D_SUBJECT, bias=True)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1
        h_last = h[torch.arange(input_ids.shape[0], device=input_ids.device), last]
        return self.head(h_last)  # [B, 4096]


def resolve_marker_ids(tokenizer):
    """Marker token + canonical neighbors from the AV prompt tokenization."""
    ids = tokenizer.encode(AV_PROMPT, add_special_tokens=False)
    inj = tokenizer.encode(INJ_CHAR, add_special_tokens=False)
    assert len(inj) == 1, f"{INJ_CHAR!r} is not a single token: {inj}"
    inj_id = inj[0]
    matches = [i for i, t in enumerate(ids) if t == inj_id]
    assert len(matches) == 1 and 0 < matches[0] < len(ids) - 1, (inj_id, matches, ids)
    p = matches[0]
    return inj_id, ids[p - 1], ids[p + 1]


# ------------------------------------------------------------------------- data
def stream_rows(path, columns, n_max=None, skip_first=0):
    """Row-group streaming read; returns dict of column -> list / [N,4096] np array."""
    pf = pq.ParquetFile(path)
    out = {c: [] for c in columns}
    seen = 0
    taken = 0
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and taken >= n_max:
            break
        n = pf.metadata.row_group(rg_idx).num_rows
        if seen + n <= skip_first:  # skip via metadata, no read
            seen += n
            continue
        rg = pf.read_row_group(rg_idx, columns=columns)
        start = max(0, skip_first - seen)
        seen += n
        take = n - start if n_max is None else min(n_max - taken, n - start)
        rg = rg.slice(start, take)
        taken += take
        for c in columns:
            col = rg.column(c)
            if c == "activation_vector":
                a = col.combine_chunks()
                out[c].append(a.values.to_numpy(zero_copy_only=False).reshape(len(a), -1).astype(np.float32))
            else:
                out[c].extend(col.to_pylist())
    if "activation_vector" in out:
        out["activation_vector"] = np.concatenate(out["activation_vector"], axis=0)
    return out


def doc_is_val(doc_id) -> bool:
    # Stable across processes (no PYTHONHASHSEED dependence).
    import zlib
    return zlib.crc32(str(doc_id).encode()) % VAL_DOC_MOD == 0


def tokenize_av_rows(tokenizer, responses, acts, doc_ids):
    """-> (train, val) lists of (resp_ids, act_row_idx); skips malformed/overlong."""
    prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(AV_PROMPT, add_special_tokens=False)
    train, val, n_skip = [], [], 0
    for i, r in enumerate(responses):
        if INJ_CHAR in r or extract_explanation(r) is None:
            n_skip += 1
            continue
        ids = tokenizer.encode(r, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prompt_ids) + len(ids) > MAX_LEN:
            n_skip += 1
            continue
        (val if doc_is_val(doc_ids[i]) else train).append((ids, i))
    skip_rate = n_skip / max(len(responses), 1)
    assert skip_rate < 0.01, f"AV skip rate {skip_rate:.2%}"
    print(f"[av data] {len(train)} train / {len(val)} val rows, {n_skip} skipped ({skip_rate:.3%})", flush=True)
    return prompt_ids, train, val


def tokenize_ar_rows(tokenizer, prompts, doc_ids):
    train, val, n_skip = [], [], 0
    for i, s in enumerate(prompts):
        assert s.startswith(CRITIC_PREFIX), s[:60]
        if INJ_CHAR in s:
            n_skip += 1
            continue
        ids = [tokenizer.bos_token_id] + tokenizer.encode(s, add_special_tokens=False)
        if len(ids) > MAX_LEN:
            n_skip += 1
            continue
        (val if doc_is_val(doc_ids[i]) else train).append((ids, i))
    skip_rate = n_skip / max(len(prompts), 1)
    assert skip_rate < 0.01, f"AR skip rate {skip_rate:.2%}"
    print(f"[ar data] {len(train)} train / {len(val)} val rows, {n_skip} skipped ({skip_rate:.3%})", flush=True)
    return train, val


def pad_batch(seqs, pad_id, device):
    mx = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), mx), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), mx), dtype=torch.long, device=device)
    for r, s in enumerate(seqs):
        ids[r, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        attn[r, : len(s)] = 1
    return ids, attn


def dir_mse(pred, gold):
    pn = normalize_activation(pred.float(), MSE_SCALE)
    gn = normalize_activation(gold.float(), MSE_SCALE)
    return ((pn - gn) ** 2).mean(dim=-1)  # [N]


def cosine_lr(optim, base_lr, step, total, warmup):
    lr = base_lr * step / max(warmup, 1) if step < warmup else (
        base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))
    )
    for g in optim.param_groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)
    return lr


def grouped_adamw(named_params, base_lr, new_layer_names, new_lr_scale):
    """AdamW with fresh layers (proj / head) at base_lr * new_lr_scale."""
    new, old = [], []
    for n, p in named_params:
        (new if any(n.startswith(k) for k in new_layer_names) else old).append(p)
    assert new, f"no params matched {new_layer_names}"
    return torch.optim.AdamW(
        [{"params": old, "lr_scale": 1.0}, {"params": new, "lr_scale": new_lr_scale}],
        lr=base_lr, betas=(0.9, 0.95), weight_decay=0.0,
    )


def save_ckpt(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.bfloat16() if v.is_floating_point() else v for k, v in model.state_dict().items()}, path)
    print(f"[save] {path}", flush=True)


def load_ckpt(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    model.load_state_dict(sd)
    print(f"[load] {path}", flush=True)


def wandb_init(args, phase):
    wandb.init(
        entity=args.wandb_entity, project=args.wandb_project,
        group=args.batch_id, name=f"{args.batch_id}_{phase}", config=vars(args),
    )


# --------------------------------------------------------------------- av_sft
def run_av_sft(args, tokenizer, device):
    data = stream_rows(
        Path(args.data_dir) / "av_sft_shuf.parquet",
        ["response", "activation_vector", "doc_id"], n_max=args.max_rows,
    )
    prompt_ids, train, val = tokenize_av_rows(
        tokenizer, data["response"], data["activation_vector"], data["doc_id"],
    )
    acts = torch.from_numpy(data["activation_vector"])

    inj_id, left_id, right_id = resolve_marker_ids(tokenizer)
    av = TinyAV(args.backbone, inj_id, left_id, right_id).to(device)
    optim = grouped_adamw(av.named_parameters(), args.av_lr, ("proj.",), args.new_lr_scale)

    total_steps = args.av_epochs * len(train) // args.av_batch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    eval_every = max(1, total_steps // 25)
    print(f"[av_sft] {total_steps} steps, eval every {eval_every}", flush=True)
    wandb_init(args, "av_sft")

    pad_id = tokenizer.pad_token_id
    p_len = len(prompt_ids)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train))
    cursor = 0
    examples_seen = tokens_seen = 0

    def batch_forward(items, train_mode):
        nonlocal tokens_seen
        seqs = [prompt_ids + ids for ids, _ in items]
        ids_t, attn = pad_batch(seqs, pad_id, device)
        v = torch.stack([acts[i] for _, i in items]).to(device)
        av.vectors_ref[0] = v
        try:
            logits = av(input_ids=ids_t, attention_mask=attn).logits
        finally:
            av.vectors_ref[0] = None
        labels = ids_t.clone()
        labels[attn == 0] = -100
        labels[:, :p_len] = -100
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        if train_mode:
            tokens_seen += int(attn.sum().item())
        return loss

    for step in range(total_steps):
        if cursor + args.av_batch > len(order):
            order = rng.permutation(len(train))
            cursor = 0
        items = [train[i] for i in order[cursor : cursor + args.av_batch]]
        cursor += args.av_batch
        examples_seen += len(items)

        av.train()
        lr = cosine_lr(optim, args.av_lr, step, total_steps, args.warmup_steps)
        loss = batch_forward(items, train_mode=True)
        optim.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
        optim.step()

        if step % 20 == 0:
            print(f"[av_sft {step}/{total_steps}] loss {loss.item():.4f} gn {gn:.2f} lr {lr:.2e}", flush=True)
            wandb.log({"loss": loss.item(), "grad_norm": gn.item(), "lr": lr,
                       "examples_seen": examples_seen, "tokens_seen": tokens_seen}, step=step)

        if step % eval_every == 0 or step == total_steps - 1:
            av.eval()
            with torch.no_grad():
                vloss = np.mean([
                    batch_forward([val[j] for j in range(s, min(s + args.av_batch, min(len(val), 2048)))],
                                  train_mode=False).item()
                    for s in range(0, min(len(val), 2048), args.av_batch)
                ])
            print(f"[av_sft {step}] VAL loss {vloss:.4f}", flush=True)
            wandb.log({"val/loss": vloss, "examples_seen": examples_seen, "tokens_seen": tokens_seen}, step=step)
            if step > 0 and step % (eval_every * 5) == 0:
                save_ckpt(av, Path(args.save_dir) / f"av_sft/step_{step:06d}.pt")

    save_ckpt(av, Path(args.save_dir) / "av_sft/final.pt")
    wandb.finish()


# --------------------------------------------------------------------- ar_sft
def run_ar_sft(args, tokenizer, device):
    data = stream_rows(
        Path(args.data_dir) / "ar_sft_shuf_clean.parquet",
        ["prompt", "activation_vector", "doc_id"], n_max=args.max_rows,
    )
    train, val = tokenize_ar_rows(tokenizer, data["prompt"], data["doc_id"])
    acts = torch.from_numpy(data["activation_vector"])

    ar = TinyAR(args.backbone).to(device)
    optim = grouped_adamw(ar.named_parameters(), args.ar_lr, ("head.",), args.new_lr_scale)

    total_steps = args.ar_epochs * len(train) // args.ar_batch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    eval_every = max(1, total_steps // 25)
    print(f"[ar_sft] {total_steps} steps, eval every {eval_every}", flush=True)
    wandb_init(args, "ar_sft")

    # FVE baselines on val golds (predict-the-mean).
    val_gold = torch.stack([acts[i] for _, i in val[:2048]])
    baseline_mse, _ = compute_predict_mean_baselines(val_gold, MSE_SCALE)
    print(f"[ar_sft] val predict-mean baseline mse {baseline_mse:.4f}", flush=True)

    pad_id = tokenizer.pad_token_id
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train))
    cursor = 0
    examples_seen = tokens_seen = 0

    for step in range(total_steps):
        if cursor + args.ar_batch > len(order):
            order = rng.permutation(len(train))
            cursor = 0
        items = [train[i] for i in order[cursor : cursor + args.ar_batch]]
        cursor += args.ar_batch
        examples_seen += len(items)

        ar.train()
        lr = cosine_lr(optim, args.ar_lr, step, total_steps, args.warmup_steps)
        ids_t, attn = pad_batch([ids for ids, _ in items], pad_id, device)
        gold = torch.stack([acts[i] for _, i in items]).to(device)
        pred = ar(ids_t, attn)
        loss = dir_mse(pred, gold).mean()
        optim.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
        optim.step()
        tokens_seen += int(attn.sum().item())

        if step % 20 == 0:
            fve = 1 - loss.item() / baseline_mse
            print(f"[ar_sft {step}/{total_steps}] mse {loss.item():.4f} (train FVE~{fve:.1%}) gn {gn:.2f} lr {lr:.2e}", flush=True)
            wandb.log({"loss": loss.item(), "train_fve": fve, "grad_norm": gn.item(), "lr": lr,
                       "examples_seen": examples_seen, "tokens_seen": tokens_seen}, step=step)

        if step % eval_every == 0 or step == total_steps - 1:
            ar.eval()
            mses, preds = [], []
            with torch.no_grad():
                for s in range(0, min(len(val), 2048), args.ar_batch):
                    chunk = val[s : s + args.ar_batch]
                    ids_t, attn = pad_batch([ids for ids, _ in chunk], pad_id, device)
                    gold = torch.stack([acts[i] for _, i in chunk]).to(device)
                    pred = ar(ids_t, attn)
                    mses.append(dir_mse(pred, gold).cpu())
                    preds.append(pred.cpu())
            val_mse = torch.cat(mses).mean().item()
            # shuffled control: same preds scored against permuted golds
            P = torch.cat(preds)
            G = val_gold[: P.shape[0]]
            perm = torch.randperm(P.shape[0], generator=torch.Generator().manual_seed(0))
            shuf_mse = dir_mse(P, G[perm]).mean().item()
            fve, fve_shuf = 1 - val_mse / baseline_mse, 1 - shuf_mse / baseline_mse
            print(f"[ar_sft {step}] VAL mse {val_mse:.4f} FVE {fve:.1%} (shuffled {fve_shuf:.1%})", flush=True)
            wandb.log({"val/mse": val_mse, "val/fve": fve, "val/fve_shuffled": fve_shuf,
                       "examples_seen": examples_seen, "tokens_seen": tokens_seen}, step=step)
            if step > 0 and step % (eval_every * 5) == 0:
                save_ckpt(ar, Path(args.save_dir) / f"ar_sft/step_{step:06d}.pt")

    save_ckpt(ar, Path(args.save_dir) / "ar_sft/final.pt")
    wandb.finish()


# ------------------------------------------------------------------------- rl
def score_explanations(ar, tokenizer, explanations, golds, device, micro=32):
    """-> rewards list (None where extraction failed / overlong)."""
    pad_id = tokenizer.pad_token_id
    idxs, seqs = [], []
    for i, e in enumerate(explanations):
        if e is None or INJ_CHAR in e:
            continue
        ids = [tokenizer.bos_token_id] + tokenizer.encode(
            CRITIC_TEMPLATE.format(explanation=e), add_special_tokens=False)
        if len(ids) > MAX_LEN:
            continue
        idxs.append(i)
        seqs.append(ids)
    rewards = [None] * len(explanations)
    ar.eval()
    with torch.no_grad():
        for s in range(0, len(seqs), micro):
            ids_t, attn = pad_batch(seqs[s : s + micro], pad_id, device)
            gold = torch.stack([golds[i] for i in idxs[s : s + micro]]).to(device)
            mse = dir_mse(ar(ids_t, attn), gold)
            for j, i in enumerate(idxs[s : s + micro]):
                m = mse[j].item()
                rewards[i] = -m if math.isfinite(m) else None
    return rewards


def rl_eval(av, ar, tokenizer, eval_acts, prompt_ids, device, baseline_mse, max_new):
    """Greedy end-to-end eval: FVE, shuffled-v control, extraction rate, samples."""
    av.eval()
    n = eval_acts.shape[0]
    ids_t = torch.tensor([prompt_ids] * n, dtype=torch.long, device=device)
    av.vectors_ref[0] = eval_acts.to(device)
    try:
        with torch.no_grad():
            out = av.generate(
                input_ids=ids_t, attention_mask=torch.ones_like(ids_t),
                max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        av.vectors_ref[0] = None
    texts = [tokenizer.decode(out[g, len(prompt_ids):], skip_special_tokens=True) for g in range(n)]
    expls = [extract_explanation(t) for t in texts]
    golds = [eval_acts[i] for i in range(n)]
    rewards = score_explanations(ar, tokenizer, expls, golds, device)
    perm = np.random.default_rng(0).permutation(n)
    rewards_shuf = score_explanations(ar, tokenizer, expls, [eval_acts[i] for i in perm], device)
    valid = [r for r in rewards if r is not None]
    valid_shuf = [r for r in rewards_shuf if r is not None]
    fve = 1 - (-np.mean(valid)) / baseline_mse if valid else float("nan")
    fve_shuf = 1 - (-np.mean(valid_shuf)) / baseline_mse if valid_shuf else float("nan")
    ext = len(valid) / n
    return fve, fve_shuf, ext, texts, expls, rewards


def run_rl(args, tokenizer, device):
    inj_id, left_id, right_id = resolve_marker_ids(tokenizer)
    av = TinyAV(args.backbone, inj_id, left_id, right_id).to(device)
    load_ckpt(av, Path(args.save_dir) / "av_sft/final.pt")
    # Fresh construction, NOT deepcopy: deepcopy keeps hook closures bound to
    # the ORIGINAL module's proj/vectors_ref (functions are atomic to deepcopy),
    # so the ref would silently inject with the trainable proj.
    ref = TinyAV(args.backbone, inj_id, left_id, right_id).to(device)
    ref.load_state_dict(av.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ar = TinyAR(args.backbone).to(device)
    load_ckpt(ar, Path(args.save_dir) / "ar_sft/final.pt")

    optim = torch.optim.AdamW(av.parameters(), lr=args.rl_lr, betas=(0.9, 0.95), weight_decay=0.0)
    ar_optim = torch.optim.AdamW(ar.parameters(), lr=args.rl_critic_lr, betas=(0.9, 0.95), weight_decay=0.0)

    data = stream_rows(Path(args.data_dir) / "rl_shuf.parquet",
                       ["activation_vector"], n_max=args.rl_rows)
    acts = torch.from_numpy(data["activation_vector"])
    eval_data = stream_rows(Path(args.data_dir) / "rl_shuf.parquet",
                            ["activation_vector"], n_max=args.eval_n, skip_first=480_000)
    eval_acts = torch.from_numpy(eval_data["activation_vector"])
    baseline_mse, _ = compute_predict_mean_baselines(eval_acts, MSE_SCALE)
    print(f"[rl] {len(acts)} train rows, {len(eval_acts)} eval rows, baseline {baseline_mse:.4f}", flush=True)
    wandb_init(args, "rl")

    prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(AV_PROMPT, add_special_tokens=False)
    p_len = len(prompt_ids)
    pad_id = tokenizer.pad_token_id
    rng = np.random.default_rng(args.seed)
    B, G = args.rl_batch_prompts, args.rl_group_size
    examples_seen = 0
    eval_table = []

    for step in range(args.rl_steps):
        t0 = time.time()
        row_idx = rng.choice(len(acts), size=B, replace=False)
        v_prompts = acts[row_idx]  # [B, 4096]
        v_full = v_prompts.repeat_interleave(G, dim=0)  # [B*G, 4096]
        examples_seen += B

        # ---- rollout (constant prompt -> no padding) ----
        av.eval()
        ids_t = torch.tensor([prompt_ids] * (B * G), dtype=torch.long, device=device)
        av.vectors_ref[0] = v_full
        try:
            with torch.no_grad():
                gen = av.generate(
                    input_ids=ids_t, attention_mask=torch.ones_like(ids_t),
                    max_new_tokens=args.rl_max_new, do_sample=True, temperature=args.rl_temp,
                    top_p=1.0, top_k=0, pad_token_id=pad_id,
                )
        finally:
            av.vectors_ref[0] = None
        texts = [tokenizer.decode(gen[g, p_len:], skip_special_tokens=True) for g in range(B * G)]
        expls = [extract_explanation(t) for t in texts]

        # ---- reward ----
        golds = [v_full[i] for i in range(B * G)]
        rewards = score_explanations(ar, tokenizer, expls, golds, device)
        rewards_f = torch.tensor([(-2.0 if r is None else r) for r in rewards], device=device)
        adv = torch.zeros_like(rewards_f)
        for b in range(B):
            grp = rewards_f[b * G : (b + 1) * G]
            adv[b * G : (b + 1) * G] = (grp - grp.mean()) / (grp.std() + 1e-6)

        # ---- GRPO update (one update per batch: old_logp = new_logp.detach()) ----
        # Rows that GENERATED the marker char are excluded: a response-region
        # marker with matching neighbors (the model echoing "<concept>㊗</concept>")
        # would break the injection hook's count assert in the update forward.
        av.train()
        optim.zero_grad()
        seq_lens = [int((gen[g] != pad_id).sum().item()) for g in range(B * G)]
        upd_rows = [g for g in range(B * G) if not (gen[g, p_len:] == inj_id).any()]
        n_upd = len(upd_rows)
        kls, losses = [], []
        for s in range(0, n_upd, args.rl_micro):
            rows = upd_rows[s : s + args.rl_micro]
            mb = gen[rows]
            mb_attn = (mb != pad_id).long()
            mb_attn[:, :p_len] = 1
            v_mb = v_full[rows]
            av.vectors_ref[0] = v_mb
            try:
                logits = av(input_ids=mb, attention_mask=mb_attn).logits
            finally:
                av.vectors_ref[0] = None
            ref.vectors_ref[0] = v_mb
            try:
                with torch.no_grad():
                    ref_logits = ref(input_ids=mb, attention_mask=mb_attn).logits
            finally:
                ref.vectors_ref[0] = None
            logp = F.log_softmax(logits.float(), dim=-1)
            ref_logp_full = F.log_softmax(ref_logits.float(), dim=-1)
            del logits, ref_logits
            mb_losses = []
            for r, g in enumerate(rows):
                L = seq_lens[g]
                if L <= p_len:
                    continue
                tgt = mb[r, p_len:L]
                pos = torch.arange(p_len - 1, L - 1, device=device)
                new_lp = logp[r].index_select(0, pos).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                ref_lp = ref_logp_full[r].index_select(0, pos).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                ratio = torch.exp(new_lp - new_lp.detach())  # ==1, REINFORCE-style grad
                surrogate = torch.minimum(ratio * adv[g], torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv[g])
                delta = ref_lp.detach() - new_lp
                kl = torch.exp(delta) - delta - 1.0
                mb_losses.append(-(surrogate - args.rl_kl_beta * kl).mean())
                kls.append(kl.detach().mean().item())
            if mb_losses:
                mb_loss = torch.stack(mb_losses).sum() / max(n_upd, 1)
                mb_loss.backward()
                losses.append(mb_loss.item() * max(n_upd, 1) / len(mb_losses))
            del logp, ref_logp_full
        gn = torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
        optim.step()

        # ---- AR co-training on this step's explanations ----
        ar.train()
        ar_optim.zero_grad()
        crit = [(i, e) for i, e in enumerate(expls)
                if e is not None and INJ_CHAR not in e]
        crit_seqs = []
        crit_idx = []
        for i, e in crit:
            ids = [tokenizer.bos_token_id] + tokenizer.encode(
                CRITIC_TEMPLATE.format(explanation=e), add_special_tokens=False)
            if len(ids) <= MAX_LEN:
                crit_seqs.append(ids)
                crit_idx.append(i)
        ar_loss_val = float("nan")
        if crit_seqs:
            tot = len(crit_seqs)
            acc = 0.0
            for s in range(0, tot, args.rl_micro):
                ids_t2, attn2 = pad_batch(crit_seqs[s : s + args.rl_micro], pad_id, device)
                gold = torch.stack([v_full[i] for i in crit_idx[s : s + args.rl_micro]]).to(device)
                closs = dir_mse(ar(ids_t2, attn2), gold).mean() * (ids_t2.shape[0] / tot)
                closs.backward()
                acc += closs.item()
            torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
            ar_optim.step()
            ar_loss_val = acc

        valid = [r for r in rewards if r is not None]
        fve = 1 - (-np.mean(valid)) / baseline_mse if valid else float("nan")
        log = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "reward_mean": float(np.mean(valid)) if valid else float("nan"),
            "fve": fve, "extraction_rate": len(valid) / (B * G),
            "kl_mean": float(np.mean(kls)) if kls else 0.0,
            "grad_norm": gn.item(), "ar_loss": ar_loss_val,
            "examples_seen": examples_seen, "wall_s": time.time() - t0,
        }
        print(f"[rl {step}/{args.rl_steps}] r {log['reward_mean']:.3f} FVE {fve:.1%} "
              f"ext {log['extraction_rate']:.0%} kl {log['kl_mean']:.4f} ar {ar_loss_val:.4f} "
              f"t {log['wall_s']:.0f}s", flush=True)

        if step % args.rl_eval_every == 0 or step == args.rl_steps - 1:
            fve_e, fve_shuf, ext, texts_e, expls_e, rewards_e = rl_eval(
                av, ar, tokenizer, eval_acts[:64], prompt_ids, device, baseline_mse, args.rl_max_new)
            log.update({"eval/fve": fve_e, "eval/fve_shuffled": fve_shuf,
                        "eval/fve_gap": fve_e - fve_shuf, "eval/extraction_rate": ext})
            print(f"  [rl eval@{step}] FVE {fve_e:.1%} shuffled {fve_shuf:.1%} ext {ext:.0%}", flush=True)
            for i in (0, 1, 2):
                print(f"    [sample {i} r={rewards_e[i]}] {texts_e[i]}", flush=True)
            for i in range(min(8, len(texts_e))):
                eval_table.append([step, i, rewards_e[i] if rewards_e[i] is not None else -2.0, texts_e[i]])
            log["eval/samples"] = wandb.Table(columns=["step", "idx", "reward", "text"], data=list(eval_table))
        wandb.log(log, step=step)

        if (step + 1) % args.rl_save_every == 0 or step == args.rl_steps - 1:
            save_ckpt(av, Path(args.save_dir) / f"rl/av_step_{step + 1:06d}.pt")
            save_ckpt(ar, Path(args.save_dir) / f"rl/ar_step_{step + 1:06d}.pt")

    save_ckpt(av, Path(args.save_dir) / "rl/av_final.pt")
    save_ckpt(ar, Path(args.save_dir) / "rl/ar_final.pt")
    wandb.finish()


# ------------------------------------------------------------------------ eval
def run_eval(args, tokenizer, device):
    """Final held-out eval of SFT-only and (if present) post-RL checkpoints."""
    inj_id, left_id, right_id = resolve_marker_ids(tokenizer)
    eval_data = stream_rows(
        Path(args.data_dir) / "rl_shuf.parquet",
        ["activation_vector", "detokenized_text_truncated"], n_max=args.eval_n, skip_first=480_000)
    eval_acts = torch.from_numpy(eval_data["activation_vector"])
    contexts = eval_data["detokenized_text_truncated"]
    baseline_mse, _ = compute_predict_mean_baselines(eval_acts, MSE_SCALE)
    prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(AV_PROMPT, add_special_tokens=False)

    # AR-on-gold ceiling: held-out av_sft docs' Claude explanations through the AR.
    av_data = stream_rows(Path(args.data_dir) / "av_sft_shuf.parquet",
                          ["response", "activation_vector", "doc_id"], n_max=100_000)
    gold_rows = [(extract_explanation(r), i) for i, r in enumerate(av_data["response"])
                 if doc_is_val(av_data["doc_id"][i]) and extract_explanation(r) is not None]
    gold_rows = gold_rows[: args.eval_n]
    gold_acts = torch.from_numpy(av_data["activation_vector"][[i for _, i in gold_rows]])
    gold_baseline, _ = compute_predict_mean_baselines(gold_acts, MSE_SCALE)

    results = {}
    for tag, av_path, ar_path in [
        ("sft", Path(args.save_dir) / "av_sft/final.pt", Path(args.save_dir) / "ar_sft/final.pt"),
        ("rl", Path(args.save_dir) / "rl/av_final.pt", Path(args.save_dir) / "rl/ar_final.pt"),
    ]:
        if not av_path.exists():
            continue
        av = TinyAV(args.backbone, inj_id, left_id, right_id).to(device)
        load_ckpt(av, av_path)
        ar = TinyAR(args.backbone).to(device)
        load_ckpt(ar, ar_path)

        rewards_gold = score_explanations(
            ar, tokenizer, [e for e, _ in gold_rows], [gold_acts[i] for i in range(len(gold_rows))], device)
        vg = [r for r in rewards_gold if r is not None]
        fve_gold = 1 - (-np.mean(vg)) / gold_baseline if vg else float("nan")

        fve, fve_shuf, ext, texts, expls, rewards = rl_eval(
            av, ar, tokenizer, eval_acts, prompt_ids, device, baseline_mse, args.rl_max_new)
        perm = np.random.default_rng(0).permutation(len(texts))
        rewards_shuf = score_explanations(
            ar, tokenizer, expls, [eval_acts[i] for i in perm], device)
        results[tag] = {"fve": fve, "fve_shuffled": fve_shuf, "fve_gap": fve - fve_shuf,
                        "extraction_rate": ext, "ar_on_gold_fve": fve_gold,
                        "baseline_mse": baseline_mse, "gold_baseline_mse": gold_baseline,
                        "n": len(texts)}
        # Per-example records for plotting / spot-checks.
        rec_path = Path(args.save_dir) / f"eval_records_{tag}.jsonl"
        with rec_path.open("w") as f:
            for i in range(len(texts)):
                f.write(json.dumps({
                    "idx": i, "reward": rewards[i], "reward_shuffled": rewards_shuf[i],
                    "context": contexts[i], "text": texts[i],
                }) + "\n")
        with (Path(args.save_dir) / f"eval_records_gold_{tag}.jsonl").open("w") as f:
            for i, (e, _) in enumerate(gold_rows):
                f.write(json.dumps({"idx": i, "reward": rewards_gold[i], "explanation": e}) + "\n")
        print(f"\n===== {tag}: end-to-end FVE {fve:.1%} | shuffled {fve_shuf:.1%} | "
              f"gap {fve - fve_shuf:.1%} | ext {ext:.0%} | AR-on-gold FVE {fve_gold:.1%}", flush=True)
        print(f"===== {tag}: sample explanations (full, with source context)", flush=True)
        for i in range(min(10, len(texts))):
            print(f"\n--- [{tag} {i}] reward={rewards[i]}", flush=True)
            print(f"CONTEXT (end): ...{contexts[i][-400:]}", flush=True)
            print(f"AV OUTPUT: {texts[i]}", flush=True)
        del av, ar
        torch.cuda.empty_cache()

    out = Path(args.save_dir) / "eval_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[eval] results -> {out}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


# ------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True,
                   choices=["av_sft", "ar_sft", "rl", "eval"])
    p.add_argument("--data-dir", default="/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--backbone", default="google/gemma-3-270m")
    p.add_argument("--batch-id", default="tn1", help="3-char wandb batch id + group")
    p.add_argument("--wandb-entity", default="MATS10-CS-JB")
    p.add_argument("--wandb-project", default="cot-oracle")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="cap SFT steps (smoke)")
    p.add_argument("--warmup-steps", type=int, default=100)
    # SFT
    p.add_argument("--new-lr-scale", type=float, default=10.0,
                   help="fresh proj/head layers train at lr * this")
    p.add_argument("--av-lr", type=float, default=1e-4)
    p.add_argument("--av-batch", type=int, default=32)  # 262k-vocab logits: 64 is ~13GB+grad just for logits
    p.add_argument("--av-epochs", type=int, default=2)
    p.add_argument("--ar-lr", type=float, default=1e-4)
    p.add_argument("--ar-batch", type=int, default=64)
    p.add_argument("--ar-epochs", type=int, default=2)
    # RL
    p.add_argument("--rl-steps", type=int, default=250)
    p.add_argument("--rl-batch-prompts", type=int, default=16)
    p.add_argument("--rl-group-size", type=int, default=8)
    p.add_argument("--rl-max-new", type=int, default=170)
    p.add_argument("--rl-temp", type=float, default=1.0)
    p.add_argument("--rl-lr", type=float, default=5e-6)
    p.add_argument("--rl-critic-lr", type=float, default=1e-5)
    p.add_argument("--rl-kl-beta", type=float, default=0.01)
    p.add_argument("--rl-micro", type=int, default=16)
    p.add_argument("--rl-rows", type=int, default=200_000)
    p.add_argument("--rl-eval-every", type=int, default=10)
    p.add_argument("--rl-save-every", type=int, default=50)
    p.add_argument("--eval-n", type=int, default=256)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    assert torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if tokenizer.pad_token_id is None:  # pythia/GPT-2 family: pad with eos.
        # Consequence: the RL update masks the final eos out of the loss
        # (termination is shaped by SFT only). SFT is unaffected — pad_batch
        # builds masks from explicit lengths, not by comparing to pad_id.
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None
    global INJ_CHAR, AV_PROMPT
    INJ_CHAR = resolve_marker(tokenizer)
    AV_PROMPT = AV_PROMPT_TMPL.format(m=INJ_CHAR)
    print(f"[marker] {INJ_CHAR!r} pad_id={tokenizer.pad_token_id}")

    {"av_sft": run_av_sft, "ar_sft": run_ar_sft, "rl": run_rl, "eval": run_eval}[args.phase](args, tokenizer, device)


if __name__ == "__main__":
    main()
