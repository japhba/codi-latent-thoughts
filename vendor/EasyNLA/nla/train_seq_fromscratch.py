"""From-scratch SEQ2SEQ NLA: inject a whole document's activation trajectory at once, verbalize
into one short code, reconstruct ALL of the activations contiguously.

This is the sequence analog of `train_rl_fromscratch.py` (which auto-encodes ONE activation at a
time). Here a "row" is a whole document:

  - AV (actor, vanilla small Llama): reads the activation trajectory [v_0..v_{T-1}] fed as INPUT
    EMBEDDINGS (activation dim == AV hidden, so each v_t is a pseudo-token embedding, normalized to
    ‖·‖=√d·inject_scale). After [bos, v_0..v_{T-1}] it emits a short token "code". Full fine-tuned.
  - AR (critic, vanilla NLACriticModel): reads [bos] code [<READ> × T], and at each of the T read
    slots predicts v̂_t. Reconstruction = masked dir-MSE over all T positions.
  - M (frozen tiny subject+prior): source of the activations AND the English prior scoring the code.

Bootstrap = the same differentiable Gumbel straight-through warm-up that works for the single-vector
case (pure RL-from-scratch can't bootstrap): hard-sample the code (so FVE is on real tokens), do a
teacher-forced AV forward for the code logits, feed the AR straight-through one-hots
`y = onehot(code) + (softmax − softmax.detach())` so the reconstruction gradient flows back through
the code into the AV. α = KL(AV‖M) English prior on the code (delayed/ramped); β = grounded
faithfulness (frozen M, given the code, must predict the document's tokens). GRPO is dropped — it
provably can't bootstrap a from-scratch encoder, and this experiment is about the warm-up.

Run (single GPU):
  python -m nla.train_seq_fromscratch --data-dir /workspace/nla/artifacts/seqdata \
    --save-dir /workspace/nla/runs/s0 --warmup-steps 2000 --seq-len 256 --max-new-tokens 48 \
    --batch-prompts 32 --alpha 0.1 --beta 0.0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

import wandb

from nla.models import NLACriticModel
from nla.schema import normalize_activation, resolve_target_scale


# ----------------------------------------------------------------------------- data
def load_doc_sequences(parquet_path, d, seq_len, n_max=None):
    """Returns (acts_pad [n, seq_len, d] f32, toks_pad [n, seq_len] i64, lens [n] i64).

    Each document's trajectory is truncated to the first `seq_len` positions and right-padded with
    zeros; `lens` holds the true (pre-pad) length so the loss can mask the padding.
    """
    pf = pq.ParquetFile(parquet_path)
    A, Tk, L = [], [], []
    for rg in range(pf.num_row_groups):
        if n_max is not None and len(L) >= n_max:
            break
        t = pf.read_row_group(rg, columns=["activations", "tokens", "seq_len"])
        ac = t.column("activations").combine_chunks()
        tc = t.column("tokens").combine_chunks()
        a_flat = ac.values.to_numpy(zero_copy_only=False).astype(np.float32)
        a_off = ac.offsets.to_numpy()
        t_flat = tc.values.to_numpy(zero_copy_only=False).astype(np.int64)
        t_off = tc.offsets.to_numpy()
        lens = t.column("seq_len").to_numpy()
        for i in range(len(lens)):
            Ti = int(lens[i])
            use = min(Ti, seq_len)
            A.append(a_flat[a_off[i]:a_off[i + 1]].reshape(Ti, d)[:use])
            Tk.append(t_flat[t_off[i]:t_off[i + 1]][:use])
            L.append(use)
            if n_max is not None and len(L) >= n_max:
                break
    n = len(L)
    acts_pad = np.zeros((n, seq_len, d), np.float32)
    toks_pad = np.zeros((n, seq_len), np.int64)
    for i in range(n):
        acts_pad[i, :L[i]] = A[i]
        toks_pad[i, :L[i]] = Tk[i]
    return acts_pad, toks_pad, np.array(L, np.int64)


def dir_mse(pred, gold, scale):
    """Per-row direction MSE = MSE(normalize(pred), normalize(gold)) over the last dim."""
    pn = normalize_activation(pred, scale)
    gn = normalize_activation(gold, scale)
    return ((pn - gn) ** 2).mean(dim=-1)


# ----------------------------------------------------------------------------- AV input
def build_av_prefix(actor, v_seq, act_mask, bos_id, d, inject_scale):
    """[bos, v_0..v_{T-1}] as AV input embeddings (activations normalized to ‖·‖=√d·inject_scale,
    pads zeroed). Returns (embeds [B,1+T,d], mask [B,1+T], position_ids [B,1+T])."""
    B, T = v_seq.shape[0], v_seq.shape[1]
    av_emb = actor.get_input_embeddings().weight
    v_unit = normalize_activation(v_seq.float(), inject_scale * (d ** 0.5)) * act_mask[..., None]   # [B,T,d]
    bos_e = av_emb[bos_id].view(1, 1, -1).expand(B, 1, -1).to(v_unit.dtype)
    embeds = torch.cat([bos_e, v_unit], dim=1)                                # [B,1+T,d]
    mask = torch.cat([torch.ones(B, 1, dtype=act_mask.dtype, device=act_mask.device), act_mask], dim=1)
    pos = (mask.long().cumsum(-1) - 1).clamp_min(0)                           # contiguous; code stays adjacent
    return embeds, mask, pos


@torch.no_grad()
def sample_code(actor, prefix_emb, prefix_mask, prefix_pos, Tc, temp, suppress, av_emb, device):
    """Autoregressively sample a Tc-token code from the AV given the activation-embed prefix.
    Manual KV-cache loop (full control over the right-padded prefix mask + adjacent position_ids)."""
    B = prefix_emb.shape[0]
    out = actor(inputs_embeds=prefix_emb, attention_mask=prefix_mask, position_ids=prefix_pos, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1].float()
    mask, last_pos = prefix_mask, prefix_pos[:, -1]
    codes = []
    for _ in range(Tc):
        logits = logits.index_fill(-1, suppress, float("-inf"))
        nxt = torch.multinomial(F.softmax(logits / temp, -1), 1)             # [B,1]
        codes.append(nxt)
        last_pos = last_pos + 1
        mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=device)], dim=1)
        emb = av_emb[nxt.squeeze(1)].unsqueeze(1)                            # [B,1,d]
        out = actor(inputs_embeds=emb, attention_mask=mask, position_ids=last_pos.unsqueeze(1),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1].float()
    return torch.cat(codes, dim=1)                                           # [B,Tc]


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="dir with seq.parquet + fromscratch_meta.json")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--batch-prompts", type=int, default=32, help="documents per step")
    p.add_argument("--seq-len", type=int, default=256, help="max activations per doc (T); longer docs truncated")
    p.add_argument("--max-new-tokens", type=int, default=48, help="code length Tc")
    p.add_argument("--warmup-temp", type=float, default=1.0)
    # AV/AR architecture (vanilla init)
    p.add_argument("--av-layers", type=int, default=6)
    p.add_argument("--av-heads", type=int, default=4)
    p.add_argument("--ar-layers", type=int, default=6)
    p.add_argument("--inject-scale", type=float, default=1.0,
                   help="activation input-embeds are normalized to ‖·‖=√d·inject_scale")
    # objective
    p.add_argument("--alpha", type=float, default=0.1, help="English-prior KL(AV‖M) weight")
    p.add_argument("--alpha-warmup-frac", type=float, default=0.1)
    p.add_argument("--alpha-delay-frac", type=float, default=0.4,
                   help="recon-only for this fraction of steps before ramping the prior in")
    p.add_argument("--beta", type=float, default=0.0,
                   help="grounded faithfulness: frozen M, given the code, must predict the doc tokens")
    p.add_argument("--rel-pairs", type=int, default=1024,
                   help="relational reader: ordered code pairs scored per step")
    p.add_argument("--rel-weight", type=float, default=1.0,
                   help="weight of the pairwise relational term (0 = skip pair computation)")
    p.add_argument("--ce-weight", type=float, default=0.0,
                   help="pu14 prior-distillation: per step, sample a continuation of each doc's REAL "
                        "token prefix from the FROZEN PRIOR (generative use — still only-a-prior), "
                        "and teacher-force the AV on it with CE. Encoder-side supervision: there is "
                        "no reader to split (cf. pu11), and it initializes/holds the assignment at "
                        "the faithful orbit point (identifiability via prior-derived initialization).")
    p.add_argument("--ce-temp", type=float, default=0.7, help="sampling temp for prior continuations")
    p.add_argument("--rel-near-frac", type=float, default=0.5,
                   help="fraction of pairs drawn from K_act nearest neighbours (rest uniform)")
    p.add_argument("--reader", choices=["ar", "mprobe", "relational"], default="ar",
                   help="mprobe = PER-POSITION frozen-M identity reader: M reads [bos, c_1..c_Tc] and "
                        "its layer --m-read-layer state at position j must equal v_j (phase-aligned, "
                        "j=1..min(Tc,Tb-1)). No trainable reader anywhere -> identifiable up to "
                        "M-synonymy (zero_nla/nla_identifiability.md), and every code token gets a LOCAL "
                        "gradient (dense credit assignment — the fix for the single-target rung-1 "
                        "ceiling of pu3-pu5).")
    p.add_argument("--m-read-layer", type=int, default=None,
                   help="M hidden_states layer for the mprobe reader (default: the data's extraction "
                        "layer K, so the identity target is exactly M's own state)")
    p.add_argument("--anchor-weight", type=float, default=0.0,
                   help="pu11 data-anchoring (reader=ar only): each step the TRAINABLE AR also fits "
                        "ground-truth code pairs from the corpus — real text presented in the code "
                        "slot, its own activation trajectory as the target. No frozen reader, no "
                        "teacher; the reader's decode convention is pinned by DATA (per-position "
                        "distillation of M). Failure mode to watch: distribution splitting (AR keeps "
                        "a second, cipher convention for off-manifold codes).")
    p.add_argument("--compress-frac", type=float, default=None,
                   help="pu10 compression: from this fraction of training, shrink the code to "
                        "--compress-code-len and switch the recon target to the doc's POOLED deep "
                        "state (batch-centered mean over the last --compress-pool real positions) vs "
                        "the code's pooled states. A transcript no longer fits; the only way to steer "
                        "the pooled state with few tokens is M-semantic compression (summary-grade "
                        "codes) — the purist test of abstraction beyond re-enactment.")
    p.add_argument("--compress-code-len", type=int, default=8)
    p.add_argument("--compress-pool", type=int, default=8)
    p.add_argument("--probe-anneal", action="store_true",
                   help="adiabatic probe annealing (two-timescale, JEPA-adjacent but with a HARD "
                        "endpoint): v̂_j = (1-γ_t)·W(state_j) + γ_t·state_j with trainable W (eye "
                        "init) and γ: 0 until 0.25·steps, linear to 1.0 by 0.75·steps. Early: pu2-"
                        "grade dense gradients (rotation freedom). End: exact identity — relabeling "
                        "capacity is REMOVED, not just slowed (unlike an EMA target, which is a "
                        "slowly trainable reader and reopens the cipher orbit at the EMA timescale).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--critic-lr", type=float, default=3e-3)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-rows", type=int, default=None, help="cap #documents loaded")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="cot-oracle")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "fromscratch_meta.json").read_text())
    d = meta["d_model"]
    K = meta["layer_index"]
    read_id = meta["read_id"]
    inject_id = meta["inject_id"]
    bos_id, eos_id, pad_id = meta["bos_id"], meta["eos_id"], meta["pad_id"]
    mse_scale = resolve_target_scale(meta["mse_scale"], d)
    tok = AutoTokenizer.from_pretrained(meta["tokenizer_dir"])
    V = len(tok)
    # The AV must never EMIT the control tokens (<INJECT>/<READ>): they'd muddy the AR readout / are
    # out-of-distribution for the prior. Suppress at sample time AND mask in the teacher-forced logits.
    suppress = torch.tensor([inject_id, read_id], device=device)
    Tc = args.max_new_tokens
    T = args.seq_len
    m_read_layer = args.m_read_layer if args.m_read_layer is not None else K
    print(f"[cfg] d={d} vocab={V} K={K} read={read_id} mse_scale={mse_scale:.3f} T={T} Tc={Tc} "
          f"reader={args.reader}" + (f" m_read_layer={m_read_layer}" if args.reader == "mprobe" else ""), flush=True)

    max_pos = 1 + T + Tc + 16

    def small_llama(n_layers, heads):
        return LlamaConfig(
            vocab_size=V, hidden_size=d, num_hidden_layers=n_layers, num_attention_heads=heads,
            num_key_value_heads=heads, intermediate_size=4 * d, max_position_embeddings=max_pos,
            tie_word_embeddings=True, attention_dropout=0.0,
            bos_token_id=bos_id, eos_token_id=eos_id, pad_token_id=pad_id,
        )

    actor = AutoModelForCausalLM.from_config(small_llama(args.av_layers, args.av_heads)).to(device)
    actor.train()
    n_av = sum(t.numel() for t in actor.parameters())
    critic = NLACriticModel.from_config(small_llama(args.ar_layers, args.av_heads), eye_init=True).to(device)
    critic.train()
    n_ar = sum(t.numel() for t in critic.parameters())
    probe = None
    if args.probe_anneal:
        assert args.reader == "mprobe", "--probe-anneal only applies to the mprobe reader"
        probe = torch.nn.Linear(d, d).to(device)
        with torch.no_grad():
            probe.weight.copy_(torch.eye(d))
            probe.bias.zero_()
        print("[anneal] trainable probe (eye init), gamma 0->1 over [0.25, 0.75] of training", flush=True)
    if args.compress_frac is not None:
        assert args.reader == "mprobe", "--compress-frac only applies to the mprobe reader"
        print(f"[compress] phase B from {args.compress_frac:.2f}: code len {Tc}->{args.compress_code_len}, "
              f"target = batch-centered POOLED deep state (last {args.compress_pool} doc positions)", flush=True)
    rel_cal = None
    if args.reader == "relational":
        # pu13: ONLY-NL-prior bias. Pairwise loss |K_act(i,j) - s·K_text(i,j) - b|²; K_text from
        # black-box prior scoring (symmetric-PMI-style co-predictiveness); no reader of any kind.
        # The affine (s,b) on similarity VALUES cannot relabel meanings.
        rel_cal = torch.nn.Parameter(torch.tensor([1.0, 0.0], device=device))
        print(f"[relational] pairs/step={args.rel_pairs} near_frac={args.rel_near_frac} — no reader; "
              "prior enters as geometry (K_text) + manifold (alpha)", flush=True)
    prior = AutoModelForCausalLM.from_pretrained(meta["model_dir"]).to(device).eval()
    for t_ in prior.parameters():
        t_.requires_grad_(False)
    print(f"[models] AV={n_av/1e6:.2f}M AR={n_ar/1e6:.2f}M prior(frozen)="
          f"{sum(t_.numel() for t_ in prior.parameters())/1e6:.2f}M | max_pos={max_pos}", flush=True)

    opt = torch.optim.AdamW(actor.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    copt = torch.optim.AdamW(
        list(critic.parameters()) + (list(probe.parameters()) if probe is not None else [])
        + ([rel_cal] if rel_cal is not None else []),
        lr=args.critic_lr, betas=(0.9, 0.95), weight_decay=0.0)

    # --- data + FVE baseline (predict-the-mean over ALL real positions) ---
    acts_pad, toks_pad, lens = load_doc_sequences(data_dir / "seq.parquet", d, T, n_max=args.max_rows)
    n_docs = len(lens)
    flat_mask = (np.arange(T)[None, :] < lens[:, None])                       # [n_docs, T] bool
    flat_acts = torch.from_numpy(acts_pad[flat_mask])                         # [sum_len, d] all real activations
    sub = flat_acts[: min(len(flat_acts), 50000)]
    mu = sub.mean(0, keepdim=True)
    fve_base = dir_mse(mu.expand(len(sub), -1), sub, mse_scale).mean().item()
    # Per-position-mean baseline: predict each position t's dataset-mean activation. Residual-stream
    # trajectories are dominated by POSITIONAL structure (position t looks alike across docs), which a
    # single global mean misses — so the global-mean FVE flatters. FVE vs the positional mean (and the
    # shuffled-target gap) is the HONEST "does the code describe THIS document beyond its position" signal.
    sum_pos = (acts_pad * flat_mask[..., None]).sum(0)                        # [T,d]
    mu_pos = sum_pos / flat_mask.sum(0).clip(min=1)[:, None]                  # [T,d]
    mu_pos_t = torch.from_numpy(mu_pos.astype(np.float32))
    mu_pos_dev = mu_pos_t.to(device)                                          # positional means (centering)
    sub_n = min(n_docs, 4000)
    dpos = dir_mse(mu_pos_t[None].expand(sub_n, -1, -1).reshape(-1, d),
                   torch.from_numpy(acts_pad[:sub_n]).reshape(-1, d), mse_scale).reshape(sub_n, T)
    msk = torch.from_numpy(flat_mask[:sub_n])
    base_pos = ((dpos * msk).sum() / msk.sum()).item()
    print(f"[data] {n_docs} docs, {len(flat_acts)} activations (mean len {lens.mean():.0f}) | "
          f"dirMSE baselines: global-mean={fve_base:.4f}  positional-mean={base_pos:.4f}", flush=True)

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, group=args.wandb_group,
                   config={**vars(args), "n_av": n_av, "n_ar": n_ar, "d_model": d,
                           "n_docs": n_docs, "fve_base": fve_base, "base_pos": base_pos})
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    B = args.batch_prompts
    av_emb = actor.get_input_embeddings().weight
    ar_emb = critic.backbone.get_input_embeddings().weight
    pe = prior.get_input_embeddings().weight

    for wstep in range(args.warmup_steps):
        t0 = time.time()
        bidx = rng.integers(0, n_docs, size=B)
        Tb = int(lens[bidx].max())                                           # pad to batch-max (free speedup)
        v_seq = torch.from_numpy(acts_pad[bidx, :Tb]).to(device).float()     # [B,Tb,d]
        doc_tok = torch.from_numpy(toks_pad[bidx, :Tb]).to(device)           # [B,Tb]
        act_mask = (torch.arange(Tb, device=device)[None, :] <
                    torch.from_numpy(lens[bidx]).to(device)[:, None]).float()  # [B,Tb]

        # ---- AV: read activation trajectory, sample code, teacher-force for STE ----
        in_phaseB = args.compress_frac is not None and wstep >= args.compress_frac * args.warmup_steps
        Tc_t = args.compress_code_len if in_phaseB else Tc                   # pu10: code budget shrinks in phase B
        prefix_emb, prefix_mask, prefix_pos = build_av_prefix(actor, v_seq, act_mask, bos_id, d, args.inject_scale)
        actor.eval()
        resp = sample_code(actor, prefix_emb, prefix_mask, prefix_pos, Tc_t, args.warmup_temp, suppress, av_emb, device)
        actor.train()
        code_emb_av = av_emb[resp]                                           # [B,Tc_t,d]
        tf_emb = torch.cat([prefix_emb, code_emb_av], dim=1)                 # [B,1+Tb+Tc_t,d]
        tf_mask = torch.cat([prefix_mask, torch.ones(B, Tc_t, dtype=prefix_mask.dtype, device=device)], dim=1)
        tf_pos = (tf_mask.long().cumsum(-1) - 1).clamp_min(0)
        logits = actor(inputs_embeds=tf_emb, attention_mask=tf_mask, position_ids=tf_pos).logits
        # code token j is at seq index 1+Tb+j, predicted by logits at index Tb+j
        clog = logits[:, 1 + Tb - 1: 1 + Tb - 1 + Tc_t, :].float().index_fill(-1, suppress, -1e4)  # [B,Tc_t,V]
        soft = F.softmax(clog, -1)
        y = F.one_hot(resp, V).float() + (soft - soft.detach())             # straight-through one-hots

        if args.reader == "ar":
            # ---- AR: reconstruct ALL Tb activations via Tb <READ> slots ----
            code_emb_ar = y @ ar_emb                                        # [B,Tc,d]
            bos_ar = ar_emb[bos_id].view(1, 1, -1).expand(B, 1, -1)
            read_ar = ar_emb[read_id].view(1, 1, -1).expand(B, Tb, -1)
            ar_in = torch.cat([bos_ar, code_emb_ar, read_ar], dim=1)        # [B,1+Tc+Tb,d]
            ar_mask = torch.cat([torch.ones(B, 1 + Tc, dtype=torch.long, device=device),
                                 act_mask.long()], dim=1)
            cout = critic(inputs_embeds=ar_in, attention_mask=ar_mask)
            read_h = cout.backbone_last_hidden[:, 1 + Tc: 1 + Tc + Tb, :].float()  # [B,Tb,d] at the read slots
            read_h = normalize_activation(read_h, mse_scale)
            pred = critic.value_head(read_h.to(critic.value_head.weight.dtype)).float()  # [B,Tb,d]
            dmse = dir_mse(pred.reshape(-1, d), v_seq.reshape(-1, d), mse_scale).reshape(B, Tb)
            recon_loss = (dmse * act_mask).sum() / act_mask.sum()
            rec_mask = act_mask
            v_tgt = v_seq
            if args.anchor_weight > 0:
                # pu11 anchor stream: an independent batch of REAL documents fed in the code slot
                # ([bos, w_1..w_Ta, READ×Ta]); the AR must decode them to their OWN trajectories.
                aidx = rng.integers(0, n_docs, size=B)
                Ta = int(lens[aidx].max())
                a_v = torch.from_numpy(acts_pad[aidx, :Ta]).to(device).float()
                a_tok = torch.from_numpy(toks_pad[aidx, :Ta]).to(device)
                a_mask = (torch.arange(Ta, device=device)[None, :] <
                          torch.from_numpy(lens[aidx]).to(device)[:, None]).float()
                a_in = torch.cat([bos_ar.expand(B, 1, -1), ar_emb[a_tok],
                                  ar_emb[read_id].view(1, 1, -1).expand(B, Ta, -1)], dim=1)
                a_am = torch.cat([torch.ones(B, 1, dtype=torch.long, device=device), a_mask.long(),
                                  a_mask.long()], dim=1)
                a_out = critic(inputs_embeds=a_in, attention_mask=a_am)
                a_h = normalize_activation(a_out.backbone_last_hidden[:, 1 + Ta:, :].float(), mse_scale)
                a_pred = critic.value_head(a_h.to(critic.value_head.weight.dtype)).float()
                a_dmse = dir_mse(a_pred.reshape(-1, d), a_v.reshape(-1, d), mse_scale).reshape(B, Ta)
                anchor_loss = (a_dmse * a_mask).sum() / a_mask.sum()
        elif args.reader == "relational":
            # ---- pu13: NO reader. Pairwise geometry match: K_act (positionally-centered pooled
            # activation cosine) vs K_text (prior-only co-predictiveness, marginal-subtracted).
            # Relational terms see code swaps — the per-sample swap lemma does not apply.
            vc = (v_seq - mu_pos_dev[:Tb]) * act_mask[..., None]
            u = F.normalize(vc.sum(1) / act_mask.sum(1, keepdim=True), dim=-1)   # [B,d]
            K_act = u @ u.T                                                       # [B,B]
            # marginal: mean-token logP_M(c | bos), soft targets through STE
            bos_pe = pe[bos_id].view(1, 1, -1).expand(B, 1, -1)
            mlp = F.log_softmax(prior(inputs_embeds=torch.cat([bos_pe, y @ pe], 1)).logits.float(), -1)
            lp_marg = (y * mlp[:, :-1, :]).sum(-1).mean(1)                        # [B] (logits k predict token k+1)
            # ordered pair sample: near (by K_act) + uniform
            with torch.no_grad():
                P = min(args.rel_pairs, B * (B - 1))
                n_near = int(P * args.rel_near_frac) // B
                near = (K_act - 2 * torch.eye(B, device=device)).topk(max(n_near, 1), dim=1).indices  # [B,n_near]
                pi_n = torch.arange(B, device=device).repeat_interleave(near.shape[1])
                pj_n = near.reshape(-1)
                n_rand = max(P - len(pi_n), B)
                pi_r = torch.randint(0, B, (n_rand,), device=device)
                pj_r = (pi_r + torch.randint(1, B, (n_rand,), device=device)) % B
                pi, pj = torch.cat([pi_n, pi_r]), torch.cat([pj_n, pj_r])
            # conditional: mean-token logP_M(c_j | bos, c_i) on the sampled pairs
            pair_in = torch.cat([bos_pe[pi], (y @ pe)[pi], (y @ pe)[pj]], dim=1)  # [P,1+2Tc_t,d]
            plp = F.log_softmax(prior(inputs_embeds=pair_in).logits.float(), -1)
            lp_cond = (y[pj] * plp[:, Tc_t:-1, :]).sum(-1).mean(1)                # logits Tc..2Tc-1 predict c_j
            K_text = lp_cond - lp_marg[pj]                                        # directed PMI-style similarity
            tgt = K_act[pi, pj]
            recon_loss = args.rel_weight * ((tgt - rel_cal[0] * K_text - rel_cal[1]) ** 2).mean()
            with torch.no_grad():
                kt, ta = K_text.detach(), tgt.detach()
                rel_corr = ((kt - kt.mean()) * (ta - ta.mean())).mean() / (kt.std() * ta.std() + 1e-8)
            pred = v_tgt = rec_mask = None
        else:
            # ---- mprobe: frozen M reads [bos, c_1..c_Tc]; its layer-K state at position j must
            # equal v_j (identity readout, phase-aligned: c_j and doc token w_j share RoPE pos j).
            # One LOCAL target per code token -> dense credit assignment; no trainable reader.
            pm_in = torch.cat([pe[bos_id].view(1, 1, -1).expand(B, 1, -1), y @ pe], dim=1)  # [B,1+Tc_t,d]
            hs = prior(inputs_embeds=pm_in, output_hidden_states=True).hidden_states[m_read_layer]
            gamma_t = 1.0 if probe is None else min(1.0, max(0.0, (wstep / args.warmup_steps - 0.25) / 0.5))
            if in_phaseB:
                # pu10 phase B: POOLED deep-state matching — code's mean state vs the doc's mean
                # state over its last K real positions, both batch-centered (kills the RoPE/position
                # offset). A transcript can't fit in Tc_t tokens; only M-semantic compression scores.
                Kp = args.compress_pool
                pos_idx = torch.arange(Tb, device=device)[None, :]
                lens_b = torch.from_numpy(lens[bidx]).to(device)[:, None]
                maskK = ((pos_idx >= lens_b - Kp) & (pos_idx < lens_b)).float()          # [B,Tb]
                pooled_doc = (v_seq * maskK[..., None]).sum(1) / maskK.sum(1, keepdim=True)
                pooled_code = hs[:, 1:1 + Tc_t].float().mean(1)                          # [B,d]
                pc = pooled_code - pooled_code.mean(0, keepdim=True)
                pd = pooled_doc - pooled_doc.mean(0, keepdim=True)
                pred = pc.unsqueeze(1)
                v_tgt = pd.unsqueeze(1)
                rec_mask = torch.ones(B, 1, device=device)
                recon_loss = dir_mse(pc, pd, mse_scale).mean()
            else:
                J = min(Tc_t, Tb - 1)
                base = normalize_activation(hs[:, 1:1 + J].float(), mse_scale)   # [B,J,d] states at pos 1..J
                if probe is not None and gamma_t < 1.0:                          # pu9 adiabatic anneal W->I
                    pred = (1.0 - gamma_t) * probe(base) + gamma_t * base
                else:
                    pred = base
                v_tgt = v_seq[:, 1:1 + J]                                        # [B,J,d] doc states at pos 1..J
                rec_mask = act_mask[:, 1:1 + J]
                dmse = dir_mse(pred.reshape(-1, d), v_tgt.reshape(-1, d), mse_scale).reshape(B, J)
                recon_loss = (dmse * rec_mask).sum() / rec_mask.sum()

        # ---- English prior: KL(AV‖M) per code token (delayed/ramped) ----
        delay = int(args.alpha_delay_frac * args.warmup_steps)
        ramp = max(1, int(args.alpha_warmup_frac * args.warmup_steps))
        alpha_t = args.alpha * min(1.0, max(0, wstep - delay) / ramp)
        if alpha_t > 0:
            p_in = torch.cat([pe[bos_id].view(1, 1, -1).expand(B, 1, -1), (y @ pe)[:, :-1, :]], dim=1)
            p_logp = F.log_softmax(prior(inputs_embeds=p_in).logits.float(), -1)   # M log-dist over code
            log_soft = F.log_softmax(clog, -1)
            term = (soft * (log_soft - p_logp)).masked_fill(soft == 0, 0.0)
            kl_prior = term.sum(-1).mean()
        else:
            kl_prior = torch.zeros((), device=device)

        # ---- grounded faithfulness: M reads [bos, code, doc_tokens], predict the doc (delayed/ramped) ----
        beta_t = args.beta * min(1.0, max(0, wstep - delay) / ramp)
        if beta_t > 0:
            m_in = torch.cat([pe[bos_id].view(1, 1, -1).expand(B, 1, -1), y @ pe, pe[doc_tok]], dim=1)
            m_logp = F.log_softmax(prior(inputs_embeds=m_in).logits.float(), -1)   # [B,1+Tc+Tb,V]
            # doc token w_t sits at m_in index Tc+1+t, so the logit at index Tc+t predicts it. Score the
            # doc's next-token NLL for t=1..Tb-1 (skip t=0, the trivial leading bos).
            pred_lp = m_logp[:, Tc + 1: Tc + Tb, :].gather(-1, doc_tok[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B,Tb-1]
            fmask = act_mask[:, 1:]
            faith = -(pred_lp * fmask).sum() / fmask.sum()
        else:
            faith = torch.zeros((), device=device)

        anchor_loss = anchor_loss if (args.reader == "ar" and args.anchor_weight > 0) else torch.zeros((), device=device)
        ce_loss = torch.zeros((), device=device)
        if args.ce_weight > 0:
            # pu14 prior-distillation: sample the PRIOR's continuation of each doc's real prefix
            # (left-padded for correct generation), teacher-force the AV on it. Encoder-side
            # supervision through the prior's generative use — no reader exists to split.
            with torch.no_grad():
                lens_b = torch.from_numpy(lens[bidx]).to(device)
                lp_ids = torch.full_like(doc_tok, pad_id)
                lp_mask = torch.zeros_like(doc_tok)
                for bi in range(B):
                    Lb = int(lens_b[bi])
                    lp_ids[bi, Tb - Lb:] = doc_tok[bi, :Lb]
                    lp_mask[bi, Tb - Lb:] = 1
                gen = prior.generate(input_ids=lp_ids, attention_mask=lp_mask,
                                     max_new_tokens=Tc_t, min_new_tokens=Tc_t, do_sample=True,
                                     temperature=args.ce_temp, top_p=1.0, top_k=0,
                                     pad_token_id=pad_id, suppress_tokens=[inject_id, read_id])
                tgt_cont = gen[:, Tb:Tb + Tc_t]                               # [B,Tc_t]
            ce_emb = torch.cat([prefix_emb, av_emb[tgt_cont]], dim=1)
            ce_mask = torch.cat([prefix_mask, torch.ones(B, Tc_t, dtype=prefix_mask.dtype, device=device)], dim=1)
            ce_pos = (ce_mask.long().cumsum(-1) - 1).clamp_min(0)
            ce_logits = actor(inputs_embeds=ce_emb, attention_mask=ce_mask, position_ids=ce_pos).logits
            ce_pred = ce_logits[:, 1 + Tb - 1: 1 + Tb - 1 + Tc_t, :].float()
            ce_loss = F.cross_entropy(ce_pred.reshape(-1, V), tgt_cont.reshape(-1))
        loss = recon_loss + alpha_t * kl_prior + beta_t * faith + args.anchor_weight * anchor_loss \
            + args.ce_weight * ce_loss
        opt.zero_grad()
        copt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
        opt.step()
        copt.step()

        if wstep % 10 == 0 and args.reader == "relational":
            with torch.no_grad():
                n_uniq = len({tuple(resp[i].tolist()) for i in range(B)})
            print(f"warmup {wstep:04d} | rel {recon_loss.item():.4f} | corr {rel_corr.item():+.3f} | "
                  f"ce {ce_loss.item():.3f} | cal s{rel_cal[0].item():.2f} b{rel_cal[1].item():+.2f} | "
                  f"KL {kl_prior.item():.2f} a{alpha_t:.2f} | uniq {n_uniq}/{B} | Tb {Tb} | {time.time()-t0:.1f}s", flush=True)
            if wstep % 50 == 0:
                print("    e.g. " + repr(tok.decode(resp[0], skip_special_tokens=False))[:200], flush=True)
            if not args.no_wandb:
                wandb.log({"warmup/rel_loss": recon_loss.item(), "warmup/rel_corr": rel_corr.item(),
                           "warmup/kl_prior": kl_prior.item(), "warmup/alpha_t": alpha_t,
                           "warmup/code_uniq": n_uniq / B}, step=wstep)
        elif wstep % 10 == 0:
            with torch.no_grad():
                fve = 1.0 - recon_loss.item() / fve_base
                shuf_idx = torch.randperm(B, device=device)
                Jp = pred.shape[1]
                shuf = (dir_mse(pred.reshape(-1, d), v_tgt[shuf_idx].reshape(-1, d), mse_scale).reshape(B, Jp)
                        * rec_mask).sum() / rec_mask.sum()
                fve_shuf = 1.0 - shuf.item() / fve_base
                fve_pos = 1.0 - recon_loss.item() / base_pos     # doc-specific FVE (honest headline)
                gap = fve - fve_shuf                              # doc-specific signal (independent check)
                n_uniq = len({tuple(resp[i].tolist()) for i in range(B)})
            print(f"warmup {wstep:04d} | FVEpos {fve_pos*100:5.1f}% gap {gap*100:4.1f}% | "
                  f"FVE {fve*100:4.1f}%(shuf {fve_shuf*100:4.1f}%) | recon {recon_loss.item():.3f} | "
                  f"KL {kl_prior.item():.2f} a{alpha_t:.2f} | faith {faith.item():.3f} b{beta_t:.2f} | "
                  + (f"anch {anchor_loss.item():.3f} | " if args.anchor_weight > 0 else "")
                  + f"uniq {n_uniq}/{B} | Tb {Tb} | {time.time()-t0:.1f}s", flush=True)
            if wstep % 50 == 0:
                print("    e.g. " + repr(tok.decode(resp[0], skip_special_tokens=False))[:200], flush=True)
            if not args.no_wandb:
                wandb.log({"warmup/fve": fve, "warmup/fve_shuf": fve_shuf, "warmup/fve_pos": fve_pos,
                           "warmup/gap": gap, "warmup/recon": recon_loss.item(),
                           "warmup/fve_pct": fve * 100, "warmup/code_uniq": n_uniq / B,
                           "warmup/kl_prior": kl_prior.item(), "warmup/alpha_t": alpha_t,
                           "warmup/faith": faith.item(), "warmup/beta_t": beta_t, "warmup/Tb": Tb}, step=wstep)
        if (wstep + 1) % max(1, args.warmup_steps // 10) == 0:
            d_out = save_dir / f"warmup_{wstep+1:06d}"
            d_out.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(str(d_out / "av"))
            torch.save(critic.state_dict(), d_out / "ar.pt")
            if probe is not None:
                torch.save(probe.state_dict(), d_out / "probe.pt")

    print("done.", flush=True)
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
