"""From-scratch NLA GRPO: vanilla AV + AR, emergent code + English-language prior.

Unlike `train_rl_self_contained.py` (which warm-starts AV/AR from a subject model and SFTs
them first), this trains a Natural Language Autoencoder FROM SCRATCH on a tiny model:

  - AV (actor): a vanilla-initialised small Llama (NOT inherited from any model). Reads an
    activation `v` injected at a <INJECT> marker (Karvonen ADD-hook, residual after layer 1)
    and emits a short token "code". Full fine-tuned (no LoRA — LoRA on a random frozen base
    can't learn).
  - AR (critic): a vanilla-initialised NLACriticModel (truncated Llama + identity value head),
    co-trained from random. Reads `[<bos>] code [<READ>]`, predicts v̂ at the <READ> position.
  - M (frozen): the tiny pretrained subject+prior model. Subject = source of `v`; English
    prior = scores P(code) so the bottleneck stays readable instead of an arbitrary cipher.

Objective per rollout:  reward = −dirMSE(AR(code), v)  +  α · mean_t logP_M(code_t | code_<t)
GRPO normalises per-prompt over G samples; an entropy bonus keeps exploration alive.

Why this differs from the tag-gated reward: a random AV never emits `<explanation>` tags, so
the original −2.0 failure floor → zero advantage → no learning. Here every rollout is scored.

Run (single GPU):
  python -m nla.train_rl_fromscratch --data-dir /workspace/nla/artifacts/data \
    --save-dir /workspace/nla/runs/r0 --num-steps 2000 --batch-prompts 64 --group-size 16
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

import wandb

from nla.injection import karvonen_inject_in_residual
from nla.models import NLACriticModel
from nla.schema import normalize_activation, resolve_target_scale


# ----------------------------------------------------------------------------- hook
def register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1, inject_scale=1.0):
    """Forward hook on `model.model.layers[layer_idx]` that ADD-injects vectors_ref[0]
    ([N, d], one row per sequence, in input_ids row order) at each row's marker. Hard-asserts
    the marker count matches N inside karvonen_inject_in_residual (loud, not silent)."""
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
        if input_ids is None or resid.shape[1] < 2:   # decode step (single new token): no-op
            return output
        v = vectors_ref[0]
        if v is None:
            return output
        injected = karvonen_inject_in_residual(input_ids, resid, v, inj_id, left_id, right_id, inject_scale=inject_scale)
        return injected if rest is None else (injected, *rest)

    model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    inner = model
    while hasattr(inner, "model") and not hasattr(inner, "layers"):
        inner = inner.model
    inner.layers[layer_idx].register_forward_hook(layer_hook)


# ----------------------------------------------------------------------------- data
def load_activations(parquet_path, n_max=None):
    """Returns (acts [N,d] float32, conts [N,KCONT] int64) — the activation and its continuation."""
    pf = pq.ParquetFile(parquet_path)
    av, ct, n = [], [], 0
    for rg in range(pf.num_row_groups):
        if n_max is not None and n >= n_max:
            break
        t = pf.read_row_group(rg, columns=["activation_vector", "continuation"])
        a = t.column("activation_vector").combine_chunks()
        av.append(a.values.to_numpy(zero_copy_only=False).reshape(len(a), -1).astype(np.float32))
        c = t.column("continuation").combine_chunks()
        ct.append(c.values.to_numpy(zero_copy_only=False).reshape(len(c), -1).astype(np.int64))
        n += len(a)
    acts, conts = np.concatenate(av, 0), np.concatenate(ct, 0)
    if n_max is not None:
        acts, conts = acts[:n_max], conts[:n_max]
    return acts, conts


def dir_mse(pred, gold, scale):
    """Per-sample direction MSE = MSE(normalize(pred), normalize(gold)). Returns [N]."""
    pn = normalize_activation(pred, scale)
    gn = normalize_activation(gold, scale)
    return ((pn - gn) ** 2).mean(dim=-1)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="dir with rl.parquet + fromscratch_meta.json")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-steps", type=int, default=2000)
    p.add_argument("--batch-prompts", type=int, default=64)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--min-new-tokens", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    # AV/AR architecture (vanilla init)
    p.add_argument("--av-layers", type=int, default=6)
    p.add_argument("--av-heads", type=int, default=4)
    p.add_argument("--ar-layers", type=int, default=6)
    p.add_argument("--inject-scale", type=float, default=1.0,
                   help="multiply the Karvonen-injected v component (h+scale*||h||*v/||v||). "
                        ">1 makes the code more activation-dependent (stronger encoder signal).")
    p.add_argument("--tie-m-embed", action="store_true",
                   help="freeze AV+AR token embeddings (and the AV's tied unembedding) to frozen M's "
                        "embedding table. Purist structural anchor: no extra loss term, but the "
                        "channel's token geometry = the prior's, so the cipher relabeling symmetry "
                        "is broken toward M-semantics. v lives in M-space, so with an M-geometry "
                        "unembedding the initialization-nearest code is the semantic one. "
                        "(pu1 result: freezing kills the STE bootstrap — prefer --tie-m-embed-init.)")
    p.add_argument("--tie-m-embed-init", action="store_true",
                   help="initialise AV+AR token embeddings from M's table but keep them TRAINABLE "
                        "(inductive bias without the bootstrap-killing freeze).")
    p.add_argument("--tie-ar-embed-frozen", action="store_true",
                   help="freeze ONLY the AR's input embeddings to M's table; AV fully free.")
    p.add_argument("--reader", choices=["ar", "mprobe", "anneal"], default="ar",
                   help="reconstruction reader. 'ar' = co-trained NLACriticModel (cipher-capable). "
                        "'mprobe' = FROZEN M reads [bos, code]; v̂ = linear probe on M's layer "
                        "--m-read-layer state at the last code token. The reader has no trainable "
                        "relabeling capacity beyond one linear map, so high FVE requires the code to "
                        "be text whose M-meaning carries v — the cipher channel is closed by "
                        "construction, with recon as the ONLY loss (purist). The residual stream "
                        "makes 'paraphrase the context' the init-nearest solution (probe≈eye when "
                        "read layer == v's layer). 'anneal' = AR bootstraps, then blends to mprobe "
                        "(λ: 0 until 0.3*steps, linear to 1.0 by 0.7*steps).")
    p.add_argument("--m-read-layer", type=int, default=4,
                   help="M hidden_states layer the mprobe reads (4 = M's 2/3-depth semantic layer)")
    p.add_argument("--init-av", default=None,
                   help="warm-start the AV from a prior stage's saved av/ dir (position-curriculum "
                        "chaining; still tabula-rasa — the init is our own earlier stage)")
    p.add_argument("--probe-mode", choices=["linear", "eye", "eyebias"], default="linear",
                   help="mprobe head. 'linear' = trainable W,b (pu2 result: W absorbs a rotation of "
                        "M's semantic space -> meaningful-but-ARBITRARY codebook). 'eye' = W=I,b=0 "
                        "fully FROZEN: v̂ IS M's normalized state of the code — the only way to "
                        "reconstruct is to put M in the state the context put it in (quote/"
                        "paraphrase). 'eyebias' = frozen I, trainable bias (recentering only — "
                        "cannot relabel).")
    # objective
    p.add_argument("--alpha", type=float, default=0.1, help="English-prior fluency weight")
    p.add_argument("--alpha-warmup-frac", type=float, default=0.1,
                   help="ramp alpha 0->alpha linearly over this fraction of steps, AFTER the delay.")
    p.add_argument("--alpha-delay-frac", type=float, default=0.4,
                   help="keep alpha=0 (recon-only) for this fraction of steps before ramping fluency "
                        "in. Recon bootstrapping is fragile — ANY early fluency pressure prevents it, "
                        "so establish reconstruction first, then make the code readable.")
    p.add_argument("--beta", type=float, default=0.0,
                   help="GROUNDED-faithfulness weight. Frozen M reads the code and must reproduce v's "
                        "layer-K representation, so the code MEANS the same to a grounded English reader "
                        "(M) as the context did — improves semantic quality. M is only a frozen scorer; "
                        "the AV bottleneck stays tabula-rasa. Same delay/ramp schedule as alpha.")
    p.add_argument("--entropy-coef", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="differentiable Gumbel-straight-through autoencoder warm-up BEFORE GRPO. "
                        "RL-from-scratch can't bootstrap (the random encoder routes v into the code "
                        "too weakly for the AR to decode, so the recon advantage is ~0). The warm-up "
                        "makes the bottleneck differentiable so the AR's reconstruction gradient flows "
                        "back through the code into the AV, directly forcing v to be encoded. 0 = skip.")
    p.add_argument("--warmup-temp", type=float, default=1.0)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--wandb-project", default="cot-oracle")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    assert args.reader == "ar" or args.num_steps == 0, "mprobe/anneal readers are warmup-only (no GRPO phase)"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "fromscratch_meta.json").read_text())
    d = meta["d_model"]
    K = meta["layer_index"]          # subject extraction layer (for the grounded-faithfulness reader)
    kcont = meta["kcont"]            # length of the stored continuation
    mk = meta["marker"]
    inj_id, left_id, right_id, read_id = mk["inject_id"], mk["left_id"], mk["right_id"], mk["read_id"]
    bos_id, eos_id, pad_id = meta["bos_id"], meta["eos_id"], meta["pad_id"]
    prompt_ids = mk["prompt_ids"]
    plen = len(prompt_ids)
    mse_scale = resolve_target_scale(meta["mse_scale"], d)
    tok = AutoTokenizer.from_pretrained(meta["tokenizer_dir"])
    V = len(tok)
    # The AV must never EMIT the control tokens: a generated <INJECT> would make the layer
    # hook's marker scan find >1 marker per row (assert fail); <READ> would muddy the AR readout.
    # Suppress them at sample time AND mask them in the old/new logits so the importance ratio
    # is computed under the same (suppressed) distribution that actually generated the tokens.
    suppress = torch.tensor([inj_id, read_id], device=device)
    print(f"[cfg] d={d} vocab={V} marker_prompt={prompt_ids} inj={inj_id} read={read_id} "
          f"mse_scale={mse_scale:.3f}", flush=True)

    max_pos = plen + args.max_new_tokens + 8

    def small_llama(n_layers, heads):
        return LlamaConfig(
            vocab_size=V, hidden_size=d, num_hidden_layers=n_layers, num_attention_heads=heads,
            num_key_value_heads=heads, intermediate_size=4 * d, max_position_embeddings=max_pos,
            tie_word_embeddings=True, attention_dropout=0.0,
            bos_token_id=bos_id, eos_token_id=eos_id, pad_token_id=pad_id,
        )

    # --- AV (vanilla, full-FT) ---
    if args.init_av:
        actor = AutoModelForCausalLM.from_pretrained(args.init_av).to(device)
        actor.config.max_position_embeddings = max(actor.config.max_position_embeddings, max_pos)
        print(f"[init] AV warm-started from {args.init_av}", flush=True)
    else:
        actor = AutoModelForCausalLM.from_config(small_llama(args.av_layers, args.av_heads)).to(device)
    actor.train()
    n_av = sum(t.numel() for t in actor.parameters())
    # --- AR (vanilla, co-trained) ---
    critic = NLACriticModel.from_config(small_llama(args.ar_layers, args.av_heads), eye_init=True).to(device)
    critic.train()
    n_ar = sum(t.numel() for t in critic.parameters())
    # --- M (frozen subject+prior) ---
    prior = AutoModelForCausalLM.from_pretrained(meta["model_dir"]).to(device).eval()  # fp32; tiny
    for t in prior.parameters():
        t.requires_grad_(False)
    if args.tie_m_embed or args.tie_m_embed_init:
        E = prior.get_input_embeddings().weight.data
        actor.get_input_embeddings().weight.data.copy_(E)      # tied -> lm_head shares this Parameter
        critic.backbone.get_input_embeddings().weight.data.copy_(E)
        if args.tie_m_embed:                                    # freeze variant (kills bootstrap, kept for record)
            actor.get_input_embeddings().weight.requires_grad_(False)
            critic.backbone.get_input_embeddings().weight.requires_grad_(False)
        print(f"[tieM] AV+AR token embeddings {'frozen to' if args.tie_m_embed else 'initialised from'} M's table", flush=True)
    if args.tie_ar_embed_frozen:
        critic.backbone.get_input_embeddings().weight.data.copy_(prior.get_input_embeddings().weight.data)
        critic.backbone.get_input_embeddings().weight.requires_grad_(False)
        print("[tieAR] AR input embeddings frozen to M's table (AV free)", flush=True)
    probe = None
    if args.reader in ("mprobe", "anneal"):
        probe = torch.nn.Linear(d, d).to(device)
        with torch.no_grad():
            probe.weight.copy_(torch.eye(d))
            probe.bias.zero_()
        if args.probe_mode in ("eye", "eyebias"):
            probe.weight.requires_grad_(False)
            if args.probe_mode == "eye":
                probe.bias.requires_grad_(False)
        print(f"[mprobe] reader = frozen M layer {args.m_read_layer} + probe mode={args.probe_mode}", flush=True)
    print(f"[models] AV={n_av/1e6:.2f}M AR={n_ar/1e6:.2f}M prior(frozen)={sum(t.numel() for t in prior.parameters())/1e6:.2f}M", flush=True)

    vectors_ref = [None]
    register_karvonen_hook(actor, vectors_ref, inj_id, left_id, right_id, layer_idx=1, inject_scale=args.inject_scale)

    opt = torch.optim.AdamW(actor.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    copt = torch.optim.AdamW(
        list(critic.parameters()) + ([t for t in probe.parameters() if t.requires_grad] if probe is not None else []),
        lr=args.critic_lr, betas=(0.9, 0.95), weight_decay=0.0)

    # --- data + FVE baseline (predict-the-mean) ---
    acts, conts = load_activations(data_dir / "rl.parquet", n_max=args.max_rows)
    acts_t = torch.from_numpy(acts)
    mu = acts_t[: min(len(acts_t), 8000)].mean(0, keepdim=True)
    fve_base = dir_mse(mu.expand(min(len(acts_t), 8000), -1), acts_t[: min(len(acts_t), 8000)], mse_scale).mean().item()
    print(f"[data] {len(acts)} activations | predict-the-mean dirMSE baseline = {fve_base:.4f}", flush=True)

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, group=args.wandb_group,
                   config={**vars(args), "n_av": n_av, "n_ar": n_ar, "d_model": d, "fve_base": fve_base})
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_every = max(1, args.num_steps // 25)

    rng = np.random.default_rng(args.seed)
    B, G = args.batch_prompts, args.group_size
    N = B * G
    prompt_row = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    examples_seen = tokens_seen = 0

    # ===================== differentiable AE warm-up (Gumbel straight-through) =====================
    # Bootstrap the autoencoder before GRPO. Hard-sample a fixed-length code (so the AR/AV see real
    # tokens), then a teacher-forced AV forward gives the code logits; straight-through one-hots
    # (forward = the sampled code, backward = softmax) feed the AR via input embeddings, so the
    # reconstruction gradient flows back through the code into the AV. Both AV and AR train.
    Tc = args.max_new_tokens
    for wstep in range(args.warmup_steps):
        t0 = time.time()
        bidx = rng.integers(0, len(acts), size=N)
        v = torch.from_numpy(acts[bidx]).to(device).float()                  # [N, d]
        cont = torch.from_numpy(conts[bidx]).to(device)                      # [N, KCONT] continuation ids
        ids = prompt_row.unsqueeze(0).expand(N, -1).contiguous()
        actor.eval()
        vectors_ref[0] = v
        with torch.no_grad():
            gen = actor.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                min_new_tokens=Tc, max_new_tokens=Tc, do_sample=True, temperature=args.warmup_temp,
                top_p=1.0, top_k=0, repetition_penalty=1.0, pad_token_id=pad_id,
                suppress_tokens=[inj_id, read_id, eos_id], return_dict_in_generate=True)
        vectors_ref[0] = None
        seq = gen.sequences
        resp = seq[:, plen:]                                                  # [N, Tc] (no eos/control)
        actor.train()
        vectors_ref[0] = v
        logits = actor(input_ids=seq, attention_mask=torch.ones_like(seq)).logits
        vectors_ref[0] = None
        # suppress with a large FINITE negative (not -inf): softmax there underflows to 0, but
        # log_softmax stays finite, so the KL term 0*log_soft = 0 instead of 0*(-inf) = NaN.
        clog = logits[:, plen - 1:-1, :].float().index_fill(-1, suppress, -1e4)   # [N, Tc, V]
        soft = F.softmax(clog, -1)
        y = F.one_hot(resp, V).float() + (soft - soft.detach())              # straight-through one-hots
        ar_loss = probe_loss = None
        if args.reader in ("ar", "anneal"):
            ar_emb = critic.backbone.get_input_embeddings().weight           # [V, d]
            code_emb = y @ ar_emb                                            # [N, Tc, d]
            bos_e = ar_emb[bos_id].view(1, 1, -1).expand(N, 1, -1)
            read_e = ar_emb[read_id].view(1, 1, -1).expand(N, 1, -1)
            ar_in = torch.cat([bos_e, code_emb, read_e], dim=1)              # [N, Tc+2, d]
            critic.train()
            cout = critic(inputs_embeds=ar_in,
                          attention_mask=torch.ones(N, Tc + 2, dtype=torch.long, device=device))
            bb = normalize_activation(cout.backbone_last_hidden[:, -1].float(), mse_scale)
            pred = critic.value_head(bb.to(critic.value_head.weight.dtype)).float()
            ar_loss = dir_mse(pred, v, mse_scale).mean()
        if args.reader in ("mprobe", "anneal"):
            # frozen M reads [bos, code]; v̂ = probe(M's layer-K state at the last code token).
            # Gradient flows through frozen M into the AV (STE) — recon through the prior's
            # understanding is the ONLY path, so the cipher channel is closed by construction.
            pm_e = prior.get_input_embeddings().weight
            pm_in = torch.cat([pm_e[bos_id].view(1, 1, -1).expand(N, 1, -1), y @ pm_e], dim=1)
            hs = prior(inputs_embeds=pm_in, output_hidden_states=True).hidden_states[args.m_read_layer]
            mb = normalize_activation(hs[:, -1].float(), mse_scale)
            pred_p = probe(mb)
            probe_loss = dir_mse(pred_p, v, mse_scale).mean()
        if args.reader == "ar":
            recon_loss = ar_loss
        elif args.reader == "mprobe":
            recon_loss, pred = probe_loss, pred_p
        else:                                                                # anneal: AR bootstraps, M-probe takes over
            lam = min(1.0, max(0.0, (wstep - 0.3 * args.warmup_steps) / (0.4 * args.warmup_steps)))
            recon_loss = (1.0 - lam) * ar_loss + lam * probe_loss
            pred = pred_p if lam >= 0.5 else pred
        # English-prior fluency on the SOFT code (differentiable): expected log P_M(code) =
        # soft cross-entropy under the frozen prior M (teacher-forced on the soft code). Gradient
        # flows through M into the AV (M frozen, so don't no_grad). alpha ramps in so the AE
        # establishes reconstruction before fluency pulls the code toward English.
        delay = int(args.alpha_delay_frac * args.warmup_steps)
        ramp = max(1, int(args.alpha_warmup_frac * args.warmup_steps))
        alpha_t = args.alpha * min(1.0, max(0, wstep - delay) / ramp)
        # Fluency = KL(AV || M) per token, NOT max log P_M: maximizing likelihood collapses to a
        # repeated high-frequency token (fluent but empty). KL matches M's full DISTRIBUTION (diverse
        # English) — its entropy term forbids the repeated-token collapse, the RLHF reference-KL.
        if alpha_t > 0:
            pe = prior.get_input_embeddings().weight                          # [V, d]
            p_in = torch.cat([pe[bos_id].view(1, 1, -1).expand(N, 1, -1), (y @ pe)[:, :-1, :]], dim=1)
            p_logp = F.log_softmax(prior(inputs_embeds=p_in).logits.float(), -1)  # [N, Tc, V] M log-dist
            log_soft = F.log_softmax(clog, -1)                               # AV log-dist
            term = (soft * (log_soft - p_logp)).masked_fill(soft == 0, 0.0)  # avoid 0*(-inf)=NaN
            kl_prior = term.sum(-1).mean()
        else:
            kl_prior = torch.zeros((), device=device)
        # GROUNDED faithfulness: a frozen English reader (M) reads the code and must reproduce v's
        # representation at the extraction layer. Only a SEMANTICALLY faithful code makes grounded M
        # arrive at the same state as the context did → improves semantic quality. M is a frozen
        # scorer (gradient flows through its activations to the AV, never into M's weights), so the
        # AV/AR bottleneck stays tabula-rasa. Same delay/ramp schedule as the prior.
        beta_t = args.beta * min(1.0, max(0, wstep - delay) / ramp)
        if beta_t > 0:
            # M reads [bos, soft_code, continuation] teacher-forced; faith = NLL of the activation's
            # REAL continuation under frozen M given the code. Low NLL = the code lets grounded M
            # predict what the context actually goes on to say → the code captures v's predictive
            # content (semantic faithfulness). Gradient flows through M into the code; M stays frozen.
            me = prior.get_input_embeddings().weight
            m_in = torch.cat([me[bos_id].view(1, 1, -1).expand(N, 1, -1), y @ me, me[cont]], dim=1)
            m_logp = F.log_softmax(prior(inputs_embeds=m_in).logits.float(), -1)   # [N, 1+Tc+KCONT, V]
            ll = m_logp[:, Tc:Tc + kcont, :].gather(-1, cont.unsqueeze(-1)).squeeze(-1)  # [N, KCONT]
            cmask = ((((cont == eos_id).cumsum(1)) == 0) | (cont == eos_id)).float()    # to & incl 1st eos
            faith = -(ll * cmask).sum() / cmask.sum()                        # mean continuation NLL
        else:
            faith = torch.zeros((), device=device)
        loss = recon_loss + alpha_t * kl_prior + beta_t * faith
        opt.zero_grad()
        copt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
        opt.step()
        copt.step()
        if wstep % 10 == 0:
            with torch.no_grad():
                fve = 1.0 - recon_loss.item() / fve_base
                shuf = dir_mse(pred.detach(), v[torch.randperm(N, device=device)], mse_scale).mean().item()
                fve_shuf = 1.0 - shuf / fve_base
                n_uniq = len({tuple(resp[i].tolist()) for i in range(N)})
                fve_p = (1.0 - probe_loss.item() / fve_base) if probe_loss is not None else None
            print(f"warmup {wstep:04d} | FVE {fve*100:5.1f}% (shuf {fve_shuf*100:4.1f}%) | "
                  + (f"fveP {fve_p*100:5.1f}% | " if fve_p is not None else "")
                  + f"recon {recon_loss.item():.3f} | KL {kl_prior.item():.2f} a{alpha_t:.2f} | "
                  f"faith {faith.item():.3f} b{beta_t:.2f} | uniq {n_uniq}/{N} | {time.time()-t0:.1f}s", flush=True)
            if wstep % 50 == 0:
                print("    e.g. " + repr(tok.decode(resp[0], skip_special_tokens=False))[:160], flush=True)
            if not args.no_wandb:
                wandb.log({"warmup/fve": fve, "warmup/fve_shuf": fve_shuf, "warmup/recon": recon_loss.item(),
                           "warmup/fve_pct": fve * 100, "warmup/code_uniq": n_uniq / N,
                           "warmup/kl_prior": kl_prior.item(), "warmup/alpha_t": alpha_t,
                           "warmup/faith": faith.item(), "warmup/beta_t": beta_t,
                           **({"warmup/fve_probe": fve_p} if fve_p is not None else {})}, step=wstep)
        if (wstep + 1) % max(1, args.warmup_steps // 10) == 0:
            d_out = save_dir / f"warmup_{wstep+1:06d}"
            d_out.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(str(d_out / "av"))
            torch.save(critic.state_dict(), d_out / "ar.pt")
            if probe is not None:
                torch.save(probe.state_dict(), d_out / "probe.pt")
    if args.warmup_steps:
        print(f"[warmup] done ({args.warmup_steps} steps)", flush=True)

    for step in range(args.num_steps):
        t0 = time.time()
        idx = rng.integers(0, len(acts), size=B)
        v_prompts = torch.from_numpy(acts[idx]).to(device).float()          # [B, d]
        v_batch = v_prompts.repeat_interleave(G, dim=0)                       # [N, d], row-major groups
        input_ids = prompt_row.unsqueeze(0).expand(N, -1).contiguous()       # [N, plen] (identical -> no pad)

        # ---- rollout (vectorized, single generate) ----
        actor.eval()
        vectors_ref[0] = v_batch
        with torch.no_grad():
            gen = actor.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                min_new_tokens=args.min_new_tokens, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                repetition_penalty=1.0, pad_token_id=pad_id, eos_token_id=eos_id,
                suppress_tokens=[inj_id, read_id],
                return_dict_in_generate=True, output_logits=True,
            )
        vectors_ref[0] = None
        seq = gen.sequences                                                  # [N, plen+new]
        new = seq.shape[1] - plen
        resp = seq[:, plen:]                                                 # [N, new]
        old_logits = torch.stack(gen.logits, dim=1).float().index_fill(-1, suppress, float("-inf"))
        old_lp = F.log_softmax(old_logits, -1).gather(-1, resp.unsqueeze(-1)).squeeze(-1)  # [N,new]
        is_eos = resp == eos_id
        resp_mask = ((is_eos.cumsum(1) == 0) | is_eos).float()               # incl first eos, excl trailing pad
        rlen = resp_mask.sum(1).clamp(min=1)                                  # [N]

        # ---- AR forward (grad ON; reward uses detached pred, co-train uses grad) ----
        Lr = int(rlen.max().item())
        ar_ids = torch.full((N, Lr + 2), pad_id, dtype=torch.long, device=device)
        ar_mask = torch.zeros((N, Lr + 2), dtype=torch.long, device=device)
        ar_ids[:, 0] = bos_id
        for i in range(N):
            k = int(rlen[i].item())
            ar_ids[i, 1:1 + k] = resp[i, :k]
            ar_ids[i, 1 + k] = read_id
            ar_mask[i, : 2 + k] = 1
        critic.train()
        cout = critic(input_ids=ar_ids, attention_mask=ar_mask)
        last = ar_mask.sum(1) - 1
        bb = cout.backbone_last_hidden[torch.arange(N, device=device), last].float()
        bb = normalize_activation(bb, mse_scale)
        pred = critic.value_head(bb.to(critic.value_head.weight.dtype)).float()   # [N, d]

        # ---- English-prior fluency: mean_t logP_M(resp_t | bos, resp_<t) ----
        pr_ids = torch.full((N, Lr + 1), pad_id, dtype=torch.long, device=device)
        pr_ids[:, 0] = bos_id
        pr_ids[:, 1:1 + Lr] = resp[:, :Lr]
        with torch.no_grad():
            pl = F.log_softmax(prior(input_ids=pr_ids).logits.float(), -1)    # [N, Lr+1, V]
            tgt = resp[:, :Lr]
            tok_lp = pl[:, :Lr, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # [N, Lr]
            m = resp_mask[:, :Lr]
            fluency = (tok_lp * m).sum(1) / m.sum(1).clamp(min=1)              # [N] mean-token logp

        # ---- reward + group-relative advantage ----
        recon = -dir_mse(pred.detach(), v_batch, mse_scale)                   # [N], in [-~2, 0]
        warm = int(args.alpha_warmup_frac * args.num_steps)
        alpha_t = args.alpha * (min(1.0, step / warm) if warm > 0 else 1.0)
        reward = recon + alpha_t * fluency
        rw = reward.view(B, G)
        adv = ((rw - rw.mean(1, keepdim=True)) / (rw.std(1, keepdim=True) + 1e-6)).view(N).detach()

        # ---- AV GRPO update (single forward, full-FT) ----
        actor.train()
        vectors_ref[0] = v_batch
        new_logits = actor(input_ids=seq, attention_mask=torch.ones_like(seq)).logits  # [N, L, V]
        vectors_ref[0] = None
        # response tokens are seq[:, plen:]; predicted by logits[:, plen-1:-1]. Mask control
        # tokens to match the suppressed sampling distribution (consistent importance ratio).
        pred_logits = new_logits[:, plen - 1:-1, :].float().index_fill(-1, suppress, float("-inf"))
        plp = F.log_softmax(pred_logits, -1)                                  # [N, new, V]
        new_lp = plp.gather(-1, resp.unsqueeze(-1)).squeeze(-1)              # [N,new]
        ratio = torch.exp(new_lp - old_lp.detach())
        surr = torch.minimum(ratio * adv[:, None], torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv[:, None])
        pe = plp.exp()
        ent = -(pe * plp.masked_fill(pe == 0, 0.0)).sum(-1)                   # [N,new]; avoid 0*(-inf)=NaN
        m = resp_mask
        pg_loss = -(surr * m).sum() / m.sum()
        ent_loss = -(ent * m).sum() / m.sum()
        av_loss = pg_loss + args.entropy_coef * ent_loss
        opt.zero_grad()
        av_loss.backward()
        av_gn = torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        opt.step()

        # ---- AR co-train (supervised dirMSE on the same codes; detached from AV) ----
        crit_loss = dir_mse(pred, v_batch, mse_scale).mean()
        copt.zero_grad()
        crit_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
        copt.step()

        # ---- metrics ----
        examples_seen += N
        tokens_seen += int(m.sum().item())
        with torch.no_grad():
            recon_mean = recon.mean().item()
            fve = 1.0 - (-recon_mean) / fve_base
            shuf = -dir_mse(pred.detach(), v_batch[torch.randperm(N, device=device)], mse_scale).mean().item()
            fve_shuf = 1.0 - (-shuf) / fve_base
            n_unique = len({tuple(resp[i, : int(rlen[i])].tolist()) for i in range(N)})
        if step % 10 == 0:
            print(f"step {step:04d} | FVE {fve*100:5.1f}% (shuf {fve_shuf*100:4.1f}%) | "
                  f"flu {fluency.mean().item():.2f} a{alpha_t:.2f} | ent {ent_loss.item()*-1:.2f} | "
                  f"uniq {n_unique}/{N} | len {rlen.float().mean().item():.1f} | "
                  f"crit {crit_loss.item():.3f} | {time.time()-t0:.1f}s", flush=True)

        log = {
            "step": step, "examples_seen": examples_seen, "tokens_seen": tokens_seen,
            "reward": reward.mean().item(), "recon": recon_mean, "fluency": fluency.mean().item(),
            "alpha_t": alpha_t,
            "fve": fve, "fve_pct": fve * 100, "fve_shuffled": fve_shuf, "fve_baseline": fve_base,
            "entropy": -ent_loss.item(), "code_unique_frac": n_unique / N, "resp_len": rlen.float().mean().item(),
            "av_loss": av_loss.item(), "pg_loss": pg_loss.item(), "critic_loss": crit_loss.item(),
            "av_grad_norm": float(av_gn), "adv_std": adv.std().item(), "wall_s": time.time() - t0,
        }

        # ---- periodic eval: decode + log sample codes (read the emergent verbalizations) ----
        if args.eval_every and step % args.eval_every == 0:
            samples = []
            for i in range(min(8, N)):
                code = tok.decode(resp[i, : int(rlen[i])], skip_special_tokens=False)
                samples.append([step, round(recon[i].item(), 3), round(fluency[i].item(), 2), code])
            for s in samples[:3]:
                print(f"    [code r={s[1]} flu={s[2]}] {s[3]!r}", flush=True)
            if not args.no_wandb:
                log["samples"] = wandb.Table(columns=["step", "recon", "fluency", "code"], data=samples)

        if not args.no_wandb:
            wandb.log(log, step=step)
        if (step + 1) % save_every == 0:
            d_out = save_dir / f"iter_{step+1:06d}"
            d_out.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(str(d_out / "av"))
            torch.save(critic.state_dict(), d_out / "ar.pt")

    print("done.", flush=True)
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
