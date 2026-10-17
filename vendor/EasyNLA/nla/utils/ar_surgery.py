"""AR depth-annealing controller (an1 paradigm).

The an1 setup feeds the CONTEXT to both AV and AR (bottleneck z prepended to
the AR's input). Since the reconstruction target — the subject's layer-42
final-token activation — is a deterministic function of the context, a
deep-enough AR recomputes it from ctx alone and the reward goes z-independent
(FVE -> 1 with zero bottleneck signal; GRPO group sigma -> 0). The mitigation
is to ANNEAL THE AR'S DEPTH: when the no-z (ctx-only) arm shows the AR is
bypassing the bottleneck, physically DELETE decoder blocks from the middle of
its stack (freezing is not enough — a frozen deep AR still CONTAINS the
recompute circuit), heal, cool down, repeat. Equilibrium: a shallow AR that
needs z to fill the computation gap — z must verbalize what the deleted
layers used to compute.

Trigger signal (NOT raw FVE — raw FVE can't distinguish "AR healed and uses
z" from "AR recomputes from ctx"):
    cut when  fve_noz >= tau                     (ctx-only arm near ceiling)
          or  fve_full - fve_noz <= eps AND fve_full >= tau
                                                 (z adds ~nothing at high FVE)
subject to cooldown, a heal window after each surgery, and a min-depth floor.

Surgery mechanics: the cut is a contiguous, interleave-aligned span (the
Qwen3.6 hybrid stacks (linear,linear,linear,full)x… — cutting whole groups of
`full_attention_interval` keeps the pattern regular), chosen as the aligned
span closest to 2/3 of the CURRENT depth (the ctx-recompute capability is
back-loaded: cg1 depth skyline 0.35@12 -> 0.75@38), clamped to
[anneal_lo, anneal_hi) in ORIGINAL block coordinates and to blocks strictly
above the value-head-feeding tail. config.num_hidden_layers/layer_types are
kept in sync, so critic_latest checkpoints stay self-describing and resume
Just Works; the controller's own state (keep-list, surgery log, heal window)
lives in <save_dir>/anneal_state.json.

All ranks must call apply() with the same plan at the same step (the trainer
broadcasts rank0's decision); the caller rebuilds critic_optim/critic_trainable
afterwards (fresh Adam moments post-surgery are intended — the loss landscape
just changed).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class AnnealController:
    def __init__(self, args, n_layers_now: int, save_dir: Path, resume: bool):
        self.tau = args.anneal_tau
        self.eps = args.anneal_eps
        self.cut_n = args.anneal_cut
        self.lo = args.anneal_lo
        self.hi = args.anneal_hi
        self.cooldown = args.anneal_cooldown
        self.heal_len = args.anneal_heal
        self.min_layers = args.anneal_min_layers
        self.path = Path(save_dir) / "anneal_state.json"

        if resume and self.path.exists():
            st = json.loads(self.path.read_text())
            self.keep = st["keep"]
            self.surgeries = st["surgeries"]
            self.heal_until = st["heal_until"]
            self.last_surgery_step = st["last_surgery_step"]
            assert len(self.keep) == n_layers_now, (
                f"anneal_state keep-list ({len(self.keep)}) disagrees with the "
                f"loaded critic depth ({n_layers_now}) — mixed-up save dir?")
        else:
            # Fresh run: current blocks ARE original indices 0..n-1.
            self.keep = list(range(n_layers_now))
            self.surgeries = []
            self.heal_until = -1
            self.last_surgery_step = -10**9

    # ------------------------------------------------------------------
    def enabled(self) -> bool:
        return self.tau > 0

    def policy_frozen(self, step: int) -> bool:
        """True during the post-surgery heal window (skip policy updates)."""
        return step < self.heal_until

    def n_layers(self) -> int:
        return len(self.keep)

    # ------------------------------------------------------------------
    def decide(self, step: int, fve_full: float, fve_noz: float) -> list[int] | None:
        """Rank0, on eval ticks. Returns the ORIGINAL block indices to delete,
        or None. Deterministic given (state, inputs) — but the trainer still
        broadcasts the plan so ranks can never disagree."""
        if not self.enabled():
            return None
        if step < self.heal_until or step - self.last_surgery_step < self.cooldown:
            return None
        if len(self.keep) - self.cut_n < self.min_layers:
            return None
        if not (fve_full == fve_full and fve_noz == fve_noz):  # NaN guard
            return None
        bypass = (fve_noz >= self.tau) or (
            fve_full - fve_noz <= self.eps and fve_full >= self.tau)
        if not bypass:
            return None
        return self.pick_cut()

    def pick_cut(self) -> list[int] | None:
        """Aligned contiguous span of cut_n CURRENT blocks nearest 2/3 depth,
        restricted to original indices in [lo, hi). Returns original indices."""
        n = len(self.keep)
        # candidate start positions (in CURRENT coordinates), aligned to cut_n
        cands = []
        for s in range(0, n - self.cut_n + 1, self.cut_n):
            span = self.keep[s: s + self.cut_n]
            if span[0] >= self.lo and span[-1] < self.hi:
                cands.append((abs(s + self.cut_n / 2 - 2 * n / 3), span))
        if not cands:
            return None
        cands.sort(key=lambda t: t[0])
        return list(cands[0][1])

    # ------------------------------------------------------------------
    def apply(self, critic, plan: list[int], step: int, rank0: bool) -> dict:
        """Delete the planned blocks IN PLACE on this rank's critic. Returns a
        summary dict for logging. Caller must rebuild critic_optim afterwards."""
        drop = set(plan)
        assert drop <= set(self.keep), f"plan {plan} not a subset of keep-list"
        backbone = critic.backbone
        layers = backbone.model.layers
        assert len(layers) == len(self.keep), (
            f"critic has {len(layers)} layers but keep-list says {len(self.keep)}")

        keep_pos = [i for i, orig in enumerate(self.keep) if orig not in drop]
        new_layers = nn.ModuleList([layers[i] for i in keep_pos])
        # Reindex per-layer bookkeeping (KV-cache indices etc.). We run the
        # critic teacher-forced without cache, but stale indices out of range
        # would still break config-validation paths — keep them consistent.
        for new_i, lyr in enumerate(new_layers):
            if hasattr(lyr, "layer_idx"):
                lyr.layer_idx = new_i
            for sub_name in ("self_attn", "linear_attn", "attn", "mixer"):
                sub = getattr(lyr, sub_name, None)
                if sub is not None and hasattr(sub, "layer_idx"):
                    sub.layer_idx = new_i
        backbone.model.layers = new_layers

        cfg = backbone.config
        if getattr(cfg, "layer_types", None) is not None:
            cfg.layer_types = [cfg.layer_types[i] for i in keep_pos]
        cfg.num_hidden_layers = len(new_layers)

        self.keep = [self.keep[i] for i in keep_pos]
        self.last_surgery_step = step
        self.heal_until = step + self.heal_len
        rec = {"step": step, "cut": sorted(plan), "n_layers_after": len(self.keep),
               "heal_until": self.heal_until}
        self.surgeries.append(rec)
        if rank0:
            self.save()
            print(f"[anneal] SURGERY @step {step}: deleted original blocks "
                  f"{sorted(plan)} -> {len(self.keep)} layers remain "
                  f"(keep={self.keep}); healing until step {self.heal_until}",
                  flush=True)
        torch.cuda.empty_cache()
        return rec

    def save(self) -> None:
        self.path.write_text(json.dumps({
            "keep": self.keep, "surgeries": self.surgeries,
            "heal_until": self.heal_until,
            "last_surgery_step": self.last_surgery_step,
        }, indent=1))
