"""Vendored vllm-lens steering for MODERN vLLM (>=0.26) — the an1/B300 path.

Why this exists: the pinned rollout stack (vllm==0.19.0 + vllm-lens==1.1.0 +
utils/patch_vllm_lens.py) cannot run on Blackwell Ultra (sm_103) — 0.19 has no
cu130 wheels and its dependency set no longer resolves on py3.12/pip>=26 — and
vllm-lens's hook is hard-matched to 0.19's GPUModelRunner (0.22+ refactor made
it silently no-op; see docs/vllm-lens-setup.md). This module vendors the
STEERING half of ``vllm_lens._worker_ext`` with every utils/patch_vllm_lens.py
hunk pre-applied:

  (1) norm_match anchored to the FULL residual stream (output[0]+output[1]);
  (2) O(1) ``_steering_id`` config lookup (offline path only — we have no
      async path, so the per-request prefix scan is dropped entirely);
  (3) steering-apply counter;
  (4) per-request coverage log {covered, applied, positions, orphaned};
  (5) dict-aware seq_lens (vLLM v1 per-KV-group attn metadata).

Differences from vllm-lens:
  * NO monkey-patching, NO entry-point plugin. The trainer passes
    ``worker_extension_cls="nla.utils.vlens_port.NLASteerExtension"`` to
    ``LLM(...)`` and generates through :func:`steer_generate`.
  * NO activation capture — the RL trainer only steers.
  * ``SteeringVector`` is a plain picklable class with the same field names.
  * FAIL-LOUD: the vllm-lens hook wrapped everything in try/except-warn, which
    is exactly how a version drift becomes silently-unsteered rollouts. Here
    the first hook error prints a full traceback and re-raises — a dead engine
    beats corrupted training.

Requirements at run time:
  * ``enforce_eager=True`` and ``enable_prefix_caching=False`` engine args
    (hooks don't fire under compiled graphs; prefix caching would reuse KV
    computed under a DIFFERENT request's injection).
  * The EasyNLA repo root on PYTHONPATH **in the environment that launches the
    engine** — vLLM worker processes unpickle SteeringVector/import the
    extension by qualified name.
"""

from __future__ import annotations

import os
import pickle
import traceback
from typing import Any

import torch

_STEER_DEBUG = os.environ.get("NLA_STEER_DEBUG") == "1"

# ---------------------------------------------------------------------------
# SteeringVector — same field names/semantics as vllm_lens._helpers.types
# ---------------------------------------------------------------------------


class SteeringVector:
    """Picklable steering vector (drop-in for vllm_lens.SteeringVector).

    activations: [n_layers, d] (broadcast) or [n_layers, n_pos, d]
    (position-specific; positions taken from ``position_indices``).
    """

    __slots__ = ("activations", "layer_indices", "scale", "norm_match", "position_indices")

    def __init__(self, activations: torch.Tensor, layer_indices: list[int],
                 scale: float = 1.0, norm_match: bool = False,
                 position_indices: list[int] | None = None):
        assert activations.dim() in (2, 3), f"activations must be 2-D or 3-D, got {activations.shape}"
        assert activations.shape[0] == len(layer_indices), (
            f"activations dim-0 {activations.shape[0]} != len(layer_indices) {len(layer_indices)}")
        self.activations = activations
        self.layer_indices = list(layer_indices)
        self.scale = float(scale)
        self.norm_match = bool(norm_match)
        self.position_indices = list(position_indices) if position_indices is not None else None

    @property
    def layer_index_map(self) -> dict[int, int]:
        """Maps actual model layer index -> index into ``activations`` dim-0."""
        return {li: i for i, li in enumerate(self.layer_indices)}

    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Worker-side steering (runs inside vLLM worker processes)
# ---------------------------------------------------------------------------


def _get_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    m: Any = model
    if hasattr(m, "language_model") and hasattr(m.language_model, "model"):
        return m.language_model.model.layers
    if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "layers"):
        return m.model.decoder.layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    raise AttributeError(f"Cannot find decoder layers on {type(model).__name__}")


def norm_match(residual: torch.Tensor, steering: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Karvonen norm-match: h' = h + ||h|| * v/||v|| (arXiv:2512.15674)."""
    r_norm = residual.float().norm(dim=-1, keepdim=True)
    v_norm = steering.float().norm(dim=-1, keepdim=True)
    return (steering * (r_norm / (v_norm + eps))).to(residual.dtype)


def _apply_steering(configs, layer_idx, target, start, end, abs_start,
                    norm_ref=None, log_key=None, steer_count=None, steer_log=None):
    """Apply matching steering vectors to a token slice in-place.

    norm_ref anchors norm_match to the FULL residual stream for (hidden,
    residual)-tuple layers — patch fix (1). Coverage accounting = fix (3)+(4).
    """
    if norm_ref is None:
        norm_ref = target
    n_tokens = end - start
    for cfg in configs:
        if layer_idx not in cfg.layer_index_map:
            continue
        act_idx = cfg.layer_index_map[layer_idx]
        vec = cfg.activations[act_idx].to(target.dtype)

        if vec.dim() == 1:  # broadcast to all positions of the slice
            v = vec.unsqueeze(0)
            if cfg.norm_match:
                v = norm_match(norm_ref[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale
        else:  # position-specific
            pos_indices = (cfg.position_indices if cfg.position_indices is not None
                           else list(range(vec.shape[0])))
            abs_end = abs_start + n_tokens
            if log_key is not None and steer_log is not None and any(
                    abs_start <= _p < abs_end for _p in pos_indices):
                _steer_entry(steer_log, log_key)["covered"] += 1
            for pi, abs_pos in enumerate(pos_indices):
                if pi >= vec.shape[0]:
                    break
                if abs_pos < abs_start or abs_pos >= abs_end:
                    continue
                rel = abs_pos - abs_start + start
                v = vec[pi]
                if cfg.norm_match:
                    v = norm_match(norm_ref[rel], v)
                if _STEER_DEBUG:
                    print(f"[vlens_port DEBUG] layer={layer_idx} abs_pos={abs_pos} "
                          f"rel={rel} ||target[rel]||={target[rel].float().norm():.3f} "
                          f"||norm_ref[rel]||={norm_ref[rel].float().norm():.3f} "
                          f"||v_scaled||={(v * cfg.scale).float().norm():.3f} "
                          f"target_is_normref={target.data_ptr() == norm_ref.data_ptr()}",
                          flush=True)
                target[rel] = target[rel] + v * cfg.scale
                if steer_count is not None:
                    steer_count[0] += 1
                if log_key is not None and steer_log is not None:
                    e = _steer_entry(steer_log, log_key)
                    e["applied"] += 1
                    e["positions"].append(int(abs_pos))


def _steer_entry(steer_log: dict, log_key: str) -> dict:
    return steer_log.setdefault(
        log_key, {"covered": 0, "applied": 0, "positions": [], "orphaned": 0})


def _hook_inner(ext: "NLASteerExtension", layer_idx: int, output):
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    if not ext._nla_steering_data:
        return None  # nothing staged — the common case outside rollout windows

    runner = ext.model_runner
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        raise AttributeError(
            "GPUModelRunner has no 'input_batch' — vLLM internals drifted past "
            "this port (see nla/utils/vlens_port.py header). Steering CANNOT "
            "run; refusing to continue silently.")
    num_reqs = input_batch.num_reqs
    if num_reqs == 0:
        return None
    req_ids = input_batch.req_ids
    requests = getattr(runner, "requests", None)
    if requests is None:
        raise AttributeError("GPUModelRunner has no 'requests' dict — vLLM drift.")

    # Token-slice mapping (request i -> flat rows [qsl[i], qsl[i+1]) at absolute
    # positions starting seq_lens[i]-n_query): source it from the RUNNER's
    # persistent buffers, NOT from per-layer attn metadata — vLLM 0.26's
    # backend metadata (FlashInfer / GDN) no longer carries query_start_loc on
    # ANY entry for hybrid models, which silently killed steering on every
    # step of the first an1 bring-up. The classic (V1) GPUModelRunner keeps
    # self.query_start_loc as a CpuGpuBuffer and self.seq_lens as a GPU tensor,
    # updated every step before the forward.
    query_start_loc = None
    seq_lens = None
    qsl_buf = getattr(runner, "query_start_loc", None)
    if qsl_buf is not None:
        qsl = qsl_buf.gpu if hasattr(qsl_buf, "gpu") else qsl_buf
        query_start_loc = qsl[: num_reqs + 1]
    sl_buf = getattr(runner, "seq_lens", None)
    if sl_buf is not None:
        sl = sl_buf.gpu if hasattr(sl_buf, "gpu") else sl_buf
        seq_lens = sl[:num_reqs]
    if query_start_loc is None:
        # Fallback: fish in the per-layer attn metadata (pre-0.26 semantics).
        fctx = get_forward_context()
        attn_metadata = fctx.attn_metadata
        if attn_metadata is None:
            return None
        if isinstance(attn_metadata, list):
            attn_metadata = attn_metadata[0]
            if attn_metadata is None:
                return None
        if hasattr(attn_metadata, "values"):
            for _m in attn_metadata.values():
                if query_start_loc is None and getattr(_m, "query_start_loc", None) is not None:
                    query_start_loc = _m.query_start_loc
                if seq_lens is None and getattr(_m, "seq_lens", None) is not None:
                    seq_lens = _m.seq_lens
                if query_start_loc is not None and seq_lens is not None:
                    break
        else:
            query_start_loc = getattr(attn_metadata, "query_start_loc", None)
            seq_lens = getattr(attn_metadata, "seq_lens", None)
    if query_start_loc is None:
        # FAIL LOUD: without the slice mapping, steering CANNOT run — and a
        # skipped step is exactly the silent-dead-injection failure mode.
        raise RuntimeError(
            f"[vlens_port] no query_start_loc available at layer {layer_idx} "
            f"(runner {type(runner).__module__}) — steering cannot map request "
            f"token slices; vLLM internals drifted past this port.")

    # --- Phase 1: per-request steering configs (O(1) _steering_id lookup) ----
    per_req_steering: list[list[SteeringVector]] = []
    per_req_log_key: list[str | None] = []
    needs_steering = False
    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = requests.get(req_id)
        extra = (req_state.sampling_params.extra_args
                 if req_state is not None and req_state.sampling_params is not None else None)
        sid = (extra or {}).get("_steering_id")
        configs = list(ext._nla_steering_data.get(sid, [])) if sid is not None else []
        per_req_steering.append(configs)
        per_req_log_key.append(sid)
        # ORPHAN: request claims steering but the payload is gone — the exact
        # silent-lost-injection mechanism (patch fix (4)). Count once per fwd.
        if sid is not None and not configs and layer_idx == ext._nla_first_layer_idx:
            _steer_entry(ext._nla_steer_log, sid)["orphaned"] += 1
        if configs:
            needs_steering = True

    # Only clone/norm-ref when some config targets THIS layer (patch perf fix).
    if needs_steering:
        needs_steering = any(
            layer_idx in cfg.layer_index_map
            for cfgs in per_req_steering for cfg in cfgs)
    if not needs_steering:
        return None

    # --- Phase 2: apply ------------------------------------------------------
    if isinstance(output, tuple):
        modified_output = (output[0].clone(),) + tuple(output[1:])
        target = modified_output[0]
        # (hidden_states, residual) layers defer the residual add to the next
        # layer's fused add-RMSNorm: the TRUE stream is output[0] + output[1].
        norm_ref = (output[0] + output[1]
                    if len(output) > 1 and output[1] is not None else output[0])
    else:
        modified_output = output.clone()
        target = modified_output
        norm_ref = target

    for i in range(num_reqs):
        if not per_req_steering[i]:
            continue
        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        n_query = end - start
        if seq_lens is not None:
            sl = seq_lens[i]
            sl_val = sl.item() if isinstance(sl, torch.Tensor) else int(sl)
            abs_start = int(sl_val - n_query)
        else:
            abs_start = 0  # prefill-from-0 fallback
        _apply_steering(per_req_steering[i], layer_idx, target, start, end,
                        abs_start, norm_ref, log_key=per_req_log_key[i],
                        steer_count=ext._nla_steer_count, steer_log=ext._nla_steer_log)
    return modified_output


def _make_hook(ext: "NLASteerExtension", layer_idx: int):
    def hook(_module, _inputs, output):
        try:
            return _hook_inner(ext, layer_idx, output)
        except Exception:
            # FAIL LOUD (vendored-port policy): print once with traceback, then
            # re-raise — a dead engine beats silently-unsteered rollouts.
            if not ext._nla_hook_error_printed:
                ext._nla_hook_error_printed = True
                print(f"[vlens_port] FATAL hook error at layer {layer_idx}:\n"
                      f"{traceback.format_exc()}", flush=True)
            raise
    return hook


class NLASteerExtension:
    """Worker extension mixin (``worker_extension_cls``) for steering.

    vLLM adds this class as a base of Worker; methods are callable via
    ``llm.collective_rpc("<name>")``. State is instance-level, initialized
    lazily in :meth:`nla_install_hooks`.
    """

    # Markers the trainer's lens-vintage guard greps for:
    #   norm_ref full-stream fix / O(1) lookup / counter / coverage log /
    #   dict-aware seq_lens — all pre-applied by construction (see header).

    def nla_install_hooks(self) -> int:
        """Register a forward hook on every decoder layer. Idempotent.

        Returns the number of decoder layers hooked (sanity check trainer-side).
        """
        if getattr(self, "_nla_hooks_installed", False):
            return self._nla_n_hooked
        from vllm.model_executor.models.utils import PPMissingLayer

        self._nla_hooks_installed = True
        self._nla_steering_data: dict[str, list[SteeringVector]] = {}
        self._nla_steer_count = [0]
        self._nla_steer_log: dict[str, dict] = {}
        self._nla_warned_no_qsl = False
        self._nla_hook_error_printed = False

        print(f"[vlens_port] runner={type(self.model_runner).__module__}."
              f"{type(self.model_runner).__name__}", flush=True)
        layers = _get_layers(self.model_runner.model)
        n = 0
        first_idx = None
        for layer_idx, layer in enumerate(layers):
            if isinstance(layer, PPMissingLayer):
                continue
            if first_idx is None:
                first_idx = layer_idx
            layer.register_forward_hook(_make_hook(self, layer_idx))
            n += 1
        self._nla_first_layer_idx = first_idx if first_idx is not None else 0
        self._nla_n_hooked = n
        return n

    def nla_set_steering_data(self, key: str, pickled_data: bytes) -> None:
        sv_list: list[SteeringVector] = pickle.loads(pickled_data)
        device = next(self.model_runner.model.parameters()).device
        dtype = next(self.model_runner.model.parameters()).dtype
        num_layers = len(_get_layers(self.model_runner.model))
        vectors = []
        for sv in sv_list:
            for idx in sv.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(f"layer_index {idx} out of range [0, {num_layers})")
            vectors.append(SteeringVector(
                activations=sv.activations.to(device=device, dtype=dtype),
                layer_indices=sv.layer_indices, scale=sv.scale,
                norm_match=sv.norm_match, position_indices=sv.position_indices))
        self._nla_steering_data[key] = vectors

    def nla_clear_steering_data(self, key: str) -> None:
        self._nla_steering_data.pop(key, None)

    def nla_clear_all_steering_data(self) -> None:
        if getattr(self, "_nla_hooks_installed", False):
            self._nla_steering_data.clear()

    def nla_read_reset_steer_count(self) -> int:
        if not getattr(self, "_nla_hooks_installed", False):
            return 0  # nothing hooked yet -> nothing written (0, not -1: the
            # counter mechanism IS available on this backend)
        c = self._nla_steer_count[0]
        self._nla_steer_count[0] = 0
        return c

    def nla_read_reset_steer_log(self) -> dict:
        if not getattr(self, "_nla_hooks_installed", False):
            return {}
        log = {k: dict(v) for k, v in self._nla_steer_log.items()}
        self._nla_steer_log.clear()
        return log


# ---------------------------------------------------------------------------
# Trainer-side offline generate wrapper (replaces vllm-lens's LLM.generate patch)
# ---------------------------------------------------------------------------

WORKER_EXT = "nla.utils.vlens_port.NLASteerExtension"


def steer_generate(llm, prompts, sampling_params_list):
    """Offline steered generate — vendored twin of vllm-lens's patched
    ``LLM.generate``. Pops ``apply_steering_vectors`` from each request's
    ``extra_args``, RPCs the payload to workers keyed ``_steer_{idx}``, tags
    the request with ``_steering_id``, generates, then clears worker state.

    Requires the engine to have been built with
    ``worker_extension_cls=WORKER_EXT`` (else the RPCs raise AttributeError —
    loud by design) and ``enable_prefix_caching=False``.
    """
    steering_payloads: dict[str, bytes] = {}
    for idx, sp in enumerate(sampling_params_list):
        extra = sp.extra_args or {}
        vectors = extra.pop("apply_steering_vectors", None)
        if vectors is not None:
            sid = f"_steer_{idx}"
            steering_payloads[sid] = pickle.dumps(vectors)
            if sp.extra_args is None:
                sp.extra_args = {}
            sp.extra_args["_steering_id"] = sid

    if steering_payloads and not getattr(llm, "_nla_hooks_installed", False):
        n_hooked = llm.collective_rpc("nla_install_hooks")
        llm._nla_hooks_installed = True
        print(f"[vlens_port] hooks installed on {n_hooked} decoder layers/worker", flush=True)

    for sid, payload in steering_payloads.items():
        llm.collective_rpc("nla_set_steering_data", args=(sid, payload))
    try:
        outputs = llm.generate(prompts, sampling_params_list)
    finally:
        if steering_payloads:
            llm.collective_rpc("nla_clear_all_steering_data")
    return outputs
