"""fd1 SFT (pod): full-depth AO on CODI band+z activations, variable positions/sources.

Base Qwen3-8B + LoRA r128. Samples built on the fly by fd1_data (random source subsets x
random position subsets x random chain subsets, no position tags, random-order targets).
Eval on cadence: per-source profile, count curve, donor + twin controls, LB strata,
text skyline. Stationarity halt; fail-early on donor gap; resumable.

Throughput (2026-08-16): multi-GPU via torchrun (unwrapped model + manual LoRA grad
all-reduce), token-budget length-sorted micro-batching, vectorized marker injection,
hidden-slice CE (lm_head only on target positions — no [B,L,V] logits materialization),
background-thread sample prefetch, rank-sharded evals + val matrix.

  torchrun --standalone --nproc-per-node N fd1_sft.py [--smoke] [--resume] [--tag a]
"""
import argparse, datetime, json, os, random, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/root/fd1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("FD1_DATA", "/root/fd1/data")
CKPT_LOCAL = os.environ.get("FD1_CKPT", "/root/fd1_ckpt/fd1")
DUR = os.environ.get("FD1_DUR", "/workspace/jbauer/fd1")
BASE = "Qwen/Qwen3-8B"
MARKER_ID, INJ_SCALE = 149705, 150.0
TOKBUDGET = int(os.environ.get("FD1_TOKBUDGET", 6144))
MAXMICRO = 16
ATTN = os.environ.get("FD1_ATTN", "sdpa")             # sdpa | flash_attention_2
PACK = os.environ.get("FD1_PACK", "0") == "1"          # varlen sequence packing (needs FA2)
COMPILE = os.environ.get("FD1_COMPILE", "0") == "1"    # torch.compile the transformer body
VALONLY = bool(os.environ.get("FD1_VALONLY"))          # resume -> one val matrix -> exit (parity checks)
assert not PACK or ATTN == "flash_attention_2", "packing needs the FA2 varlen path"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--sources", default="", help="comma list restricting sources (e.g. band-only: L9,L18,L27,L35,LF)")
    ap.add_argument("--valloss-every", type=int, default=0, help=">0: per-{input x label} val-loss matrix cadence")
    ap.add_argument("--converge", action="store_true", help="halt on per-CELL val-loss double-log flatness, not aggregate F1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--schedule", default="const", choices=["const", "cosine"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ckpt_dir = CKPT_LOCAL + args.tag
    os.makedirs(DUR, exist_ok=True)

    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import wandb
    from fd1_data import Pool, build_sample, render_acts, canon, PREAMBLE, PREAMBLE_TXT, text_block, SRC as SRC_ALL

    RANK = int(os.environ.get("RANK", "0"))
    WORLD = int(os.environ.get("WORLD_SIZE", "1"))
    if WORLD > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=120))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert args.batch % WORLD == 0
    PER_RANK = args.batch // WORLD
    r0 = RANK == 0

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE)
    assert tok.decode([MARKER_ID]) == "㈎"
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 attn_implementation=ATTN).to(dev)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if r0: model.print_trainable_parameters()
    embed = model.get_input_embeddings()
    base_lm = model.get_base_model()
    body, lm_head = base_lm.model, base_lm.lm_head
    # separate compiled callable: generate() keeps hitting the eager body (kv-cache shapes
    # would trigger recompile storms), train/val forwards go through fwd_body
    fwd_body = torch.compile(body, dynamic=True) if COMPILE else body
    tparams = [p for p in model.parameters() if p.requires_grad]

    gsm, syn = Pool("gsm"), Pool("syn")
    gev = Pool("gsmeval")
    # extra thought-only train pools (no qh, no eval/val role) — data scale-up
    extra = [Pool(n) for n in os.environ.get("FD1_EXTRA_POOLS", "").split(",") if n]
    if r0:
        print(f"pools: gsm ok {len(gsm.ok)}/{len(gsm.meta)}, syn ok {len(syn.ok)}/{len(syn.meta)}, "
              f"gsmeval ok {len(gev.ok)}/{len(gev.meta)}, extra {[len(p.ok) for p in extra]} "
              f"| world {WORLD} per-rank {PER_RANK} tokbudget {TOKBUDGET}")

    # pinned holdout — same derivation as nx1/nx2 (identical meta ordering)
    heldset = set(gsm.ok[-200:])
    held_rows = gsm.ok[-200:][:100]
    gsm_train = [i for i in gsm.ok if i not in heldset]
    # syn: train split rows for training; test_skeleton (ok) for eval
    syn_train = [i for i in syn.ok if syn.meta[i]["split"] == "train"]
    syn_skel = [i for i in syn.ok if syn.meta[i]["split"] == "test_skeleton"][:100]
    syn.ok_train = syn_train
    TRAIN_POOLS = [(gsm, None), (syn, None)] + [(p, p.ok) for p in extra]  # rows filled below
    # twin map: card_id -> gold values of partner (if partner captured & correct)
    bycard = {}
    for i in syn.ok: bycard.setdefault(syn.meta[i]["card_id"], i)
    twin_gold = {}
    for i in syn_skel:
        p = syn.meta[i].get("twin_of")
        if p and p in bycard: twin_gold[i] = [c["val"] for c in syn.meta[bycard[p]]["chain"]]
    if r0: print(f"train rows: gsm {len(gsm_train)}, syn {len(syn_train)}; held 100, syn_skel {len(syn_skel)}, twins {len(twin_gold)}")

    # test 207 with LB strata (nx convention)
    ours_path = next(p for p in ("/root/nx2/data/ours_full.jsonl",
                                 "/workspace-vast/jbauer/bt1/nx1/pod_salvage/work/data/ours_full.jsonl")
                     if os.path.exists(p))
    flags = {json.loads(l)["idx"]: json.loads(l) for l in open(ours_path)}
    for m in gev.meta:
        f = flags[m["idx"]]
        m["stratum"] = "LB" if f["lb"] else ("cnl" if f["correct"] and f["correct_k0"] else "wrong")
    lb = [i for i, m in enumerate(gev.meta) if m["stratum"] == "LB"]
    cnl = [i for i, m in enumerate(gev.meta) if m["stratum"] == "cnl"][:60]
    wr = [i for i, m in enumerate(gev.meta) if m["stratum"] == "wrong"][:60]
    test_rows = lb + cnl + wr
    if args.smoke:
        held_rows, test_rows, syn_skel = held_rows[:16], test_rows[:24], syn_skel[:8]

    def nums_in(text):
        out = set()
        for mt in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
            try: out.add(canon(mt))
            except ValueError: pass
        return out

    def f1_of(text, gold):
        E, G = nums_in(text), set(gold)
        if not E or not G: return 0.0
        hit = len(E & G)
        p, r = hit / len(E), hit / len(G)
        return 2 * p * r / (p + r) if hit else 0.0

    def encode(prompt, target=None):
        txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, enable_thinking=False, tokenize=False)
        pids = tok(txt, add_special_tokens=False).input_ids
        if target is None: return pids, None
        tids = tok(target, add_special_tokens=False).input_ids + [tok.eos_token_id]
        return pids + tids, [-100] * len(pids) + tids

    def inject(E, ids_dev, vecs_list):
        """vectorized marker replacement: one stacked H2D copy + one scatter"""
        lens = [len(v) for v in vecs_list]
        if not sum(lens): return
        mask = ids_dev == MARKER_ID
        assert mask.sum(1).tolist() == lens, (mask.sum(1).tolist(), lens)
        V = torch.from_numpy(np.stack([v for vs in vecs_list for v in vs])).to(dev)
        E[mask] = (V / V.norm(dim=1, keepdim=True) * INJ_SCALE).to(E.dtype)

    def embed_batch(seqs, vecs_list):
        B = len(seqs); L = max(len(s) for s in seqs)
        ids = torch.full((B, L), tok.pad_token_id or tok.eos_token_id, dtype=torch.long)
        att = torch.zeros(B, L, dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
        ids_dev = ids.to(dev)
        E = embed(ids_dev)
        inject(E, ids_dev, vecs_list)
        return E, att.to(dev), ids

    @torch.no_grad()
    def gen_batch(prompts_vecs):
        outs = []
        for s0 in range(0, len(prompts_vecs), 48):
            chunk = prompts_vecs[s0:s0 + 48]
            seqs = [encode(p)[0] for p, _ in chunk]
            # left-pad for generation
            L = max(len(s) for s in seqs)
            ids = torch.full((len(chunk), L), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros(len(chunk), L, dtype=torch.long)
            for b, s in enumerate(seqs):
                ids[b, L - len(s):] = torch.tensor(s); att[b, L - len(s):] = 1
            ids_dev = ids.to(dev)
            E = embed(ids_dev)
            inject(E, ids_dev, [v for _, v in chunk])
            g = model.generate(inputs_embeds=E, attention_mask=att.to(dev),
                               max_new_tokens=args.max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            outs += [tok.decode(s, skip_special_tokens=True) for s in g]
        return outs

    def slice_ce(E, att, labels_dev, pos_ids=None):
        """token-level CE via lm_head on target positions only (no full-vocab logits).
        pos_ids with resets + att=None on a [1,T] row -> FA2 block-diagonal varlen path.
        Packed shift is safe: each segment starts with >=1 prompt (-100) label, so the
        cross-boundary prediction is always masked."""
        h = fwd_body(inputs_embeds=E, attention_mask=att, position_ids=pos_ids,
                     use_cache=False).last_hidden_state
        lb = labels_dev[:, 1:]
        sel = lb != -100
        ce = F.cross_entropy(lm_head(h[:, :-1][sel]).float(), lb[sel], reduction="none")
        return ce, sel

    LISTQ = "List every intermediate result computed (any order)."
    def cell_prompt(pool, i, sources, tpos, pos_mode="thoughts", donor_pool_rows=None, shift=0):
        j = i
        if shift and donor_pool_rows is not None:
            j = donor_pool_rows[(donor_pool_rows.index(i) + shift) % len(donor_pool_rows)]
        acts, vecs = render_acts(pool, j, sources, pos_mode, tpos)
        return (PREAMBLE + acts + f"\n\nQuestion: {LISTQ}", vecs)

    ALLSRC = args.sources.split(",") if args.sources else ["L9", "L18", "L27", "L35", "LF", "Z"]
    SRCS = ALLSRC if args.sources else None

    def evaluate(step):
        model.eval()
        gold_h = [[c["val"] for c in gsm.meta[r]["chain"]] for r in held_rows]
        def mean_f1(pv, gold):
            return float(np.mean([f1_of(x, g) for x, g in zip(gen_batch(pv), gold)]))
        thunks = []
        # per-source profile (all 6 thought positions), own + donor
        for src in ALLSRC + ["ALL"]:
            S = ALLSRC if src == "ALL" else [src]
            for tag, shift in ((f"held_{src}", 0), (f"held_{src}_donor", 13)):
                thunks.append(lambda S=S, tag=tag, shift=shift: {tag: mean_f1(
                    [cell_prompt(gsm, r, S, list(range(6)), donor_pool_rows=held_rows, shift=shift) for r in held_rows],
                    gold_h)})
        # count curve on Z and ALL — one shared rng stream in canonical cell order (continuity with 1-GPU runs)
        cr = random.Random(1234)
        count_draws = {}
        for src_name, S in (("Z", ["Z"]), ("ALL", ALLSRC)):
            for k in (1, 3):
                count_draws[(src_name, k)] = [sorted(cr.sample(range(6), k)) for _ in held_rows]
        for src_name, S in (("Z", ["Z"]), ("ALL", ALLSRC)):
            for k in (1, 3):
                thunks.append(lambda S=S, src_name=src_name, k=k: {f"held_{src_name}_k{k}": mean_f1(
                    [cell_prompt(gsm, r, S, tp) for r, tp in zip(held_rows, count_draws[(src_name, k)])], gold_h)})
        # q-region only (recompute channel, reported separately)
        thunks.append(lambda: {"held_qonly": mean_f1(
            [cell_prompt(gsm, r, ["L27", "LF"], [], pos_mode="q_only") for r in held_rows], gold_h)})
        # text skyline (sanity)
        thunks.append(lambda: {"held_text": mean_f1(
            [(PREAMBLE_TXT + f"Transcript: {text_block(gsm.meta[r], cr)}\n\nQuestion: {LISTQ}", [])
             for r in held_rows[:50]], gold_h[:50])})
        # gsm test 207 strata, ALL sources
        gold_t = [[c["val"] for c in gev.meta[r]["chain"]] for r in test_rows]
        def test_fn(shift):
            pv = [cell_prompt(gev, r, ALLSRC, list(range(6)), donor_pool_rows=test_rows, shift=shift) for r in test_rows]
            f1s = [f1_of(x, g) for x, g in zip(gen_batch(pv), gold_t)]
            out = {("test_donor" if shift else "test"): float(np.mean(f1s))}
            if not shift:
                for st in ("LB", "cnl", "wrong"):
                    out[f"test_{st}"] = float(np.mean([f for f, r in zip(f1s, test_rows) if gev.meta[r]["stratum"] == st]))
            return out
        thunks += [lambda: test_fn(0), lambda: test_fn(13)]
        # syn test_skeleton: ALL + Z cells, donor, twin cross-score
        if syn_skel:
            gold_s = [[c["val"] for c in syn.meta[r]["chain"]] for r in syn_skel]
            def syn_fn(tag, S, shift):
                pv = [cell_prompt(syn, r, S, list(range(6)), donor_pool_rows=syn_skel, shift=shift) for r in syn_skel]
                texts = gen_batch(pv)
                out = {tag: float(np.mean([f1_of(x, g) for x, g in zip(texts, gold_s)]))}
                if tag == "syn_ALL":
                    tw = [(x, twin_gold[r]) for x, r in zip(texts, syn_skel) if r in twin_gold]
                    if tw: out["syn_twin_x"] = float(np.mean([f1_of(x, g) for x, g in tw]))
                return out
            for tag, S, shift in (("syn_ALL", ALLSRC, 0), ("syn_Z", ["Z"], 0), ("syn_ALL_donor", ALLSRC, 13)):
                thunks.append(lambda tag=tag, S=S, shift=shift: syn_fn(tag, S, shift))
        # rank-sharded execution, merged everywhere
        local = {}
        for k, fn in enumerate(thunks):
            if k % WORLD == RANK: local.update(fn())
        res = local
        if WORLD > 1:
            gath = [None] * WORLD
            dist.all_gather_object(gath, local)
            res = {}
            for d in gath: res.update(d)
        model.train()
        if r0:
            wandb.log({f"eval/{k}": v for k, v in res.items()}, step=step)
            with open(f"{DUR}/fd1_evals.jsonl", "a") as f:
                f.write(json.dumps(dict(step=step, ts=time.time(), tag=args.tag, **res)) + "\n")
            prof = " ".join(f"{s} {res[f'held_{s}']:.2f}/{res[f'held_{s}_donor']:.2f}" for s in ALLSRC + ["ALL"])
            print(f"[eval @{step}] {prof} | test {res['test']:.3f}/{res['test_donor']:.3f} "
                  f"| syn {res.get('syn_ALL', -1):.3f}/{res.get('syn_ALL_donor', -1):.3f} twin_x {res.get('syn_twin_x', -1):.3f} "
                  f"| text {res['held_text']:.3f} qonly {res['held_qonly']:.3f}", flush=True)
        return res

    # ---- per-{input x label} val-loss matrix (teacher-forced CE on fixed held samples)
    VAL_INPUTS = [(s, ([s], "thoughts", list(range(6)))) for s in ALLSRC] + \
                 [("ALL", (ALLSRC, "thoughts", list(range(6))))] + \
                 ([("qonly", (["L27", "LF"], "q_only", []))] if "L27" in ALLSRC else []) + \
                 [("text", None)]
    VAL_LABELS = ["allvals", "stepval", "allsents", "stepsent", "answer", "ops", "count", "givens"]
    VN = 48
    val_cells = {}
    if args.valloss_every:
        import random as _rnd
        import zlib
        for iname, spec in VAL_INPUTS:
            for lab in VAL_LABELS:
                crng = _rnd.Random(zlib.crc32(f"{iname}|{lab}".encode()))
                samples = [build_sample(gsm, crng, mode=("text" if iname == "text" else "act"),
                                        i=held_rows[j % len(held_rows)], kind=lab, spec=spec)
                           for j in range(VN)]
                val_cells[f"{iname}|{lab}"] = [(encode(s["prompt"], s["target"]), s["vecs"]) for s in samples]
        # strata-resolved cells on gsmeval (LB / cnl / wrong). answer@st targets GOLD;
        # answerpred@wrong targets CODI's own (wrong) pred — the read-vs-solve monitor
        # in CE space (reader: pred-CE low, gold-CE high; recomputer: the reverse).
        if not args.sources:
            strows = {st: [i for i, m in enumerate(gev.meta) if m["stratum"] == st]
                      for st in ("LB", "cnl", "wrong")}
            SPECS_ST = [("ALL", (ALLSRC, "thoughts", list(range(6)))),
                        ("Z", (["Z"], "thoughts", list(range(6)))),
                        ("qonly", (["L27", "LF"], "q_only", []))]
            for iname, spec in SPECS_ST:
                for lab in ("allvals", "answer"):
                    for st in ("LB", "cnl", "wrong"):
                        crng = _rnd.Random(zlib.crc32(f"{iname}|{lab}@{st}".encode()))
                        rs = strows[st]
                        samples = [build_sample(gev, crng, mode="act", i=rs[j % len(rs)],
                                                kind=lab, spec=spec) for j in range(VN)]
                        val_cells[f"{iname}|{lab}@{st}"] = [(encode(s["prompt"], s["target"]), s["vecs"])
                                                            for s in samples]
                crng = _rnd.Random(zlib.crc32(f"{iname}|answerpred@wrong".encode()))
                rs = strows["wrong"]
                spl = []
                for j in range(VN):
                    i = rs[j % len(rs)]
                    s = build_sample(gev, crng, mode="act", i=i, kind="answer", spec=spec)
                    s["target"] = gev.meta[i]["pred"] or "0"
                    spl.append(s)
                val_cells[f"{iname}|answerpred@wrong"] = [(encode(s["prompt"], s["target"]), s["vecs"])
                                                          for s in spl]
        if r0: print(f"val-loss matrix: {len(val_cells)} cells x {VN} samples", flush=True)

    vl_hist = {}
    @torch.no_grad()
    def val_matrix(step):
        model.eval()
        local = {}
        for k, (cname, items) in enumerate(val_cells.items()):
            if k % WORLD != RANK: continue
            tot, ntok = 0.0, 0
            for s0 in range(0, len(items), 16):
                ch = items[s0:s0 + 16]
                E, att, ids = embed_batch([e[0][0] for e in ch], [v for (_, v) in ch])
                L = E.shape[1]
                labels = torch.full((len(ch), L), -100, dtype=torch.long)
                for b, ((_, lab), _) in enumerate(ch):
                    labels[b, :len(lab)] = torch.tensor(lab)
                ce, _ = slice_ce(E, att, labels.to(dev))
                tot += float(ce.sum()); ntok += int(ce.numel())
            local[cname] = tot / ntok
        out = local
        if WORLD > 1:
            gath = [None] * WORLD
            dist.all_gather_object(gath, local)
            merged = {}
            for d in gath: merged.update(d)
            out = {c: merged[c] for c in val_cells}
        for c, v in out.items():
            vl_hist.setdefault(c, []).append((step, v))
        model.train()
        if r0 and not VALONLY:
            wandb.log({f"valloss/{k}": v for k, v in out.items()}, step=step)
            with open(f"{DUR}/fd1_valloss_{args.tag}.jsonl", "a") as f:
                f.write(json.dumps(dict(step=step, ts=time.time(), **out)) + "\n")
        # per-cell double-log slopes over trailing window
        slopes = {}
        for k, h in vl_hist.items():
            if len(h) >= 8:
                if h[-1][1] < 0.05:      # floored cell: converged by definition, log-slope is noise
                    slopes[k] = 0.0
                    continue
                xs = np.log([s for s, _ in h[-8:]]); ys = np.log([max(v, 1e-4) for _, v in h[-8:]])
                slopes[k] = float(np.polyfit(xs, ys, 1)[0])
        if slopes and r0:
            worst = sorted(slopes.items(), key=lambda x: x[1])[:5]
            print(f"[val @{step}] mean CE {np.mean(list(out.values())):.4f} | steepest cells: "
                  + " ".join(f"{k} {v:+.3f}" for k, v in worst), flush=True)
        return slopes

    TRAIN_POOLS[0] = (gsm, gsm_train); TRAIN_POOLS[1] = (syn, syn_train)
    POOL_W = [len(r) for _, r in TRAIN_POOLS]
    if r0: print(f"train mix: {[(p.meta[0]['src'] if p.meta else '?', w) for (p, _), w in zip(TRAIN_POOLS, POOL_W)]}")

    opt = torch.optim.AdamW(tparams, lr=args.lr, fused=True)
    import math
    def lr_mult(s):
        w = min(1.0, (s + 1) / args.warmup)
        if args.schedule == "cosine":
            t = min(1.0, max(0.0, (s - args.warmup) / max(1, args.steps - args.warmup)))
            return w * 0.5 * (1 + math.cos(math.pi * t))
        return w
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    step0 = 0
    if args.resume and os.path.isdir(ckpt_dir):
        steps = sorted(int(d.split("step")[-1]) for d in os.listdir(ckpt_dir))
        if os.environ.get("FD1_RESUME_STEP"):
            steps = [int(os.environ["FD1_RESUME_STEP"])]
        if steps:
            sd = torch.load(f"{ckpt_dir}/step{steps[-1]}/state.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(sd["lora"], strict=False)
            opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
            for g in opt.param_groups: g["lr"] = args.lr   # allow LR override on resume
            step0 = sd["step"]
            random.setstate(sd["pyrng"]); torch.set_rng_state(sd["torng"])
            if r0: print(f"resumed from step {step0} (lr {args.lr})")

    if VALONLY:   # parity mode: one val matrix on the resumed weights, dump CEs, exit
        val_matrix(step0 or 1)
        if r0:
            print("VALONLY " + json.dumps({k: h[-1][1] for k, h in vl_hist.items()}), flush=True)
        if WORLD > 1:
            dist.barrier(); dist.destroy_process_group()
        return

    if r0:
        wandb.init(project="nlarl", group="fd1", name=f"fd1_sft_{args.tag}", resume="allow",
                   id=f"fd1sft{args.tag}", config=vars(args) | dict(base=BASE, scale=INJ_SCALE, world=WORLD,
                                                                    attn=ATTN, pack=PACK, compile=COMPILE))

    if args.smoke:
        args.steps = step0 + 20

    def save(step):
        d = f"{ckpt_dir}/step{step}"
        os.makedirs(d, exist_ok=True)
        lora = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items() if "lora_" in k}
        torch.save(dict(lora=lora, opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                        pyrng=random.getstate(), torng=torch.get_rng_state()), f"{d}/state.pt")
        torch.save(lora, f"{d}/lora.pt")
        subprocess.run(["rsync", "-a", f"{d}/lora.pt", f"{DUR}/fd1{args.tag}_step{step}_lora.pt"], check=False)
        # prune old full states (keep last 2); slim loras persist in DUR
        old = sorted(int(x.split("step")[-1]) for x in os.listdir(ckpt_dir))[:-2]
        for s in old:
            subprocess.run(["rm", "-rf", f"{ckpt_dir}/step{s}"], check=False)

    def build_step(step):
        """CPU-side batch build for one optimizer step (runs in prefetch thread):
        PER_RANK samples -> length-sorted token-budget micro groups, pinned tensors."""
        srng = random.Random(args.seed * 1000003 + step + RANK * 611953)
        samples = []
        for _ in range(PER_RANK):
            pi = srng.choices(range(len(TRAIN_POOLS)), weights=POOL_W)[0]
            pool, rows = TRAIN_POOLS[pi]
            samples.append(build_sample(pool, srng, i=srng.choice(rows), srcs=SRCS))
        enc = [encode(s["prompt"], s["target"]) for s in samples]
        for e, s in zip(enc, samples):
            assert sum(1 for t in e[0] if t == MARKER_ID) == len(s["vecs"])
        order = sorted(range(PER_RANK), key=lambda j: len(enc[j][0]))
        if PACK:
            # varlen packing: greedy fill into flattened [1,T] rows (zero padding);
            # per-segment position_ids resets trigger FA2 block-diagonal attention
            groups, cur, tot = [], [], 0
            for j in order:
                Lj = len(enc[j][0])
                if cur and tot + Lj > TOKBUDGET:
                    groups.append(cur); cur, tot = [], 0
                cur.append(j); tot += Lj
            if cur: groups.append(cur)
            micros = []
            for gr in groups:
                ids = torch.cat([torch.tensor(enc[j][0]) for j in gr]).unsqueeze(0)
                labels = torch.cat([torch.tensor(enc[j][1]) for j in gr]).unsqueeze(0)
                pos = torch.cat([torch.arange(len(enc[j][0])) for j in gr]).unsqueeze(0)
                allv = [v for j in gr for v in samples[j]["vecs"]]
                V = torch.from_numpy(np.stack(allv)).pin_memory() if allv else None
                tcnts = [sum(1 for x in enc[j][1] if x != -100) for j in gr]
                micros.append(dict(ids=ids.pin_memory(), pos=pos.pin_memory(), labels=labels.pin_memory(),
                                   vecs=V, cells=[samples[j]["cell"] for j in gr], tcnts=tcnts))
            return micros
        groups, cur, curmax = [], [], 0
        for j in order:
            Lj = len(enc[j][0])
            if cur and (max(curmax, Lj) * (len(cur) + 1) > TOKBUDGET or len(cur) >= MAXMICRO):
                groups.append(cur); cur, curmax = [], 0
            cur.append(j); curmax = max(curmax, Lj)
        if cur: groups.append(cur)
        micros = []
        for gr in groups:
            L = max(len(enc[j][0]) for j in gr)
            ids = torch.full((len(gr), L), tok.pad_token_id or tok.eos_token_id, dtype=torch.long)
            att = torch.zeros(len(gr), L, dtype=torch.long)
            labels = torch.full((len(gr), L), -100, dtype=torch.long)
            for b, j in enumerate(gr):
                seq, lab = enc[j]
                ids[b, :len(seq)] = torch.tensor(seq); att[b, :len(seq)] = 1
                labels[b, :len(lab)] = torch.tensor(lab)
            allv = [v for j in gr for v in samples[j]["vecs"]]
            V = torch.from_numpy(np.stack(allv)).pin_memory() if allv else None
            micros.append(dict(ids=ids.pin_memory(), att=att.pin_memory(), labels=labels.pin_memory(),
                               vecs=V, cells=[samples[j]["cell"] for j in gr]))
        return micros

    pref = ThreadPoolExecutor(1)
    online_acc = {}
    f1_hist, best = [], -1
    model.train()
    t0 = time.time()
    fut = pref.submit(build_step, step0)
    for step in range(step0, args.steps):
        micros = fut.result()
        if step + 1 < args.steps:
            fut = pref.submit(build_step, step + 1)
        opt.zero_grad(set_to_none=True)
        M = len(micros)
        tot_loss = 0.0
        for g in micros:
            ids = g["ids"].to(dev, non_blocking=True)
            labels = g["labels"].to(dev, non_blocking=True)
            E = embed(ids)
            if g["vecs"] is not None:
                V = g["vecs"].to(dev, non_blocking=True)
                E[ids == MARKER_ID] = (V / V.norm(dim=1, keepdim=True) * INJ_SCALE).to(E.dtype)
            if PACK:
                ce, sel = slice_ce(E, None, labels, pos_ids=g["pos"].to(dev, non_blocking=True))
                cnts = g["tcnts"]
            else:
                ce, sel = slice_ce(E, g["att"].to(dev, non_blocking=True), labels)
                cnts = sel.sum(1).tolist()
            (ce.mean() / M).backward()
            tot_loss += ce.mean().item() / M
            for cell, part, n in zip(g["cells"], torch.split(ce.detach(), cnts), cnts):
                if n:
                    a = online_acc.setdefault(cell, [0.0, 0])
                    a[0] += float(part.mean()); a[1] += 1
        if WORLD > 1:   # manual LoRA grad sync (unwrapped model)
            flat = torch.cat([p.grad.reshape(-1) for p in tparams])
            dist.all_reduce(flat)
            flat /= WORLD
            off = 0
            for p in tparams:
                n = p.grad.numel(); p.grad.copy_(flat[off:off + n].view_as(p.grad)); off += n
            del flat
        torch.nn.utils.clip_grad_norm_(tparams, 1.0)
        opt.step(); sched.step()
        if (step + 1) % 100 == 0:
            merged = online_acc
            if WORLD > 1:
                gath = [None] * WORLD
                dist.all_gather_object(gath, online_acc)
                merged = {}
                for d in gath:
                    for k, v in d.items():
                        a = merged.setdefault(k, [0.0, 0]); a[0] += v[0]; a[1] += v[1]
            if r0 and merged:
                with open(f"{DUR}/fd1_online_{args.tag}.jsonl", "a") as f:
                    f.write(json.dumps(dict(step=step + 1, ts=time.time(),
                            **{k: [v[0] / v[1], v[1]] for k, v in merged.items()})) + "\n")
            online_acc = {}
        if step % 10 == 0:
            lt = torch.tensor([tot_loss], device=dev)
            if WORLD > 1:
                dist.all_reduce(lt); lt /= WORLD
            if r0:
                dt = (time.time() - t0) / max(1, step - step0 + 1)
                print(f"step {step} loss {lt.item():.4f} ({dt:.1f}s/step, M{M})", flush=True)
                wandb.log({"train/loss": lt.item()}, step=step)
        if args.valloss_every and (step + 1) % args.valloss_every == 0:
            slopes = val_matrix(step + 1)
            if args.converge and slopes and step + 1 >= 5000:
                mx = max(abs(v) for v in slopes.values())
                if mx < 0.02:
                    if r0:
                        print(f"CONVERGED: all {len(slopes)} val cells flat (max |slope| {mx:.3f}); stopping.")
                        save(step + 1)
                    break
        if (step + 1) % args.eval_every == 0 or (args.smoke and step + 1 == args.steps):
            res = evaluate(step + 1)
            if r0: save(step + 1)
            if WORLD > 1: dist.barrier()
            key = res["held_ALL"]
            f1_hist.append(key)
            if key > best: best = key
            gap = res["held_ALL"] - res["held_ALL_donor"]
            if step + 1 >= 3 * args.eval_every and gap < 0.02:
                if r0: print(f"FAIL-EARLY: donor gap {gap:.3f} < 0.02 — acts not being read; aborting for redesign.")
                break
            if not args.converge and len(f1_hist) >= 8 and max(f1_hist[-3:]) < max(f1_hist[:-3]) + 0.005:
                if r0: print(f"STATIONARY: held_ALL plateaued ({[f'{v:.3f}' for v in f1_hist[-5:]]}); stopping.")
                break
    pref.shutdown(wait=False)
    if WORLD > 1:
        dist.barrier()
        dist.destroy_process_group()
    if r0: print("FD1 SFT DONE")

if __name__ == "__main__":
    main()
