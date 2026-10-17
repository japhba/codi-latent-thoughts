"""fd1 post-hoc control battery (pod): evaluate a trained fd1 lora over the full
control grid. The variable-subset training makes every cell in-distribution.

  python3 fd1_battery.py --lora /root/fd1_ckpt/fd1a/stepN/lora.pt --out battery_a.json

Cells (held 100 gsm rows unless noted; F1 vs gold chain, greedy):
  grid:        source x count k in {1..6} own; donor at k=6
  pairs:       Z+L18, Z+L27, Z+LF, L18+L27, L9+L18 (k=6)
  shufpres:    Z and ALL with marker/vec presentation order shuffled (order-in-vector test)
  partdonor:   Z k=6 with position t swapped to donor's z_t, t=1..6 (per-position reliance)
  placebo:     gaussian vectors norm 150 (floor)
  opsq/countq: ops and step-count questions per source (exact/F1)
  wrongpred:   test wrong-stratum rows, answer question — does AO track CODI's (wrong)
               pred over gold answer? (reads the computation, not the problem)
  twins:       syn twin pairs — own-gold vs twin-gold score + value-overlap baseline
"""
import argparse, json, random, re, sys

sys.path.insert(0, "/root/fd1")
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from fd1_data import Pool, render_acts, canon, PREAMBLE, SRC, OPNAME

BASE = "Qwen/Qwen3-8B"
MARKER_ID, INJ_SCALE = 149705, 150.0
LISTQ = "List every intermediate result computed (any order)."
OPSQ = "Which arithmetic operations occur in the computation?"
CNTQ = "How many computation steps are there in total?"
ANSQ = "What is the final answer of the computation?"

ap = argparse.ArgumentParser()
ap.add_argument("--lora", required=True)
ap.add_argument("--out", default="battery.json")
ap.add_argument("--n", type=int, default=100)
args = ap.parse_args()

dev = "cuda"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
model = get_peft_model(model, LoraConfig(r=128, lora_alpha=256, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
sd = torch.load(args.lora, map_location="cpu", weights_only=False)
sd = {k: v for k, v in sd.get("lora", sd).items() if ".ref." not in k}
_, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected
model.eval()
embed = model.get_input_embeddings()

gsm, syn, gev = Pool("gsm"), Pool("syn"), Pool("gsmeval")
held = gsm.ok[-200:][:args.n]
import os
ours_path = next(p for p in ("/root/nx2/data/ours_full.jsonl",
                             "/workspace-vast/jbauer/bt1/nx1/pod_salvage/work/data/ours_full.jsonl")
                 if os.path.exists(p))
flags = {json.loads(l)["idx"]: json.loads(l) for l in open(ours_path)}
wrong_rows = [i for i, m in enumerate(gev.meta)
              if not flags[m["idx"]]["lb"] and not (flags[m["idx"]]["correct"] and flags[m["idx"]]["correct_k0"])][:60]

def encode(prompt):
    txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, enable_thinking=False, tokenize=False)
    return tok(txt, add_special_tokens=False).input_ids

@torch.no_grad()
def gen(pv, max_new=96):
    outs = []
    for s0 in range(0, len(pv), 48):
        chunk = pv[s0:s0 + 48]
        seqs = [encode(p) for p, _ in chunk]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(chunk), L), tok.eos_token_id, dtype=torch.long)
        att = torch.zeros(len(chunk), L, dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, L - len(s):] = torch.tensor(s); att[b, L - len(s):] = 1
        E = embed(ids.to(dev))
        for b, (_, vecs) in enumerate(chunk):
            mpos = (ids[b] == MARKER_ID).nonzero().flatten().tolist()
            assert len(mpos) == len(vecs), (len(mpos), len(vecs))
            for p, v in zip(mpos, vecs):
                z = torch.from_numpy(np.ascontiguousarray(v)).to(dev)
                E[b, p] = (z / z.norm() * INJ_SCALE).to(E.dtype)
        g = model.generate(inputs_embeds=E, attention_mask=att.to(dev), max_new_tokens=max_new,
                           do_sample=False, pad_token_id=tok.eos_token_id)
        outs += [tok.decode(s, skip_special_tokens=True) for s in g]
    return outs

def nums_in(t):
    out = set()
    for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", t):
        try: out.add(canon(m))
        except ValueError: pass
    return out

def f1(t, gold):
    E, G = nums_in(t), set(gold)
    if not E or not G: return 0.0
    h = len(E & G)
    p, r = h / len(E), h / len(G)
    return 2 * p * r / (p + r) if h else 0.0

def prompt_for(pool, i, sources, tpos, q=LISTQ, vec_override=None, pos_mode="thoughts"):
    acts, vecs = render_acts(pool, i, sources, pos_mode, tpos)
    if vec_override is not None: vecs = vec_override
    return (PREAMBLE + acts + f"\n\nQuestion: {q}", vecs)

R = {}
gold_h = [[c["val"] for c in gsm.meta[r]["chain"]] for r in held]
def cellmean(pv, golds=None):
    golds = golds or gold_h
    return float(np.mean([f1(x, g) for x, g in zip(gen(pv), golds)]))

# 1. grid
for src in SRC + ["ALL"]:
    S = SRC if src == "ALL" else [src]
    for k in range(1, 7):
        cr = random.Random(1000 + k)
        pv = [prompt_for(gsm, r, S, sorted(cr.sample(range(6), k))) for r in held]
        R[f"grid_{src}_k{k}"] = cellmean(pv)
    pv = [prompt_for(gsm, held[(j + 13) % len(held)], S, list(range(6))) for j, r in enumerate(held)]
    R[f"grid_{src}_donor"] = cellmean(pv)
    print(f"grid {src}: " + " ".join(f"k{k} {R[f'grid_{src}_k{k}']:.2f}" for k in range(1, 7)) + f" donor {R[f'grid_{src}_donor']:.2f}", flush=True)

# 2. pairs (k=6)
for a, b in (("Z", "L18"), ("Z", "L27"), ("Z", "LF"), ("L18", "L27"), ("L9", "L18")):
    pv = [prompt_for(gsm, r, [a, b], list(range(6))) for r in held]
    R[f"pair_{a}+{b}"] = cellmean(pv)
print("pairs:", {k: round(v, 3) for k, v in R.items() if k.startswith("pair")}, flush=True)

# 3. presentation shuffle (vectors permuted among marker slots of each source group)
sr = random.Random(7)
for name, S in (("Z", ["Z"]), ("ALL", SRC)):
    pv = []
    for r in held:
        acts, vecs = render_acts(gsm, r, S, "thoughts", list(range(6)))
        perm = list(range(6)); sr.shuffle(perm)
        shuf = []
        for gidx in range(len(S)):
            grp = vecs[gidx * 6:(gidx + 1) * 6]
            shuf += [grp[p] for p in perm]      # same shuffle within every group
        pv.append((PREAMBLE + acts + f"\n\nQuestion: {LISTQ}", shuf))
    R[f"shufpres_{name}"] = cellmean(pv)
print("shufpres:", R["shufpres_Z"], R["shufpres_ALL"], flush=True)

# 4. partial donor on Z (k=6, position t from donor row)
for t in range(6):
    pv = []
    for j, r in enumerate(held):
        d = held[(j + 13) % len(held)]
        vecs = [np.asarray(gsm.z[r, u], dtype=np.float32) for u in range(6)]
        vecs[t] = np.asarray(gsm.z[d, t], dtype=np.float32)
        acts, _ = render_acts(gsm, r, ["Z"], "thoughts", list(range(6)))
        pv.append((PREAMBLE + acts + f"\n\nQuestion: {LISTQ}", vecs))
    R[f"partdonor_z{t+1}"] = cellmean(pv)
print("partdonor:", " ".join(f"z{t+1} {R[f'partdonor_z{t+1}']:.2f}" for t in range(6)), flush=True)

# 5. placebo
pr = np.random.default_rng(0)
pv = []
for r in held:
    acts, vecs = render_acts(gsm, r, ["Z"], "thoughts", list(range(6)))
    pv.append((PREAMBLE + acts + f"\n\nQuestion: {LISTQ}",
               [pr.standard_normal(4096).astype(np.float32) for _ in range(6)]))
R["placebo_Z"] = cellmean(pv)
print("placebo:", R["placebo_Z"], flush=True)

# 6. ops / count questions
for src in ("Z", "L18", "ALL"):
    S = SRC if src == "ALL" else [src]
    pv = [prompt_for(gsm, r, S, list(range(6)), q=OPSQ) for r in held]
    texts = gen(pv, max_new=32)
    accs = []
    for x, r in zip(texts, held):
        m = gsm.meta[r]
        gold_ops = {OPNAME[c["step"]["op"]] for c in m["chain"] if c["step"]}
        if not gold_ops: continue
        got = {o for o in OPNAME.values() if o in x.lower()}
        accs.append(len(got & gold_ops) / max(1, len(got | gold_ops)))
    R[f"ops_{src}"] = float(np.mean(accs))
    pv = [prompt_for(gsm, r, S, list(range(6)), q=CNTQ) for r in held]
    texts = gen(pv, max_new=8)
    R[f"count_{src}"] = float(np.mean([
        1.0 if re.search(rf"\b{len(gsm.meta[r]['chain']) + 1}\b", x) else 0.0 for x, r in zip(texts, held)]))
print("ops/count:", {k: round(v, 3) for k, v in R.items() if k.startswith(("ops", "count"))}, flush=True)

# 7. wrong-row pred tracking (test wrong stratum): answer question
for src in ("Z", "ALL"):
    S = SRC if src == "ALL" else [src]
    pv = [prompt_for(gev, r, S, list(range(6)), q=ANSQ) for r in wrong_rows]
    texts = gen(pv, max_new=16)
    hitp = hitg = 0; rows_out = []
    for x, r in zip(texts, wrong_rows):
        m = gev.meta[r]
        E = nums_in(x)
        hp, hg = (m["pred"] in E), (m["answer"] in E)
        hitp += hp; hitg += hg
        rows_out.append(dict(idx=m["idx"], pred=m["pred"], gold=m["answer"], gen=x, hit_pred=hp, hit_gold=hg))
    R[f"wrong_{src}_hitpred"] = hitp / len(wrong_rows)
    R[f"wrong_{src}_hitgold"] = hitg / len(wrong_rows)
    if src == "ALL": R["wrong_rows"] = rows_out
print(f"wrongpred: Z pred {R['wrong_Z_hitpred']:.2f} vs gold {R['wrong_Z_hitgold']:.2f} | "
      f"ALL pred {R['wrong_ALL_hitpred']:.2f} vs gold {R['wrong_ALL_hitgold']:.2f}", flush=True)

# 7b. qonly wrongpred — question-region acts causally precede the thoughts, so a
# recomputing reader should hit GOLD at its solve rate, a state-reader hits neither;
# two-ended dissociation vs the act-read cells above (which should hit PRED).
pv = [prompt_for(gev, r, ["L27", "LF"], [], q=ANSQ, pos_mode="q_only") for r in wrong_rows]
texts = gen(pv, max_new=16)
hitp = hitg = 0
for x, r in zip(texts, wrong_rows):
    m = gev.meta[r]; E = nums_in(x)
    hitp += (m["pred"] in E); hitg += (m["answer"] in E)
R["wrong_qonly_hitpred"] = hitp / len(wrong_rows)
R["wrong_qonly_hitgold"] = hitg / len(wrong_rows)
print(f"wrongpred qonly: pred {R['wrong_qonly_hitpred']:.2f} vs gold {R['wrong_qonly_hitgold']:.2f}", flush=True)

# 8. twins (all captured twin pairs regardless of correctness of partner; own row must be ok)
bycard = {}
for i in range(len(syn.meta)): bycard.setdefault(syn.meta[i]["card_id"], i)
tw_rows = [(i, bycard[syn.meta[i]["twin_of"]]) for i in syn.ok
           if syn.meta[i].get("twin_of") and syn.meta[i]["twin_of"] in bycard][:100]
if tw_rows:
    pv = [prompt_for(syn, i, SRC, list(range(6))) for i, _ in tw_rows]
    texts = gen(pv)
    own_s, twin_s, ovl = [], [], []
    for x, (i, p) in zip(texts, tw_rows):
        go = [c["val"] for c in syn.meta[i]["chain"]]
        gt = [c["val"] for c in syn.meta[p]["chain"]]
        own_s.append(f1(x, go)); twin_s.append(f1(x, gt))
        ovl.append(len(set(go) & set(gt)) / max(1, len(set(go) | set(gt))))
    R["twin_own"], R["twin_cross"], R["twin_overlap"] = map(lambda v: float(np.mean(v)), (own_s, twin_s, ovl))
    R["twin_n"] = len(tw_rows)
    print(f"twins n={len(tw_rows)}: own {R['twin_own']:.3f} cross {R['twin_cross']:.3f} value-overlap {R['twin_overlap']:.3f}", flush=True)

json.dump(R, open(args.out, "w"), indent=1)
print("BATTERY DONE ->", args.out)
