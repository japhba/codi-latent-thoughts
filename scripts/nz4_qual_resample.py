"""Resampled verbalizations for the qualitative NLA card (fig_05): thought t of a
problem (z_{t+1} or h_{t+1}^(24)) read by the matching nz4 reader at the warmstart
(step 0) and the final adapter (step 3200). Per (item, step): one greedy decode
(pipeline check against the greedy reads on record: nz4_posgen at step 3200, the dense
scan at step 0 for t = 0) plus --n samples at the rollout distribution of training
(temperature 1.0, top_p 1, no top_k, 256 new tokens; run_config.yaml of nz4_{z,h24}),
torch seed reset per item. Injection recipe identical to nz4_score_scan.py /
nz4_posgen.py: layer-1 norm-matched ADD at the single marker. Thought t = the uid's
t-th row in {z,h24}_rl.parquet (row order within a uid is t = 0..5, the nz4_posgen.py
convention; its t=0 read equals the scan read).

  python3 nz4_qual_resample.py --which z|h24 --items g5096:3 g5148:0 [--n 2] [--seed 0]
Output: artifacts/nz4_qual_resample_{which}.jsonl, one row per gen
  {"uid", "t", "step", "decode" ("greedy"|"sample"), "k", "seed", "gen" (<explanation> body)}
"""
import argparse, json, os, re

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
ROOT = os.environ.get("CODI_ROOT", "/nfs/nhome/live/jbauer/codi-latent-thoughts")
WS = os.environ.get("NLA_WS", f"{ROOT}/checkpoints/nla_warmstart_nz1")
MARKER_ID = 149705
STEPS = (0, 3200)

ap = argparse.ArgumentParser()
ap.add_argument("--which", required=True, choices=["z", "h24"])
ap.add_argument("--items", nargs="+", required=True, help="uid:t pairs, t = 0..5")
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
SAVE = os.environ.get("NLA_SAVE", f"{ROOT}/checkpoints/nla_nz4_{args.which}")
PARQ = os.environ.get("NLA_PARQ", f"{ROOT}/experiments/bt1/nz1/data/{args.which}_rl.parquet")
POSGEN = os.path.join(ART, "nz4_posgen_3200.jsonl" if args.which == "z" else "nz4_posgen_h24_3200.jsonl")
SCAN = os.path.join(ART, f"nz4_score_scan_{args.which}.jsonl")
OUT = os.path.join(ART, f"nz4_qual_resample_{args.which}.jsonl")
ITEMS = [(u, int(t)) for u, t in (it.split(":") for it in args.items)]

pf = pq.ParquetFile(PARQ)
doc = pq.read_table(PARQ, columns=["doc_id"])["doc_id"].to_pylist()
starts = [0]
for rg in range(pf.num_row_groups):
    starts.append(starts[-1] + pf.metadata.row_group(rg).num_rows)
VEC, prompt_msgs = {}, None
for uid, t in ITEMS:
    gi = [i for i, d in enumerate(doc) if d == uid][t]
    rg = max(r for r in range(pf.num_row_groups) if starts[r] <= gi)
    tb = pf.read_row_group(rg, columns=["activation_vector", "prompt", "doc_id"])
    assert tb["doc_id"][gi - starts[rg]].as_py() == uid
    VEC[(uid, t)] = torch.tensor(tb["activation_vector"][gi - starts[rg]].as_py(), dtype=torch.bfloat16, device="cuda")
    prompt_msgs = tb["prompt"][gi - starts[rg]].as_py()
    print(f"{args.which} {uid} t={t}: parquet row {gi}", flush=True)

tok = AutoTokenizer.from_pretrained(WS + "/av")
base = AutoModelForCausalLM.from_pretrained(WS + "/av", dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
base.eval()

STATE = {}
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if h.shape[1] > 1 and "vec" in STATE:          # prefill only
        v = STATE["vec"]
        for b in range(h.shape[0]):
            hv = h[b, STATE["pos"]]
            h[b, STATE["pos"]] = hv + hv.norm() * (v / v.norm())
    return out

txt = tok.apply_chat_template(
    [{"role": r["role"], "content": r["content"].replace("<INJECT>", tok.decode([MARKER_ID]))}
     for r in prompt_msgs], add_generation_prompt=True, enable_thinking=False, tokenize=False)
pids = tok(txt, add_special_tokens=False).input_ids
assert pids.count(MARKER_ID) == 1
STATE["pos"] = pids.index(MARKER_ID)


def body(g):
    ex = re.search(r"<explanation>(.*?)(</explanation>|$)", g, re.S)
    return ex.group(1).strip() if ex else g.strip()


@torch.no_grad()
def gen(model, n, sample):
    ids = torch.tensor([pids] * n, device="cuda")
    kw = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0) if sample else dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=256, pad_token_id=tok.eos_token_id, **kw)
    return [body(tok.decode(s[len(pids):], skip_special_tokens=True)) for s in g]


ref = {(r["uid"], r["t"], 3200): body(r["gen"]) for r in map(json.loads, open(POSGEN)) if r["kind"] == "own"}
ref |= {(r["uid"], 0, 0): r["gen"] for r in map(json.loads, open(SCAN)) if r["step"] == 0}
rows = []
for step in STEPS:
    model = base if step == 0 else PeftModel.from_pretrained(base, f"{SAVE}/iter_{step:06d}").eval()
    hnd = (model.get_base_model() if step else model).model.layers[1].register_forward_hook(hook)
    for uid, t in ITEMS:
        STATE["vec"] = VEC[(uid, t)]
        greedy = gen(model, 1, False)[0]
        r0 = ref.get((uid, t, step))
        print(f"step {step} {uid} t={t}: greedy reproduces record: {'no record' if r0 is None else greedy == r0}", flush=True)
        rows.append(dict(uid=uid, t=t, step=step, decode="greedy", k=0, seed=None, gen=greedy))
        torch.manual_seed(args.seed)
        for k, s in enumerate(gen(model, args.n, True), 1):
            rows.append(dict(uid=uid, t=t, step=step, decode="sample", k=k, seed=args.seed, gen=s))
    hnd.remove()
    if step:
        model.unload()
for r in rows:
    print(f"\n===== {r['uid']} t={r['t']} step {r['step']} {r['decode']} {r['k']} =====\n{r['gen']}", flush=True)
with open(OUT, "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
print(f"wrote {OUT}")
