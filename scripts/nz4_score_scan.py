"""nz4 dense score scan: regenerate verbalizations from the FULL nz4 checkpoint
ladder (warmstart step 0 + all 100 iter_* adapters, every 4 steps) for the same
held-out (doc-hash val) problems as the nz1 scan, dump full gens for the
answer/judged-intermediates scoring that feeds the writeup's NLA figure.

  python3 nz4_score_scan.py --which z|h24 [--n 60] [--bs 16]
Output: /workspace-vast/jbauer/bt1/nz4/score_scan_{which}.jsonl (one row per gen,
append, resumable by step). Recipe identical to nz1_score_scan.py: layer-1
norm-matched ADD injection at the single marker, greedy 256 tokens,
<explanation> body extracted.
"""
import argparse, glob, json, os, re, sys

sys.path.insert(0, "/workspace-vast/jbauer/activation_oracles_dev/third_party/EasyNLA")
import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.val_split import is_val_doc, val_doc_permille

MARKER_ID = 149705
WS = "/workspace-vast/jbauer/bt1/nz1/warmstart"
DATA = "/workspace-vast/jbauer/bt1/nz1/data"
FD1DATA = "/workspace-vast/jbauer/bt1/fd1/data"
SAVE = "/workspace-vast/jbauer/exp/training/nz4_{w}"
OUTD = "/workspace-vast/jbauer/bt1/nz4"

ap = argparse.ArgumentParser()
ap.add_argument("--which", required=True, choices=["z", "h24"])
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--bs", type=int, default=16)
args = ap.parse_args()

def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"

def nums_in(text):
    out = set()
    for mt in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        try: out.add(canon(mt))
        except (ValueError, OverflowError): pass
    return out

meta = {}
for l in open(f"{FD1DATA}/gsm_meta.jsonl"):
    m = json.loads(l)
    vals = [canon(v) for v in m["intermediates"] if v != m["answer"]]
    meta[m["uid"]] = dict(chain=vals, answer=canon(m["answer"]))

tbl = pq.read_table(f"{DATA}/{args.which}_rl.parquet")
prompt_msgs = tbl["prompt"][0].as_py()
PERMILLE = val_doc_permille(50000, len(tbl))
docs, seen = [], set()
for i in range(len(tbl)):
    uid = tbl["doc_id"][i].as_py()
    if uid in seen or not is_val_doc(uid, PERMILLE) or uid not in meta: continue
    seen.add(uid)
    docs.append((uid, np.array(tbl["activation_vector"][i].as_py(), dtype=np.float32)))
    if len(docs) >= args.n: break
print(f"{args.which}: scanning {len(docs)} held-out problems", flush=True)

tok = AutoTokenizer.from_pretrained(WS + "/av")
base = AutoModelForCausalLM.from_pretrained(WS + "/av", torch_dtype=torch.bfloat16,
                                            attn_implementation="sdpa").to("cuda")
base.eval()

STATE = {}
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if h.shape[1] > 1 and "pos" in STATE:          # prefill only
        for b, (p, v) in enumerate(zip(STATE["pos"], STATE["vec"])):
            hv = h[b, p]
            h[b, p] = hv + hv.norm() * (v / v.norm())
    return out
def layer1(m):
    return (m.get_base_model() if hasattr(m, "get_base_model") else m).model.layers[1]

txt = tok.apply_chat_template(
    [{"role": r["role"], "content": r["content"].replace("<INJECT>", tok.decode([MARKER_ID]))}
     for r in prompt_msgs], add_generation_prompt=True, enable_thinking=False, tokenize=False)
pids = tok(txt, add_special_tokens=False).input_ids
assert pids.count(MARKER_ID) == 1

@torch.no_grad()
def gen_all(model, docs):
    hnd = layer1(model).register_forward_hook(hook)
    outs = []
    try:
        for s0 in range(0, len(docs), args.bs):
            ch = docs[s0:s0 + args.bs]
            L = len(pids)
            ids = torch.tensor([pids] * len(ch)).to("cuda")
            att = torch.ones_like(ids)
            STATE["pos"] = [pids.index(MARKER_ID)] * len(ch)
            STATE["vec"] = [torch.tensor(v, dtype=torch.bfloat16, device="cuda") for _, v in ch]
            g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=256,
                               do_sample=False, pad_token_id=tok.eos_token_id)
            outs += [tok.decode(s[L:], skip_special_tokens=True) for s in g]
    finally:
        hnd.remove()
        STATE.clear()
    return outs

iters = sorted(int(os.path.basename(p).split("_")[1])
               for p in glob.glob(SAVE.format(w=args.which) + "/iter_*"))
CKPTS = [(0, None)] + [(s, SAVE.format(w=args.which) + f"/iter_{s:06d}") for s in iters]
outp = f"{OUTD}/score_scan_{args.which}.jsonl"
done = set()
if os.path.exists(outp):
    done = {json.loads(l)["step"] for l in open(outp)}
    print(f"resume: {len(done)} steps already scanned", flush=True)

from tqdm.auto import tqdm
for step, adir in tqdm(CKPTS, desc=f"ckpts {args.which}"):
    if step in done:
        continue
    if adir is None:
        model = base
    else:
        model = PeftModel.from_pretrained(base, adir)
        model.eval()
    gens = gen_all(model, docs)
    with open(outp, "a") as f:
        for (uid, _), g in zip(docs, gens):
            m = meta[uid]
            ex = re.search(r"<explanation>(.*?)(</explanation>|$)", g, re.S)
            body = ex.group(1).strip() if ex else g.strip()
            E = nums_in(body)
            chain = set(m["chain"])
            rec = len(E & chain) / len(chain) if chain else None
            f.write(json.dumps(dict(step=step, uid=uid, gen=body,
                                    chain_recall=rec, answer_hit=int(m["answer"] in E))) + "\n")
    print(f"ckpt {step}: done ({len(gens)} gens)", flush=True)
    if adir is not None:
        model = model.unload()   # strip adapter, restore clean base
print(f"SCAN OUT -> {outp}")
print(f"NZ4 SCAN {args.which} DONE")
