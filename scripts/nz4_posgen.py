"""nz4 per-position generation scan for the ordering figure (fig_08): verbalize
EACH thought position t=1..6 of the held-out problems through the FINAL nz4
reader (iter_003200) of the given read object (z or h24), plus a donor-fed
control (the same reader fed the position-t vector of a +7-shifted problem,
scored against the own problem's gold chain — the chance floor). Row schema
matches the nx2 posgen fig_08 reads:
  {"uid", "t" (0-5), "kind" ("own"|"donor"), "gen", "gold" (ordered chain values)}

  python3 nz4_posgen.py --which z|h24
Output: /workspace-vast/jbauer/bt1/nz4/posgen_3200.jsonl (z, historical name) or
        /workspace-vast/jbauer/bt1/nz4/posgen_h24_3200.jsonl
"""
import argparse, json, os, re, sys

sys.path.insert(0, "/workspace-vast/jbauer/activation_oracles_dev/third_party/EasyNLA")
import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.val_split import is_val_doc, val_doc_permille

ap = argparse.ArgumentParser()
ap.add_argument("--which", required=True, choices=["z", "h24"])
args = ap.parse_args()

MARKER_ID = 149705
WS = "/workspace-vast/jbauer/bt1/nz1/warmstart"
DATA = "/workspace-vast/jbauer/bt1/nz1/data"
FD1DATA = "/workspace-vast/jbauer/bt1/fd1/data"
CKPT = f"/workspace-vast/jbauer/exp/training/nz4_{args.which}/iter_003200"
OUT = ("/workspace-vast/jbauer/bt1/nz4/posgen_3200.jsonl" if args.which == "z"
       else "/workspace-vast/jbauer/bt1/nz4/posgen_h24_3200.jsonl")
N, BS, SHIFT = 60, 16, 7


def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"


meta = {}
for l in open(f"{FD1DATA}/gsm_meta.jsonl"):
    m = json.loads(l)
    meta[m["uid"]] = [canon(v) for v in m["intermediates"]]

tbl = pq.read_table(f"{DATA}/{args.which}_rl.parquet")
prompt_msgs = tbl["prompt"][0].as_py()
PERMILLE = val_doc_permille(50000, len(tbl))
byuid = {}
for i in range(len(tbl)):
    uid = tbl["doc_id"][i].as_py()
    if not is_val_doc(uid, PERMILLE) or uid not in meta:
        continue
    byuid.setdefault(uid, []).append(np.array(tbl["activation_vector"][i].as_py(),
                                              dtype=np.float32))
uids = [u for u in byuid if len(byuid[u]) == 6][:N]
print(f"posgen: {len(uids)} problems x 6 positions x own/donor", flush=True)

tok = AutoTokenizer.from_pretrained(WS + "/av")
base = AutoModelForCausalLM.from_pretrained(WS + "/av", torch_dtype=torch.bfloat16,
                                            attn_implementation="sdpa").to("cuda")
model = PeftModel.from_pretrained(base, CKPT)
model.eval()

STATE = {}
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if h.shape[1] > 1 and "pos" in STATE:
        for b, (p, v) in enumerate(zip(STATE["pos"], STATE["vec"])):
            hv = h[b, p]
            h[b, p] = hv + hv.norm() * (v / v.norm())
    return out

hnd = model.get_base_model().model.layers[1].register_forward_hook(hook)

txt = tok.apply_chat_template(
    [{"role": r["role"], "content": r["content"].replace("<INJECT>", tok.decode([MARKER_ID]))}
     for r in prompt_msgs], add_generation_prompt=True, enable_thinking=False, tokenize=False)
pids = tok(txt, add_special_tokens=False).input_ids
assert pids.count(MARKER_ID) == 1

@torch.no_grad()
def gen_batch(vecs):
    outs = []
    for s0 in range(0, len(vecs), BS):
        ch = vecs[s0:s0 + BS]
        L = len(pids)
        ids = torch.tensor([pids] * len(ch)).to("cuda")
        STATE["pos"] = [pids.index(MARKER_ID)] * len(ch)
        STATE["vec"] = [torch.tensor(v, dtype=torch.bfloat16, device="cuda") for v in ch]
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                           max_new_tokens=256, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        outs += [tok.decode(s[L:], skip_special_tokens=True) for s in g]
    STATE.clear()
    return outs

from tqdm.auto import tqdm
with open(OUT, "w") as f:
    for t in tqdm(range(6), desc="positions"):
        own_vecs = [byuid[u][t] for u in uids]
        donor_vecs = [byuid[uids[(j + SHIFT) % len(uids)]][t] for j in range(len(uids))]
        for kind, vecs in (("own", own_vecs), ("donor", donor_vecs)):
            gens = gen_batch(vecs)
            for u, g in zip(uids, gens):
                ex = re.search(r"<explanation>(.*?)(</explanation>|$)", g, re.S)
                body = ex.group(1).strip() if ex else g.strip()
                f.write(json.dumps(dict(uid=u, t=t, kind=kind, gen=body,
                                        gold=meta[u])) + "\n")
        f.flush()
hnd.remove()
print(f"wrote {OUT}")
print(f"NZ4 POSGEN {args.which} DONE")
