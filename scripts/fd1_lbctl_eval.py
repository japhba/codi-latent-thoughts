"""AO (fd1c converged lora) on the LB-vs-control cells, matching the NLA eval:
rows = 41 load-bearing + 41 correct non-LB GSM8k-test rows (same sets as
bt1/nz2/{lb41,ctl41}_capture.jsonl, matched by gsmeval idx). Reads: Z k=1 and
L27 k=1 per thought position, Z/ALL joint, qonly. Labels: answer question +
allvals question (exact training phrasings). Scores: answer hit; vr-hit (LB) /
intermediates recall (ctl); vp/cf controls on LB rows.
-> /workspace-vast/jbauer/bt1/fd1/lbctl_eval.json     (1 GPU, ~minutes)
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from fd1_data import Pool, render_acts, canon, PREAMBLE, SRC

BASE = "Qwen/Qwen3-8B"
MARKER_ID, INJ_SCALE = 149705, 150.0
ANSQ = "What is the final answer of the computation?"
VALQ = "List every intermediate result computed (any order)."
DUR = "/workspace-vast/jbauer/bt1/fd1/dur"
NZ2 = "/workspace-vast/jbauer/bt1/nz2"
OUT = "/workspace-vast/jbauer/bt1/fd1/lbctl_eval.json"
LORA = f"{DUR}/fd1c_step66000_lora.pt"

dev = "cuda"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to(dev)
model = get_peft_model(model, LoraConfig(r=128, lora_alpha=256, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.eval()
embed = model.get_input_embeddings()
sd = torch.load(LORA, map_location="cpu", weights_only=False)
sd = {k: v for k, v in sd.get("lora", sd).items() if ".ref." not in k}
_, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected

gev = Pool("gsmeval")
idx2row = {m["idx"]: i for i, m in enumerate(gev.meta)}
lb = [json.loads(l) for l in open(f"{NZ2}/lb41_capture.jsonl")]
ctl = [json.loads(l) for l in open(f"{NZ2}/ctl41_capture.jsonl")]
wrong = [json.loads(l) for l in open(f"{NZ2}/wrong41_capture.jsonl")]
cells = {"lb": [(idx2row[r["idx"]], r) for r in lb if r["idx"] in idx2row],
         "ctl": [(idx2row[r["idx"]], r) for r in ctl if r["idx"] in idx2row],
         "wrong": [(idx2row[r["idx"]], r) for r in wrong if r["idx"] in idx2row]}
print({k: len(v) for k, v in cells.items()}, flush=True)

def nums_in(text):
    out = set()
    for mt in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        try: out.add(canon(mt))
        except ValueError: pass
    return out

def encode(prompt):
    txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, enable_thinking=False, tokenize=False)
    return tok(txt, add_special_tokens=False).input_ids

@torch.no_grad()
def gen(pv, max_new=32):
    outs = []
    for s0 in range(0, len(pv), 48):
        chunk = pv[s0:s0 + 48]
        seqs = [encode(p) for p, _ in chunk]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(chunk), L), tok.eos_token_id, dtype=torch.long)
        att = torch.zeros(len(chunk), L, dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, L - len(s):] = torch.tensor(s); att[b, L - len(s):] = 1
        ids_dev = ids.to(dev)
        E = embed(ids_dev)
        vecs_list = [v for _, v in chunk]
        if sum(len(v) for v in vecs_list):
            mask = ids_dev == MARKER_ID
            assert mask.sum(1).tolist() == [len(v) for v in vecs_list]
            V = torch.from_numpy(np.stack([v for vs in vecs_list for v in vs])).to(dev)
            E[mask] = (V / V.norm(dim=1, keepdim=True) * INJ_SCALE).to(E.dtype)
        g = model.generate(inputs_embeds=E, attention_mask=att.to(dev),
                           max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        outs += [tok.decode(s, skip_special_tokens=True) for s in g]
    return outs

READS = ([(f"Z_k1_p{p+1}", ["Z"], [p], "thoughts") for p in range(6)]
         + [(f"L27_k1_p{p+1}", ["L27"], [p], "thoughts") for p in range(6)]
         + [("Z_joint", ["Z"], list(range(6)), "thoughts"),
            ("ALL_joint", list(SRC), list(range(6)), "thoughts"),
            ("qonly", ["L27", "LF"], [], "q_only")])

res = {}
gens_f = open("/workspace-vast/jbauer/bt1/fd1/lbctl_gens.jsonl", "w")
for cell, rows in cells.items():
    for rname, S, tpos, pm in READS:
        for qname, q in (("ans", ANSQ), ("vals", VALQ)):
            pv = []
            for gi, _ in rows:
                acts, vecs = render_acts(gev, gi, S, pm, tpos)
                pv.append((PREAMBLE + acts + f"\n\nQuestion: {q}", vecs))
            texts = gen(pv, max_new=16 if qname == "ans" else 48)
            for x, (gi, meta) in zip(texts, rows):
                gens_f.write(json.dumps(dict(cell=cell, read=rname, q=qname,
                                             idx=meta["idx"], text=x)) + "\n")
            hits = {"ans": 0.0, "vr": 0.0, "int": 0.0, "vp": 0.0, "cf": 0.0, "pred": 0.0}
            for x, (gi, meta) in zip(texts, rows):
                E = nums_in(x)
                hits["ans"] += canon(meta["answer"]) in E
                if cell == "lb":
                    hits["vr"] += canon(meta["vr"]) in E
                    hits["vp"] += canon(meta["vp"]) in E
                    hits["cf"] += canon(meta["cf"]) in E
                else:
                    chain = {canon(v) for v in meta["intermediates"]
                             if str(v) != str(meta["answer"])}
                    if chain:
                        hits["int"] += len(E & chain) / len(chain)
                    if cell == "wrong":
                        hits["pred"] += canon(meta["pred"]) in E
            n = len(rows)
            res[f"{cell}.{rname}.{qname}"] = {k: v / n for k, v in hits.items()}
            print(f"{cell:3s} {rname:10s} {qname:4s} " +
                  " ".join(f"{k}={v/n:.3f}" for k, v in hits.items() if v or k == 'ans'),
                  flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"wrote {OUT}", flush=True)
