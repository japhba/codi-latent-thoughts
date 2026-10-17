"""Text-combine control for the fd1 AO answer-read decomposition (lbctl cells):
can the q-only reader recover the answer once the key intermediate is HANDED OVER as
text? Isolates the reader's combine capacity from its thought-decoding.

Reads (all q-only acts, L27+LF at question positions, no thought vectors):
  qonly          : baseline replication (matches lbctl_eval qonly cell)
  qonly_vr       : + "A known intermediate result of the computation is: {v}."
                   v = vr on LB rows, first non-answer intermediate on ctl/wrong rows
  qonly_donorval : + the same sentence with the NEXT row's value (misleading placebo)
Scores: answer hit (wrong rows additionally: CODI-pred hit), vr/cf/vp echo on LB.
-> /workspace-vast/jbauer/bt1/fd1/textcombine_eval.json  (1 GPU, ~minutes)
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from fd1_data import Pool, render_acts, canon, PREAMBLE

BASE = "Qwen/Qwen3-8B"
MARKER_ID, INJ_SCALE = 149705, 150.0
ANSQ = "What is the final answer of the computation?"
DUR = "/workspace-vast/jbauer/bt1/fd1/dur"
NZ2 = "/workspace-vast/jbauer/bt1/nz2"
OUT = "/workspace-vast/jbauer/bt1/fd1/textcombine_eval.json"
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
cells = {}
for cell, fn in (("lb", "lb41"), ("ctl", "ctl41"), ("wrong", "wrong41")):
    rows = [json.loads(l) for l in open(f"{NZ2}/{fn}_capture.jsonl")]
    cells[cell] = [(idx2row[r["idx"]], r) for r in rows if r["idx"] in idx2row]
print({k: len(v) for k, v in cells.items()}, flush=True)


def keyval(meta, cell):
    if cell == "lb":
        return meta["vr"]
    vals = [v for v in meta["intermediates"] if str(v) != str(meta["answer"])]
    return vals[0] if vals else meta["intermediates"][0]


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
def gen(pv, max_new=16):
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


res = {}
gens_f = open("/workspace-vast/jbauer/bt1/fd1/textcombine_gens.jsonl", "w")
for cell, rows in cells.items():
    for rname in ("qonly", "qonly_vr", "qonly_donorval"):
        pv = []
        for k, (gi, meta) in enumerate(rows):
            acts, vecs = render_acts(gev, gi, ["L27", "LF"], "q_only", [])
            extra = ""
            if rname == "qonly_vr":
                extra = (f"\n\nA known intermediate result of the computation is: "
                         f"{keyval(meta, cell)}.")
            elif rname == "qonly_donorval":
                dmeta = rows[(k + 1) % len(rows)][1]
                extra = (f"\n\nA known intermediate result of the computation is: "
                         f"{keyval(dmeta, cell)}.")
            pv.append((PREAMBLE + acts + extra + f"\n\nQuestion: {ANSQ}", vecs))
        texts = gen(pv)
        for x, (gi, meta) in zip(texts, rows):
            gens_f.write(json.dumps(dict(cell=cell, read=rname, idx=meta["idx"], text=x)) + "\n")
        hits = {"ans": 0.0, "vr_echo": 0.0, "cf": 0.0, "vp": 0.0, "pred": 0.0}
        for x, (gi, meta) in zip(texts, rows):
            E = nums_in(x)
            hits["ans"] += canon(meta["answer"]) in E
            if cell == "lb":
                hits["vr_echo"] += canon(meta["vr"]) in E
                hits["cf"] += canon(meta["cf"]) in E
                hits["vp"] += canon(meta["vp"]) in E
            if cell == "wrong":
                hits["pred"] += canon(meta["pred"]) in E
        n = len(rows)
        res[f"{cell}.{rname}"] = {k: v / n for k, v in hits.items()}
        print(f"{cell:5s} {rname:14s} " + " ".join(f"{k}={v/n:.3f}" for k, v in hits.items()),
              flush=True)
gens_f.close()
json.dump(res, open(OUT, "w"), indent=1)
print(f"wrote {OUT}", flush=True)
