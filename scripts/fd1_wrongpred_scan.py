"""Solve-creep ladder scan: wrongpred answer-reads (Z / ALL / qonly) across the fd1c
lora ladder. On CODI-wrong test rows, a state-READER hits CODI's (wrong) pred; a
RECOMPUTER hits gold. pred/gold vs training step = when (if ever) solving creeps in.

  python3 fd1_wrongpred_scan.py [--every 2000]
Resumable: skips steps already in wrongpred_scan.jsonl.
"""
import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from fd1_data import Pool, render_acts, canon, PREAMBLE, SRC

BASE = "Qwen/Qwen3-8B"
MARKER_ID, INJ_SCALE = 149705, 150.0
ANSQ = "What is the final answer of the computation?"
DUR = "/workspace-vast/jbauer/bt1/fd1/dur"
OUT = "/workspace-vast/jbauer/bt1/fd1/wrongpred_scan.jsonl"
PNG = "/workspace-vast/jbauer/bt1/fd1/wrongpred_scan.png"

ap = argparse.ArgumentParser()
ap.add_argument("--every", type=int, default=2000)
args = ap.parse_args()

dev = "cuda"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to(dev)
model = get_peft_model(model, LoraConfig(r=128, lora_alpha=256, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.eval()
embed = model.get_input_embeddings()

gev = Pool("gsmeval")
ours_path = next(p for p in ("/root/nx2/data/ours_full.jsonl",
                             "/workspace-vast/jbauer/bt1/nx1/pod_salvage/work/data/ours_full.jsonl")
                 if os.path.exists(p))
flags = {json.loads(l)["idx"]: json.loads(l) for l in open(ours_path)}
wrong_rows = [i for i, m in enumerate(gev.meta)
              if not flags[m["idx"]]["lb"] and not (flags[m["idx"]]["correct"] and flags[m["idx"]]["correct_k0"])][:60]
print(f"wrong rows: {len(wrong_rows)}", flush=True)

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

def prompt_for(i, sources, tpos, pos_mode):
    acts, vecs = render_acts(gev, i, sources, pos_mode, tpos)
    return (PREAMBLE + acts + f"\n\nQuestion: {ANSQ}", vecs)

ladder = sorted(int(re.search(r"step(\d+)_lora", p).group(1))
                for p in glob.glob(f"{DUR}/fd1c_step*_lora.pt"))
ladder = [s for s in ladder if s % args.every == 0 or s == 1000]
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["step"] for l in open(OUT)}
print(f"ladder: {len(ladder)} ckpts, {len(done)} already done", flush=True)

for step in ladder:
    if step in done: continue
    sd = torch.load(f"{DUR}/fd1c_step{step}_lora.pt", map_location="cpu", weights_only=False)
    sd = {k: v for k, v in sd.get("lora", sd).items() if ".ref." not in k}
    _, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected
    row = {"step": step}
    for name, S, pm in (("Z", ["Z"], "thoughts"), ("ALL", list(SRC), "thoughts"),
                        ("qonly", ["L27", "LF"], "q_only")):
        tpos = [] if pm == "q_only" else list(range(6))
        pv = [prompt_for(r, S, tpos, pm) for r in wrong_rows]
        texts = gen(pv)
        hp = hg = 0
        for x, r in zip(texts, wrong_rows):
            m = gev.meta[r]; E = nums_in(x)
            hp += (m["pred"] in E); hg += (m["answer"] in E)
        row[f"{name}_pred"] = hp / len(wrong_rows)
        row[f"{name}_gold"] = hg / len(wrong_rows)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"step {step}: " + " ".join(f"{n} {row[f'{n}_pred']:.2f}/{row[f'{n}_gold']:.2f}"
          for n in ("Z", "ALL", "qonly")), flush=True)

# plot: pred (solid) vs gold (dashed) per channel over log step
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = sorted((json.loads(l) for l in open(OUT)), key=lambda r: r["step"])
fig, ax = plt.subplots()
for name, col in (("ALL", "C0"), ("Z", "C1"), ("qonly", "C2")):
    xs = [r["step"] for r in rows]
    ax.plot(xs, [r[f"{name}_pred"] for r in rows], color=col, label=f"{name} hit CODI-pred")
    ax.plot(xs, [r[f"{name}_gold"] for r in rows], color=col, linestyle="--", label=f"{name} hit gold")
ax.set_xscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("hit rate on CODI-wrong rows (n=60)")
ax.set_title("solve-creep scan: answer-read tracks CODI's pred (read) vs gold (recompute)")
ax.legend()
fig.savefig(PNG, dpi=150, bbox_inches="tight")
print(f"PLOT -> {PNG}")
print("SCAN DONE")
