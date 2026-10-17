"""Resampled AO readouts for the Appendix D example card (fig_23): the fd1c reader at
the untrained init (step 0, fresh LoRA = identity) and the final checkpoint (step 66000)
reading the load-bearing exemplar (gsmeval idx 126) through the two joint thought reads
(ALL = L9/18/27/35/LF + Z, thoughts 1-6 | Z only, thoughts 1-6) under both label prompts
(final answer, 16 tokens / list intermediates, 48 tokens), recipe as fd1_qual_scan.py.
Per (step, read, question): one greedy readout (pipeline check: must reproduce
artifacts/fd1_qual_scan.jsonl) plus --n samples at T = 1 (top_p 1, no top_k).

The gsmeval capture arrays are not in the local backup (R2 only, 238 GB into a 246 GB
tar stream), so the problem's z / h band are RECAPTURED here with the CODI organism
(checkpoint-7580, codi_repro strict loader) by fd1_capture.py's rollout at batch size 1
(raw question + eos + bocot, fp16 storage); the greedy reproduction checks the capture.

  python3 fd1_qual_resample.py [--idx 126] [--n 2] [--seed 0]
Output: artifacts/fd1_qual_resample.jsonl
  {"idx", "step", "read", "q", "decode" ("greedy"|"sample"), "k", "seed", "text"}
"""
import argparse, json, os, sys, tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
ROOT = Path(os.environ.get("CODI_ROOT", "/nfs/nhome/live/jbauer/codi-latent-thoughts"))
CODI_CKPT = ROOT / "checkpoints/codi_qwen3-8b/checkpoint-7580"
AO_DIR = ROOT / "checkpoints/ao_fd1c"
META = ROOT / "experiments/bt1/fd1/data/gsmeval_meta.jsonl"
SCAN = HERE.parent / "artifacts/fd1_qual_scan.jsonl"
OUT = HERE.parent / "artifacts/fd1_qual_resample.jsonl"
BASE = "Qwen/Qwen3-8B"
BAND = [9, 18, 27, 35, 36]          # hidden_states indices, 36 = post-final-norm output
MARKER_ID, INJ_SCALE = 149705, 150.0
ANSQ = "What is the final answer of the computation?"
VALQ = "List every intermediate result computed (any order)."
QS = (("ans", ANSQ, 16), ("vals", VALQ, 48))
STEPS = (0, 66000)

ap = argparse.ArgumentParser()
ap.add_argument("--idx", type=int, default=126)
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
meta = next(m for m in map(json.loads, open(META)) if m["idx"] == args.idx)

# ---- recapture z [6,4096] and h [8 = qend, bocot, t1..t6, 5 = band, 4096] (fd1_capture.py, B = 1)
from codi_repro.organism import load
codi = load(CODI_CKPT, json.load(open(HERE.parent / "artifacts/base_config.json")))
ctok = codi.tokenizer
with torch.no_grad():
    ids = torch.tensor([ctok(meta["question"]).input_ids + [ctok.eos_token_id, ctok.convert_tokens_to_ids("<|bocot|>")]], device="cuda")
    att = torch.ones_like(ids)
    o = codi.codi(input_ids=ids, attention_mask=att, use_cache=True, output_hidden_states=True)
    H = torch.empty(8, len(BAND), 4096)
    H[0] = torch.stack([o.hidden_states[l][0, -3] for l in BAND]).float().cpu()   # qend
    H[1] = torch.stack([o.hidden_states[l][0, -1] for l in BAND]).float().cpu()   # bocot
    pkv, latent, Z = o.past_key_values, o.hidden_states[-1][:, -1:, :], []
    for t in range(6):
        z = codi.prj(latent).to(latent.dtype)
        Z.append(z[0, 0].float().cpu())
        att = torch.cat([att, torch.ones(1, 1, dtype=torch.long, device="cuda")], 1)
        o = codi.codi(inputs_embeds=z, use_cache=True, past_key_values=pkv, attention_mask=att, output_hidden_states=True)
        H[2 + t] = torch.stack([o.hidden_states[l][0, -1] for l in BAND]).float().cpu()
        pkv, latent = o.past_key_values, o.hidden_states[-1][:, -1:, :]
del codi, o, pkv, latent
torch.cuda.empty_cache()

# one-row fd1 pool so the reads go through fd1_data.render_acts unchanged
pool_dir = tempfile.mkdtemp(prefix="fd1pool_")
with open(f"{pool_dir}/one_meta.jsonl", "w") as f:
    f.write(json.dumps(meta) + "\n")
np.save(f"{pool_dir}/one_z.npy", torch.stack(Z)[None].numpy().astype(np.float16))
np.save(f"{pool_dir}/one_h.npy", H[None].numpy().astype(np.float16))
os.environ["FD1_DATA"] = pool_dir
sys.path.insert(0, str(HERE))
from fd1_data import Pool, render_acts, PREAMBLE, SRC
pool = Pool("one")
print(f"idx {args.idx}: recaptured, z norm {torch.stack(Z).norm(dim=-1).mean():.1f}", flush=True)

# ---- AO reads
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(BASE)
assert tok.decode([MARKER_ID]) == "㈎"
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
model = get_peft_model(model, LoraConfig(r=128, lora_alpha=256, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.eval()
embed = model.get_input_embeddings()


@torch.no_grad()
def gen(prompt, vecs, max_new, n, sample):
    txt = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = torch.tensor([tok(txt, add_special_tokens=False).input_ids], device="cuda")
    E = embed(ids)
    mask = ids == MARKER_ID
    assert int(mask.sum()) == len(vecs)
    V = torch.from_numpy(np.stack(list(vecs))).to("cuda")
    E[mask] = (V / V.norm(dim=1, keepdim=True) * INJ_SCALE).to(E.dtype)
    kw = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0) if sample else dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    out = model.generate(inputs_embeds=E.repeat(n, 1, 1), attention_mask=torch.ones(n, ids.shape[1], dtype=torch.long, device="cuda"),
                         max_new_tokens=max_new, pad_token_id=tok.eos_token_id, **kw)
    return [tok.decode(s, skip_special_tokens=True) for s in out]


scan = {(r["step"], r["read"], r["q"]): r["text"] for r in map(json.loads, open(SCAN)) if r["idx"] == args.idx}
READS = [("ALL_joint", list(SRC)), ("Z_joint", ["Z"])]
rows = []
for step in STEPS:                                  # ascending: the fresh init runs first
    if step:
        sd = torch.load(AO_DIR / f"fd1c_step{step}_lora.pt", map_location="cpu", weights_only=True)
        assert not model.load_state_dict(sd, strict=False).unexpected_keys
    for rname, S in READS:
        acts, vecs = render_acts(pool, 0, S, "thoughts", list(range(6)))
        for qname, q, max_new in QS:
            prompt = PREAMBLE + acts + f"\n\nQuestion: {q}"
            g = gen(prompt, vecs, max_new, 1, False)[0]
            print(f"step {step} {rname} {qname}: greedy reproduces scan: {g == scan[(step, rname, qname)]}", flush=True)
            rows.append(dict(idx=args.idx, step=step, read=rname, q=qname, decode="greedy", k=0, seed=None, text=g))
            torch.manual_seed(args.seed)
            for k, s in enumerate(gen(prompt, vecs, max_new, args.n, True), 1):
                rows.append(dict(idx=args.idx, step=step, read=rname, q=qname, decode="sample", k=k, seed=args.seed, text=s))
for r in rows:
    print(f"\n===== step {r['step']} {r['read']} {r['q']} {r['decode']} {r['k']} =====\n{r['text']}", flush=True)
with open(OUT, "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
print(f"wrote {OUT}")
