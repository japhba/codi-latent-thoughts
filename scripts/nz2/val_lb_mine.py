"""LB labels for VAL docs (pod, 1 GPU): subject CODI answer with 6 thoughts vs
0 thoughts (bocot -> eot immediately, latent loop skipped). LB = correct with
thoughts AND wrong without. Prints the train-dist LB base rate.
-> /root/lbeval/val_lb.jsonl  (doc_id, pred6, pred0, correct6, correct0, lb)
"""
import json, os, re, sys

sys.path.insert(0, "/root/lbeval/CODI")
os.environ.setdefault("HF_HOME", "/root/nz2/hf_cache")
import torch
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
from src.model import CODI

DOCS = "/root/lbeval/val_docs.jsonl"
OUT = "/root/lbeval/val_lb.jsonl"

def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"

ckpt = snapshot_download("cds-jb/codi_qwen3-8b-answer_only")
model = CODI.from_pretrained(checkpoint_path=ckpt, model_name_or_path="Qwen/Qwen3-8B",
                             lora_r=128, lora_alpha=32, num_latent=6, use_prj=True,
                             prj_dim=4096, prj_no_ln=False, prj_dropout=0.0, dtype="bfloat16")
model.eval().to("cuda")
tok = model.tokenizer
if tok.pad_token_id is None:
    tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")
bot = tok.convert_tokens_to_ids("<|bocot|>")
eot = tok.convert_tokens_to_ids("<|eocot|>")
eos = tok.eos_token_id
embd = model.get_embd(model.codi, model.model_name)

@torch.no_grad()
def rollout(questions, n_thoughts, ans_max=24):
    B = len(questions)
    b = tok(questions, return_tensors="pt", padding="longest", padding_side="left")
    ids = torch.cat([b["input_ids"], torch.tensor([[eos, bot]]).expand(B, 2)], 1).to("cuda")
    att = torch.cat([b["attention_mask"], torch.ones(B, 2, dtype=torch.long)], 1).to("cuda")
    o = model.codi(input_ids=ids, attention_mask=att, use_cache=True, output_hidden_states=True)
    pkv, latent = o.past_key_values, o.hidden_states[-1][:, -1:, :]
    for _t in range(n_thoughts):
        z = model.prj(latent).to(latent.dtype)
        att = torch.cat([att, torch.ones(B, 1, dtype=torch.long, device="cuda")], 1)
        o = model.codi(inputs_embeds=z, use_cache=True, past_key_values=pkv,
                       attention_mask=att, output_hidden_states=True)
        pkv, latent = o.past_key_values, o.hidden_states[-1][:, -1:, :]
    step = embd(torch.tensor([[eot]], device="cuda")).expand(B, 1, -1)
    preds = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device="cuda")
    for _t in range(ans_max):
        att = torch.cat([att, torch.ones(B, 1, dtype=torch.long, device="cuda")], 1)
        o = model.codi(inputs_embeds=step, use_cache=True, past_key_values=pkv, attention_mask=att)
        pkv = o.past_key_values
        nxt = o.logits[:, -1, :len(tok)].argmax(-1)
        for bi in range(B):
            if not done[bi]:
                preds[bi].append(nxt[bi].item())
                done[bi] |= nxt[bi].eq(eos).item()
        if done.all(): break
        step = embd(nxt.unsqueeze(1))
    texts = [tok.decode(p, skip_special_tokens=True).strip() for p in preds]
    return texts

def last_num(t):
    m = re.findall(r"-?[\d,]+(?:\.\d+)?", t)
    return canon(m[-1]) if m else ""

docs = [json.loads(l) for l in open(DOCS)]
print(f"{len(docs)} docs", flush=True)
BS = 48
out = open(OUT, "w")
n6 = n0 = nlb = 0
for s0 in tqdm(range(0, len(docs), BS)):
    chunk = docs[s0:s0 + BS]
    qs = [d["question"] for d in chunk]
    t6 = rollout(qs, 6)
    t0 = rollout(qs, 0)
    for d, a6, a0 in zip(chunk, t6, t0):
        p6, p0 = last_num(a6), last_num(a0)
        c6, c0 = int(p6 == canon(d["answer"])), int(p0 == canon(d["answer"]))
        lb = int(c6 and not c0)
        n6 += c6; n0 += c0; nlb += lb
        out.write(json.dumps(dict(doc_id=d["doc_id"], pred6=p6, pred0=p0,
                                  correct6=c6, correct0=c0, lb=lb)) + "\n")
    out.flush()
out.close()
N = len(docs)
print(f"correct with 6 thoughts: {n6}/{N} = {n6/N:.3f}")
print(f"correct with 0 thoughts: {n0}/{N} = {n0/N:.3f}")
print(f"LOAD-BEARING (c6 & !c0): {nlb}/{N} = {nlb/N:.3f}")
print("MINE DONE")
