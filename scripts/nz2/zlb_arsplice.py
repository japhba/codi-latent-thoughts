"""AR-roundtrip splice test (pod, 1 GPU): text-space edit -> AR-encode -> splice
into the SUBJECT -> answer readout.

Cases: zlb_gens.jsonl (LB val docs, z strata, 32 samples/read) whose
<explanation> surfaces EXACTLY ONE true chain intermediate v (not a question
number, != gold answer). Dedup by explanation body; cap CASE_CAP distinct
bodies per (doc, stratum).

Per case:
  expl_orig  = the AV explanation body
  expl_swap  = same with every numeral occurrence of v -> v' (2v; +7/+13 on
               collision with question/chain/answer numbers)
  zhat_orig  = direction(critic_latest AR(expl_orig))  x ||z_read||
  zhat_swap  = direction(critic_latest AR(expl_swap)) x ||z_read||
Subject rollout (greedy answer, thought slot p-1 spliced as INPUT embedding,
downstream thoughts recompute naturally):
  orig       — no splice (one per doc; sanity: LB rows answer gold)
  recon_ctrl — slot <- zhat_orig   (does AR recon preserve the answer?)
  swap       — slot <- zhat_swap   (does the text edit move the answer?)
  rand       — slot <- random gaussian, matched norm (slot-sensitivity floor)
-> /root/lbeval/zlb_splice.jsonl
"""
import json, os, re, sys

sys.path.insert(0, "/root/nz2/EasyNLA")
sys.path.insert(0, "/root/lbeval/CODI")
os.environ.setdefault("HF_HOME", "/root/nz2/hf_cache")

CASE_CAP = 6
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"

def try_canon(s):
    try: return canon(s)
    except (ValueError, OverflowError): return None

def nums_in(t):
    return {c for c in (try_canon(m) for m in NUM.findall(t)) if c is not None}

if __name__ == "__main__":
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer

    docs = {d["doc_id"]: d for d in (json.loads(l) for l in open("/root/lbeval/val_docs.jsonl"))}

    # ---- cases ----
    seen_bodies = set()
    per_rs = {}
    cases = []
    n_gens = 0
    for g in (json.loads(l) for l in open("/root/lbeval/zlb_gens.jsonl")):
        n_gens += 1
        d = docs[g["doc"]]
        m = re.search(r"<explanation>(.*?)</explanation>", g["text"], re.S)
        if not m: continue
        body = m.group(1)
        key = (g["doc"], g["stratum"], body.strip())
        if key in seen_bodies: continue
        qnums = nums_in(d["question"])
        ans = canon(d["answer"])
        chain = {canon(v) for v in d["intermediates"]} - qnums - {ans}
        hits = nums_in(body) & chain
        if len(hits) != 1: continue
        rs = (g["doc"], g["stratum"])
        if per_rs.get(rs, 0) >= CASE_CAP: continue
        v = next(iter(hits))
        taboo = qnums | {canon(x) for x in d["intermediates"]} | {ans, v}
        vs = next((c for c in (canon(2 * float(v)), canon(float(v) + 7),
                               canon(float(v) + 13)) if c not in taboo), None)
        if vs is None: continue
        swapped = NUM.sub(lambda mm: vs if try_canon(mm.group()) == v else mm.group(), body)
        if swapped == body: continue
        seen_bodies.add(key)
        per_rs[rs] = per_rs.get(rs, 0) + 1
        cases.append(dict(doc=g["doc"], stratum=g["stratum"], v=v, v_swap=vs,
                          body=body, swapped=swapped))
    print(f"{n_gens} gens -> {len(cases)} distinct cases over "
          f"{len({c['doc'] for c in cases})} docs / {len(per_rs)} (doc,stratum)", flush=True)
    assert cases, "no qualifying cases"

    # ---- read vectors (norm reference) ----
    pf = pq.ParquetFile("/root/lbeval/zlb_reads.parquet")
    zvec = {}
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg).to_pylist():
            zvec[(r["doc_id"], r["stratum"])] = np.asarray(r["activation_vector"], dtype=np.float32)

    # ---- AR (critic_latest) reconstructions ----
    from nla.config import load_nla_config
    from nla.models import NLACriticModel
    from nla.utils.critic import critic_predict

    def render_critic_text(template, explanation, ctx=None):
        if "{ctx}" in template:
            return template.format(explanation=explanation, ctx="" if ctx is None else ctx)
        return template.format(explanation=explanation)

    tok = AutoTokenizer.from_pretrained("/root/nz2/warmstart/av")
    cfg = load_nla_config("/root/nz2/data/mix_rl.parquet", tok)
    print(f"mse_scale={cfg.mse_scale}", flush=True)
    critic = NLACriticModel.from_pretrained("/root/nz2/runs/nz3_plain/critic_latest",
                                            torch_dtype=torch.bfloat16).eval().to("cuda")

    @torch.no_grad()
    def ar_encode(bodies):
        preds = []
        for s0 in range(0, len(bodies), 32):
            chunk = bodies[s0:s0 + 32]
            texts = [render_critic_text(cfg.critic_prompt_template, b) for b in chunk]
            enc = tok([t for t in texts], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            p = critic_predict(critic, enc["input_ids"], enc["attention_mask"],
                               cfg.mse_scale)
            preds.append(p.cpu())
        return torch.cat(preds).float().numpy()

    pred_orig = ar_encode([c["body"] for c in cases])
    pred_swap = ar_encode([c["swapped"] for c in cases])
    # sanity: recon quality of the ORIGINAL body vs the true read vector
    cos = []
    for c, p in zip(cases, pred_orig):
        z = zvec[(c["doc"], c["stratum"])]
        cos.append(float(np.dot(p, z) / (np.linalg.norm(p) * np.linalg.norm(z))))
    print(f"AR(orig) vs true z cosine: mean {np.mean(cos):.3f} min {np.min(cos):.3f}", flush=True)
    del critic
    torch.cuda.empty_cache()

    def scaled(pred, z):
        return pred / np.linalg.norm(pred) * np.linalg.norm(z)

    rng = np.random.default_rng(0)
    jobs = []   # (case_idx | None, cond, doc, step, vec | None)
    for d in sorted({c["doc"] for c in cases}):
        jobs.append((None, "orig", d, -1, None))
    for ci, c in enumerate(cases):
        z = zvec[(c["doc"], c["stratum"])]
        step = int(c["stratum"].split("_p")[1]) - 1
        jobs.append((ci, "recon_ctrl", c["doc"], step, scaled(pred_orig[ci], z)))
        jobs.append((ci, "swap", c["doc"], step, scaled(pred_swap[ci], z)))
        jobs.append((ci, "rand", c["doc"], step,
                     scaled(rng.standard_normal(z.shape).astype(np.float32), z)))
    print(f"{len(jobs)} subject rollouts", flush=True)

    # ---- subject with splice ----
    from huggingface_hub import snapshot_download
    from src.model import CODI
    ckpt = snapshot_download("cds-jb/codi_qwen3-8b-answer_only")
    model = CODI.from_pretrained(checkpoint_path=ckpt, model_name_or_path="Qwen/Qwen3-8B",
                                 lora_r=128, lora_alpha=32, num_latent=6, use_prj=True,
                                 prj_dim=4096, prj_no_ln=False, prj_dropout=0.0, dtype="bfloat16")
    model.eval().to("cuda")
    stok = model.tokenizer
    if stok.pad_token_id is None:
        stok.add_special_tokens({"pad_token": "[PAD]"})
        stok.pad_token_id = stok.convert_tokens_to_ids("[PAD]")
    bot = stok.convert_tokens_to_ids("<|bocot|>")
    eot = stok.convert_tokens_to_ids("<|eocot|>")
    eos = stok.eos_token_id
    embd = model.get_embd(model.codi, model.model_name)

    @torch.no_grad()
    def rollout(questions, splices, ans_max=24):
        """splices: per-batch-row None or (step, np_vector[4096])."""
        B = len(questions)
        b = stok(questions, return_tensors="pt", padding="longest", padding_side="left")
        ids = torch.cat([b["input_ids"], torch.tensor([[eos, bot]]).expand(B, 2)], 1).to("cuda")
        att = torch.cat([b["attention_mask"], torch.ones(B, 2, dtype=torch.long)], 1).to("cuda")
        o = model.codi(input_ids=ids, attention_mask=att, use_cache=True, output_hidden_states=True)
        pkv, latent = o.past_key_values, o.hidden_states[-1][:, -1:, :]
        for t in range(6):
            z = model.prj(latent).to(latent.dtype)
            for bi, sp in enumerate(splices):
                if sp is not None and sp[0] == t:
                    z[bi, 0] = torch.from_numpy(sp[1]).to(z.device, z.dtype)
            att = torch.cat([att, torch.ones(B, 1, dtype=torch.long, device="cuda")], 1)
            o = model.codi(inputs_embeds=z, use_cache=True, past_key_values=pkv,
                           attention_mask=att, output_hidden_states=True)
            pkv, latent = o.past_key_values, o.hidden_states[-1][:, -1:, :]
        step_e = embd(torch.tensor([[eot]], device="cuda")).expand(B, 1, -1)
        preds = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device="cuda")
        for _t in range(ans_max):
            att = torch.cat([att, torch.ones(B, 1, dtype=torch.long, device="cuda")], 1)
            o = model.codi(inputs_embeds=step_e, use_cache=True, past_key_values=pkv,
                           attention_mask=att)
            pkv = o.past_key_values
            nxt = o.logits[:, -1, :len(stok)].argmax(-1)
            for bi in range(B):
                if not done[bi]:
                    preds[bi].append(nxt[bi].item())
                    done[bi] |= nxt[bi].eq(eos).item()
            if done.all(): break
            step_e = embd(nxt.unsqueeze(1))
        return [stok.decode(p, skip_special_tokens=True).strip() for p in preds]

    BS = 32
    out = open("/root/lbeval/zlb_splice.jsonl", "w")
    for s0 in range(0, len(jobs), BS):
        chunk = jobs[s0:s0 + BS]
        qs = [docs[j[2]]["question"] for j in chunk]
        sps = [None if j[1] == "orig" else (j[3], j[4]) for j in chunk]
        texts = rollout(qs, sps)
        for (ci, cond, d, step, _), txt in zip(chunk, texts):
            ms = NUM.findall(txt)
            pred = try_canon(ms[-1]) if ms else ""
            row = dict(cond=cond, doc=d, answer_text=txt, pred=pred,
                       gold=canon(docs[d]["answer"]))
            if ci is not None:
                c = cases[ci]
                row.update(case=ci, stratum=c["stratum"], v=c["v"], v_swap=c["v_swap"],
                           body=c["body"], swapped=c["swapped"])
            out.write(json.dumps(row) + "\n")
        out.flush()
        print(f"{min(s0 + BS, len(jobs))}/{len(jobs)}", flush=True)
    out.close()
    print("SPLICE DONE", flush=True)
