"""AO qualitative progression over training (post: Appendix F): greedy readouts of
the fd1c reader at checkpoints across training, on the SAME three exemplar GSM8k
test problems the final-checkpoint example card uses (one load-bearing, one correct
control, one CODI-wrong; picked with the identical rule from the final-checkpoint
gens, so the progression ends exactly at the published card).

Per checkpoint step in STEPS (0 = the untrained init: fresh LoRA, identity adapter)
and per problem: the two joint thought reads (ALL = h at L9/18/27/35/36 + z,
thoughts 1-6 | z only, thoughts 1-6) x the two label prompts (final answer / list
intermediate values). Steps run in ascending order so the fresh-init generations
happen before any checkpoint overwrites the adapter.

-> /workspace-vast/jbauer/bt1/fd1/dur/fd1_qual_scan.jsonl (resumable by
   (step, cell, read, q))

srun -p general --qos=high --gres=gpu:1 --mem=96G --time=1:00:00 \
  --job-name=fd1qs_jb bash scripts/slurm/run_fd1_qual_scan.sh
"""
import functools
import json
import os
import re
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "scripts", "bt1"))
os.environ.setdefault("FD1_DATA", "/workspace-vast/jbauer/bt1/fd1/data")
os.environ.setdefault("HF_HOME", f"/workspace-vast/{os.environ.get('USER', 'jbauer')}/hf_cache")
print = functools.partial(print, flush=True)

BASE = "Qwen/Qwen3-8B"
MARKER_ID, INJ_SCALE = 149705, 150.0
ANSQ = "What is the final answer of the computation?"
VALQ = "List every intermediate result computed (any order)."
DUR = "/workspace-vast/jbauer/bt1/fd1/dur"
NZ2 = "/workspace-vast/jbauer/bt1/nz2"
GENS = "/workspace-vast/jbauer/bt1/fd1/lbctl_gens.jsonl"
OUT = f"{DUR}/fd1_qual_scan.jsonl"
STEPS = [0, 1000, 8000, 66000]
DEV = "cuda"


def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"


def nums_in(t):
    out = set()
    for mt in re.findall(r"-?\d[\d,]*(?:\.\d+)?", t):
        try: out.add(canon(mt))
        except (ValueError, OverflowError): pass
    return out


def pick(gens, caps):
    """One idx per cell exemplifying the aggregate finding (verbatim from the
    example-card figure so the progression covers the same problems)."""
    out = {}
    for idx, m in caps["lb"].items():
        a, q = gens.get(("lb", "ALL_joint", "ans", idx), ""), gens.get(("lb", "qonly", "ans", idx), "")
        g = canon(m["answer"])
        if g in nums_in(a) and g not in nums_in(q):
            out["lb"] = idx; break
    for idx, m in caps["ctl"].items():
        a, q = gens.get(("ctl", "ALL_joint", "ans", idx), ""), gens.get(("ctl", "qonly", "ans", idx), "")
        g = canon(m["answer"])
        if g in nums_in(a) and g in nums_in(q):
            out["ctl"] = idx; break
    for idx, m in caps["wrong"].items():
        a, q = gens.get(("wrong", "ALL_joint", "ans", idx), ""), gens.get(("wrong", "qonly", "ans", idx), "")
        p, g = canon(m["pred"]), canon(m["answer"])
        if p != g and p in nums_in(a) and p not in nums_in(q):
            out["wrong"] = idx; break
    assert set(out) == {"lb", "ctl", "wrong"}, out
    return out


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fd1_data import Pool, render_acts, PREAMBLE, SRC

    gens = {}
    for l in open(GENS):
        r = json.loads(l)
        gens[(r["cell"], r["read"], r["q"], r["idx"])] = r["text"].strip()
    caps = {}
    for cell, fn in (("lb", "lb41"), ("ctl", "ctl41"), ("wrong", "wrong41")):
        caps[cell] = {r["idx"]: r for r in map(json.loads, open(f"{NZ2}/{fn}_capture.jsonl"))}
    picked = pick(gens, caps)
    print(f"picked {picked}")

    tok = AutoTokenizer.from_pretrained(BASE)
    assert tok.decode([MARKER_ID]) == "㈎"
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV)
    model = get_peft_model(model, LoraConfig(
        r=128, lora_alpha=256, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.eval()
    embed = model.get_input_embeddings()

    gev = Pool("gsmeval")
    idx2row = {m["idx"]: i for i, m in enumerate(gev.meta)}

    done = set()
    if os.path.exists(OUT):
        done = {(r["step"], r["cell"], r["read"], r["q"]) for r in map(json.loads, open(OUT))}
        print(f"resume: {len(done)} already done")

    READS = [("ALL_joint", list(SRC)), ("Z_joint", ["Z"])]

    @torch.no_grad()
    def gen(prompt, vecs, max_new):
        txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = torch.tensor([tok(txt, add_special_tokens=False).input_ids], device=DEV)
        E = embed(ids)
        mask = ids == MARKER_ID
        assert int(mask.sum()) == len(vecs)
        V = torch.from_numpy(np.stack(list(vecs))).to(DEV)
        E[mask] = (V / V.norm(dim=1, keepdim=True) * INJ_SCALE).to(E.dtype)
        out = model.generate(inputs_embeds=E, attention_mask=torch.ones_like(ids),
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0], skip_special_tokens=True)

    for step in sorted(STEPS):                     # ascending: fresh init runs first
        if step > 0:
            sd = torch.load(f"{DUR}/fd1c_step{step}_lora.pt", map_location="cpu", weights_only=True)
            missing = model.load_state_dict(sd, strict=False)
            assert not missing.unexpected_keys, missing.unexpected_keys[:3]
        for cell, idx in picked.items():
            gi = idx2row[idx]
            for rname, S in READS:
                for qname, q, max_new in (("ans", ANSQ, 16), ("vals", VALQ, 48)):
                    if (step, cell, rname, qname) in done:
                        continue
                    acts, vecs = render_acts(gev, gi, S, "thoughts", list(range(6)))
                    text = gen(PREAMBLE + acts + f"\n\nQuestion: {q}", vecs, max_new)
                    with open(OUT, "a") as f:
                        f.write(json.dumps(dict(step=step, cell=cell, read=rname,
                                                q=qname, idx=idx, text=text)) + "\n")
                    print(f"step {step:>5} {cell:>5} {rname:>9} {qname:>4}: {text!r}")
    print("FD1-QUALSCAN-DONE")


if __name__ == "__main__":
    main()
