"""fd1 capture (pod): CODI rollout harvesting fed-back z AND residual band h.

  python3 fd1_capture.py --list capture_gsm.jsonl --name gsm [--limit 256]

Outputs to /root/fd1/data/:
  {name}_z.npy    [N,6,4096]  fp16 — fed-back thought vectors (same convention as nx1)
  {name}_h.npy    [N,8,5,4096] fp16 — positions [qend, bocot, t1..t6] x streams [L9,L18,L27,L35,Lfin]
                  (Lfin at thought t is the pre-prj state that spawns z_{t+1}; bocot spawns z1)
  {name}_meta.jsonl — input row + pred + correct
"""
import argparse, json, os, sys

DATA = os.environ.get("FD1_DATA", "/root/fd1/data")
LISTS = os.environ.get("FD1_LISTS", os.path.dirname(DATA.rstrip("/")) or "/root/fd1")
sys.path.insert(0, next(p for p in ("/root/nx2/CODI",
                                    "/workspace-vast/jbauer/activation_oracles_dev/third_party/CODI")
                        if os.path.isdir(p)))
BAND = [int(x) for x in os.environ.get("FD1_BAND", "9,18,27,35,36").split(",")]
# hidden_states indices; 36 = final (Qwen3-8B has 36 layers, idx0=embeddings)

def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import re
    import torch
    torch.cuda.set_device(args.device)
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm
    from src.model import CODI

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
    def rollout(questions, ans_max=24):
        B = len(questions)
        b = tok(questions, return_tensors="pt", padding="longest", padding_side="left")
        ids = torch.cat([b["input_ids"], torch.tensor([[eos, bot]]).expand(B, 2)], 1).to("cuda")
        att = torch.cat([b["attention_mask"], torch.ones(B, 2, dtype=torch.long)], 1).to("cuda")
        o = model.codi(input_ids=ids, attention_mask=att, use_cache=True, output_hidden_states=True)
        H = torch.empty(B, 8, len(BAND), 4096)
        H[:, 0] = torch.stack([o.hidden_states[l][:, -3] for l in BAND], 1).float().cpu()  # qend
        H[:, 1] = torch.stack([o.hidden_states[l][:, -1] for l in BAND], 1).float().cpu()  # bocot
        pkv, latent = o.past_key_values, o.hidden_states[-1][:, -1:, :]
        zs = []
        for t in range(6):
            z = model.prj(latent).to(latent.dtype)
            zs.append(z[:, 0].float().cpu())
            att = torch.cat([att, torch.ones(B, 1, dtype=torch.long, device="cuda")], 1)
            o = model.codi(inputs_embeds=z, use_cache=True, past_key_values=pkv,
                           attention_mask=att, output_hidden_states=True)
            H[:, 2 + t] = torch.stack([o.hidden_states[l][:, -1] for l in BAND], 1).float().cpu()
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
        return torch.stack(zs, 1), H, texts

    rows = [json.loads(l) for l in open(f"{LISTS}/{args.list}")]
    if args.nshards > 1: rows = rows[args.shard::args.nshards]
    if args.limit: rows = rows[:args.limit]
    os.makedirs(DATA, exist_ok=True)
    N = len(rows)
    Z = np.empty((N, 6, 4096), dtype=np.float16)
    Hh = np.empty((N, 8, len(BAND), 4096), dtype=np.float16)
    fout = open(f"{DATA}/{args.name}_meta.jsonl", "w")
    ncorr = 0
    for s0 in tqdm(range(0, N, args.bs), desc=f"capture:{args.name}"):
        chunk = rows[s0:s0 + args.bs]
        z, H, texts = rollout([r["question"] for r in chunk])
        Z[s0:s0 + len(chunk)] = z.numpy().astype(np.float16)
        Hh[s0:s0 + len(chunk)] = H.numpy().astype(np.float16)
        for r, p in zip(chunk, texts):
            m = re.findall(r"-?[\d,]+(?:\.\d+)?", p)
            pc = canon(m[-1]) if m else ""
            corr = int(pc == r["answer"]); ncorr += corr
            fout.write(json.dumps({**r, "pred": pc, "correct": corr}) + "\n")
        fout.flush()
    fout.close()
    np.save(f"{DATA}/{args.name}_z.npy", Z)
    np.save(f"{DATA}/{args.name}_h.npy", Hh)
    print(f"{args.name}: {N} rows, codi-correct {ncorr/N:.3f}, "
          f"z norm {np.linalg.norm(Z[0].astype(np.float32), axis=-1).mean():.1f}, "
          f"h norms per stream {[f'{np.linalg.norm(Hh[0,2,i].astype(np.float32)):.0f}' for i in range(len(BAND))]}")
    print(f"CAPTURE DONE {args.name}")

if __name__ == "__main__":
    main()
