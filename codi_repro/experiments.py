import json
import re

import numpy as np
import torch
from tqdm.auto import tqdm

from .analysis import ROOT, equal, last_number, read, write
from .models import load_organism, load_reader


@torch.no_grad()
def rollout(model, questions, zero=False, donor=None, active=(), capture=False):
    """Incremental thought transplantation; h24 is captured after block 24.

    The twin path uses question + bot, as in the original twin campaign. The
    capture path uses raw question + eos + bot and explicit masks, as in FD1.
    Neither path renormalizes the model's post-final-norm hidden state.
    """
    device = next(model.parameters()).device
    tok = model.tokenizer
    batch = tok(questions, return_tensors='pt', padding='longest', padding_side='left').to(device)
    b = len(questions)
    delimiters = [tok.eos_token_id, tok.bot_id] if capture else [tok.bot_id]
    ids = torch.cat([batch.input_ids, torch.tensor([delimiters] * b, device=device)], 1)
    mask = torch.cat([batch.attention_mask, torch.ones(b, len(delimiters), dtype=torch.long, device=device)], 1)
    out = model.codi(input_ids=ids, attention_mask=mask, use_cache=True, output_hidden_states=True)
    zs, hs = [], []
    for t in range(0 if zero else 6):
        z = model.prj(out.hidden_states[-1][:, -1:]).to(model.codi.dtype)
        if t in active: z = donor[:, t:t+1].to(device, dtype=model.codi.dtype)
        mask = torch.cat([mask, torch.ones(b, 1, dtype=mask.dtype, device=device)], 1)
        kwargs = {'attention_mask': mask} if capture else {}
        out = model.codi(inputs_embeds=z, past_key_values=out.past_key_values, use_cache=True, output_hidden_states=True, **kwargs)
        zs.append(z[:, 0].float().cpu())
        hs.append(out.hidden_states[24][:, -1].float().cpu())
    embed = model.codi.get_input_embeddings()
    current = embed(torch.full((b, 1), tok.eot_id, device=device, dtype=torch.long))
    finished = torch.zeros(b, dtype=torch.bool, device=device)
    sequences = [[] for _ in range(b)]
    for _ in range(24 if capture else 64):
        mask = torch.cat([mask, torch.ones(b, 1, dtype=mask.dtype, device=device)], 1)
        kwargs = {'attention_mask': mask} if capture else {}
        out = model.codi(inputs_embeds=current, past_key_values=out.past_key_values, use_cache=True, **kwargs)
        ceiling = len(tok) if capture else model.codi.config.vocab_size - 1
        nxt = out.logits[:, -1, :ceiling].argmax(-1)
        for j in range(b):
            if not finished[j]:
                token = nxt[j].item()
                if token == tok.eos_token_id: finished[j] = True
                else: sequences[j].append(token)
        if finished.all(): break
        current = embed(nxt[:, None])
    texts = [tok.decode(s, skip_special_tokens=capture) for s in sequences]
    return texts, zs, hs


def twin_run(args):
    model = load_organism(args)
    pairs = read('artifacts/twins.json')
    if args.limit: pairs = pairs[:args.limit]
    results = []
    suffix = ' Output only the answer and nothing else.'
    question = lambda p, k: p[k].strip().replace('  ', ' ') + suffix
    for start in tqdm(range(0, len(pairs), args.batch_size), desc='Twin batches'):
        batch = pairs[start:start + args.batch_size]
        orig = [question(p, 'question') for p in batch]
        twin = [question(p, 'twin_question') for p in batch]
        po, zo, _ = rollout(model, orig)
        pt, zt, _ = rollout(model, twin)
        no, _, _ = rollout(model, orig, zero=True)
        nt, _, _ = rollout(model, twin, zero=True)
        donor, own = torch.stack(zo, 1), torch.stack(zt, 1)
        conditions = {'identity': pt}
        conditions['repatchz_p3'], _, _ = rollout(model, twin, donor=own, active=[2])
        conditions['revz_all6'], _, _ = rollout(model, twin, donor=donor, active=range(6))
        for t in tqdm(range(6), desc='Single positions', leave=False):
            conditions[f'revz_p{t+1}'], _, _ = rollout(model, twin, donor=donor, active=[t])
        for j, pair in enumerate(batch):
            gate = equal(last_number(po[j]), pair['gold']) and equal(last_number(pt[j]), pair['cf_twin']) and not equal(last_number(no[j]), pair['gold']) and not equal(last_number(nt[j]), pair['cf_twin'])
            results.append({**pair, 'gate_passed': gate, 'gating_outputs': {'original_six': po[j], 'twin_six': pt[j], 'original_zero': no[j], 'twin_zero': nt[j]}, 'conds': {c: {'text': texts[j], 'pred': last_number(texts[j])} for c, texts in conditions.items()}})
        write(args.out / 'twins.json', results)
    # Save complete outputs before failing on a changed gating population.
    assert all(r['gate_passed'] for r in results), 'Some historical pairs no longer satisfy gating; inspect full outputs. Do not silently redefine the population.'
    assert all(r['conds']['identity']['pred'] == r['conds']['repatchz_p3']['pred'] for r in results), 'Own-vector repatching changed an answer.'


@torch.no_grad()
def reader_run(args):
    model, tok = load_reader(args)
    payload = np.load(ROOT / f'artifacts/vectors_{args.stream}.npz')
    uids, vectors = payload['uids'].tolist(), payload['vectors']
    n = len(uids)
    marker = 149705
    messages = [{'role': r['role'], 'content': r['content'].replace('<INJECT>', tok.decode([marker]))} for r in read(f'artifacts/prompt_{args.stream}.json')]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pids = tok(text, add_special_tokens=False).input_ids
    assert pids.count(marker) == 1
    state = {}
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:
            v = state['vectors']
            hv = h[:, pids.index(marker)]
            h[:, pids.index(marker)] = hv + hv.norm(dim=-1, keepdim=True) * (v / v.norm(dim=-1, keepdim=True))
        return out
    handle = model.get_base_model().model.layers[1].register_forward_hook(hook)
    old = {(r['uid'], r['t'], r['kind']): r for r in read('artifacts/readouts.json') if r['src'] == args.stream}
    rows = []
    count = min(args.limit, n) if args.limit else n
    for t in tqdm(range(6), desc='Thought positions'):
        for kind in ['own', 'donor']:
            for start in range(0, count, args.batch_size):
                idx = np.arange(start, min(start + args.batch_size, count))
                donors = idx if kind == 'own' else (idx + 7) % n
                state['vectors'] = torch.tensor(vectors[donors, t], dtype=torch.bfloat16, device='cuda')
                ids = torch.tensor([pids] * len(idx), device='cuda')
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
                for j, i in enumerate(idx):
                    gen = tok.decode(output[j, len(pids):], skip_special_tokens=True)
                    match = re.search(r'<explanation>(.*?)(</explanation>|$)', gen, re.S)
                    body = match.group(1).strip() if match else gen.strip()
                    rows.append({**old[(uids[i], t, kind)], 'gen': body})
                write(args.out / 'readouts.json', rows)
    handle.remove()


def capture_run(args):
    model = load_organism(args)
    payload = np.load(ROOT / f'artifacts/vectors_{args.stream}.npz')
    uids = payload['uids'].tolist()
    if args.limit: uids = uids[:args.limit]
    problems = {r['uid']: r for r in read('artifacts/problems.json')}
    zs, hs, records = [], [], []
    for start in tqdm(range(0, len(uids), args.batch_size), desc='Capture batches'):
        batch = uids[start:start + args.batch_size]
        texts, z, h = rollout(model, [problems[u]['question'] for u in batch], capture=True)
        # Historical extraction stored fp16 captures before converting to fp32 parquet.
        zs.extend(torch.stack(z, 1).numpy().astype(np.float16).astype(np.float32))
        hs.extend(torch.stack(h, 1).numpy().astype(np.float16).astype(np.float32))
        records.extend({**problems[u], 'text': text} for u, text in zip(batch, texts))
        write(args.out / 'capture_records.json', records)
    vectors = np.stack(zs if args.stream == 'z' else hs)
    np.savez_compressed(args.out / f'vectors_{args.stream}.npz', uids=np.array(uids), vectors=vectors)
    write(args.out / 'capture_comparison.json', {'stream': args.stream, 'n': len(uids), 'mean_absolute_error_vs_bundled': float(np.abs(vectors - payload['vectors'][:len(uids)]).mean()), 'note': 'Batch sizes and attention kernels can change bf16 generations and extracted vectors.'})


def run(args):
    if args.batch_size is None: args.batch_size = read('manifest.json')['protocols'][args.experiment]['batch_size']
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / 'run_config.json', {'experiment': args.experiment, 'organism': read('manifest.json')['organism'], 'reader_step': 3200, 'stream': args.stream, 'batch_size': args.batch_size, 'limit': args.limit, 'torch': torch.__version__})
    if args.experiment == 'controls':
        from .controls import main
        main(args)
    elif args.experiment == 'twins': twin_run(args)
    elif args.experiment == 'readouts': reader_run(args)
    else:
        assert args.stream is not None, 'Capture requires --stream z or h24.'
        capture_run(args)
