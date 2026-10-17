"""Fixed-checkpoint LogitLens baselines for the exact NLA evaluation cohorts."""
import torch
import gc
import numpy as np
from tqdm.auto import tqdm

from .analysis import ROOT, read, write
from .av_analysis import select_lens
from .lenses import decode
from .models import load_organism, snapshot


@torch.no_grad()
def run(args):
    selection = select_lens()
    assert (selection['hidden_state'], selection['thought']) == (36, 1)
    problems = read('artifacts/problems.json')
    uids = [p['uid'] for p in problems]
    model = load_organism(args)
    tok = model.tokenizer
    best, comparisons = [], []
    for start in tqdm(range(0, len(problems), args.batch_size), desc='Matched-cohort first thoughts'):
        batch = problems[start:start + args.batch_size]
        prompt = tok([p['question'] for p in batch], return_tensors='pt', padding='longest', padding_side='left').to('cuda')
        ids = torch.cat([prompt.input_ids, torch.tensor([[tok.eos_token_id, tok.bot_id]] * len(batch), device='cuda')], 1)
        mask = torch.cat([prompt.attention_mask, torch.ones(len(batch), 2, dtype=torch.long, device='cuda')], 1)
        out = model.codi(input_ids=ids, attention_mask=mask, use_cache=True, output_hidden_states=True)
        z = model.prj(out.hidden_states[-1][:, -1:]).to(model.codi.dtype)
        mask = torch.cat([mask, torch.ones(len(batch), 1, dtype=mask.dtype, device='cuda')], 1)
        out = model.codi(inputs_embeds=z, past_key_values=out.past_key_values, attention_mask=mask, use_cache=True, output_hidden_states=True)
        best.extend(out.hidden_states[-1][:, -1].float().cpu().numpy().astype(np.float16))
        comparisons.extend(torch.stack([z[:, 0], out.hidden_states[24][:, -1]], 1).float().cpu().numpy().astype(np.float16))
    checkpoint = args.organism_dir if args.organism_dir else snapshot(read('manifest.json')['organism'], args.cache_dir)
    state = torch.load(checkpoint / 'pytorch_model.bin', mmap=True, map_location='cpu', weights_only=True)
    head = state['codi.base_model.model.lm_head.weight'].float()
    norm = state['codi.base_model.model.model.norm.weight'].float()
    encoded_digits = [tok.encode(str(d), add_special_tokens=False) for d in range(10)]
    assert all(len(ids) == 1 for ids in encoded_digits)
    digit_ids = [ids[0] for ids in encoded_digits]
    del out, z, model
    gc.collect()
    torch.cuda.empty_cache()
    head, norm = head.to('cuda'), norm.to('cuda')
    torch.set_float32_matmul_precision('highest')

    def readout(vectors, post_norm):
        result = []
        for start in range(0, len(vectors), args.readout_batch_size):
            h = torch.tensor(vectors[start:start + args.readout_batch_size], dtype=torch.float32, device='cuda')
            logp = decode(h, head, norm, post_norm=post_norm).log_softmax(-1)
            probabilities, tokens = logp.topk(5, -1)
            for ids, values in zip(tokens.cpu().tolist(), probabilities.cpu().tolist()):
                result.append({'top_ids': ids, 'top_tokens': [tok.decode([i]) for i in ids], 'top_logp': values, 'top1_digit': digit_ids.index(ids[0]) if ids[0] in digit_ids else None})
        return result

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / 'av_best_lens_vectors.npz', uids=np.array(uids), vectors=np.array(best))
    best_reads = dict(zip(uids, readout(np.array(best), True)))
    comparisons = dict(zip(uids, comparisons))
    streams, audit = {}, {}
    for src, column in [('z', 0), ('h24', 1)]:
        payload = np.load(ROOT / f'artifacts/vectors_{src}.npz')
        ordered = payload['uids'].tolist()
        same_reads = readout(payload['vectors'][:, 0], False)
        original = payload['vectors'][:, 0]
        fresh = np.array([comparisons[u][column] for u in ordered]).astype(np.float32)
        cosine = (original * fresh).sum(-1) / (np.linalg.norm(original, axis=-1) * np.linalg.norm(fresh, axis=-1))
        audit[src] = {'min_cosine_vs_saved_first_vector': float(cosine.min()), 'mean_cosine_vs_saved_first_vector': float(cosine.mean()), 'mean_absolute_error': float(np.abs(original - fresh).mean())}
        streams[src] = {'uids': ordered, 'best_top1_digit': [best_reads[u]['top1_digit'] for u in ordered], 'same_input_top1_digit': [r['top1_digit'] for r in same_reads], 'best_readouts': [best_reads[u] for u in ordered], 'same_input_readouts': same_reads}
    write(args.out / 'av_lens_readouts.json', {'organism': read('manifest.json')['organism'], 'selection': selection, 'streams': streams, 'fresh_capture': {'batch_size': args.batch_size, 'prompt': 'raw question + EOS + BOT', 'cached_padding_mask': True, 'stored_dtype': 'float16', 'torch': torch.__version__, 'first_vector_comparison': audit}, 'readout': {'dtype': 'float32', 'final_hidden_state': 'head directly, no second norm', 'same_input': 'exact bundled first-thought h24/z, final norm once', 'token_digit_ids': digit_ids}})
