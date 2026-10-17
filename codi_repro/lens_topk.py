"""Top-k LogitLens / J-lens readouts at every layer and thought position, for a chance-controlled readability test.

Recaptures the 156 reader evaluation questions exactly as all_layers.capture does (raw question + EOS + BOT,
batch size 1, saved h24/z vectors swapped in), then stores the top-K token ids of both lenses for all 37 hidden
states and six thought positions. The top-5 lists are checked against the saved all-layer readouts.
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .all_layers import cohorts, record
from .analysis import ROOT, read, write
from .curve_lenses import METHODS


def main(args):
    import torch
    from .lenses import decode, transport
    from .models import load_organism
    torch.set_float32_matmul_precision('highest')
    model = load_organism(args)
    _, _, evaluation = cohorts()
    saved = read('artifacts/all_layer_lens_readouts.json')
    assert [p['uid'] for p in evaluation] == saved['uids']
    acts = np.stack([record(model, p['question'])[0] for p in tqdm(evaluation, desc='Capture')])[:, 1:]  # [n, 6, 37, d]
    uids = saved['uids']
    for src, hs in [('h24', 24), ('z', 0)]:
        old = np.load(ROOT / f'artifacts/vectors_{src}.npz')
        for uid, vec in zip(old['uids'], old['vectors']):
            acts[uids.index(uid), :, hs] = vec
    state = torch.load(args.organism_dir / 'pytorch_model.bin', mmap=True, map_location='cpu', weights_only=True)
    head = state['codi.base_model.model.lm_head.weight'].float().to('cuda')
    norm = state['codi.base_model.model.model.norm.weight'].float().to('cuda')
    lens = torch.load(args.jacobians, mmap=True, map_location='cpu', weights_only=True)
    n = len(uids)
    top = np.empty((len(METHODS), 37, n, 6, args.k), dtype=np.int32)
    for hs in tqdm(range(37), desc='Top-k readouts'):
        jacobian = None if hs == 36 else (lens['J_emb'] if hs == 0 else lens['J'][hs - 1]).to('cuda')
        h = torch.tensor(acts[:, :, hs].reshape(-1, acts.shape[-1]), dtype=torch.float32, device='cuda')
        for m, method in enumerate(METHODS):
            x = h if method == 'LogitLens' or hs == 36 else transport(h, jacobian)
            top[m, hs] = decode(x, head, norm, post_norm=hs == 36).topk(args.k, -1).indices.reshape(n, 6, args.k).cpu().numpy()
    pairs = [(top[m, r['hidden_state'], i, r['thought'] - 1, :5], r['top_ids']) for m, method in enumerate(METHODS)
             for i, uid in enumerate(uids) for r in saved['methods'][method][uid]]
    top1 = float(np.mean([new[0] == old[0] for new, old in pairs]))
    overlap = float(np.mean([len(set(new.tolist()) & set(old)) / 5 for new, old in pairs]))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / 'lens_topk.npz', top_ids=top, uids=np.array(uids), methods=np.array(METHODS))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.organism_dir)
    write(args.out / 'lens_topk_tokens.json', {str(i): tok.decode([i]) for i in sorted({int(i) for i in np.unique(top)})})
    print(f'agreement with the saved top-5 readouts: top-1 {top1:.4f}, top-5 overlap {overlap:.4f}', flush=True)
    assert top1 > .9 and overlap > .9, 'Recaptured readouts disagree with the saved all-layer readouts.'
    write(args.out / 'lens_topk.json', {'k': args.k, 'top1_agreement_with_saved': top1, 'top5_overlap_with_saved': overlap, 'n': n, 'prompt': 'raw question + EOS + BOT'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--organism-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--jacobians', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--k', type=int, default=100)
    main(parser.parse_args())
