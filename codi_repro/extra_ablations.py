"""Masked reruns of the post's remaining whole-trace ablations, on the saved rv6 clean thoughts.

Adds the four conditions of the main-text ablation figure that recurrent_controls does not
cover: first four thoughts removed, thoughts shuffled, the next test problem's thoughts, and
moment-matched Gaussian thoughts. All are frozen replays of saved thought inputs.
"""
import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .analysis import equal, last_number, read, write
from .controlled_rollout import rollout
from .models import load_organism


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    rows = read('artifacts/controls.json')
    parts = [torch.load(p, weights_only=True) for p in sorted((args.controls / 'captures').glob('*.pt'))]
    assert [r['i'] for part in parts for r in part['rows']] == [r['i'] for r in rows]
    clean = torch.cat([part['z'] for part in parts])
    identity = [t for part in parts for t in part['texts']]
    zero = [t for part in parts for t in part['zero_texts']]
    suffix = ' Output only the answer and nothing else.'
    questions = [r['question'].strip().replace('  ', ' ') + suffix for r in rows]
    model = load_organism(args)

    # same moment-matched noise and donor definitions as recurrent_controls (seed 11)
    mean, std = clean.mean(0), clean.std(0)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(11)) * std[None] + mean[None]
    g = torch.Generator().manual_seed(11)
    perms = []
    for _ in range(len(rows)):
        p = torch.randperm(6, generator=g)
        while (p == torch.arange(6)).all(): p = torch.randperm(6, generator=g)
        perms.append(p)
    shuffled = torch.stack([z[p] for z, p in zip(clean, perms)])
    conditions = {'dropfirst_4': (2, clean[:, 4:]), 'shuffle': (6, shuffled), 'swap_neighbor': (6, clean.roll(-1, 0)), 'gauss': (6, noise)}

    # own-vector replay must reproduce the saved six-thought answers
    check = rollout(model, questions[:args.batch_size], frozen=clean[:args.batch_size])
    assert check['texts'] == identity[:args.batch_size], 'Own-vector replay differs from the saved rv6 decodes.'

    eligible = [k for k, (r, a, b) in enumerate(zip(rows, identity, zero)) if equal(last_number(a), r['gold']) and not equal(last_number(b), r['gold'])]
    scores, predictions = [], {}
    for name, (thoughts, frozen) in tqdm(conditions.items(), desc='Conditions'):
        texts = []
        for start in tqdm(range(0, len(rows), args.batch_size), desc=name, leave=False):
            texts += rollout(model, questions[start:start + args.batch_size], thoughts=thoughts, frozen=frozen[start:start + args.batch_size])['texts']
        predictions[name] = {'texts': texts, 'predictions': [last_number(t) for t in texts]}
        lb = sum(equal(last_number(texts[k]), rows[k]['gold']) for k in eligible)
        scores.append({'condition': name, 'correct': sum(equal(last_number(t), r['gold']) for t, r in zip(texts, rows)), 'n': len(rows), 'load_bearing_correct': lb, 'load_bearing_n': len(eligible)})
        print(json.dumps(scores[-1]), flush=True)
        write(args.out / 'extra_scores.json', scores)
        write(args.out / 'extra_predictions.json', predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--organism-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    main(parser.parse_args())
