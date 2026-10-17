"""Reader learning curves with a shared, literal answer-leading-digit diagnostic."""
import re
import gzip
import json

import matplotlib.pyplot as plt
import numpy as np

from .analysis import ROOT, canonical_number, read, write


def leading_digit(text):
    match = re.search(r'\d', str(text))
    return int(match.group()) if match else None


def first_number_digit(text):
    # Score one literal mention, rather than rewarding every number in a long read.
    match = re.search(r'-?\d[\d,]*(?:\.\d+)?', text)
    return leading_digit(match.group()) if match else None


def select_lens(root=ROOT):
    problems = read('artifacts/lens_problems.json', root)
    heldout = read('artifacts/problems.json', root)
    assert min(p['idx'] for p in heldout) >= 5000
    mask = np.array([p['ds_index'] < 5000 and p['correct'] for p in problems])
    data = np.load(root / 'artifacts/lens_readouts.npz')
    target = data['digit_ids'][[leading_digit(p['gold']) for p in problems]]
    # One fixed layer/position is chosen globally, with every reader-eval row excluded.
    hits = (data['top_ids'][0, :, :, 1:, 0] == target[None, :, None])[:, mask].mean(1)
    h, t = np.unravel_index(hits.argmax(), hits.shape)
    return {'hidden_state': int(h), 'thought': int(t + 1), 'selection_n': int(mask.sum()), 'selection_indices': [4000, 4999], 'selection_rate': float(hits[h, t]), 'criterion': 'full-vocabulary top-1 answer-leading-digit match', 'candidate_rates': hits.tolist(), 'selection_prompt': 'answer-only question + EOS + BOT', 'evaluation_prompt': 'raw question + EOS + BOT'}


def summarize(root=ROOT):
    with gzip.open(root / 'artifacts/av_history.json.gz', 'rt') as handle: rows = json.load(handle)
    problems = {p['uid']: p for p in read('artifacts/problems.json', root)}
    lens = read('artifacts/av_lens_readouts.json', root)
    final_reads = {(r['src'], r['uid']): r['gen'] for r in read('artifacts/readouts.json', root) if r['t'] == 0 and r['kind'] == 'own'}
    selection = select_lens(root)
    assert lens['selection'] == selection
    assert lens['organism'] == read('manifest.json', root)['organism']
    assert len({(r['src'], r['step'], r['uid']) for r in rows}) == len(rows) == 96120
    for row in rows:
        problem = problems[row['uid']]
        values = {canonical_number(v) for v in re.findall(r'-?\d[\d,]*(?:\.\d+)?', row['gen'])}
        assert row['answer_mentioned'] == (canonical_number(problem['answer']) in values)
        eligible = set(map(canonical_number, problem['intermediates'])) - {canonical_number(problem['answer'])}
        assert (row['intermediate_matched'] is not None) == bool(eligible)
        if row['step'] == 3200: assert row['gen'] == final_reads[(row['src'], row['uid'])]
    out = {'metric': 'First numeric literal in the NLA description starts with the gold answer digit; LogitLens top-1 token is that digit.', 'selection': selection, 'streams': {}}
    for src in ['z', 'h24']:
        vectors = np.load(root / f'artifacts/vectors_{src}.npz')
        uids = vectors['uids'].tolist()
        target = np.array([leading_digit(problems[u]['answer']) for u in uids])
        baseline = lens['streams'][src]
        assert baseline['uids'] == uids
        for kind in ['best', 'same_input']:
            for row, digit in zip(baseline[f'{kind}_readouts'], baseline[f'{kind}_top1_digit']):
                tokens = lens['readout']['token_digit_ids']
                predicted = tokens.index(row['top_ids'][0]) if row['top_ids'][0] in tokens else None
                assert predicted == digit == row['top1_digit']
        steps = sorted({r['step'] for r in rows if r['src'] == src})
        rates, complete, value_match = [], [], []
        for step in steps:
            batch = [r for r in rows if (r['src'], r['step']) == (src, step)]
            assert [r['uid'] for r in batch] == uids
            rates.append(float(np.mean([first_number_digit(r['gen']) == y for r, y in zip(batch, target)])))
            complete.append(float(np.mean([r['answer_mentioned'] for r in batch])))
            eligible = [r['intermediate_matched'] for r in batch if r['intermediate_matched'] is not None]
            value_match.append(float(np.mean(eligible)))
        counts = np.bincount(target, minlength=10)
        out['streams'][src] = {'n': len(uids), 'intermediate_n': len(eligible), 'uids': uids, 'steps': steps, 'first_number_leading_digit': rates, 'complete_answer_mentioned': complete, 'judged_intermediate_mentioned': value_match, 'best_lens_count': int(sum(p == y for p, y in zip(baseline['best_top1_digit'], target))), 'same_input_lens_count': int(sum(p == y for p, y in zip(baseline['same_input_top1_digit'], target))), 'majority_digit_rate': float(counts.max() / len(target)), 'label_shuffle_rate': float(sum(counts[d] for d in baseline['best_top1_digit'] if d is not None) / len(target) ** 2)}
    return out


def ema(values, halflife=25):
    alpha = 1 - .5 ** (1 / halflife)
    state, weight, result = 0., 0., []
    for value in values:
        state = (1 - alpha) * state + alpha * value
        weight = (1 - alpha) * weight + alpha
        result.append(state / weight)
    return result


def reproduce(out, root=ROOT):
    summary = summarize(root)
    write(out / 'av_summary.json', summary)
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    for ax, src in zip(axes, ['h24', 'z']):
        data = summary['streams'][src]
        ax.plot(data['steps'], data['first_number_leading_digit'], color='C0', alpha=.2)
        ax.plot(data['steps'], ema(data['first_number_leading_digit']), color='C0', label='NLA: first number')
        ax.axhline(data['best_lens_count'] / data['n'], color='C2', linestyle='--', label='Best LogitLens: h36, t1')
        ax.axhline(data['same_input_lens_count'] / data['n'], color='C3', linestyle=':', label='LogitLens: same input')
        ax.axhline(data['majority_digit_rate'], color='C4', linestyle='-.', label='Empirical majority digit')
        ax.set(title=(r'$\mathbf{h}^{(24)}$' if src == 'h24' else r'$\mathbf{z}$') + f' reader: {data["n"]} held-out problems', ylabel='Answer-digit match', ylim=(-.05, 1.05))
    axes[0].legend(loc='center left')
    axes[-1].set(xlabel='Reader training step')
    fig.tight_layout()
    fig.savefig(out / 'nla_digit_training.png')
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    for ax, src in zip(axes, ['h24', 'z']):
        data = summary['streams'][src]
        for color, (key, label) in enumerate([('complete_answer_mentioned', 'Complete answer mentioned'), ('judged_intermediate_mentioned', 'Any intermediate (judge)')]):
            ax.plot(data['steps'], data[key], color=f'C{color}', alpha=.2)
            ax.plot(data['steps'], ema(data[key]), color=f'C{color}', label=label)
        ax.set(title=(r'$\mathbf{h}^{(24)}$' if src == 'h24' else r'$\mathbf{z}$') + ' reader, thought 1', ylabel='Value-mention rate', ylim=(0, 1))
    axes[0].legend()
    axes[-1].set(xlabel='Reader training step')
    fig.tight_layout()
    fig.savefig(out / 'nla_value_training.png')
    plt.close(fig)
    return summary
