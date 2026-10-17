"""Population scores from compact saved lens distributions; no torch required."""
import json
import re

import matplotlib.pyplot as plt
import numpy as np

from .analysis import ROOT, canonical_number, read, write

METHODS = ['LogitLens', 'J-lens']


def numeric_mentions(text):
    return {canonical_number(s) for s in re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))}


def targets(problems):
    values = {name: [] for name in ['answer_leading_digit', 'first_reference_leading_digit', 'last_reference_leading_digit', 'novel_reference_leading_digit', 'single_token_answer', 'single_token_novel_reference', 'single_token_novel_distinct_from_answer_digit']}
    for problem in problems:
        if not problem['correct']:
            for name in values: values[name].append(None)
            continue
        gold = canonical_number(problem['gold'])
        question = numeric_mentions(problem['question'])
        chain = [canonical_number(v) for v in problem['chain']]
        novel = [v for v in chain if v not in question and v != gold]
        single = [v for v in novel if v in [str(d) for d in range(10)]]
        distinct = [v for v in single if v != re.search(r'\d', gold).group()]
        row = [gold, chain[0], chain[-1], novel[0] if novel else None, gold if gold in [str(d) for d in range(10)] else None, single[0] if single else None, distinct[0] if distinct else None]
        for name, value in zip(values, row): values[name].append(value)
    return values


def score(data, problems):
    correct = np.array([p['correct'] for p in problems])
    digit_ids = data['digit_ids']
    results = {}
    for name, values in targets(problems).items():
        idx = np.array([i for i, v in enumerate(values) if correct[i] and v is not None])
        labels = np.array([int(re.search(r'\d', values[i]).group()) for i in idx])
        frequency = np.bincount(labels, minlength=10) / len(labels)
        logp = np.take(data['digit_logp'], idx, axis=2)
        ranks = np.take(data['digit_rank'], idx, axis=2)
        top = np.take(data['top_ids'], idx, axis=2)
        target_ids = digit_ids[labels][None, None, :, None, None]
        top1 = top[..., 0] == target_ids[..., 0]
        top5 = (top == target_ids).any(-1)
        digit_top1 = logp.argmax(-1) == labels[None, None, :, None]
        gather = labels[None, None, :, None, None]
        target_logp = np.take_along_axis(logp, gather, -1)[..., 0]
        target_rank = np.take_along_axis(ranks, gather, -1)[..., 0]
        top1_digit_hits = top[..., :1, None] == digit_ids
        top5_digit_hits = (top[..., None] == digit_ids).any(-2)
        # Exact expectation of a uniformly permuted target-label assignment,
        # preserving the cohort's actual digit frequencies rather than assuming 10%.
        shuffled_top1 = (top1_digit_hits[..., 0, :] * frequency).sum(-1)
        shuffled_top5 = (top5_digit_hits * frequency).sum(-1)
        shuffled_digit_top1 = frequency[logp.argmax(-1)]
        answer_digits = np.array([int(re.search(r'\d', problems[i]['gold']).group()) for i in idx])
        conditioned_null = np.empty_like(shuffled_top5)
        conditioned_digit_null = np.empty_like(shuffled_digit_top1)
        for answer_digit in np.unique(answer_digits):
            group = answer_digits == answer_digit
            group_frequency = np.bincount(labels[group], minlength=10) / group.sum()
            conditioned_null[:, :, group] = (top5_digit_hits[:, :, group] * group_frequency).sum(-1)
            conditioned_digit_null[:, :, group] = group_frequency[logp[:, :, group].argmax(-1)]
        results[name] = {'n': len(idx), 'ds_indices': [problems[i]['ds_index'] for i in idx], 'label_counts': np.bincount(labels, minlength=10).tolist(), 'majority_digit_rate': float(frequency.max()), 'full_vocab_top1_count': top1.sum(2).tolist(), 'full_vocab_top5_count': top5.sum(2).tolist(), 'full_vocab_top1': top1.mean(2).tolist(), 'full_vocab_top5': top5.mean(2).tolist(), 'shuffled_full_vocab_top1': shuffled_top1.mean(2).tolist(), 'shuffled_full_vocab_top5': shuffled_top5.mean(2).tolist(), 'shuffled_top5_given_answer_digit': conditioned_null.mean(2).tolist(), 'digit_only_top1': digit_top1.mean(2).tolist(), 'shuffled_digit_only_top1': shuffled_digit_top1.mean(2).tolist(), 'shuffled_digit_top1_given_answer_digit': conditioned_digit_null.mean(2).tolist(), 'mean_full_vocab_probability': np.exp(target_logp).mean(2).tolist(), 'median_full_vocab_rank': np.median(target_rank, axis=2).tolist()}
    return {'n': len(problems), 'correct_n': int(correct.sum()), 'methods': METHODS, 'positions': ['BOT', '1', '2', '3', '4', '5', '6'], 'hidden_states': list(range(37)), 'targets': results, 'scope': 'GSM8k-Aug-NL train 4000..5199, conditioned on saved correct rollout; reference-number labels are numeric mentions, not verified reasoning states.'}


def plots(summary, out):
    for name, filename, label in [('answer_leading_digit', 'lens_answer.png', 'Answer leading digit'), ('single_token_novel_reference', 'lens_novel.png', 'Reference digits, no prompt numeral')]:
        target = summary['targets'][name]
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
        for method, ax in enumerate(axes):
            rates = np.array(target['full_vocab_top5'][method])
            im = ax.imshow(rates, origin='lower', aspect='auto', vmin=0, vmax=1)
            ax.set(title=METHODS[method], ylabel='Hidden-state index', yticks=[0, 8, 16, 24, 32, 36], xticks=range(7), xticklabels=summary['positions'])
            fig.colorbar(im, ax=ax, label='Full-vocabulary top-5 rate')
        axes[-1].set_xlabel('BOT / thought position')
        fig.suptitle(f'{label} (n={target["n"]})')
        fig.tight_layout(); fig.savefig(out / filename); plt.close(fig)
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    for ax, target_name, position in zip(axes, ['answer_leading_digit', 'single_token_novel_distinct_from_answer_digit'], [1, 2]):
        target = summary['targets'][target_name]
        for method, label in enumerate(METHODS):
            own = np.array(target['full_vocab_top5'][method])[:, position]
            null_name = 'shuffled_full_vocab_top5' if position == 1 else 'shuffled_top5_given_answer_digit'
            shuffled = np.array(target[null_name][method])[:, position]
            ax.plot(range(37), own, label=label)
            ax.plot(range(37), shuffled, linestyle='--', label=f'{label}: shuffled labels')
        ax.set(title=('Answer digit, thought 1' if target_name == 'answer_leading_digit' else 'Distinct reference value, thought 2') + f' (n={target["n"]})', ylabel='Full-vocabulary top-5 rate', ylim=(0, 1))
        ax.legend()
    axes[-1].set(xlabel='Hidden-state index (0 = fed vector)', xticks=[0, 8, 16, 24, 32, 36])
    fig.suptitle('Corrected lenses at fixed thought positions')
    fig.tight_layout(); fig.savefig(out / 'lens_controls.png'); plt.close(fig)
    target = summary['targets']['single_token_novel_distinct_from_answer_digit']
    difference = np.array(target['digit_only_top1']) - np.array(target['shuffled_digit_top1_given_answer_digit'])
    bound = np.abs(difference).max()
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    for method, ax in enumerate(axes):
        im = ax.imshow(difference[method], origin='lower', aspect='auto', cmap='RdBu_r', vmin=-bound, vmax=bound)
        ax.set(title=METHODS[method], ylabel='Hidden-state index', yticks=[0, 8, 16, 24, 32, 36], xticks=range(7), xticklabels=summary['positions'])
        fig.colorbar(im, ax=ax, label='Accuracy gain')
    axes[-1].set_xlabel('BOT / thought position')
    fig.suptitle(f'Reference value distinct from answer digit (n={target["n"]})')
    fig.tight_layout(); fig.savefig(out / 'lens_digits.png'); plt.close(fig)


def examples(data, problems, out, inputs):
    tokens = read('lens_token_strings.json', inputs)
    eligible = targets(problems)['single_token_novel_reference']
    selected = [i for i, p in enumerate(problems) if p['correct'] and eligible[i] is not None][:3]
    parts = ['# Corrected lens examples', '', 'First three saved correct examples with a single-token reference number not among the question\'s numeric literals. Selected by metadata, without inspecting lens predictions. All decoded top-five tokens are printed in full.', '']
    for i in selected:
        problem = problems[i]
        parts.extend([f'## Dataset row {problem["ds_index"]}', '', problem['question'], '', '**Reference CoT:** ' + problem['cot'], '', f'**Gold:** {problem["gold"]}; **saved CODI prediction:** {problem["pred"]}; **target reference number:** {eligible[i]}', '', '| Source | Thought | Method | Top five tokens |', '|---|---|---|---|'])
        for hs in [0, 24, 36]:
            for position in [1, 6]:
                for method, label in enumerate(METHODS):
                    full = ', '.join(json.dumps(tokens[str(t)], ensure_ascii=False).replace('|', '\\|') for t in data['top_ids'][method, hs, i, position])
                    parts.append(f'| {"fed vector" if hs == 0 else f"h{hs}"} | {position} | {label} | {full} |')
        parts.append('')
    (out / 'lens_examples.md').write_text('\n'.join(parts) + '\n')


def reproduce(out, root=ROOT, inputs=None):
    inputs = inputs if inputs is not None else root / 'artifacts'
    with np.load(inputs / 'lens_readouts.npz') as archive: data = {name: archive[name] for name in archive.files}
    problems = read('lens_problems.json', inputs)
    assert data['digit_logp'].shape == (2, 37, len(problems), 7, 10)
    assert np.isfinite(data['digit_logp']).all()
    for name in ['digit_logp', 'digit_rank', 'top_ids', 'top_logp']:
        np.testing.assert_array_equal(data[name][0, 36], data[name][1, 36])
    summary = score(data, problems)
    summary['normalization_audit'] = read('lens_run.json', inputs)['normalization_audit']
    write(out / 'lens_summary.json', summary)
    plots(summary, out)
    examples(data, problems, out, inputs)
    return summary
