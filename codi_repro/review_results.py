"""Summarize the rv6 all-thought evaluation and controlled reruns (statistics only; figures/ draws the post's figures)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .analysis import ROOT, equal, last_number, read, write
from .all_layer_analysis import scores as lens_scores


def estimate(values, groups):
    values = np.asarray(values, dtype=float)
    assert len(values) == len(groups)
    if not len(values): return {'n': 0, 'rate': None, 'ci95': None, 'reason': 'No eligible pairs; this contrast is unidentifiable.'}
    unique = list(dict.fromkeys(groups))
    counts = np.array([sum(g == u for g in groups) for u in unique])
    totals = np.array([values[np.array([g == u for g in groups])].sum() for u in unique])
    draws = np.random.default_rng(11).integers(len(unique), size=(5000, len(unique)))
    boot = totals[draws].sum(1) / counts[draws].sum(1)
    return {'n': len(values), 'groups': len(unique), 'rate': float(values.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist()}


def performance(folder, out):
    data = read(folder / 'baseline.json')
    rows = data['rows']
    six = np.array([equal(r['six_pred'], r['gold']) for r in rows])
    zero = np.array([equal(r['zero_pred'], r['gold']) for r in rows])
    groups = [r['i'] for r in rows]
    categories = {'Both correct': int((six & zero).sum()), 'Six only': int((six & ~zero).sum()), 'Zero only': int((~six & zero).sum()), 'Neither': int((~six & ~zero).sum())}
    summary = {'six': estimate(six, groups), 'zero': estimate(zero, groups), 'paired_gain': estimate(six.astype(int) - zero.astype(int), groups), 'categories': categories, 'n': len(rows), 'config': data['config']}
    write(out / 'performance_summary.json', summary)
    return summary


def ablations(folder, out):
    rows = read(folder / 'baseline.json')['rows']
    eligible = [i for i, r in enumerate(rows) if equal(r['six_pred'], r['gold']) and not equal(r['zero_pred'], r['gold'])]
    groups = [rows[i]['i'] for i in eligible]
    predictions = read(folder / 'predictions.json')
    scores = {}
    for result in read(folder / 'controls_scores.json'):
        name = result['condition']
        texts = predictions[name]['texts']
        assert len(texts) == len(rows)
        values = [equal(last_number(texts[i]), rows[i]['gold']) for i in eligible]
        scores[name] = estimate(values, groups)
    assert len(scores) == 61
    write(out / 'ablation_summary.json', {'six_only_n': len(eligible), 'conditions': scores})


def nla(scores_path, out):
    scores = read(scores_path)['scores']
    summary = {}
    for src in ['h24', 'z']:
        series = sorted([r for r in scores if r['src'] == src and r['kind'] == 'own'], key=lambda r: r['step'])
        assert [r['step'] for r in series] == [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200]
        assert all(r['answer']['n'] == 60 and r['intermediate']['n'] == (55 if src == 'h24' else 49) for r in series)
        summary[src] = series
    write(out / 'all_thought_curve_summary.json', summary)


def twins(folder, out):
    rows = read(folder / 'baseline.json')
    assert len(rows) == 296
    scores, predictions = {}, {}
    for condition in sorted((folder / 'conditions').iterdir()):
        data = [r for p in sorted(condition.glob('*.json')) for r in json.loads(p.read_text())]
        assert [r['pair_index'] for r in data] == list(range(len(rows)))
        predictions[condition.name] = [r['pred'] for r in data]
        scores[condition.name] = {}
        for population in ['all', 'both_correct', 'both_six_only']:
            indices = [i for i, r in enumerate(rows) if population == 'all' or r[population]]
            groups = [rows[i]['doc'] for i in indices]
            scores[condition.name][population] = {name: estimate([equal(data[i]['pred'], rows[i][gold]) for i in indices], groups) for name, gold in [('adopt', 'gold'), ('keep', 'cf_twin')]}
    assert len(scores) == 98
    contrasts, fidelity_summary = {}, {}
    for src in ['z', 'h24']:
        fidelity = read(folder / f'language_fidelity_{src}.json')
        own = predictions[f'nla_{src}_own_all6']
        edited = predictions[f'nla_{src}_edited_all6']
        sham = predictions[f'nla_{src}_sham_all6']
        fidelity_summary[src] = {}
        for population in ['all', 'both_correct', 'both_six_only']:
            selected = [i for i, r in enumerate(rows) if population == 'all' or r[population]]
            cosines = np.asarray([fidelity['cosine_own'][i] for i in selected]).flatten()
            changed = np.asarray([fidelity['edit_present'][i] for i in selected])
            fidelity_summary[src][population] = {'pairs': len(selected), 'vectors': len(cosines), 'own_cosine_mean': float(cosines.mean()), 'own_cosine_median': float(np.median(cosines)), 'own_cosine_min': float(cosines.min()), 'edited_pairs': int(changed.any(1).sum()), 'edited_vectors': int(changed.sum())}
        for population in ['both_correct', 'both_six_only']:
            indices = [i for i, r in enumerate(rows) if r[population] and any(fidelity['edit_present'][i])]
            stable = [i for i in indices if equal(own[i], rows[i]['cf_twin'])]
            contrast = {}
            for cohort, chosen in [('eligible', indices), ('roundtrip_preserved', stable)]:
                groups = [rows[i]['doc'] for i in chosen]
                contrast[cohort] = {'n': len(chosen)}
                for method, outputs in [('own', own), ('edited', edited), ('sham', sham)]:
                    contrast[cohort][method] = {name: estimate([equal(outputs[i], rows[i][gold]) for i in chosen], groups) for name, gold in [('adopt', 'gold'), ('keep', 'cf_twin')]}
                difference = [int(equal(edited[i], rows[i]['gold'])) - int(equal(sham[i], rows[i]['gold'])) for i in chosen]
                contrast[cohort]['edited_minus_sham_adoption'] = estimate(difference, groups)
                retention_difference = [int(equal(edited[i], rows[i]['cf_twin'])) - int(equal(sham[i], rows[i]['cf_twin'])) for i in chosen]
                contrast[cohort]['edited_minus_sham_retention'] = estimate(retention_difference, groups)
            contrasts[f'{src}_{population}'] = contrast
    summary = {'n': len(rows), 'original_problems': len({r['doc'] for r in rows}), 'both_correct': sum(r['both_correct'] for r in rows), 'both_six_only': sum(r['both_six_only'] for r in rows), 'conditions': scores, 'language_contrasts': contrasts, 'directional_fidelity': fidelity_summary, 'confidence_intervals': '5000 paired bootstrap draws clustered by original problem; seed 11'}
    write(out / 'twin_summary.json', summary)
    return summary


def verify_review(root=ROOT):
    folder = root / 'artifacts/rv6'
    manifest = read(folder / 'bundle.json')
    for name, expected in manifest['files_sha256'].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected, f'Corrected evidence checksum mismatch: {name}'
    for name, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, f'Corrected source checksum mismatch: {name}'
    scores = read(folder / 'scores/all_thought_scores.json')
    assert len(scores['scores']) == 36 and scores['positions'] == list(range(1, 7))
    assert len(read(folder / 'scores/all_thought_judgments.json')) == 2160
    return {'group': manifest['group'], 'files': len(manifest['files_sha256']), 'reader_ensembles': 2160, 'reader_descriptions': 12960, 'organism': manifest['organism']}


def reproduce(out, root=ROOT):
    out.mkdir(parents=True, exist_ok=True)
    folder = root / 'artifacts/rv6'
    performance_summary = performance(folder / 'controls', out)
    ablations(folder / 'controls', out)
    nla(folder / 'scores/all_thought_scores.json', out)
    twin_summary = twins(folder / 'twins', out)
    summary = {'performance': performance_summary, 'ablations': read(out / 'ablation_summary.json'), 'all_thought_readers': read(out / 'all_thought_curve_summary.json'), 'twins': twin_summary, 'reader_layer_lenses': lens_scores(scope='reader_layers'), 'all_layer_lens_skylines': lens_scores(scope='all_layers')}
    write(out / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', type=Path)
    parser.add_argument('--scores', type=Path)
    parser.add_argument('--twins', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.controls:
        performance(args.controls, args.out)
        ablations(args.controls, args.out)
    if args.scores: nla(args.scores, args.out)
    if args.twins: twins(args.twins, args.out)
