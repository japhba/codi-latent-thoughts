from pathlib import Path
import hashlib
import json
import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]


def read(name, root=ROOT):
    return json.loads((root / name).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def last_number(text):
    values = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return float(values[-1]) if values else None


def equal(a, b):
    return a is not None and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)


def canonical_number(value):
    f = float(str(value).replace(',', '').replace('$', '').rstrip('%.'))
    return str(round(f)) if abs(f-round(f)) < 1e-9 else f'{f:g}'


def rate(values):
    return {'correct': int(sum(values)), 'n': len(values), 'rate': float(np.mean(values))}


def verify(root=ROOT):
    manifest = read('manifest.json', root)
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, f'Artifact checksum mismatch: {name}'
    for name, expected in manifest['source_files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, f'Vendored source mismatch: {name}'
    rows = read('artifacts/controls.json', root)
    assert len(rows) == 1319 and [r['i'] for r in rows] == list(range(1319))
    for r in rows:
        for condition, text in r['raw'].items():
            parsed = last_number(text)
            assert parsed == r['pred'][condition], (r['i'], condition, text, parsed, r['pred'][condition])
    assert all(r['pred']['drop_6'] == r['pred']['trunc_5'] for r in rows)
    gens = read('artifacts/readouts.json', root)
    verdicts = read('artifacts/verdicts.json', root)
    key = lambda r: (r['src'], r['uid'], r['t'], r['kind'])
    assert len({key(r) for r in gens}) == len(gens) == 1440
    assert len({key(r) for r in verdicts}) == len(verdicts) == 1248
    problems = {r['uid']: r for r in read('artifacts/problems.json', root)}
    eligible = set()
    for r in gens:
        p = problems[r['uid']]
        gold = {canonical_number(v) for v in p['intermediates']}
        assert gold == set(r['gold']), (r['uid'], gold, r['gold'])
        if gold - {canonical_number(p['answer'])}: eligible.add(key(r))
    assert {key(r) for r in verdicts} == eligible
    for src, eligible in [('z', 49), ('h24', 55)]:
        vectors = np.load(root / f'artifacts/vectors_{src}.npz')
        assert vectors['vectors'].shape == (60, 6, 4096)
        assert set(vectors['uids']) == {r['uid'] for r in gens if r['src'] == src}
        for t in range(6):
            for kind in ['own', 'donor']:
                assert sum((r['src'], r['t'], r['kind']) == (src, t, kind) for r in verdicts) == eligible
    lens_problems = read('artifacts/lens_problems.json', root)
    assert [p['ds_index'] for p in lens_problems] == list(range(4000, 5200))
    assert sum(p['correct'] for p in lens_problems) == 973
    assert read('artifacts/lens_run.json', root)['organism'] == manifest['organism']
    assert read('artifacts/lens_head_audit.json', root)['ordinary_token_rows_identical']
    assert read('artifacts/lens_head_audit.json', root)['final_norm_identical']
    best_vectors = np.load(root / 'artifacts/av_best_lens_vectors.npz')
    assert best_vectors['vectors'].shape == (108, 4096)
    assert best_vectors['uids'].tolist() == list(problems)
    av_lens = read('artifacts/av_lens_readouts.json', root)
    assert av_lens['organism'] == manifest['organism']
    assert (av_lens['selection']['hidden_state'], av_lens['selection']['thought']) == (36, 1)
    from .curve_lenses import AO_HIDDEN_STATES, METHODS, adjudicate, ensembles, tasks
    ensemble = read('artifacts/curve_lens_ensemble_readouts.json', root)
    assert ensemble['organism'] == manifest['organism']
    assert ensemble['jacobian_sha256'] == read('artifacts/lens_provenance.json', root)['jacobian_sha256']
    assert ensemble['fit_metadata'] == read('artifacts/lens_provenance.json', root)['fit_metadata']
    ao = np.load(root / 'artifacts/curve_lens_ao_ensemble_vectors.npz')
    assert ao['vectors'].shape == (48, 6, 6, 4096) and ao['hidden_states'].tolist() == AO_HIDDEN_STATES
    assert ao['uids'].tolist() == ensemble['streams']['ao']['uids']
    old_ao = np.load(root / 'artifacts/curve_lens_ao_vectors.npz')
    assert np.array_equal(ao['vectors'][:, 0, -1], old_ao['vectors'])
    for src, hs in [('h24', [24]), ('z', [0]), ('ao', AO_HIDDEN_STATES)]:
        stream = ensemble['streams'][src]
        assert stream['hidden_states'] == hs
        for method in METHODS:
            assert list(stream['methods'][method]) == stream['uids']
            for uid in stream['uids']:
                records = stream['methods'][method][uid]
                assert len(records) == 6 * len(hs)
                assert {(r['hidden_state'], r['thought']) for r in records} == {(h, t) for h in hs for t in range(1, 7)}
                assert all(len(r['top_ids']) == len(r['top_tokens']) == len(r['top_logp']) == 5 for r in records)
                assert all(np.isfinite(r['top_logp']).all() for r in records)
        for uid in stream['uids']:
            for a, b in zip(stream['methods']['LogitLens'][uid], stream['methods']['J-lens'][uid]):
                assert (a['hidden_state'], a['thought']) == (b['hidden_state'], b['thought'])
                if a['hidden_state'] == 36: assert a == b
    assert len(ensembles(root)) == 432
    inputs = read('artifacts/curve_lens_ensemble_judge_inputs.json', root)
    assert inputs == tasks(root) and len(inputs) == 832
    judged = read('artifacts/curve_lens_ensemble_verdicts.json', root)
    key = lambda r: (r['src'], r['uid'], r['t'], r['kind'])
    assert len(judged) == len({key(r) for r in judged}) == len(inputs)
    assert {key(r) for r in judged} == {key(r) for r in inputs}
    assert all(isinstance(r['matched'], bool) for r in judged)
    extracted = read('artifacts/curve_lens_ensemble_extractions.json', root)
    extraction_key = lambda r: (r['src'], r['uid'], r['method'])
    assert len(extracted) == len({extraction_key(r) for r in extracted}) == 432
    bykey = {extraction_key(r): r for r in extracted}
    for row in ensembles(root):
        tokens = set.union(set(), *(set(json.loads(line)) for line in row['gen'].splitlines()))
        for item in bykey[extraction_key(row)]['values']:
            assert item['token'] and item['token'] in tokens
            assert item['value'] == canonical_number(item['value'])
    input_bykey = {key(r): r for r in inputs}
    for row in judged:
        method, _ = row['kind'].split(':')
        values = adjudicate(bykey[(row['src'], row['uid'], method)], root)
        assert row['matches'] == [v for v in values if v['value'] in input_bykey[key(row)]['gold']]
        assert row['matched'] == bool(row['matches'])
    from .all_layer_analysis import verify as verify_all_layers
    from .review_results import verify_review
    all_layers = verify_all_layers(root)
    return {'artifact_files': len(manifest['files']), 'source_files': len(manifest['source_files']), 'controls': len(rows), 'readouts': len(gens), 'verdicts': len(verdicts), 'lens_rollouts': len(lens_problems), 'lens_correct_rollouts': 973, 'av_baseline_questions': 108, 'curve_lens_ensembles': 432, 'curve_lens_extractions': len(extracted), 'curve_lens_judgments': len(judged), 'all_layer_lenses': all_layers, 'corrected_reruns': verify_review(root)}



def reproduce(root=ROOT):
    """Validate the bundled evidence, write the statistics (and the J-lens fit curve) to results/, then draw every post figure into figs/."""
    out = root / 'results'
    out.mkdir(parents=True, exist_ok=True)
    verification = verify(root)
    from .review_results import reproduce as reproduce_review
    summary = reproduce_review(out, root)
    from .all_layer_analysis import profile
    from .all_layers import plot_fit
    profile(out, root)
    plot_fit(read('artifacts/layer_lens_fit.json', root)['history'], out)  # J-lens fit convergence (training control, not in the post)
    write(out / 'validation.json', verification)
    write(out / 'summary.json', summary)
    figures(root)
    print(out.resolve())


def figures(root=ROOT):
    """Run every figures/fig_*.py; each draws one post figure (or its main/appendix pair) into figs/."""
    import subprocess, sys
    for script in tqdm(sorted((root / 'figures').glob('fig_*.py')), desc='Post figures'):
        subprocess.run([sys.executable, str(script)], cwd=root, check=True)
    print((root / 'figs').resolve())
