"""Bundle compact corrected-run evidence without copying weights or captures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from tqdm.auto import tqdm

from .analysis import ROOT, read, write
from .lenses import file_hash


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    controls = args.runs / 'controls/full'
    target = args.out / 'controls'
    target.mkdir(exist_ok=True)
    baseline = read(controls / 'baseline.json')
    baseline['rows'] = [{k: r[k] for k in ['i', 'question', 'gold', 'six_text', 'zero_text', 'six_pred', 'zero_pred']} for r in baseline['rows']]
    write(target / 'baseline.json', baseline)
    scores = read(controls / 'controls_scores.json')
    assert len(scores) == 61
    write(target / 'controls_scores.json', scores)
    results = {}
    for score in tqdm(scores, desc='Bundle ablation responses'):
        name = score['condition']
        parts = [torch.load(p, weights_only=True) for p in sorted((controls / 'conditions' / name).glob('*.pt'))]
        texts = [text for part in parts for text in part['texts']]
        predictions = [value for part in parts for value in part['predictions']]
        assert len(texts) == len(predictions) == 1319
        results[name] = {'texts': texts, 'predictions': predictions}
    write(target / 'predictions.json', results)
    twins = args.runs / 'twins/full'
    target = args.out / 'twins'
    target.mkdir(exist_ok=True)
    for name in ['baseline.json', 'config.json', 'critic_z_provenance.json', 'critic_h24_provenance.json', 'language_fidelity_z.json', 'language_fidelity_h24.json']:
        shutil.copy2(twins / name, target / name)
    critic_steps = {}
    for src, directory in [('z', args.critic_z_dir), ('h24', args.critic_h24_dir)]:
        step_file = directory / 'saved_at_step.txt'
        step = int(step_file.read_text().strip())
        assert step == 3200
        provenance = read(twins / f'critic_{src}_provenance.json')
        for name, expected in provenance['files_sha256'].items():
            assert file_hash(directory / name) == expected, f'AR checkpoint mismatch: {src}/{name}'
        critic_steps[src] = {'step': step, 'saved_at_step_sha256': file_hash(step_file), 'saved_at_step_text': step_file.read_text()}
    write(target / 'critic_steps.json', critic_steps)
    assert len(list((twins / 'conditions').iterdir())) == 98
    for name in ['conditions', 'descriptions_z', 'descriptions_h24']: shutil.copytree(twins / name, target / name, dirs_exist_ok=True)
    readers = args.out / 'readers'
    readers.mkdir(exist_ok=True)
    manifest = read('manifest.json')
    for src in ['z', 'h24']:
        source = args.runs / 'readers' / src
        config = read(source / f'reader_{src}_config.json')
        assert config['adapter_hashes']['3200'] == manifest['readers'][src]['weights_sha256']
        descriptions = [json.loads(line) for line in (source / f'reader_{src}.jsonl').read_text().splitlines()]
        keys = {(r['step'], r['uid'], r['thought'], r['kind']) for r in descriptions}
        assert len(keys) == len(descriptions) == 6480
        for name in [f'reader_{src}.jsonl', f'reader_{src}_config.json']: shutil.copy2(source / name, readers / name)
    source = args.runs / 'scores'
    assert len(list((source / 'extractions').glob('*.json'))) == 2160
    shutil.copytree(source, args.out / 'scores', dirs_exist_ok=True)
    files = {str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.rglob('*')) if p.is_file() and p.name != 'bundle.json'}
    sources = [ROOT / 'codi_repro' / name for name in ['analysis.py', '__main__.py', 'controlled_rollout.py', 'recurrent_controls.py', 'all_thought_readers.py', 'all_thought_scores.py', 'twin_controls.py', 'judge.py', 'review_results.py', 'bundle_review.py']]
    write(args.out / 'bundle.json', {'group': 'rv6', 'organism': manifest['organism'], 'reader_step': 3200, 'files_sha256': files, 'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, 'external_only': ['Raw residual captures', 'Large reader/AR weights', 'Intermediate AR direction tensors'], 'publication_status': 'Not published'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/rv6')
    parser.add_argument('--critic-z-dir', type=Path, required=True)
    parser.add_argument('--critic-h24-dir', type=Path, required=True)
    main(parser.parse_args())
