"""Reevaluate the full GSM8k cohort with masked, regenerated interventions."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from tqdm.auto import tqdm

from .analysis import equal, last_number, read, write
from .controlled_rollout import rollout
from .models import load_organism


def save_batch(path, data):
    partial = path.with_suffix('.partial')
    torch.save(data, partial)
    partial.replace(path)


def main(args):
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(11)
    rows = read('artifacts/controls.json')
    if args.limit: rows = rows[:args.limit]
    assert len(rows) >= 2
    model = load_organism(args)
    suffix = ' Output only the answer and nothing else.'
    questions = [r['question'].strip().replace('  ', ' ') + suffix for r in rows]
    config = {'organism': read('manifest.json')['organism'], 'group': args.group, 'batch_size': args.batch_size, 'n': len(rows), 'i': [r['i'] for r in rows], 'torch': torch.__version__, 'prompt': 'question + answer-only instruction + BOT', 'cached_padding_mask': True, 'max_new': 64, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'rollout_sha256': hashlib.sha256(Path(rollout.__code__.co_filename).read_bytes()).hexdigest(), 'seed': 11}
    config_path = args.out / 'controls_config.json'
    if args.resume and config_path.is_file(): assert read(config_path.name, args.out) == config
    write(config_path, config)
    import wandb
    run_id = f'{args.group}-recurrent-controls' + ('-smoke' if args.limit else '')
    run = wandb.init(project='codi-latent-thoughts', group=args.group, name=run_id, id=run_id, resume='allow', config=config)
    captures = args.out / 'captures'
    captures.mkdir(exist_ok=True)
    for start in tqdm(range(0, len(rows), args.batch_size), desc='Clean six/zero rollouts'):
        path = captures / f'{start:06d}.pt'
        if args.resume and path.is_file(): continue
        assert not path.exists(), 'Use --resume for existing outputs.'
        batch = questions[start:start + args.batch_size]
        original = rollout(model, batch)
        zero = rollout(model, batch, thoughts=0)
        replay = rollout(model, batch, frozen=original['z'])
        assert replay['texts'] == original['texts'], 'Own-vector replay changed an answer.'
        torch.testing.assert_close(replay['z'], original['z'], rtol=0, atol=0)
        save_batch(path, {'start': start, 'rows': rows[start:start + len(batch)], 'texts': original['texts'], 'zero_texts': zero['texts'], 'z': original['z']})
        run.log({'phase': 'capture', 'captured_problems': min(start + args.batch_size, len(rows))})
    captured = [torch.load(p, weights_only=True) for p in sorted(captures.glob('*.pt'))]
    assert [r['i'] for part in captured for r in part['rows']] == [r['i'] for r in rows]
    clean = torch.cat([part['z'] for part in captured])
    identity = [text for part in captured for text in part['texts']]
    zero = [text for part in captured for text in part['zero_texts']]
    accuracy = sum(equal(last_number(text), r['gold']) for text, r in zip(identity, rows)) / len(rows)
    assert args.limit or accuracy > .2, 'The masked baseline is clearly broken; inspect complete outputs.'
    write(args.out / 'baseline.json', {'config': config, 'rows': [{**r, 'six_text': six, 'zero_text': none, 'six_pred': last_number(six), 'zero_pred': last_number(none)} for r, six, none in zip(rows, identity, zero)]})
    donor = clean.roll(-1, 0)
    mean, std = clean.mean(0), clean.std(0)
    generator = torch.Generator().manual_seed(11)
    noise = torch.randn(clean.shape, generator=generator) * std[None] + mean[None]
    replacements = {'donor': donor, 'mean': mean[None].expand_as(clean), 'noise': noise, 'zero': torch.zeros_like(clean)}
    conditions = [(f'{kind}_p{pos + 1}_{mode}', kind, pos, mode) for kind in replacements for pos in range(6) for mode in ['frozen', 'regenerated']]
    conditions += [(f'trunc_{n}_regenerated', 'trunc', n, 'regenerated') for n in range(1, 6)]
    conditions += [(f'drop_p{pos + 1}_frozen', 'drop', pos, 'frozen') for pos in range(6)]
    conditions += [(name, name, -1, 'frozen') for name in ['reverse', 'repeat_z1']]
    conditions_dir = args.out / 'conditions'
    conditions_dir.mkdir(exist_ok=True)
    scores = []
    for name, kind, pos, mode in tqdm(conditions, desc='Ablation conditions'):
        folder = conditions_dir / name
        folder.mkdir(exist_ok=True)
        began = time.perf_counter()
        for start in tqdm(range(0, len(rows), args.batch_size), desc=name, leave=False):
            path = folder / f'{start:06d}.pt'
            if args.resume and path.is_file(): continue
            assert not path.exists(), 'Use --resume for existing outputs.'
            batch = questions[start:start + args.batch_size]
            original = clean[start:start + len(batch)]
            if kind in replacements:
                patched = rollout(model, batch, replacement={'z': replacements[kind][start:start + len(batch)]}, positions=[pos], frozen=original if mode == 'frozen' else None)
            elif kind == 'trunc': patched = rollout(model, batch, thoughts=pos)
            elif kind == 'drop': patched = rollout(model, batch, thoughts=5, frozen=torch.cat([original[:, :pos], original[:, pos + 1:]], 1))
            elif kind == 'reverse': patched = rollout(model, batch, frozen=original.flip(1))
            elif kind == 'repeat_z1': patched = rollout(model, batch, frozen=original[:, :1].expand_as(original))
            else: raise ValueError(kind)
            save_batch(path, {'start': start, 'condition': name, 'texts': patched['texts'], 'predictions': [last_number(text) for text in patched['texts']]})
        parts = [torch.load(p, weights_only=True) for p in sorted(folder.glob('*.pt'))]
        texts = [text for part in parts for text in part['texts']]
        assert len(texts) == len(rows)
        eligible = [index for index, (r, a, b) in enumerate(zip(rows, identity, zero)) if equal(last_number(a), r['gold']) and not equal(last_number(b), r['gold'])]
        assert eligible, 'The baseline yielded no six-only problems.'
        all_correct = sum(equal(last_number(text), r['gold']) for text, r in zip(texts, rows))
        lb_correct = sum(equal(last_number(texts[index]), rows[index]['gold']) for index in eligible)
        scores.append({'condition': name, 'correct': all_correct, 'n': len(rows), 'load_bearing_correct': lb_correct, 'load_bearing_n': len(eligible), 'seconds': time.perf_counter() - began})
        write(args.out / 'controls_scores.json', scores)
        run.log({'condition': name, 'accuracy': all_correct / len(rows), 'load_bearing_accuracy': lb_correct / len(eligible), 'conditions_complete': len(scores)})
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--organism-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--group', default='rv6')
    parser.add_argument('--resume', action='store_true')
    main(parser.parse_args())
