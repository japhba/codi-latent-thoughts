"""Matched six-thought twin transplants and final AV/AR language edits.

Capture every solver-verified candidate before selecting fresh both-correct or
six-only subsets. Raw columns contain pre-normalization block outputs. Language
edits change complete numeric intermediate literals only, without using answers
or givens. Random controls match their displacement from the own reconstruction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import torch
from tqdm.auto import tqdm

from .analysis import ROOT, canonical_number, equal, last_number, read, write
from .controlled_rollout import rollout
from .models import load_organism, load_reader
from .recurrent_controls import save_batch


def candidates(args):
    rows = read('artifacts/verified_twin_candidates.json')
    return rows[:args.limit] if args.limit else rows


def questions(rows, side):
    field = 'question' if side == 'original' else 'twin_question'
    return [r[field].strip().replace('  ', ' ') + ' Output only the answer and nothing else.' for r in rows]


def capture(args):
    model = load_organism(args)
    rows = candidates(args)
    folder = args.out / 'captures'
    folder.mkdir(exist_ok=True)
    for start in tqdm(range(0, len(rows), args.batch_size), desc='Twin capture'):
        path = folder / f'{start:06d}.pt'
        if args.resume and path.is_file(): continue
        assert not path.exists(), 'Use --resume for existing captures.'
        batch = rows[start:start + args.batch_size]
        result = {'start': start, 'rows': batch}
        for side in ['original', 'twin']:
            q = questions(batch, side)
            full = rollout(model, q, capture=True)
            own = rollout(model, q, replacement=full, positions=range(6), kind='column')
            assert own['texts'] == full['texts'], 'Own raw-column replay changed answers.'
            torch.testing.assert_close(own['z'], full['z'], atol=0, rtol=0)
            result[side] = {k: v.to(torch.bfloat16) if torch.is_tensor(v) else v for k, v in full.items()}
            result[side]['zero_texts'] = rollout(model, q, thoughts=0)['texts']
        save_batch(path, result)
    parts = [torch.load(p, weights_only=True) for p in sorted(folder.glob('*.pt'))]
    assert [r for part in parts for r in part['rows']] == rows
    records = []
    for part in parts:
        for j, row in enumerate(part['rows']):
            texts = {f'{side}_{mode}': part[side]['texts' if mode == 'six' else 'zero_texts'][j] for side in ['original', 'twin'] for mode in ['six', 'zero']}
            both = equal(last_number(texts['original_six']), row['gold']) and equal(last_number(texts['twin_six']), row['cf_twin'])
            lb = both and not equal(last_number(texts['original_zero']), row['gold']) and not equal(last_number(texts['twin_zero']), row['cf_twin'])
            records.append({**row, 'pair_index': len(records), 'outputs': texts, 'both_correct': both, 'both_six_only': lb})
    write(args.out / 'baseline.json', records)


def intermediate_map(row):
    mapping = {}
    for original, twin in zip(row['chain'], row['chain_twin'], strict=True):
        a, b = canonical_number(original), canonical_number(twin)
        if a == b or a in {canonical_number(row['gold']), canonical_number(row['g'])} or b in {canonical_number(row['cf_twin']), canonical_number(row['g_twin'])}: continue
        mapping.setdefault(b, set()).add(a)
    # Conflicting references cannot identify an unambiguous literal edit.
    return {value: next(iter(targets)) for value, targets in mapping.items() if len(targets) == 1}


def edit_intermediates(text, mapping):
    replaced = []
    def replace(match):
        token = match.group()
        value = canonical_number(token)
        if value not in mapping: return token
        replaced.append({'before': token, 'after': mapping[value]})
        return mapping[value]
    changed = re.sub(r'(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|\.\d)', replace, text)
    return changed, replaced


@torch.no_grad()
def describe(args):
    parts = [torch.load(p, weights_only=True) for p in sorted((args.out / 'captures').glob('*.pt'))]
    rows = candidates(args)
    vectors = {side: torch.cat([p[side][args.stream] for p in parts]) for side in ['original', 'twin']}
    model, tok = load_reader(args)
    marker = 149705
    messages = [{'role': r['role'], 'content': r['content'].replace('<INJECT>', tok.decode([marker]))} for r in read(f'artifacts/prompt_{args.stream}.json')]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pids = tok(prompt, add_special_tokens=False).input_ids
    assert pids.count(marker) == 1
    at, state = pids.index(marker), {}
    def inject(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1:
            v = state['vectors']
            original = h[:, at]
            h[:, at] = original + original.norm(dim=-1, keepdim=True) * (v / v.norm(dim=-1, keepdim=True))
        return output
    handle = model.get_base_model().model.layers[1].register_forward_hook(inject)
    folder = args.out / f'descriptions_{args.stream}'
    folder.mkdir(exist_ok=True)
    tasks = [(side, i, t) for side in ['original', 'twin'] for i in range(len(rows)) for t in range(6)]
    for start in tqdm(range(0, len(tasks), args.batch_size), desc=f'Final AV {args.stream}'):
        path = folder / f'{start:06d}.json'
        if args.resume and path.is_file(): continue
        assert not path.exists()
        batch = tasks[start:start + args.batch_size]
        state['vectors'] = torch.stack([vectors[side][i, t] for side, i, t in batch]).to('cuda')
        ids = torch.tensor([pids] * len(batch), device='cuda')
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        result = []
        for j, (side, i, t) in enumerate(batch):
            full = tok.decode(output[j, len(pids):], skip_special_tokens=True)
            match = re.search(r'<explanation>(.*?)(</explanation>|$)', full, re.S)
            body = match.group(1).strip() if match else full.strip()
            edited, changes = edit_intermediates(body, intermediate_map(rows[i])) if side == 'twin' else (body, [])
            result.append({'pair_index': i, 'side': side, 'thought': t + 1, 'gen': body, 'edited': edited, 'changes': changes, 'full_response': full})
        partial = path.with_suffix('.partial')
        write(partial, result)
        partial.replace(path)
    handle.remove()


@torch.no_grad()
def reconstruct(args):
    import yaml
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / 'vendor/EasyNLA'))
    from nla.models import NLACriticModel
    from nla.utils.critic import critic_predict
    from .lenses import file_hash
    tok = AutoTokenizer.from_pretrained(args.reader_base_dir)
    tok.padding_side = 'right'  # critic_predict indexes the last real token.
    template = yaml.safe_load((args.critic_dir.parent / 'nla_meta.yaml').read_text())['prompt_templates']['critic']
    assert '{ctx}' not in template, 'These final readers were trained without question context.'
    files = [args.critic_dir / 'config.json', *sorted(args.critic_dir.glob('*.safetensors'))]
    write(args.out / f'critic_{args.stream}_provenance.json', {'files_sha256': {p.name: file_hash(p) for p in files}, 'critic_template': template, 'anchor': 'last real token; right padding', 'backbone_output': 'raw truncated residual, final norm removed', 'head_dtype': 'float32', 'direction_only': True, 'norm': 'true receiver vector norm for own, edited, sham and donor conditions'})
    critic = NLACriticModel.from_pretrained(args.critic_dir, torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval().to('cuda')
    rows = [r for p in sorted((args.out / f'descriptions_{args.stream}').glob('*.json')) for r in json.loads(p.read_text())]
    assert len(rows) == len(candidates(args)) * 12
    tasks = [(r, variant) for r in rows for variant in ['gen', 'edited'] if variant == 'gen' or r['side'] == 'twin']
    folder = args.out / f'reconstruction_{args.stream}'
    folder.mkdir(exist_ok=True)
    for start in tqdm(range(0, len(tasks), args.batch_size), desc=f'Matched final AR {args.stream}'):
        path = folder / f'{start:06d}.pt'
        if args.resume and path.is_file(): continue
        assert not path.exists()
        batch = tasks[start:start + args.batch_size]
        enc = tok([template.format(explanation=r[variant]) for r, variant in batch], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        pred = critic_predict(critic, enc.input_ids, enc.attention_mask, 1.).float()
        assert torch.isfinite(pred).all() and (pred.norm(dim=-1) > 0).all()
        directions = torch.nn.functional.normalize(pred, dim=-1).cpu()
        keys = [(r['side'], r['pair_index'], r['thought'] - 1, variant) for r, variant in batch]
        save_batch(path, {'keys': keys, 'directions': directions})


def matched_sham(own, edited):
    generator = torch.Generator().manual_seed(11)
    random = torch.randn(own.shape, generator=generator)
    orthogonal = random - (random * own).sum(-1, keepdim=True) * own
    orthogonal = orthogonal - (orthogonal * own).sum(-1, keepdim=True) * own
    orthogonal = torch.nn.functional.normalize(orthogonal, dim=-1)
    # Compute the angle from the displacement to avoid sqrt(eps) perturbations
    # when the edited text and its reconstruction are exactly unchanged.
    distance = (edited - own).norm(dim=-1, keepdim=True).clamp(max=2)
    sham = (1 - distance.square() / 2) * own + distance * (1 - distance.square() / 4).clamp_min(0).sqrt() * orthogonal
    torch.testing.assert_close((sham - own).norm(dim=-1), (edited - own).norm(dim=-1), atol=2e-5, rtol=2e-4)
    return sham


def vector_controls(args):
    parts = [torch.load(p, weights_only=True) for p in sorted((args.out / 'captures').glob('*.pt'))]
    stream = args.stream
    true = {side: torch.cat([p[side][stream] for p in parts]).float() for side in ['original', 'twin']}
    dirs = {}
    for path in sorted((args.out / f'reconstruction_{stream}').glob('*.pt')):
        part = torch.load(path, weights_only=True)
        dirs.update(zip(map(tuple, part['keys']), part['directions']))
    n = len(candidates(args))
    own = torch.stack([torch.stack([dirs[('twin', i, t, 'gen')] for t in range(6)]) for i in range(n)])
    edited = torch.stack([torch.stack([dirs[('twin', i, t, 'edited')] for t in range(6)]) for i in range(n)])
    donor = torch.stack([torch.stack([dirs[('original', i, t, 'gen')] for t in range(6)]) for i in range(n)])
    cosine = (edited * own).sum(-1, keepdim=True).clamp(-1, 1)
    sham = matched_sham(own, edited)
    # Every receiver-side language condition uses exactly the same true norm.
    norm = true['twin'].norm(dim=-1, keepdim=True)
    descriptions = [r for p in sorted((args.out / f'descriptions_{stream}').glob('*.json')) for r in json.loads(p.read_text())]
    changed = {(r['pair_index'], r['thought'] - 1): bool(r['changes']) for r in descriptions if r['side'] == 'twin'}
    fidelity = {'cosine_own': (own * torch.nn.functional.normalize(true['twin'], dim=-1)).sum(-1).tolist(), 'cosine_edited': cosine[..., 0].tolist(), 'edit_present': [[changed[(i, t)] for t in range(6)] for i in range(n)]}
    write(args.out / f'language_fidelity_{stream}.json', fidelity)
    return {name: {stream: vector * norm} for name, vector in [('own', own), ('edited', edited), ('sham', sham), ('donor', donor)]}


@torch.no_grad()
def patch(args):
    model = load_organism(args)
    rows = candidates(args)
    captures = [torch.load(p, weights_only=True) for p in sorted((args.out / 'captures').glob('*.pt'))]
    assert [r for p in captures for r in p['rows']] == rows
    ranges = [('all6', list(range(6))), *[(f'p{t + 1}', [t]) for t in range(6)]]
    if args.action == 'raw':
        sources = {side: {k: torch.cat([p[side][k] for p in captures]) for k in ['z', 'h24', 'blocks']} for side in ['original', 'twin']}
        conditions = [(f'raw_{kind}_{source}_{label}', kind, sources['twin' if source == 'own' else 'original'], active) for kind in ['z', 'h24', 'column'] for source in ['own', 'donor'] for label, active in ranges]
    else:
        controls = vector_controls(args)
        conditions = [(f'nla_{args.stream}_{source}_{label}', args.stream, replacement, active) for source, replacement in controls.items() for label, active in ranges]
    for name, kind, source, active in tqdm(conditions, desc='Twin conditions'):
        folder = args.out / 'conditions' / name
        folder.mkdir(parents=True, exist_ok=True)
        for start in tqdm(range(0, len(rows), args.batch_size), desc=name, leave=False):
            path = folder / f'{start:06d}.json'
            if args.resume and path.is_file(): continue
            assert not path.exists()
            batch = rows[start:start + args.batch_size]
            replacement = {k: v[start:start + len(batch)] for k, v in source.items()}
            result = rollout(model, questions(batch, 'twin'), replacement=replacement, positions=active, kind=kind)
            if name.startswith(f'raw_{kind}_own'):
                baseline = read(args.out / 'baseline.json')
                assert result['texts'] == [r['outputs']['twin_six'] for r in baseline[start:start + len(batch)]], 'Own raw-state control failed.'
            partial = path.with_suffix('.partial')
            write(partial, [{'pair_index': start + j, 'text': text, 'pred': last_number(text)} for j, text in enumerate(result['texts'])])
            partial.replace(path)


def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(11)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {'organism': read('manifest.json')['organism'], 'n': len(candidates(args)), 'batch_size': args.batch_size, 'seed': 11, 'max_new_tokens': 64, 'attention_mask': 'all cached steps', 'column': 'raw block outputs; not KV-cache transplantation', 'reader_step': 3200, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'rollout_sha256': hashlib.sha256(Path(rollout.__code__.co_filename).read_bytes()).hexdigest(), 'group': args.group}
    path = args.out / 'config.json'
    if args.resume and path.is_file(): assert read(path) == config
    write(path, config)
    import wandb
    run_id = f'{args.group}-twins-{args.action}' + (f'-{args.stream}' if args.stream else '') + ('-smoke' if args.limit else '')
    run = wandb.init(project='codi-latent-thoughts', group=args.group, id=run_id, name=run_id, resume='allow', config=config)
    if args.action == 'capture': capture(args)
    elif args.action == 'describe': describe(args)
    elif args.action == 'reconstruct': reconstruct(args)
    else: patch(args)
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['capture', 'raw', 'describe', 'reconstruct', 'language'], required=True)
    parser.add_argument('--organism-dir', type=Path)
    parser.add_argument('--reader-base-dir', type=Path)
    parser.add_argument('--reader-dir', type=Path)
    parser.add_argument('--critic-dir', type=Path)
    parser.add_argument('--stream', choices=['z', 'h24'])
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--group', default='rv6')
    main(parser.parse_args())
