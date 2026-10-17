"""Regenerate six-position NLA evaluation curves from recovered reader weights."""
import argparse
from contextlib import nullcontext
import json
import os
import re
from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from .analysis import ROOT, read, write
from .models import load_reader
from .lenses import file_hash


def main(args):
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(11)
    model, tok = load_reader(args)
    payload = np.load(ROOT / f'artifacts/vectors_{args.stream}.npz')
    uids, vectors = payload['uids'].tolist(), payload['vectors']
    if args.limit: uids = uids[:args.limit]
    marker = 149705
    messages = [{'role': r['role'], 'content': r['content'].replace('<INJECT>', tok.decode([marker]))} for r in read(f'artifacts/prompt_{args.stream}.json')]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pids = tok(text, add_special_tokens=False).input_ids
    assert pids.count(marker) == 1
    state = {}

    def inject(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1:
            v = state['vectors']
            at = h[:, pids.index(marker)]
            h[:, pids.index(marker)] = at + at.norm(dim=-1, keepdim=True) * (v / v.norm(dim=-1, keepdim=True))
        return output

    handle = model.get_base_model().model.layers[1].register_forward_hook(inject)
    paths = {int(p.name.split('_')[1]): p for p in args.reader_root.glob('iter_*') if (p / 'adapter_model.safetensors').is_file()}
    recovered_steps = [0, *sorted(paths)]
    assert recovered_steps == [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200]
    steps = args.steps if args.steps is not None else recovered_steps
    assert set(steps) <= set(recovered_steps)
    path = args.out / f'reader_{args.stream}.jsonl'
    shards = args.out / f'reader_{args.stream}_batches'
    shards.mkdir(exist_ok=True)
    batch_paths = sorted(shards.glob('*.json'))
    assert args.resume or not batch_paths, 'Use --resume for existing output.'
    records = [row for batch_path in batch_paths for row in json.loads(batch_path.read_text())]
    completed = {(r['step'], r['uid'], r['thought'], r['kind']) for r in records}
    assert len(completed) == len(records)
    checkpoint_hashes = {str(step): file_hash(paths[step] / 'adapter_model.safetensors') for step in steps if step}
    spec = {'organism': read('manifest.json')['organism'], 'reader_base': read('manifest.json')['reader_base'], 'stream': args.stream, 'group': args.group, 'steps': steps, 'adapter_hashes': checkpoint_hashes, 'batch_size': args.batch_size, 'max_new_tokens': args.max_new, 'positions': list(range(1, 7)), 'uids': uids, 'donor_shift': 7, 'criterion': 'Independent six-description ensemble per question; explicit complete values; no cross-position composition.', 'torch': torch.__version__, 'started': time.time()}
    write(args.out / f'reader_{args.stream}_config.json', spec)
    import wandb
    run_id = f'{args.group}-all-thought-{args.stream}' + ('-smoke' if args.limit else '')
    run = wandb.init(project='codi-latent-thoughts', group=args.group, name=run_id, id=run_id, resume='allow', config={k: v for k, v in spec.items() if k != 'started'})
    for step in tqdm(steps, desc='Reader checkpoints'):
        keys = {(step, uid, t, kind) for uid in uids for t in range(1, 7) for kind in ['own', 'donor']}
        if keys <= completed: continue
        adapter = 'default' if step == 3200 else f'step_{step}'
        if step not in [0, 3200]: model.load_adapter(paths[step], adapter_name=adapter)
        if step: model.set_adapter(adapter)
        context = model.disable_adapter() if step == 0 else nullcontext()
        began = time.perf_counter()
        with context, torch.no_grad():
            tasks = [(i, t, kind) for kind in ['own', 'donor'] for t in range(6) for i, uid in enumerate(uids) if (step, uid, t + 1, kind) not in completed]
            for start in tqdm(range(0, len(tasks), args.batch_size), desc=f'{args.stream} step {step}', leave=False):
                batch = tasks[start:start + args.batch_size]
                state['vectors'] = torch.tensor(np.stack([vectors[i if kind == 'own' else (i + 7) % len(vectors), t] for i, t, kind in batch]), dtype=torch.bfloat16, device='cuda')
                ids = torch.tensor([pids] * len(batch), device='cuda')
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
                new_rows = []
                for j, (i, t, kind) in enumerate(batch):
                    full = tok.decode(output[j, len(pids):], skip_special_tokens=True)
                    match = re.search(r'<explanation>(.*?)(</explanation>|$)', full, re.S)
                    body = match.group(1).strip() if match else full.strip()
                    row = {'src': args.stream, 'step': step, 'uid': uids[i], 'thought': t + 1, 'kind': kind, 'gen': body, 'full_response': full}
                    new_rows.append(row)
                committed = shards / f'{len(batch_paths):06d}.json'
                partial = committed.with_suffix('.partial')
                with partial.open('w') as sink:
                    json.dump(new_rows, sink, ensure_ascii=False)
                    sink.flush()
                    os.fsync(sink.fileno())
                partial.replace(committed)
                batch_paths.append(committed)
                records.extend(new_rows)
                completed.update((r['step'], r['uid'], r['thought'], r['kind']) for r in new_rows)
                write(args.out / f'reader_{args.stream}_progress.json', {'step': step, 'completed_descriptions': len(completed), 'total_descriptions': len(steps) * len(uids) * 12, 'cuda_rng': torch.cuda.get_rng_state().tolist(), 'cpu_rng': torch.get_rng_state().tolist()})
        run.log({'reader_step': step, 'completed_descriptions': len(completed), 'seconds_per_checkpoint': time.perf_counter() - began})
        if step not in [0, 3200]:
            model.set_adapter('default')
            model.delete_adapter(adapter)
        torch.cuda.empty_cache()
    handle.remove()
    assert len(completed) == len(steps) * len(uids) * 12
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', choices=['z', 'h24'], required=True)
    parser.add_argument('--reader-root', type=Path, required=True)
    parser.add_argument('--reader-dir', type=Path, required=True)
    parser.add_argument('--reader-base-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-new', type=int, default=256)
    parser.add_argument('--group', default='rv6')
    parser.add_argument('--steps', type=int, nargs='+')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    main(parser.parse_args())
