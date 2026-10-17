"""Fresh, disjoint, layer-specific Jacobian calibration and all-layer readouts."""
import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .analysis import ROOT, canonical_number, read, write
from .curve_lenses import METHODS, intermediates


def cohorts():
    nla = read('artifacts/problems.json')
    ao = read('artifacts/ao_curves.json')['problems']
    fit = [p for p in read('artifacts/lens_problems.json') if 4000 <= p['ds_index'] < 4800]
    validation = [p for p in read('artifacts/lens_problems.json') if 4800 <= p['ds_index'] < 5000]
    assert {p['ds_index'] for p in fit}.isdisjoint(p['ds_index'] for p in validation)
    assert {p['ds_index'] for p in fit + validation}.isdisjoint(p['idx'] for p in nla + ao)
    questions = lambda rows: {' '.join(p['question'].split()) for p in rows}
    assert questions(fit).isdisjoint(questions(validation))
    assert questions(fit + validation).isdisjoint(questions(nla + ao))
    return fit, validation, nla + ao


def prompt_ids(model, question):
    import torch
    tokens = model.tokenizer(question, return_tensors='pt').input_ids.to('cuda')
    return torch.cat([tokens, torch.tensor([[model.tokenizer.eos_token_id, model.tokenizer.bot_id]], device='cuda')], 1)


def record(model, question):
    import torch
    with torch.no_grad():
        ids = prompt_ids(model, question)
        out = model.codi(input_ids=ids, use_cache=True, output_hidden_states=True)
        acts = [torch.stack([h[0, -1] for h in out.hidden_states])]
        for _ in range(6):
            z = model.prj(out.hidden_states[-1][:, -1:]).to(model.codi.dtype)
            out = model.codi(inputs_embeds=z, past_key_values=out.past_key_values, use_cache=True, output_hidden_states=True)
            acts.append(torch.stack([h[0, -1] for h in out.hidden_states]))
        current = model.codi.get_input_embeddings()(torch.tensor([[model.tokenizer.eot_id]], device='cuda'))
        tokens = []
        for _ in range(12):
            out = model.codi(inputs_embeds=current, past_key_values=out.past_key_values, use_cache=True)
            nxt = int(out.logits[0, -1, :model.codi.config.vocab_size - 1].argmax())
            if nxt == model.tokenizer.eos_token_id: break
            tokens.append(nxt)
            current = model.codi.get_input_embeddings()(torch.tensor([[nxt]], device='cuda'))
        return torch.stack(acts).float().cpu().numpy().astype(np.float16), model.tokenizer.decode(tokens, skip_special_tokens=True)


def capture(args):
    from .lenses import numbers
    from .models import load_organism
    model = load_organism(args)
    args.out.mkdir(parents=True, exist_ok=True)
    fit, validation, evaluation = cohorts()
    output = {}
    for name, candidates, limit in [('fit', fit, args.n_fit), ('validation', validation, 24), ('evaluation', evaluation, len(evaluation))]:
        selected, vectors = [], []
        for problem in tqdm(candidates, desc=f'All-layer {name} capture'):
            acts, answer = record(model, problem['question'])
            parsed = numbers(answer)
            correct = bool(parsed) and canonical_number(parsed[0]) == canonical_number(problem['gold'] if name != 'evaluation' else problem['answer'])
            if name != 'evaluation' and not correct: continue
            selected.append({**problem, 'captured_answer': answer, 'captured_correct': correct})
            vectors.append(acts)
            if len(selected) == limit: break
        assert len(selected) == limit, f'Insufficient correct {name} problems'
        output[name] = selected
        np.save(args.out / f'{name}_activations.npy', np.array(vectors))
        write(args.out / f'{name}_problems.json', selected)
    # Use the saved reader vectors at the two original sources; report replay agreement.
    evaluation = np.load(args.out / 'evaluation_activations.npy', mmap_mode='r+')
    uids = [p['uid'] for p in output['evaluation']]
    agreement = {}
    for src, hs in [('h24', 24), ('z', 0)]:
        old = np.load(ROOT / f'artifacts/vectors_{src}.npz')
        similarities = []
        for uid, saved in zip(old['uids'], old['vectors']):
            row = uids.index(uid)
            fresh = evaluation[row, 1:, hs].astype(np.float32)
            saved32 = saved.astype(np.float32)
            cosine = (fresh * saved32).sum(-1) / (np.linalg.norm(fresh, axis=-1) * np.linalg.norm(saved32, axis=-1))
            assert cosine.min() > .98, (uid, src, cosine.tolist())
            similarities.extend(cosine.tolist())
            evaluation[row, 1:, hs] = saved
        agreement[src] = {'minimum_cosine': min(similarities), 'mean_cosine': float(np.mean(similarities))}
    evaluation.flush()
    write(args.out / 'capture.json', {'organism': read('manifest.json')['organism'], 'prompt': 'raw question + EOS + BOT', 'batch_size': 1, 'positions': ['BOT', 'thought1', 'thought2', 'thought3', 'thought4', 'thought5', 'thought6'], 'hidden_states': list(range(37)), 'saved_reader_source_agreement': agreement, 'source_replacement': 'Exact historical h24/z vectors replace corresponding evaluation states; other layers are freshly captured.', 'fit_ids': [p['ds_index'] for p in output['fit']], 'validation_ids': [p['ds_index'] for p in output['validation']], 'evaluation_uids': uids})


def jacobian_chunks(inner, ids, zs, *, dim_batch, start_row=0, stop_row=None):
    """Yield exact output-coordinate rows for every source in one backward pass.

    Mean source positions, sum causal future target positions (paper reduction).
    Thought inputs are fixed leaves; feedback projection is not differentiated.
    """
    import torch
    dim, p0 = zs.shape[-1], ids.shape[1]
    valid = torch.arange(p0 - 1, p0 + len(zs), device=zs.device)
    leaf = torch.cat([inner.get_input_embeddings()(ids)[0].detach(), zs], 0)[None].repeat(dim_batch, 1, 1).requires_grad_(True)
    acts = {}
    handles = [block.register_forward_hook(lambda module, inp, out, l=l: acts.__setitem__(l, out)) for l, block in enumerate(inner.model.layers)]
    inner.model(inputs_embeds=leaf, use_cache=False)
    for handle in handles: handle.remove()
    target = acts[len(inner.model.layers) - 1]
    sources = [leaf] + [acts[l] for l in range(len(inner.model.layers) - 1)]
    cotangent = torch.zeros_like(target)
    batches = torch.arange(dim_batch, device=zs.device)
    end = dim if stop_row is None else stop_row
    for start in tqdm(range(start_row, end, dim_batch), desc='All-layer Jacobian rows', leave=False, mininterval=15):
        nd = min(dim_batch, end - start)
        cotangent.zero_()
        cotangent[batches[:nd, None], valid[None, :], start + batches[:nd, None]] = 1
        grads = torch.autograd.grad(target, sources, grad_outputs=cotangent, retain_graph=start + dim_batch < end)
        reduced = [grads[0][:nd, valid[1:]].float().mean(1).cpu()]
        reduced += [g[:nd, valid].float().mean(1).cpu() for g in grads[1:]]
        yield start, torch.stack(reduced), target[0, -1].detach()


def copy_atomic(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.copying')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def save_resume(state, args):
    import torch
    from .lenses import atomic_save
    state['rng'] = torch.get_rng_state(); state['cuda_rng'] = torch.cuda.get_rng_state_all()
    atomic_save(state, args.out / 'layer_lens_resume.pt')


def proxy_metrics(model, maps, n, vectors, problems):
    """Per-source first-answer-token CE/accuracy, on fixed calibration/validation data.

    A convergence diagnostic, not the blog's complete-value mention criterion.
    """
    import torch
    from .lenses import decode, transport
    head = model.codi.get_output_embeddings().weight.float()
    norm = model.codi.get_base_model().model.norm.weight.float()
    targets = torch.tensor([model.tokenizer.encode(str(p['gold']), add_special_tokens=False)[0] for p in problems], device='cuda')
    losses, accuracy = [], []
    with torch.no_grad():
        for hs in range(36):
            jacobian = (maps[hs] / n).to('cuda')
            h = torch.tensor(vectors[:, 1:, hs].astype(np.float32), device='cuda').reshape(-1, 4096)
            labels = targets.repeat_interleave(6)
            loss, right = 0., 0
            for start in range(0, len(h), 32):
                logits = decode(transport(h[start:start + 32], jacobian), head, norm)
                loss += torch.nn.functional.cross_entropy(logits, labels[start:start + 32], reduction='sum').item()
                right += (logits.argmax(-1) == labels[start:start + 32]).sum().item()
            losses.append(loss / len(h)); accuracy.append(right / len(h))
    assert np.isfinite(losses).all()
    return {'mean_ce': float(np.mean(losses)), 'mean_accuracy': float(np.mean(accuracy)), 'ce_by_hidden_state': losses, 'accuracy_by_hidden_state': accuracy}


def fit(args):
    import torch
    import wandb
    from .lenses import atomic_save, file_hash
    from .models import load_organism
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    model = load_organism(args)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    problems = json.loads((args.out / 'fit_problems.json').read_text())
    val_problems = json.loads((args.out / 'validation_problems.json').read_text())
    vectors = np.load(args.out / 'fit_activations.npy', mmap_mode='r')
    val_vectors = np.load(args.out / 'validation_activations.npy', mmap_mode='r')
    organism = read('manifest.json')['organism']
    resume_path = args.out / 'layer_lens_resume.pt'
    if args.resume:
        copy_atomic(args.shared / resume_path.name, resume_path)
        state = torch.load(resume_path, map_location='cpu', weights_only=True)
        assert state['organism'] == organism and state['dim_batch'] == args.dim_batch and state['fit_ids'] == [p['ds_index'] for p in problems]
        torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        state = {'sums': torch.zeros(36, 4096, 4096), 'count': 0, 'row_cursor': 0, 'fit_ids': [p['ds_index'] for p in problems], 'organism': organism, 'dim_batch': args.dim_batch, 'history': [], 'elapsed_seconds': 0., 'finished': False}
    run = wandb.init(project='codi-latent-thoughts', group=args.group, id=args.group, resume='allow', name=f'{args.group}-layer-jacobians', dir=str(args.out), config={'organism': organism, 'n_fit_max': len(problems), 'dim_batch': args.dim_batch, 'validation_n': len(val_problems), 'estimator': 'exact mean Jacobian, paper future-target reduction', 'checkpoint_seconds': args.checkpoint_seconds})
    began = time.monotonic()
    prior_elapsed = state['elapsed_seconds']
    saved_at = began
    with ThreadPoolExecutor(max_workers=1) as copier:
        future = None
        for i in tqdm(range(state['count'], state['count'] if state['finished'] else len(problems)), desc='Dedicated layer calibrations'):
            problem = problems[i]
            ids = prompt_ids(model, problem['question'])
            zs = torch.tensor(vectors[i, 1:, 0].astype(np.float32), dtype=torch.bfloat16, device='cuda')
            problem_began = time.monotonic()
            # Check replay independently of row iteration: a checkpoint may have
            # all rows but not yet have advanced the completed-problem counter.
            with torch.no_grad():
                inner = model.codi.get_base_model()
                leaf = torch.cat([inner.get_input_embeddings()(ids)[0], zs], 0)[None]
                replay = inner.model(inputs_embeds=leaf, use_cache=False).last_hidden_state[0, -1]
                cosine = torch.nn.functional.cosine_similarity(replay.float(), torch.tensor(vectors[i, -1, -1].astype(np.float32), device='cuda'), dim=0).item()
            assert cosine > .98, f'Teacher-forced replay mismatch: {cosine}'
            for start, chunk, _ in jacobian_chunks(inner, ids, zs, dim_batch=args.dim_batch, start_row=state['row_cursor']):
                assert torch.isfinite(chunk).all(), f'Nonfinite Jacobian at problem {i}, row {start}'
                state['sums'][:, start:start + chunk.shape[1]] += chunk
                state['row_cursor'] = start + chunk.shape[1]
                if time.monotonic() - saved_at >= args.checkpoint_seconds:
                    if future is not None: future.result()
                    state['elapsed_seconds'] = prior_elapsed + time.monotonic() - began
                    save_resume(state, args)
                    future = copier.submit(copy_atomic, resume_path, args.shared / resume_path.name)
                    saved_at = time.monotonic()
            state['count'] = i + 1; state['row_cursor'] = 0
            print({'problem': problem['ds_index'], 'count': state['count'], 'seconds': time.monotonic() - problem_began, 'replay_cosine': cosine}, flush=True)
            if state['count'] % args.eval_every == 0 or state['count'] == len(problems):
                train = proxy_metrics(model, state['sums'], state['count'], vectors[:8], problems[:8])
                val = proxy_metrics(model, state['sums'], state['count'], val_vectors, val_problems)
                row = {'n_fit': state['count'], 'train': train, 'validation': val, 'elapsed_seconds': prior_elapsed + time.monotonic() - began}
                state['history'].append(row)
                slopes = {}
                if len(state['history']) >= 4:
                    window = state['history'][-4:]
                    for name in ['train', 'validation']:
                        slopes[name] = float(np.polyfit(np.log([r['n_fit'] for r in window]), np.log([r[name]['mean_ce'] for r in window]), 1)[0])
                row['log_log_ce_slopes'] = slopes
                run.log({'n_fit': state['count'], 'train/ce': train['mean_ce'], 'validation/ce': val['mean_ce'], 'validation/accuracy': val['mean_accuracy'], **{f'{name}/log_log_ce_slope': value for name, value in slopes.items()}}, step=state['count'])
                write(args.out / 'layer_lens_fit_history.json', state['history'])
                copy_atomic(args.out / 'layer_lens_fit_history.json', args.shared / 'layer_lens_fit_history.json')
                plot_fit(state['history'], args.out)
                copy_atomic(args.out / 'layer_lens_fit.png', args.shared / 'layer_lens_fit.png')
                print(row, flush=True)
                if state['count'] >= 32 and slopes and all(abs(value) < .01 for value in slopes.values()):
                    state['finished'] = True; state['stop_reason'] = 'Both training and validation proxy CE stationary: absolute log-log slopes below 0.01 over four evaluations.'
            if state['count'] == len(problems): state['finished'] = True; state['stop_reason'] = 'Maximum calibration count reached.'
            state['elapsed_seconds'] = prior_elapsed + time.monotonic() - began
            save_resume(state, args)
            if state['finished'] or state['count'] % args.eval_every == 0 or time.monotonic() - saved_at >= args.checkpoint_seconds:
                if future is not None: future.result()
                future = copier.submit(copy_atomic, resume_path, args.shared / resume_path.name)
                saved_at = time.monotonic()
            if state['finished']: break
        if future is not None: future.result()
    assert state['finished'] and state['row_cursor'] == 0
    maps = state['sums'] / state['count']
    meta = {'organism': organism, 'n_problems': state['count'], 'selected': state['fit_ids'][:state['count']], 'validation_ids': [p['ds_index'] for p in val_problems], 'prompt': 'raw question + EOS + BOT', 'estimator': 'paper reduction, target=block-35 output', 'valid_positions': 'BOT+thought1..6 for residual maps; thought1..6 for embedding map', 'feedback': 'recorded thought inputs held fixed', 'hidden_state_to_matrix': {str(h): 'J_emb' if h == 0 else f'J[{h-1}]' for h in range(36)}, 'h36': 'Already normalized endpoint: identity, no fitted map.', 'group': args.group, 'stop_reason': state['stop_reason'], 'dim_batch': args.dim_batch, 'history': state['history']}
    atomic_save({'J_emb': maps[0], 'J': {h - 1: maps[h] for h in range(1, 36)}, 'meta': meta}, args.out / 'layer_jacobians.pt')
    copy_atomic(args.out / 'layer_jacobians.pt', args.shared / 'layer_jacobians.pt')
    write(args.out / 'layer_lens_fit.json', {**meta, 'sha256': file_hash(args.out / 'layer_jacobians.pt'), 'wandb_url': run.url})
    copy_atomic(args.out / 'layer_lens_fit.json', args.shared / 'layer_lens_fit.json')
    run.finish()


def plot_fit(history, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, layout='constrained')
    for name in ['train', 'validation']:
        axes[0].plot([r['n_fit'] for r in history], [r[name]['mean_ce'] for r in history], marker='o', label=name)
        axes[1].plot([r['n_fit'] for r in history], [r[name]['mean_accuracy'] for r in history], marker='o', label=name)
    axes[0].set(xscale='log', yscale='log', xlabel='Calibration problems', ylabel='First-answer-token CE', title='Convergence proxy')
    axes[1].set(xlabel='Calibration problems', ylabel='First-answer-token accuracy', ylim=(0, 1))
    axes[0].legend(); axes[1].legend()
    fig.savefig(out / 'layer_lens_fit.png', bbox_inches='tight'); plt.close(fig)


def readouts(args):
    import torch
    from transformers import AutoTokenizer
    from .lenses import decode, file_hash, transport, weight_hash
    from .models import fingerprint, prepare_cache, snapshot
    torch.set_num_threads(8); torch.set_float32_matmul_precision('highest')
    prepare_cache(args.cache_dir)
    organism = read('manifest.json')['organism']
    checkpoint = args.organism_dir if args.organism_dir else snapshot(organism, args.cache_dir)
    fingerprint(checkpoint / 'pytorch_model.bin', organism['weights_sha256'])
    state = torch.load(checkpoint / 'pytorch_model.bin', mmap=True, map_location='cpu', weights_only=True)
    head = state['codi.base_model.model.lm_head.weight'].float().to('cuda')
    norm = state['codi.base_model.model.model.norm.weight'].float().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    lens = torch.load(args.out / 'layer_jacobians.pt', mmap=True, map_location='cpu', weights_only=True)
    assert lens['meta']['organism'] == organism
    vectors = np.load(args.out / 'evaluation_activations.npy', mmap_mode='r')[:, 1:]
    problems = json.loads((args.out / 'evaluation_problems.json').read_text())
    uids = [p['uid'] for p in problems]
    assert set(lens['meta']['selected'] + lens['meta']['validation_ids']).isdisjoint(p['idx'] for p in problems)
    output = {method: {uid: [] for uid in uids} for method in METHODS}
    hashes, differences = {}, {}
    for hs in tqdm(range(37), desc='Fresh all-layer lens readouts'):
        jacobian = None if hs == 36 else (lens['J_emb'] if hs == 0 else lens['J'][hs - 1])
        if jacobian is not None: hashes[str(hs)] = weight_hash(jacobian); jacobian = jacobian.to('cuda')
        h = np.ascontiguousarray(vectors[:, :, hs]).reshape(-1, 4096)
        for start in range(0, len(h), 64):
            batch = torch.tensor(h[start:start + 64], dtype=torch.float32, device='cuda')
            for method in METHODS:
                transformed = batch if method == 'LogitLens' or hs == 36 else transport(batch, jacobian)
                logp = decode(transformed, head, norm, post_norm=hs == 36).log_softmax(-1)
                values, ids = logp.topk(5, -1)
                for offset, (token_ids, token_logp) in enumerate(zip(ids.cpu().tolist(), values.cpu().tolist())):
                    row, thought = divmod(start + offset, 6)
                    output[method][uids[row]].append({'hidden_state': hs, 'thought': thought + 1, 'top_ids': token_ids, 'top_tokens': [tokenizer.decode([i]) for i in token_ids], 'top_logp': token_logp})
        differences[str(hs)] = sum(a['top_ids'] != b['top_ids'] for uid in uids for a, b in zip(output['LogitLens'][uid][-6:], output['J-lens'][uid][-6:]))
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / 'all_layer_lens_readouts.json', {'organism': organism, 'uids': uids, 'methods': output, 'hidden_states': list(range(37)), 'matrix_sha256_by_hidden_state': hashes, 'jacobian_sha256': file_hash(args.out / 'layer_jacobians.pt'), 'fit_metadata': lens['meta'], 'different_top5_lists_by_hidden_state': differences, 'readout': {'k': 5, 'positions': list(range(1, 7)), 'hidden_states': list(range(37)), 'dtype': 'float32; TF32 disabled', 'normalization': 'RMSNorm once after transport; h36 directly to head', 'location_selection': 'none; every layer and thought position'}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['capture', 'fit', 'readouts'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shared', type=Path)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--organism-dir', type=Path)
    parser.add_argument('--n-fit', type=int, default=96)
    parser.add_argument('--dim-batch', type=int, default=16)
    parser.add_argument('--eval-every', type=int, default=8)
    parser.add_argument('--checkpoint-seconds', type=float, default=120)
    parser.add_argument('--group', default='jl4')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert len(args.group) == 3 and args.group.isalnum()
    {'capture': capture, 'fit': fit, 'readouts': readouts}[args.action](args)


if __name__ == '__main__': main()
