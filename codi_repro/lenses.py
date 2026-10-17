"""Fixed-checkpoint lenses; all residuals and Jacobians use explicit norm conventions."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .analysis import ROOT, read, write
from .models import fingerprint, load_organism, prepare_cache, snapshot


def rmsnorm(h, weight, eps=1e-6):
    return h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps) * weight


def decode(h, head, norm, *, post_norm=False):
    return (h if post_norm else rmsnorm(h, norm)) @ head.T


def transport(h, jacobian):
    # Stored rows index output coordinates: column-vector J h == row-vector h J.T.
    return h @ jacobian.T


def weight_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def numbers(text):
    return [(s.rstrip('0').rstrip('.') or '0') if '.' in s else s for s in re.findall(r'-?\d+(?:\.\d+)?', text)]


@torch.no_grad()
def audit(args):
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download, snapshot_download
    prepare_cache(args.cache_dir)
    manifest = read('manifest.json')
    checkpoint = args.organism_dir if args.organism_dir else snapshot(manifest['organism'], args.cache_dir)
    fingerprint(checkpoint / 'pytorch_model.bin', manifest['organism']['weights_sha256'])
    base_spec = manifest['base']
    if args.base_dir:
        base_dir = args.base_dir
    else:
        base_dir = Path(snapshot_download(repo_id=base_spec['repo_id'], revision=base_spec['revision'], token=True, cache_dir=args.cache_dir / 'hub', allow_patterns=['config.json', 'model.safetensors.index.json', 'tokenizer*', 'vocab.json', 'merges.txt', 'added_tokens.json', 'special_tokens_map.json']))
        index = json.loads((base_dir / 'model.safetensors.index.json').read_text())['weight_map']
        for filename in {index['lm_head.weight'], index['model.norm.weight']}:
            hf_hub_download(base_spec['repo_id'], filename, revision=base_spec['revision'], token=True, cache_dir=args.cache_dir / 'hub')
    index = json.loads((base_dir / 'model.safetensors.index.json').read_text())['weight_map']
    state = torch.load(checkpoint / 'pytorch_model.bin', mmap=True, map_location='cpu', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    n = len(AutoTokenizer.from_pretrained(base_dir))
    with safe_open(base_dir / index['lm_head.weight'], framework='pt', device='cpu') as handle: base_head = handle.get_tensor('lm_head.weight')
    trained = state['codi.base_model.model.lm_head.weight']
    with safe_open(base_dir / index['model.norm.weight'], framework='pt', device='cpu') as handle: base_norm = handle.get_tensor('model.norm.weight')
    norm = state['codi.base_model.model.model.norm.weight']
    result = {'base_snapshot': base_spec['revision'], 'base_tokenizer_size': n, 'checkpoint_tokenizer_size': len(tokenizer), 'base_head_size': len(base_head), 'checkpoint_head_size': len(trained), 'ordinary_token_rows_identical': torch.equal(base_head[:n], trained[:n]), 'base_ordinary_rows_sha256': weight_hash(base_head[:n]), 'checkpoint_ordinary_rows_sha256': weight_hash(trained[:n]), 'new_token_ids': {name: tokenizer.convert_tokens_to_ids(name) for name in ['<|bocot|>', '<|eocot|>']}, 'final_norm_identical': torch.equal(base_norm, norm)}
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / 'lens_head_audit.json', result)
    acts = np.load(args.acts, mmap_mode='r')
    problems = json.loads(args.problems.read_text())
    head, norm = trained.float().to('cuda'), norm.float().to('cuda')
    final = np.ascontiguousarray(acts[:, :, 36]).reshape(-1, 4096)
    correct_mask = np.array([p['correct'] for p in problems])
    target_ids = np.array([tokenizer.encode(re.search(r'\d', p['gold']).group(), add_special_tokens=False)[0] for p in problems])
    counts = {name: np.zeros(7, dtype=int) for name in ['top1_changed', 'top5_set_changed', 'correct_answer_digit_top1', 'double_norm_answer_digit_top1', 'correct_answer_digit_top5', 'double_norm_answer_digit_top5']}
    torch.set_float32_matmul_precision('highest')
    for start in tqdm(range(0, len(final), args.batch_size), desc='Detailed normalization audit'):
        stop = min(start + args.batch_size, len(final))
        h = torch.tensor(final[start:stop], dtype=torch.float32, device='cuda')
        current = decode(h, head, norm, post_norm=True).topk(5, -1).indices.cpu().numpy()
        old = decode(h, head, norm).topk(5, -1).indices.cpu().numpy()
        row = np.arange(start, stop) // 7
        position = np.arange(start, stop) % 7
        target = target_ids[row, None]
        flags = {'top1_changed': current[:, 0] != old[:, 0], 'top5_set_changed': (np.sort(current) != np.sort(old)).any(-1), 'correct_answer_digit_top1': (current[:, 0] == target[:, 0]) & correct_mask[row], 'double_norm_answer_digit_top1': (old[:, 0] == target[:, 0]) & correct_mask[row], 'correct_answer_digit_top5': (current == target).any(-1) & correct_mask[row], 'double_norm_answer_digit_top5': (old == target).any(-1) & correct_mask[row]}
        for name, matches in flags.items(): counts[name] += np.bincount(position[matches], minlength=7)
    write(args.out / 'lens_normalization_audit.json', {'positions_per_column': len(problems), 'correct_n': int(correct_mask.sum()), 'counts_by_position': {name: values.tolist() for name, values in counts.items()}, 'note': 'Same checkpoint head for both conditions: this isolates the second-normalization bug.'})


@torch.no_grad()
def readouts(args):
    from transformers import AutoTokenizer
    prepare_cache(args.cache_dir)
    spec = read('manifest.json')['organism']
    checkpoint = args.organism_dir if args.organism_dir else snapshot(spec, args.cache_dir)
    fingerprint(checkpoint / 'pytorch_model.bin', spec['weights_sha256'])
    lens_spec = read('artifacts/lens_provenance.json')
    hashes = {'activation_sha256': file_hash(args.acts), 'jacobian_sha256': file_hash(args.lens)}
    if args.recorded:
        assert all(hashes[k] == lens_spec[k] for k in hashes), 'Wrong historical lens inputs.'
    state = torch.load(checkpoint / 'pytorch_model.bin', mmap=True, map_location='cpu', weights_only=True)
    head = state['codi.base_model.model.lm_head.weight'].float().to('cuda')
    norm = state['codi.base_model.model.model.norm.weight'].float().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False) for d in range(10)]
    assert all(len(ids) == 1 for ids in digit_ids)
    digits = torch.tensor([ids[0] for ids in digit_ids], device='cuda')
    lens = torch.load(args.lens, mmap=True, map_location='cpu', weights_only=True)
    if args.recorded: assert lens['meta'] == lens_spec['fit_metadata']
    assert Path(lens['meta']['ckpt']).name == 'checkpoint-7580'
    assert lens['meta']['base'] == 'Qwen/Qwen3-8B' and lens['meta']['offset'] >= 5200
    assert lens['meta']['estimator'] == 'paper reduction, target=block-35 output'
    assert set(lens['J']) == set(range(35))
    acts = np.load(args.acts, mmap_mode='r')
    problems = json.loads(args.problems.read_text())
    assert acts.shape == (len(problems), 7, 37, 4096)
    n = len(problems)
    shape = (2, 37, n, 7)
    output = {'digit_logp': np.empty((*shape, 10), dtype=np.float32), 'digit_rank': np.empty((*shape, 10), dtype=np.int32), 'top_ids': np.empty((*shape, 5), dtype=np.int32), 'top_logp': np.empty((*shape, 5), dtype=np.float32)}
    torch.set_float32_matmul_precision('highest')
    args.out.mkdir(parents=True, exist_ok=True)
    for hs in tqdm(range(37), desc='Lens layers'):
        jacobian = lens['J_emb'] if hs == 0 else lens['J'][hs - 1] if hs < 36 else None
        if jacobian is not None:
            jacobian = jacobian.to('cuda')
            assert torch.isfinite(jacobian).all(), f'Nonfinite Jacobian at h{hs}'
        vectors = np.ascontiguousarray(acts[:, :, hs]).reshape(n * 7, 4096)
        for start in tqdm(range(0, n * 7, args.batch_size), desc=f'h{hs}', leave=False):
            stop = min(start + args.batch_size, n * 7)
            h = torch.tensor(vectors[start:stop], dtype=torch.float32, device='cuda')
            assert torch.isfinite(h).all(), f'Nonfinite saved activations at h{hs}'
            for method in range(2):
                z = h if method == 0 or hs == 36 else transport(h, jacobian)
                logits = decode(z, head, norm, post_norm=hs == 36)
                log_partition = logits.logsumexp(-1, keepdim=True)
                selected = logits[:, digits]
                rank = 1 + (logits[:, :, None] > selected[:, None, :]).sum(1)
                top, ids = logits.topk(5, -1)
                output['digit_logp'][method, hs].reshape(n * 7, 10)[start:stop] = (selected - log_partition).cpu().numpy()
                output['digit_rank'][method, hs].reshape(n * 7, 10)[start:stop] = rank.cpu().numpy()
                output['top_ids'][method, hs].reshape(n * 7, 5)[start:stop] = ids.cpu().numpy()
                output['top_logp'][method, hs].reshape(n * 7, 5)[start:stop] = (top - log_partition).cpu().numpy()
        del jacobian
    np.savez_compressed(args.out / 'lens_readouts.npz', **output, digit_ids=digits.cpu().numpy())
    write(args.out / 'lens_problems.json', problems)
    write(args.out / 'lens_token_strings.json', {str(i): tokenizer.decode([int(i)]) for i in np.unique(output['top_ids'])})
    # Isolate the old normalization bug at the final state, without changing the head.
    change = {'top1_changed': 0, 'top5_set_changed': 0, 'positions': n * 7, 'mean_kl_correct_to_double_norm': 0.}
    h36 = np.ascontiguousarray(acts[:, :, 36]).reshape(n * 7, 4096)
    for start in tqdm(range(0, n * 7, args.batch_size), desc='Normalization audit'):
        h = torch.tensor(h36[start:start + args.batch_size], dtype=torch.float32, device='cuda')
        correct = decode(h, head, norm, post_norm=True).log_softmax(-1)
        old = decode(h, head, norm, post_norm=False).log_softmax(-1)
        ct, ot = correct.topk(5, -1).indices, old.topk(5, -1).indices
        change['top1_changed'] += (ct[:, 0] != ot[:, 0]).sum().item()
        change['top5_set_changed'] += (ct.sort(-1).values != ot.sort(-1).values).any(-1).sum().item()
        change['mean_kl_correct_to_double_norm'] += (correct.exp() * (correct - old)).sum().item() / (n * 7)
    write(args.out / 'lens_run.json', {'organism': spec, 'torch': torch.__version__, 'dtype': 'float32, TF32 disabled', **hashes, 'recorded_inputs': args.recorded, 'fit_metadata': lens['meta'], 'normalization_audit': change, 'methods': ['logitlens', 'jlens'], 'hidden_states': '0=input embedding; 1..35=pre-final-norm block outputs; 36=post-final-norm output', 'positions': ['BOT', 'thought1', 'thought2', 'thought3', 'thought4', 'thought5', 'thought6'], 'z': 'exact saved hidden_states[0] at thought1..6, not an approximate re-projection'})
    print(args.out / 'lens_readouts.npz', flush=True)


@torch.no_grad()
def record(model, problem, max_tokens=8):
    """The historical lens prompt: answer-only question, EOS, BOT, no padding."""
    tokenizer = model.tokenizer
    question = problem['question'].strip().replace('  ', ' ') + ' Output only the answer and nothing else.'
    ids = tokenizer(question, return_tensors='pt').input_ids.to('cuda')
    ids = torch.cat([ids, torch.tensor([[tokenizer.eos_token_id, tokenizer.bot_id]], device='cuda')], 1)
    out = model.codi(input_ids=ids, use_cache=True, output_hidden_states=True)
    acts = [torch.stack([h[0, -1] for h in out.hidden_states])]
    zs = []
    for _ in range(6):
        z = model.prj(out.hidden_states[-1][:, -1:]).to(model.codi.dtype)
        zs.append(z[0, 0])
        out = model.codi(inputs_embeds=z, past_key_values=out.past_key_values, use_cache=True, output_hidden_states=True)
        acts.append(torch.stack([h[0, -1] for h in out.hidden_states]))
    embed = model.codi.get_input_embeddings()
    current = embed(torch.tensor([[tokenizer.eot_id]], device='cuda'))
    tokens = []
    for _ in range(max_tokens):
        out = model.codi(inputs_embeds=current, past_key_values=out.past_key_values, use_cache=True)
        nxt = int(out.logits[0, -1, :model.codi.config.vocab_size - 1].argmax())
        if nxt == tokenizer.eos_token_id: break
        tokens.append(nxt)
        current = embed(torch.tensor([[nxt]], device='cuda'))
    return ids, torch.stack(zs), torch.stack(acts), tokenizer.decode(tokens, skip_special_tokens=True)


def jacobian_rows(inner, ids, zs, sums, sums_emb, dim_batch):
    """Paper reduction: sum valid future targets, mean valid source positions.

    Recorded thought inputs are fixed leaves. This differentiates a teacher-forced
    replay, not the recursive projection/feedback path of a free-running organism.
    """
    dim = zs.shape[-1]
    p0 = ids.shape[1]
    valid = torch.arange(p0 - 1, p0 + len(zs), device=zs.device)
    thought_positions = valid[1:]
    embed = inner.get_input_embeddings()
    leaf = torch.cat([embed(ids)[0].detach(), zs], 0).unsqueeze(0).repeat(dim_batch, 1, 1).requires_grad_(True)
    acts = {}
    handles = [block.register_forward_hook(lambda module, inp, out, l=l: acts.__setitem__(l, out)) for l, block in enumerate(inner.model.layers)]
    inner.model(inputs_embeds=leaf, use_cache=False)
    for handle in handles: handle.remove()
    target = acts[len(inner.model.layers) - 1]  # pre-final-norm block output
    sources = [leaf] + [acts[l] for l in sums]
    cotangent = torch.zeros_like(target)
    batches = torch.arange(dim_batch, device=zs.device)
    for start in tqdm(range(0, dim, dim_batch), desc='Jacobian rows', leave=False):
        nd = min(dim_batch, dim - start)
        cotangent.zero_()
        cotangent[batches[:nd, None], valid[None, :], start + batches[:nd, None]] = 1
        grads = torch.autograd.grad(target, sources, grad_outputs=cotangent, retain_graph=start + dim_batch < dim)
        sums_emb[start:start + nd] += grads[0][:nd, thought_positions].float().mean(1).cpu()
        for i, layer in enumerate(sums): sums[layer][start:start + nd] += grads[i + 1][:nd, valid].float().mean(1).cpu()
    return target[0, -1].detach()


def atomic_save(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def fit(args):
    model = load_organism(args)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    problems = read('artifacts/lens_fit_problems.json')
    args.out.mkdir(parents=True, exist_ok=True)
    resume_path = args.out / 'lens_resume.pt'
    if args.resume:
        state = torch.load(resume_path, map_location='cpu', weights_only=True)
        assert state['organism'] == read('manifest.json')['organism'] and state['dim_batch'] == args.dim_batch
        torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        state = {'sums': {l: torch.zeros(4096, 4096) for l in range(35)}, 'sums_emb': torch.zeros(4096, 4096), 'count': 0, 'next_problem': 0, 'selected': [], 'organism': read('manifest.json')['organism'], 'dim_batch': args.dim_batch}
    for i in tqdm(range(state['next_problem'], len(problems)), desc='Fit problems'):
        problem = problems[i]
        ids, zs, acts, answer = record(model, problem)
        values = numbers(answer)
        eligible = bool(values) and values[0] == problem['gold'] and bool(problem['chain'])
        if eligible:
            target = jacobian_rows(model.codi.get_base_model(), ids, zs, state['sums'], state['sums_emb'], args.dim_batch)
            replay = model.codi.get_base_model().model.norm(target)
            cosine = torch.nn.functional.cosine_similarity(replay.float(), acts[-1, -1].float(), dim=0).item()
            assert cosine > .98, f'Replay mismatch at {problem["ds_index"]}: {cosine}'
            state['selected'].append(problem['ds_index']); state['count'] += 1
        state['next_problem'] = i + 1
        state['rng'] = torch.get_rng_state(); state['cuda_rng'] = torch.cuda.get_rng_state_all()
        atomic_save(state, resume_path)
        print({'ds_index': problem['ds_index'], 'answer': answer, 'eligible': eligible, 'count': state['count']}, flush=True)
        if state['count'] == args.n_fit: break
    assert state['count'] == args.n_fit, 'Supplied fit cohort exhausted before requested count.'
    atomic_save({'J': {l: matrix / state['count'] for l, matrix in state['sums'].items()}, 'J_emb': state['sums_emb'] / state['count'], 'meta': {'ckpt': 'checkpoint-7580', 'base': 'Qwen/Qwen3-8B', 'organism': state['organism'], 'selected': state['selected'], 'n_problems': state['count'], 'offset': 5200, 'estimator': 'paper reduction, target=block-35 output', 'valid_positions': 'Enc(bocot)+t1..t6; J_emb over t1..t6 only', 'feedback': 'recorded thought inputs held fixed'}}, args.out / 'jacobian_lens.pt')


def capture(args):
    model = load_organism(args)
    problems = read('artifacts/lens_problems.json')
    args.out.mkdir(parents=True, exist_ok=True)
    n = args.limit if args.limit else len(problems)
    original = np.load(args.acts, mmap_mode='r') if args.acts else None
    output = np.lib.format.open_memmap(args.out / 'lens_activations.npy', mode='w+', dtype=np.float16, shape=(n, 7, 37, 4096))
    rows = []
    recaptured = []
    for i, problem in enumerate(tqdm(problems[:n], desc='Lens capture')):
        _, _, acts, answer = record(model, problem, max_tokens=4)
        output[i] = acts.float().cpu().numpy().astype(np.float16)
        entry = {'ds_index': problem['ds_index'], 'answer': answer}
        if original is not None:
            entry['mean_absolute_error'] = float(np.abs(output[i].astype(np.float32) - original[i].astype(np.float32)).mean())
            entry['cosine_h36'] = torch.nn.functional.cosine_similarity(torch.tensor(output[i, :, -1].astype(np.float32)), torch.tensor(original[i, :, -1].astype(np.float32))).tolist()
        rows.append(entry)
        values = numbers(answer)
        prediction = values[0] if values else ''
        recaptured.append({**problem, 'pred': prediction, 'correct': prediction == problem['gold']})
        output.flush(); write(args.out / 'capture_records.json', rows)
        write(args.out / 'lens_problems.json', recaptured)
    print(args.out / 'lens_activations.npy', flush=True)


def run(args):
    torch.set_num_threads(4)
    if args.action == 'readouts': readouts(args)
    elif args.action == 'fit': fit(args)
    elif args.action == 'audit': audit(args)
    else: capture(args)
