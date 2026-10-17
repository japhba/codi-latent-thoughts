"""Judge fixed-source lens candidates pooled across all six thought positions."""
import asyncio
import json
import re
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .analysis import ROOT, canonical_number, read, write

METHODS = ['LogitLens', 'J-lens']
AO_HIDDEN_STATES = [0, 9, 18, 27, 35, 36]
EXTRACTION_PROMPT = '''Read the decoded token candidates below and list the exact numeric values explicitly expressed by individual candidates. You are not given an answer or target values.

Each JSON-array entry is ONE INDEPENDENT TOKEN CANDIDATE. Array entries and different lines are NOT consecutive output tokens. NEVER combine tokens, digits, words or symbols from different entries, even if they could form a familiar number. Never invent arithmetic. A claimed value must be expressed entirely by ONE quoted candidate.

Accept complete numeric literals, spelled-out cardinal numbers in any language, explicit fractions, and an arithmetic expression entirely contained in one candidate. A dozen means 12; hundred means 100; 六个 means 6; 四十 means 40. Numeric units alone (kilometer, 千米), vague plurals (hundreds), idioms, dates and years, ordinal words, programming identifiers, truncated word fragments, and generic words such as total or times do not express an exact numeric value. Do not convert units. If a candidate contains a different complete number, never take a digit substring. In particular, separate candidates 1 and 2 do not express 12; separate candidates fifty and five do not express 55; and thirty does not express 3.

Copy each matching candidate EXACTLY as shown (whitespace is already trimmed), and provide its value as a JSON NUMBER, without units or words. Write fractions as decimals. List each distinct candidate/value pair once. Return strict JSON with keys values (a list of objects, each with token and value), and justification (a short description of the extraction). If none of the individual candidates expresses a value, values must be an empty list. Do not include possible or guessed values.

Candidates:
{gen}'''


def numbers(text):
    return set(map(canonical_number, re.findall(r'-?\d[\d,]*(?:\.\d+)?', str(text))))


def intermediates(problem):
    return set(map(canonical_number, problem['intermediates'])) - {canonical_number(problem['answer'])}


def capture(args):
    import torch
    from .models import load_organism
    torch.set_num_threads(4)
    problems = read('artifacts/ao_curves.json')['problems']
    model = load_organism(args)
    tokenizer = model.tokenizer
    vectors = []
    with torch.no_grad():
        for start in tqdm(range(0, len(problems), args.batch_size), desc='AO six-position capture'):
            batch = problems[start:start + args.batch_size]
            prompt = tokenizer([p['question'] for p in batch], return_tensors='pt', padding='longest', padding_side='left').to('cuda')
            ids = torch.cat([prompt.input_ids, torch.tensor([[tokenizer.eos_token_id, tokenizer.bot_id]] * len(batch), device='cuda')], 1)
            mask = torch.cat([prompt.attention_mask, torch.ones(len(batch), 2, dtype=torch.long, device='cuda')], 1)
            out = model.codi(input_ids=ids, attention_mask=mask, use_cache=True, output_hidden_states=True)
            thoughts = []
            for _ in range(6):
                z = model.prj(out.hidden_states[-1][:, -1:]).to(model.codi.dtype)
                mask = torch.cat([mask, torch.ones(len(batch), 1, dtype=mask.dtype, device='cuda')], 1)
                out = model.codi(inputs_embeds=z, attention_mask=mask, past_key_values=out.past_key_values, use_cache=True, output_hidden_states=True)
                thoughts.append(torch.stack([z[:, 0]] + [out.hidden_states[h][:, -1] for h in AO_HIDDEN_STATES[1:]], 1).float().cpu().numpy().astype(np.float16))
            vectors.extend(np.stack(thoughts, axis=1))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / 'curve_lens_ao_ensemble_vectors.npz', uids=np.array([p['uid'] for p in problems]), hidden_states=np.array(AO_HIDDEN_STATES), vectors=np.array(vectors))
    write(args.out / 'curve_lens_ensemble_capture.json', {'organism': read('manifest.json')['organism'], 'prompt': 'raw question + EOS + BOT', 'batch_size': args.batch_size, 'cached_padding_mask': True, 'torch': torch.__version__, 'stored_dtype': 'float16', 'positions': list(range(1, 7)), 'hidden_states': AO_HIDDEN_STATES})


def readouts(args):
    import torch
    from transformers import AutoTokenizer
    from .lenses import decode, file_hash, transport, weight_hash
    from .models import fingerprint, prepare_cache, snapshot
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    prepare_cache(args.cache_dir)
    spec = read('manifest.json')['organism']
    checkpoint = args.organism_dir if args.organism_dir else snapshot(spec, args.cache_dir)
    fingerprint(checkpoint / 'pytorch_model.bin', spec['weights_sha256'])
    provenance = read('artifacts/lens_provenance.json')
    digest = file_hash(args.lens)
    assert digest == provenance['jacobian_sha256']
    lens = torch.load(args.lens, mmap=True, map_location='cpu', weights_only=True)
    assert lens['meta'] == provenance['fit_metadata']
    state = torch.load(checkpoint / 'pytorch_model.bin', mmap=True, map_location='cpu', weights_only=True)
    head = state['codi.base_model.model.lm_head.weight'].float().to('cuda')
    norm = state['codi.base_model.model.model.norm.weight'].float().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    ao = np.load(args.vectors)
    assert ao['hidden_states'].tolist() == AO_HIDDEN_STATES
    assert ao['vectors'].shape == (48, 6, 6, 4096)
    assert ao['uids'].tolist() == [p['uid'] for p in read('artifacts/ao_curves.json')['problems']]
    datasets = {}
    for src, hs in [('h24', 24), ('z', 0)]:
        data = np.load(ROOT / f'artifacts/vectors_{src}.npz')
        datasets[src] = {'uids': data['uids'].tolist(), 'hidden_states': [hs], 'vectors': data['vectors'][:, :, None]}
    datasets['ao'] = {'uids': ao['uids'].tolist(), 'hidden_states': AO_HIDDEN_STATES, 'vectors': ao['vectors']}
    matrices, streams = {}, {}
    for src, data in tqdm(datasets.items(), desc='Lens cohorts'):
        output = {method: {uid: [] for uid in data['uids']} for method in METHODS}
        for column, hs in tqdm(list(enumerate(data['hidden_states'])), desc=src, leave=False):
            jacobian = None if hs == 36 else (lens['J_emb'] if hs == 0 else lens['J'][hs - 1])
            if jacobian is not None:
                matrices[str(hs)] = weight_hash(jacobian)
                jacobian = jacobian.to('cuda')
                assert torch.isfinite(jacobian).all()
            vectors = np.ascontiguousarray(data['vectors'][:, :, column]).reshape(-1, 4096)
            for start in range(0, len(vectors), args.readout_batch_size):
                h = torch.tensor(vectors[start:start + args.readout_batch_size], dtype=torch.float32, device='cuda')
                assert torch.isfinite(h).all()
                for method in METHODS:
                    transformed = h if method == 'LogitLens' or hs == 36 else transport(h, jacobian)
                    logp = decode(transformed, head, norm, post_norm=hs == 36).log_softmax(-1)
                    values, ids = logp.topk(5, -1)
                    for offset, (token_ids, token_logp) in enumerate(zip(ids.cpu().tolist(), values.cpu().tolist())):
                        row, thought = divmod(start + offset, 6)
                        output[method][data['uids'][row]].append({'hidden_state': hs, 'thought': thought + 1, 'top_ids': token_ids, 'top_tokens': [tokenizer.decode([i]) for i in token_ids], 'top_logp': token_logp})
            del jacobian
        streams[src] = {'uids': data['uids'], 'hidden_states': data['hidden_states'], 'methods': output}
    # The historical fit did not save accepted IDs: conservatively report candidate overlap.
    fit_candidates = {p['ds_index'] for p in read('artifacts/lens_fit_problems.json')}
    overlaps = {src: [u for u in data['uids'] if int(u[1:]) in fit_candidates] for src, data in streams.items()}
    counts = {src: sum(a['top_ids'] != b['top_ids'] for uid in data['uids'] for a, b in zip(data['methods']['LogitLens'][uid], data['methods']['J-lens'][uid])) for src, data in streams.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / 'curve_lens_ensemble_readouts.json', {'organism': spec, 'streams': streams, 'jacobian_sha256': digest, 'matrix_sha256_by_hidden_state': matrices, 'fit_metadata': lens['meta'], 'fit_candidate_overlap': overlaps, 'different_top5_lists': counts, 'readout': {'dtype': 'float32; TF32 disabled', 'k': 5, 'positions': list(range(1, 7)), 'normalization': 'final norm once after transport; h36 directly to head', 'z': 'exact projected fed-back vector, J_emb transport', 'location_selection': 'none: fixed reader source(s), all six thoughts', 'nla_vectors': 'exact original saved reader-cohort vectors', 'ao_vectors': 'fresh captured reader-cohort vectors'}})


def ensembles(root=ROOT):
    data = read('artifacts/curve_lens_ensemble_readouts.json', root)
    assert data['organism'] == read('manifest.json', root)['organism']
    output = []
    for src, stream, hidden_states in [('h24', 'h24', [24]), ('z', 'z', [0]), ('ao_ALL', 'ao', AO_HIDDEN_STATES), ('ao_Z', 'ao', [0])]:
        cohort = data['streams'][stream]
        for method in METHODS:
            for uid in cohort['uids']:
                candidates = [r for r in cohort['methods'][method][uid] if r['hidden_state'] in hidden_states]
                assert {(r['hidden_state'], r['thought']) for r in candidates} == {(h, t) for h in hidden_states for t in range(1, 7)}
                # Labels carry no numbers and cannot introduce false value matches.
                gen = '\n'.join(json.dumps([token.strip() for token in r['top_tokens']], ensure_ascii=False) for r in candidates)
                output.append({'src': src, 'uid': uid, 'method': method, 'gen': gen})
    return output


def tasks(root=ROOT):
    problems = {p['uid']: p for p in read('artifacts/problems.json', root) + read('artifacts/ao_curves.json', root)['problems']}
    output = []
    for row in ensembles(root):
        problem = problems[row['uid']]
        for criterion, gold in [('answer', {canonical_number(problem['answer'])}), ('intermediate', intermediates(problem))]:
            if gold: output.append({'src': row['src'], 'uid': row['uid'], 't': 0, 'kind': f"{row['method']}:{criterion}", 'gen': row['gen'], 'gold': sorted(gold)})
    return output


def adjudicate(extraction, root=ROOT):
    corrections = read('artifacts/curve_lens_extraction_review.json', root)['corrections']
    values = set()
    for item in extraction['values']:
        token = item['token']
        assigned = corrections[token]['values'] if token in corrections else [item['value']]
        values.update((token, value) for value in assigned)
    return [{'token': token, 'value': value} for token, value in sorted(values)]


def verdicts(extractions, root=ROOT):
    extracted = {(r['src'], r['uid'], r['method']): r for r in extractions}
    results = []
    for row in tasks(root):
        method, _ = row['kind'].split(':')
        extraction = extracted[(row['src'], row['uid'], method)]
        matches = [v for v in adjudicate(extraction, root) if v['value'] in row['gold']]
        results.append({k: row[k] for k in ['src', 'uid', 't', 'kind']} | {'matched': bool(matches), 'matches': matches, 'model': extraction['model'], 'justification': 'Exact gold intersection of independently extracted values, with the saved gold-independent token-meaning audit.', 'raw_judge_response': extraction['raw_judge_response']})
    return results


async def classify(args):
    import httpx
    from dotenv import load_dotenv
    from .judge import complete
    load_dotenv(Path.home() / '.env')
    load_dotenv()
    rows = tasks()
    pooled = ensembles()
    model = read('manifest.json')['judge_model']
    semaphore = asyncio.Semaphore(args.concurrency)
    extractions = read(args.out / 'curve_lens_ensemble_extractions.json') if args.resume else []
    assert [(r['src'], r['uid'], r['method']) for r in extractions] == [(r['src'], r['uid'], r['method']) for r in pooled[:len(extractions)]]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'raw_extractions').mkdir(exist_ok=True)
    write(args.out / 'curve_lens_ensemble_judge_inputs.json', rows)

    async def extract(client, row):
        tokens = set.union(set(), *(set(json.loads(line)) for line in row['gen'].splitlines()))
        schema = {'type': 'object', 'properties': {'values': {'type': 'array', 'items': {'type': 'object', 'properties': {'token': {'type': 'string'}, 'value': {'type': 'number'}}, 'required': ['token', 'value'], 'additionalProperties': False}}, 'justification': {'type': 'string'}}, 'required': ['values', 'justification'], 'additionalProperties': False}
        raw = await complete(client, semaphore, EXTRACTION_PROMPT.format(gen=row['gen']), model, 8192, {'type': 'json_schema', 'json_schema': {'name': 'explicit_values', 'strict': True, 'schema': schema}})
        write(args.out / 'raw_extractions' / f"{row['src']}_{row['uid']}_{row['method']}.json", {'input': row, 'raw_judge_response': raw})
        result = json.loads(raw)
        assert isinstance(result['values'], list) and isinstance(result['justification'], str)
        for item in result['values']:
            assert item['token'] and item['token'] in tokens, f'Invented token citation: {item}'
            item['value'] = canonical_number(item['value'])
            literal = re.fullmatch(r'\$?(-?[\d,]+(?:\.\d+)?)(?:%)?', item['token'])
            if literal: assert item['value'] == canonical_number(literal.group(1)), f'Value contradicts numeric literal: {item}'
        result['values'] = [{'token': token, 'value': value} for token, value in sorted({(v['token'], v['value']) for v in result['values']})]
        return {'src': row['src'], 'uid': row['uid'], 'method': row['method'], **result, 'model': model, 'raw_judge_response': raw}

    async with httpx.AsyncClient() as client:
        for start in tqdm(range(len(extractions), len(pooled), args.concurrency), desc='Extract explicit pooled values'):
            batch = pooled[start:start + args.concurrency]
            extractions.extend(await asyncio.gather(*(extract(client, row) for row in batch)))
            write(args.out / 'curve_lens_ensemble_extractions.json', extractions)
    write(args.out / 'curve_lens_ensemble_verdicts.json', verdicts(extractions))


def run(args):
    if args.action == 'capture': capture(args)
    elif args.action == 'readouts': readouts(args)
    else: asyncio.run(classify(args))
