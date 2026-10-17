"""Gold-blind single-candidate semantics, reused across every layer ensemble."""
import argparse
import asyncio
import gzip
import json
import re
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .analysis import ROOT, canonical_number, read, write
from .curve_lenses import AO_HIDDEN_STATES, EXTRACTION_PROMPT, METHODS, intermediates

LAYER_ACCESS = {
    'reader_layers': {'h24': [24], 'z': [0], 'ao_ALL': AO_HIDDEN_STATES, 'ao_Z': [0]},
    'all_layers': {src: list(range(37)) for src in ['h24', 'z', 'ao_ALL']},
}


def pool_candidates(records, hidden_states):
    candidates = [r for r in records if r['hidden_state'] in hidden_states]
    coordinates = [(r['hidden_state'], r['thought']) for r in candidates]
    assert len(coordinates) == len(set(coordinates)) == len(hidden_states) * 6
    assert set(coordinates) == {(h, t) for h in hidden_states for t in range(1, 7)}
    return sorted({t.strip() for r in candidates for t in r['top_tokens']} - {''})


def literal(token):
    match = re.fullmatch(r'[$€£]?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)(?:[%\.])?', token)
    return canonical_number(match.group(1)) if match else None


def ensembles(root=ROOT, *, scope='all_layers'):
    data = read('artifacts/all_layer_lens_readouts.json', root)
    assert data['organism'] == read('manifest.json', root)['organism']
    cohorts = {src: np.load(root / f'artifacts/vectors_{src}.npz')['uids'].tolist() for src in ['h24', 'z']}
    cohorts['ao_ALL'] = [p['uid'] for p in read('artifacts/ao_curves.json', root)['problems']]
    cohorts['ao_Z'] = cohorts['ao_ALL']
    output = []
    for src, hidden_states in LAYER_ACCESS[scope].items():
        for method in METHODS:
            for uid in cohorts[src]:
                records = data['methods'][method][uid]
                assert {(r['hidden_state'], r['thought']) for r in records} == {(h, t) for h in range(37) for t in range(1, 7)}
                tokens = pool_candidates(records, hidden_states)
                output.append({'src': src, 'uid': uid, 'method': method, 'tokens': tokens, 'gen': json.dumps(tokens, ensure_ascii=False)})
    return output


def dictionary(root=ROOT):
    data = read('artifacts/all_layer_token_extractions.json', root)
    values = {token: [] for token in data['tokens']}
    for token in values:
        numeric = literal(token)
        if numeric is not None: values[token] = [numeric]
    corrections = read('artifacts/curve_lens_extraction_review.json', root)['corrections'] | read('artifacts/all_layer_token_review.json', root)['corrections']
    for row in data['batches']:
        for item in row['values']:
            token, value = item['token'], item['value']
            assert token in row['tokens'] and value == canonical_number(value)
            values[token].append(value)
    for token in corrections:
        if token in values: values[token] = corrections[token]['values']
    return {token: sorted(set(numeric)) for token, numeric in values.items()}


def values_for(tokens, semantics):
    return set.union(set(), *(set(semantics[token]) for token in tokens))


def scores(root=ROOT, *, scope='all_layers'):
    data = read('artifacts/all_layer_lens_readouts.json', root)
    semantics = dictionary(root)
    problems = {p['uid']: p for p in read('artifacts/problems.json', root) + read('artifacts/ao_curves.json', root)['problems']}
    rows = ensembles(root, scope=scope)
    fit_ids = set(data['fit_metadata']['selected'] + data['fit_metadata']['validation_ids'])
    assert fit_ids.isdisjoint(p['idx'] for p in problems.values())
    output = {}
    for src in LAYER_ACCESS[scope]:
        output[src] = {}
        for method in METHODS:
            cohort = [r for r in rows if (r['src'], r['method']) == (src, method)]
            output[src][method] = {}
            for criterion in ['answer', 'intermediate']:
                gold = lambda uid: {canonical_number(problems[uid]['answer'])} if criterion == 'answer' else intermediates(problems[uid])
                eligible = [r for r in cohort if gold(r['uid'])]
                extracted = [values_for(r['tokens'], semantics) for r in eligible]
                matched = [bool(values & gold(r['uid'])) for values, r in zip(extracted, eligible)]
                shuffled = np.mean([bool(values & gold(other['uid'])) for values in extracted for other in eligible])
                output[src][method][criterion] = {'count': int(sum(matched)), 'n': len(matched), 'rate': float(np.mean(matched)), 'semantic_permutation_expected_rate': float(shuffled), 'literal_count': sum(bool({literal(t) for t in r['tokens']} & gold(r['uid'])) for r in eligible), 'k_per_position_source': 5}
    return output


def verdicts(root=ROOT, *, scope='all_layers'):
    semantics = dictionary(root)
    problems = {p['uid']: p for p in read('artifacts/problems.json', root) + read('artifacts/ao_curves.json', root)['problems']}
    output = []
    for row in ensembles(root, scope=scope):
        for criterion, gold in [('answer', {canonical_number(problems[row['uid']]['answer'])}), ('intermediate', intermediates(problems[row['uid']]))]:
            if not gold: continue
            matches = [{'token': token, 'value': value} for token in row['tokens'] for value in semantics[token] if value in gold]
            output.append({'src': row['src'], 'uid': row['uid'], 'kind': f"{row['method']}:{criterion}", 'matched': bool(matches), 'matches': matches, 'gold': sorted(gold)})
    return output


def verify(root=ROOT):
    data = read('artifacts/all_layer_lens_readouts.json', root)
    calibration = read('artifacts/layer_lens_fit.json', root)
    capture = read('artifacts/all_layer_capture.json', root)
    assert data['fit_metadata']['organism'] == data['organism'] == calibration['organism'] == capture['organism'] == read('manifest.json', root)['organism']
    assert data['jacobian_sha256'] == calibration['sha256']
    resumed = read('artifacts/all_layer_resume_validation.json', root)
    assert resumed['all_equal'] is True and resumed['identical_maps'] == 36 and resumed['calibration_n'] == calibration['n_problems']
    assert set(data['matrix_sha256_by_hidden_state']) == {str(h) for h in range(36)}
    assert len(set(data['matrix_sha256_by_hidden_state'].values())) == 36
    assert len(calibration['selected']) == calibration['n_problems']
    assert calibration['selected'] == capture['fit_ids'][:calibration['n_problems']]
    assert calibration['validation_ids'] == capture['validation_ids']
    assert set(calibration['selected']).isdisjoint(calibration['validation_ids'])
    question_lists = [read(f'artifacts/all_layer_{name}_problems.json', root) for name in ['fit', 'validation', 'evaluation']]
    assert [p['ds_index'] for p in question_lists[0]] == capture['fit_ids']
    assert [p['ds_index'] for p in question_lists[1]] == capture['validation_ids']
    for problem in question_lists[0] + question_lists[1]:
        assert problem['captured_correct'] is True
        assert canonical_number(re.findall(r'-?\d+(?:\.\d+)?', problem['captured_answer'])[0]) == canonical_number(problem['gold'])
    original_questions = {p['uid']: p for p in read('artifacts/problems.json', root) + read('artifacts/ao_curves.json', root)['problems']}
    for problem in question_lists[2]:
        assert all(problem[key] == original_questions[problem['uid']][key] for key in ['question', 'answer', 'intermediates', 'cot'])
    question_sets = [{' '.join(p['question'].split()) for p in rows} for rows in question_lists]
    assert question_sets[0].isdisjoint(question_sets[1])
    assert (question_sets[0] | question_sets[1]).isdisjoint(question_sets[2])
    assert data['uids'] == capture['evaluation_uids'] and len(data['uids']) == len(set(data['uids'])) == 156
    tokens = set()
    for uid in data['uids']:
        for method in METHODS:
            records = data['methods'][method][uid]
            assert len(records) == 222
            assert {(r['hidden_state'], r['thought']) for r in records} == {(h, t) for h in range(37) for t in range(1, 7)}
            for row in records:
                assert len(row['top_ids']) == len(row['top_tokens']) == len(row['top_logp']) == 5
                assert np.isfinite(row['top_logp']).all()
                tokens.update(t.strip() for t in row['top_tokens'] if t.strip())
        for a, b in zip(data['methods']['LogitLens'][uid], data['methods']['J-lens'][uid]):
            assert (a['hidden_state'], a['thought']) == (b['hidden_state'], b['thought'])
            if a['hidden_state'] == 36: assert a == b
    extraction = read('artifacts/all_layer_token_extractions.json', root)
    assert extraction['tokens'] == sorted(tokens)
    assert [t for row in extraction['batches'] for t in row['tokens']] == [t for t in sorted(tokens) if literal(t) is None]
    assert read('artifacts/all_layer_lens_verdicts.json', root) == verdicts(root)
    assert read('artifacts/reader_layer_lens_verdicts.json', root) == verdicts(root, scope='reader_layers')
    protocol = read('artifacts/curve_evaluation_protocol.json', root)
    cohorts = {src: np.load(root / f'artifacts/vectors_{src}.npz')['uids'].tolist() for src in ['h24', 'z']}
    cohorts['ao'] = [p['uid'] for p in read('artifacts/ao_curves.json', root)['problems']]
    for src, uids in cohorts.items():
        recorded = protocol['cohorts'][src]
        assert recorded['uids'] == uids and recorded['n'] == len(uids)
        assert recorded['intermediate_n'] == sum(bool(intermediates(original_questions[u])) for u in uids)
    assert protocol['shared_nla_uids'] == sorted(set(cohorts['h24']) & set(cohorts['z']))
    with gzip.open(root / 'artifacts/av_history.json.gz', 'rt') as handle: history = json.load(handle)
    for src in ['h24', 'z']:
        by_step = {}
        for row in history:
            if row['src'] == src: by_step.setdefault(row['step'], []).append(row['uid'])
        assert all(uids == cohorts[src] for uids in by_step.values())
    summary = scores(root)
    restricted = scores(root, scope='reader_layers')
    for src, methods in restricted.items():
        skyline_src = 'ao_ALL' if src == 'ao_Z' else src
        for method, criteria in methods.items():
            for criterion, value in criteria.items():
                skyline = summary[skyline_src][method][criterion]
                assert value['n'] == skyline['n'] and value['count'] <= skyline['count']
    return {'calibration_n': calibration['n_problems'], 'validation_n': len(calibration['validation_ids']), 'evaluation_n': len(data['uids']), 'fitted_maps': 36, 'top5_readouts': len(data['uids']) * 222 * 2, 'distinct_candidates': len(tokens), 'judge_batches': len(extraction['batches']), 'scores': summary, 'reader_layer_scores': restricted, 'layer_access': LAYER_ACCESS, 'shared_nla_questions': len(protocol['shared_nla_uids'])}


async def judge(args):
    import httpx
    from dotenv import load_dotenv
    from .judge import complete
    load_dotenv(Path.home() / '.env'); load_dotenv()
    data = json.loads(args.readouts.read_text())
    tokens = sorted({t.strip() for method in METHODS for records in data['methods'][method].values() for r in records for t in r['top_tokens']} - {''})
    # A candidate's meaning must be self-contained. Judge each distinct candidate
    # once, without its question, location, method, target or adjacent candidates.
    ambiguous = [token for token in tokens if literal(token) is None]
    batches = [ambiguous[start:start + 120] for start in range(0, len(ambiguous), 120)]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'all_layer_token_extractions.json'
    state = json.loads(path.read_text()) if args.resume else {'model': read('manifest.json')['judge_model'], 'tokens': tokens, 'batches': [], 'protocol': 'Deduplicated individual candidates; exact numeric literals parsed deterministically; remaining tokens judged gold-blind in independent-candidate batches. No cross-candidate composition.'}
    assert state['tokens'] == tokens
    semaphore = asyncio.Semaphore(args.concurrency)
    schema = {'type': 'object', 'properties': {'values': {'type': 'array', 'items': {'type': 'object', 'properties': {'token': {'type': 'string'}, 'value': {'type': 'number'}}, 'required': ['token', 'value'], 'additionalProperties': False}}, 'justification': {'type': 'string'}}, 'required': ['values', 'justification'], 'additionalProperties': False}
    async def extract(client, batch):
        prompt = EXTRACTION_PROMPT.format(gen=json.dumps(batch, ensure_ascii=False))
        raw = await complete(client, semaphore, prompt, state['model'], 8192, {'type': 'json_schema', 'json_schema': {'name': 'explicit_values', 'strict': True, 'schema': schema}})
        result = json.loads(raw)
        for item in result['values']:
            assert item['token'] in batch, f'Invented token citation: {item}'
            item['value'] = canonical_number(item['value'])
        return {'tokens': batch, **result, 'raw_judge_response': raw}
    async with httpx.AsyncClient() as client:
        for start in tqdm(range(len(state['batches']), len(batches), args.concurrency), desc='Gold-blind candidate semantics'):
            state['batches'].extend(await asyncio.gather(*(extract(client, batch) for batch in batches[start:start + args.concurrency])))
            write(path, state)
    assert [row['tokens'] for row in state['batches']] == batches


def profile(out, root=ROOT):
    data = read('artifacts/all_layer_lens_readouts.json', root)
    semantics = dictionary(root)
    rows = ensembles(root)
    problems = {p['uid']: p for p in read('artifacts/problems.json', root) + read('artifacts/ao_curves.json', root)['problems']}
    records = {}
    for src in ['h24', 'ao_ALL']:
        records[src] = {}
        for method in METHODS:
            records[src][method] = {}
            cohort = [r for r in rows if (r['src'], r['method']) == (src, method)]
            for criterion in ['answer', 'intermediate']:
                gold = lambda uid: {canonical_number(problems[uid]['answer'])} if criterion == 'answer' else intermediates(problems[uid])
                eligible = [r for r in cohort if gold(r['uid'])]
                rates = []
                for hs in range(37):
                    matched = []
                    for row in eligible:
                        tokens = {token.strip() for r in data['methods'][method][row['uid']] if r['hidden_state'] == hs for token in r['top_tokens']} - {''}
                        matched.append(bool(values_for(tokens, semantics) & gold(row['uid'])))
                    rates.append(float(np.mean(matched)))
                records[src][method][criterion] = rates
    write(out / 'all_layer_lens_profile.json', records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readouts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    asyncio.run(judge(parser.parse_args()))


if __name__ == '__main__': main()
