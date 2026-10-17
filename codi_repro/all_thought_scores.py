"""Gold-blind explicit-value extraction for six-thought NLA ensembles."""
import argparse
import asyncio
import hashlib
import html
import json
from pathlib import Path
import re

import httpx
from dotenv import load_dotenv
from tqdm.auto import tqdm

from .analysis import canonical_number, read, write
from .curve_lenses import intermediates, numbers
from .judge import complete

PROMPT = '''List complete numeric values explicitly expressed in these independent descriptions. You are not given the question or any target values.
Never join fragments from separate descriptions. Accept exact numeric literals, spelled-out cardinal numbers, explicit fractions, and arithmetic expressions whose value is unambiguous within one description. A dozen expresses 12. Report the numeral of a percentage: 12% expresses 12, not 0.12. Do not infer implied quantities, round values, convert units, take digit substrings, or count dates, years, ordinals, identifiers or vague plurals. An expression such as 45 divided by 3 expresses 15 as well as its explicit operands. Copy a short exact quote containing each expressed value from ONE description, preserving its spelling and whitespace. Return each distinct quote/value pair once. Values must be JSON numbers, with fractions represented as decimals. Return strict JSON with values (objects with quote and value) and justification (a short extraction explanation). An empty values list is allowed.

List each distinct numeric VALUE only once, using its shortest sufficient exact evidence quote. Do not repeat the same value for multiple quotations. For arithmetic results, use at most six significant digits; never expand a repeating decimal indefinitely.

Descriptions:
{gen}'''


def pools(inputs):
    result = []
    for folder in inputs:
        paths = list(folder.glob('reader_*.jsonl'))
        assert len(paths) == 1, 'Score the completed, consolidated reader output.'
        records = [json.loads(line) for line in paths[0].read_text().splitlines()]
        groups = {}
        for row in records:
            group_key = (row['src'], row['step'], row['uid'], row['kind'])
            group = groups.setdefault(group_key, {})
            assert row['thought'] not in group
            group[row['thought']] = row['gen']
        for (src, step, uid, kind), group in sorted(groups.items()):
            assert set(group) == set(range(1, 7)), 'Incomplete thought ensemble.'
            descriptions = [group[t] for t in range(1, 7)]
            digest = hashlib.sha256(json.dumps(descriptions, ensure_ascii=False).encode()).hexdigest()
            result.append({'src': src, 'step': step, 'uid': uid, 'kind': kind, 'descriptions': descriptions, 'input_sha256': digest})
    assert len({key(r) for r in result}) == len(result)
    return result


def key(row):
    return row['src'], row['step'], row['uid'], row['kind']


def expressed_values(row):
    numeric = r'[$€£]?[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?'
    complete = {canonical_number(m.group().lstrip('$€£+')) for text in row['descriptions'] for m in re.finditer(r'(?<!\w)' + numeric, text)}
    rejected = [item for item in row['values'] if re.fullmatch(numeric, item['quote']) and item['value'] not in complete]
    return {item['value'] for item in row['values'] if item not in rejected}, rejected


async def classify(args):
    load_dotenv(Path.home() / '.env')
    load_dotenv()
    args.out.mkdir(parents=True, exist_ok=True)
    folder = args.out / 'extractions'
    folder.mkdir(exist_ok=True)
    records = pools(args.inputs)
    model = read('manifest.json')['judge_model']
    semaphore = asyncio.Semaphore(args.concurrency)
    schema = {'type': 'object', 'properties': {'values': {'type': 'array', 'items': {'type': 'object', 'properties': {'quote': {'type': 'string'}, 'value': {'type': 'number'}}, 'required': ['quote', 'value'], 'additionalProperties': False}}, 'justification': {'type': 'string'}}, 'required': ['values', 'justification'], 'additionalProperties': False}

    async def extract(client, row):
        path = folder / ('_'.join(map(str, key(row))) + '.json')
        if args.resume and path.is_file():
            saved = json.loads(await asyncio.to_thread(path.read_text))
            assert saved['input_sha256'] == row['input_sha256'] and saved['model'] == model
            return saved
        assert not path.exists(), 'Use --resume for existing extractions.'
        prompt = PROMPT.format(gen=json.dumps(row['descriptions'], ensure_ascii=False))
        raw_folder = args.out / 'raw_extractions'
        raw_folder.mkdir(exist_ok=True)
        attempts = []
        for attempt in range(3):
            raw = await complete(client, semaphore, prompt, model, 8192, {'type': 'json_schema', 'json_schema': {'name': 'description_values', 'strict': True, 'schema': schema}})
            attempts.append({'prompt': prompt, 'raw_response': raw})
            await asyncio.to_thread(write, raw_folder / path.name, {'input': row, 'attempts': attempts})
            result = json.loads(raw)
            for item in result['values']:
                quote = item['quote']
                decoded = html.unescape(quote)
                if not any(quote in text for text in row['descriptions']) and any(decoded in text for text in row['descriptions']):
                    item['quote'] = decoded
            invalid = [item for item in result['values'] if not item['quote'] or not any(item['quote'] in text for text in row['descriptions'])]
            if not invalid: break
            prompt = PROMPT.format(gen=json.dumps(row['descriptions'], ensure_ascii=False)) + '\nYour previous response cited spans absent from the descriptions: ' + json.dumps(invalid, ensure_ascii=False) + '\nReturn a corrected complete extraction. Copy exact substrings without adding quotation marks to the quote field.'
        assert not invalid, f'Invalid citations after correction: {invalid}'
        for item in result['values']:
            quote = item['quote']
            assert quote and any(quote in text for text in row['descriptions']), f'Invented evidence: {item}'
            item['value'] = canonical_number(item['value'])
            literal = re.fullmatch(r'\$?(-?[\d,]+(?:\.\d+)?)(?:%)?', quote)
            # The lens audit likewise parses complete literals deterministically.
            if literal: item['value'] = canonical_number(literal.group(1))
        saved = {**row, **result, 'model': model, 'raw_response': raw, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        partial = path.with_suffix('.partial')
        await asyncio.to_thread(write, partial, saved)
        partial.replace(path)
        return saved

    extracted = []
    async with httpx.AsyncClient() as client:
        tasks = [extract(client, row) for row in records]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='Gold-blind six-thought extraction'):
            extracted.append(await task)
    summarize(extracted, args.out)


def summarize(extracted, out):
    extracted = sorted(extracted, key=key)
    problems = {p['uid']: p for p in read('artifacts/problems.json')}
    groups, judgments, boundary_review = {}, [], []
    for row in extracted:
        values, rejected = expressed_values(row)
        if rejected: boundary_review.append({'key': list(key(row)), 'rejected_partial_numeric_evidence': rejected})
        p = problems[row['uid']]
        judged = {k: row[k] for k in ['src', 'step', 'uid', 'kind', 'input_sha256']}
        judged['values'] = sorted(values)
        judged['answer'] = canonical_number(p['answer']) in values
        targets = intermediates(p)
        judged['intermediate'] = bool(targets & values) if targets else None
        novel = targets - numbers(p['question'])
        judged['novel_intermediate'] = bool(novel & values) if novel else None
        judgments.append(judged)
        groups.setdefault((row['src'], row['step'], row['kind']), []).append(judged)
    scores = []
    for (src, step, kind), rows in sorted(groups.items()):
        score = {'src': src, 'step': step, 'kind': kind}
        for metric in ['answer', 'intermediate', 'novel_intermediate']:
            eligible = [r for r in rows if r[metric] is not None]
            targets = [{canonical_number(problems[r['uid']]['answer'])} if metric == 'answer' else intermediates(problems[r['uid']]) - (numbers(problems[r['uid']]['question']) if metric == 'novel_intermediate' else set()) for r in eligible]
            permutation = sum(bool(set(r['values']) & target) for r in eligible for target in targets) / len(eligible) ** 2
            score[metric] = {'correct': sum(r[metric] for r in eligible), 'n': len(eligible), 'rate': sum(r[metric] for r in eligible) / len(eligible), 'semantic_permutation': permutation}
        scores.append(score)
    write(out / 'all_thought_judgments.json', judgments)
    write(out / 'numeric_boundary_review.json', {'criterion': 'Bare numeric quotes require a complete numeric occurrence with the same canonical value in a description; a prefix of a different number is insufficient.', 'rejected': boundary_review})
    write(out / 'all_thought_scores.json', {'scores': scores, 'criterion': 'Gold-blind complete explicit values pooled across six independent descriptions; same gold sets as lens ensembles.', 'positions': list(range(1, 7)), 'model': read('manifest.json')['judge_model']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=12)
    parser.add_argument('--resume', action='store_true')
    asyncio.run(classify(parser.parse_args()))
