import asyncio
import hashlib
import json
import os

import httpx
from dotenv import load_dotenv
from tqdm.auto import tqdm

from .analysis import canonical_number, read, write

PROMPT = '''You are checking whether a model-generated text mentions any of a list of gold numeric values.

Text:
{gen}

Gold values: {gold}

Does the text semantically express AT LEAST ONE of the gold values? Count exact numerals in any formatting ("1,200", "$12", "12.0", "12%"), spelled-out numbers ("twelve", "a dozen"), and arithmetic expressions whose result is unambiguously a gold value ("45 divided by 3" matches 15). Do not count a value that only appears as a substring of a different number (15 inside 150 does not match), do not count digits inside a date, year, or year range (2010-2012 does not match 12), and do not count a different quantity that merely rounds to a gold value.

Reply with strict JSON only: {{"matched": true, "justification": "<one short sentence naming the matching span or the closest miss, without any quotation marks>"}}'''


async def complete(client, semaphore, prompt, model, max_tokens, response_format):
    for attempt in range(9):
        try:
            async with semaphore:
                async with asyncio.timeout(120):
                    response = await client.post('https://openrouter.ai/api/v1/chat/completions', headers={'Authorization': f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, json={'model': model, 'temperature': 0., 'max_tokens': max_tokens, 'response_format': response_format, 'provider': {'require_parameters': True}, 'messages': [{'role': 'user', 'content': prompt}]}, timeout=120)
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            response.raise_for_status()
            choice = response.json()['choices'][0]
            if choice['finish_reason'] == 'length':
                print(json.dumps({'retry': 'length', 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'attempt': attempt, 'usage': response.json()['usage']}), flush=True)
                assert attempt < 8, choice
                max_tokens = min(max_tokens * 2, 32768)
                await asyncio.sleep(min(2 ** attempt, 30))
                continue
            if choice['finish_reason'] == 'error':
                code = choice['error']['code']
                if code == 429 or code >= 500:
                    assert attempt < 8, choice['error']
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
            assert choice['finish_reason'] == 'stop', choice
            return choice['message']['content'].strip()
        except (TimeoutError, httpx.TransportError, httpx.HTTPStatusError) as error:
            print(json.dumps({'retry': type(error).__name__, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'attempt': attempt}), flush=True)
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code not in [429, 500, 502, 503, 504]: raise
            if attempt == 8: raise
            await asyncio.sleep(min(2 ** attempt, 30))


async def judge(client, semaphore, row, gold, model):
    prompt = PROMPT.format(gen=row['gen'].strip(), gold=', '.join(gold))
    raw = await complete(client, semaphore, prompt, model, 512, {'type': 'json_object'})
    verdict = json.loads(raw)
    assert isinstance(verdict['matched'], bool)
    return {k: row[k] for k in ['src', 'uid', 't', 'kind']} | verdict | {'model': model, 'raw_judge_response': raw}


async def main(args):
    load_dotenv()
    load_dotenv(os.path.expanduser('~/.env'))
    problems = {r['uid']: r for r in read('artifacts/problems.json')}
    rows = json.loads(args.readouts.read_text())
    tasks = []
    for r in rows:
        problem = problems[r['uid']]
        gold = sorted(set(canonical_number(v) for v in problem['intermediates']) - {canonical_number(problem['answer'])})
        if gold: tasks.append((r, gold))
    semaphore = asyncio.Semaphore(args.concurrency)
    model = read('manifest.json')['judge_model']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    async with httpx.AsyncClient() as client:
        for start in tqdm(range(0, len(tasks), args.concurrency), desc='Value-match classification'):
            batch = tasks[start:start + args.concurrency]
            results.extend(await asyncio.gather(*(judge(client, semaphore, r, g, model) for r, g in batch)))
            write(args.out, results)


def run(args):
    asyncio.run(main(args))
