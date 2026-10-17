"""LLM solve-verification of the mined load-bearing twin candidates: an independent
solver must reproduce the mechanically repropagated cf_twin, which catches hidden
branches (max/comparison problems), edits that make the problem ill-posed, and any
repropagation slip. Judge: google/gemini-3-flash-preview via OpenRouter, temperature
0, one sample per candidate, 32-way concurrency with exponential backoff.

reports/codi_writeup/lb_twins_cands.jsonl -> reports/codi_writeup/lb_twins.jsonl
(verified rows, llm answer logged; mismatches printed in full)
"""
from __future__ import annotations

import asyncio
import json
import os
import re

import httpx

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_F = os.path.join(_REPO, "reports", "codi_writeup", "lb_twins_cands.jsonl")
OUT_F = os.path.join(_REPO, "reports", "codi_writeup", "lb_twins.jsonl")
MODEL = "google/gemini-3-flash-preview"
KEY = os.environ["OPENROUTER_API_KEY"]
SEM = asyncio.Semaphore(32)

PROMPT = ("Solve this grade-school math problem. Reason step by step, then give the "
          "final numeric answer on its own line after '####'.\n\n{q}")


def last_number(text):
    tail = text.split("####")[-1].replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+\.?\d*", tail)
    return float(m[0]) if m else None


async def solve(client, q):
    for attempt in range(6):
        try:
            async with SEM:
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {KEY}"},
                    json=dict(model=MODEL, temperature=0.0, max_tokens=2048,
                              messages=[{"role": "user", "content": PROMPT.format(q=q)}]),
                    timeout=120)
            r.raise_for_status()
            return last_number(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            if attempt == 5:
                raise
            await asyncio.sleep(2 ** attempt)


async def main():
    cands = [json.loads(l) for l in open(IN_F)]
    print(f"verifying {len(cands)} candidates with {MODEL}")
    async with httpx.AsyncClient() as client:
        answers = await asyncio.gather(*[solve(client, c["twin_question"]) for c in cands])
    kept, mism = [], 0
    for c, a in zip(cands, answers):
        if a is not None and abs(a - float(c["cf_twin"])) < 1e-6:
            kept.append(dict(c, llm_answer=a))
        else:
            mism += 1
            print(f"MISMATCH {c['doc']} g {c['g']}->{c['g_twin']}: "
                  f"mechanical {c['cf_twin']} vs llm {a}\n  TQ: {c['twin_question']}")
    print(f"kept {len(kept)}/{len(cands)} ({mism} mismatches), "
          f"{len({c['doc'] for c in kept})} distinct problems")
    with open(OUT_F, "w") as f:
        for c in kept:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {OUT_F}")


if __name__ == "__main__":
    asyncio.run(main())
