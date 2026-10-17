"""an1 judged-trueness eval: does the AV's explanation verbalize the gold
cognitive state? OpenRouter (gemini-3-flash) judge, 0.0-1.0 in 0.1 steps with
a one-sentence justification. Scores content BEYOND trivial text restatement —
the scene-aware rubric lesson from the chronopercept campaign (a judge that
can't see the ctx credits paraphrase; showing it the ctx makes restatement
score 0).

Failures return None per row (caller skips) — judge flakiness must never kill
training. Concurrency is fine on OpenRouter (unlike Anthropic)."""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import time

import httpx

JUDGE_MODEL = os.environ.get("AN1_JUDGE_MODEL", "google/gemini-3-flash-preview")

PROMPT = """You are grading an interpretability system. A language model read a short text; a monitor then looked at one of the model's internal activation vectors (recorded at the final token of the text) and produced an explanation of its content. The text was written to induce a specific GOLD cognitive state, which the text deliberately never states outright.

Text the model read: {ctx}

GOLD cognitive state: {gold}

Monitor explanation: {expl}

Score how truly the explanation captures the GOLD cognitive state, 0.0-1.0 with 0.1 granularity. Only credit content that matches the gold state BEYOND what the text literally says — restating or paraphrasing the visible text scores 0.0, generic filler scores 0.0, naming the state's substance (even in other words) scores high. Reply with JSON only: {{"trueness": <float>, "justification": "<one sentence>"}}"""


def _one(client: httpx.Client, expl, gold, ctx):
    if expl is None:
        return None, None
    for attempt in range(5):
        try:
            r = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                json={"model": JUDGE_MODEL, "temperature": 0,
                      "messages": [{"role": "user",
                                    "content": PROMPT.format(ctx=ctx, gold=gold, expl=expl)}]},
                timeout=90)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S)
            d = json.loads(m.group(0))
            s = float(d["trueness"])
            if 0.0 <= s <= 1.0:
                return s, str(d.get("justification", ""))
        except Exception:
            time.sleep(min(2 ** attempt, 20))
    return None, None


def judge_trueness(explanations, golds, ctxs, concurrency=32):
    """-> (scores: list[float|None], justifications: list[str|None]), row-aligned."""
    with httpx.Client() as client, cf.ThreadPoolExecutor(concurrency) as ex:
        futs = [ex.submit(_one, client, e, g, c)
                for e, g, c in zip(explanations, golds, ctxs)]
        out = [f.result() for f in futs]
    return [o[0] for o in out], [o[1] for o in out]
