"""Judge "any gold intermediate semantically matched" for the writeup's
intermediates criterion: per verbalization, does the text semantically express AT
LEAST ONE of the gold non-answer chain values (numerals in any formatting,
spelled-out numbers, or arithmetic expressions unambiguously producing the value).
Judge: google/gemini-3-flash-preview via OpenRouter, temperature 0, 32-way
concurrency with exponential backoff, strict-JSON verdicts with justification.

Inputs:
  NLA  reports/codi_writeup/repo/artifacts/score_scan_{z,h24}_judged.jsonl
       (nz1 scan, all steps; gold = canonical gsm_meta intermediates minus the
       answer, empty-gold problems drop out, 49/60)
  NLA  reports/codi_writeup/repo/artifacts/nz4_score_scan_{z,h24}.jsonl if
       present (nz4 dense-ladder scan, 101 steps; judged under src
       "nz4_z"/"nz4_h24" so the nz1 verdicts keep their keys)
  AO   /workspace-vast/jbauer/bt1/fd1/dur/fd1_inter_gens.jsonl if present
       (VALQ readouts of feeds ALL/Z/qonly6 at every saved checkpoint, gold
       stored per row)
Output (append, resumable by key):
  reports/codi_writeup/repo/artifacts/inter_match_judged.jsonl
  rows {"kind": "nla"|"ao", "src": reader-or-feed, "step", "uid",
        "matched": bool, "justification"}
"""
from __future__ import annotations

import asyncio
import json
import os
import re

import httpx

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(_REPO, "reports", "codi_writeup", "repo", "artifacts")
AO_GENS = "/workspace-vast/jbauer/bt1/fd1/dur/fd1_inter_gens.jsonl"
META_F = "/workspace-vast/jbauer/bt1/fd1/data/gsm_meta.jsonl"
OUT_F = os.path.join(ART, "inter_match_judged.jsonl")
MODEL = "google/gemini-3-flash-preview"
KEY = os.environ["OPENROUTER_API_KEY"]
SEM = asyncio.Semaphore(int(os.environ.get("JUDGE_CONC", "16")))
CHUNK = 800   # append verdicts after every chunk so a crash keeps progress

PROMPT = """You are checking whether a model-generated text mentions any of a list of gold numeric values.

Text:
{gen}

Gold values: {gold}

Does the text semantically express AT LEAST ONE of the gold values? Count exact numerals in any formatting ("1,200", "$12", "12.0", "12%"), spelled-out numbers ("twelve", "a dozen"), and arithmetic expressions whose result is unambiguously a gold value ("45 divided by 3" matches 15). Do not count a value that only appears as a substring of a different number (15 inside 150 does not match), do not count digits inside a date, year, or year range (2010-2012 does not match 12), and do not count a different quantity that merely rounds to a gold value.

Reply with strict JSON only: {{"matched": true, "justification": "<one short sentence naming the matching span or the closest miss, without any quotation marks>"}}"""


def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"


def collect():
    META = {m["uid"]: m for m in map(json.loads, open(META_F))}
    tasks = []
    scans = [(src, os.path.join(ART, f"score_scan_{src}_judged.jsonl"))
             for src in ("z", "h24")]
    scans += [(f"nz4_{src}", os.path.join(ART, f"nz4_score_scan_{src}.jsonl"))
              for src in ("z", "h24")]
    for src, path in scans:
        if not os.path.exists(path):
            print(f"NOTE: {path} absent, skipped")
            continue
        for r in map(json.loads, open(path)):
            m = META[r["uid"]]
            gold = sorted(set(canon(v) for v in m["intermediates"]) - {canon(m["answer"])})
            if gold:
                tasks.append(dict(kind="nla", src=src, step=r["step"], uid=r["uid"],
                                  gold=gold, gen=r["gen"]))
    if os.path.exists(AO_GENS):
        for r in map(json.loads, open(AO_GENS)):
            if r["gold"]:
                tasks.append(dict(kind="ao", src=r["feed"], step=r["step"], uid=r["uid"],
                                  gold=r["gold"], gen=r["gen"]))
    else:
        print(f"NOTE: {AO_GENS} absent, judging NLA rows only")
    return tasks


async def judge(client, t):
    prompt = PROMPT.format(gen=t["gen"].strip(), gold=", ".join(t["gold"]))
    for attempt in range(9):
        try:
            async with SEM:
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {KEY}"},
                    json=dict(model=MODEL, temperature=0.0, max_tokens=512,
                              messages=[{"role": "user", "content": prompt}]),
                    timeout=120)
            if r.status_code == 429:
                # sustained provider throttling: back off patiently, up to 5 min
                await asyncio.sleep(min(10 * 2 ** attempt, 300))
                continue
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r'"matched"\s*:\s*(true|false)', txt)
            assert m, txt
            j = re.search(r'"justification"\s*:\s*"(.*)"', txt, re.S)
            return dict(kind=t["kind"], src=t["src"], step=t["step"], uid=t["uid"],
                        matched=m.group(1) == "true",
                        justification=j.group(1) if j else txt)
        except Exception:
            if attempt == 8:
                raise
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError("judge: exhausted retries (sustained 429)")


async def main():
    tasks = collect()
    done = set()
    if os.path.exists(OUT_F):
        done = {(r["kind"], r["src"], r["step"], r["uid"])
                for r in map(json.loads, open(OUT_F))}
    todo = [t for t in tasks if (t["kind"], t["src"], t["step"], t["uid"]) not in done]
    print(f"{len(tasks)} rows total, {len(done)} already judged, {len(todo)} to judge")
    if todo:
        async with httpx.AsyncClient() as client:
            for i in range(0, len(todo), CHUNK):
                chunk = todo[i:i + CHUNK]
                results = await asyncio.gather(*[judge(client, t) for t in chunk])
                with open(OUT_F, "a") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
                print(f"judged {min(i + CHUNK, len(todo))}/{len(todo)}", flush=True)
    rows = [json.loads(l) for l in open(OUT_F)]
    for kind in ("nla", "ao"):
        for src in sorted({r["src"] for r in rows if r["kind"] == kind}):
            for step in sorted({r["step"] for r in rows if r["kind"] == kind and r["src"] == src}):
                sub = [r for r in rows if (r["kind"], r["src"], r["step"]) == (kind, src, step)]
                acc = sum(r["matched"] for r in sub) / len(sub)
                print(f"{kind} {src:>6} step {step:>5}: any-intermediate matched "
                      f"{acc:.3f} (n={len(sub)})")


if __name__ == "__main__":
    asyncio.run(main())
