"""Judged permuted-gold coincidence control at the FINAL scanned checkpoint of
each nz4 reader: the same any-intermediate judge, but each verbalization is
scored against a DIFFERENT problem's gold intermediates (fixed +7 shift). The
judged own-rate minus this judged permuted-rate is the coincidence-corrected
decode used for the twin-patch figures' NLA baselines (mechanical containment
understates readers that verbalize operations, judged own-rate overstates
number-spraying readers; the subtraction needs both sides on the judge metric).

-> reports/codi_writeup/repo/artifacts/nz4_perm_control.json
"""
from __future__ import annotations

import asyncio
import json
import os

import httpx

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(_REPO, "reports", "codi_writeup", "repo", "artifacts")
META_F = "/workspace-vast/jbauer/bt1/fd1/data/gsm_meta.jsonl"
OUT_F = os.path.join(ART, "nz4_perm_control.json")
MODEL = "google/gemini-3-flash-preview"
KEY = os.environ["OPENROUTER_API_KEY"]
SEM = asyncio.Semaphore(8)
SHIFT = 7

from codi_inter_judge import PROMPT, canon, judge  # same prompt/parse/backoff


async def main():
    META = {m["uid"]: m for m in map(json.loads, open(META_F))}
    out = {}
    async with httpx.AsyncClient() as client:
        for src in ("z", "h24"):
            rows = [json.loads(l) for l in open(os.path.join(ART, f"nz4_score_scan_{src}.jsonl"))]
            fin = max(r["step"] for r in rows)
            sub = [r for r in rows if r["step"] == fin]
            golds = []
            for r in sub:
                m = META[r["uid"]]
                golds.append(sorted(set(canon(v) for v in m["intermediates"])
                                    - {canon(m["answer"])}))
            keep = [i for i in range(len(sub)) if golds[i]]
            tasks = [dict(kind="perm", src=src, step=fin, uid=sub[i]["uid"],
                          gold=golds[keep[(j + SHIFT) % len(keep)]], gen=sub[i]["gen"])
                     for j, i in enumerate(keep)]
            res = await asyncio.gather(*[judge(client, t) for t in tasks])
            rate = sum(r["matched"] for r in res) / len(res)
            out[src] = {"step": fin, "n": len(res), "judged_perm_rate": rate}
            print(f"{src}: step {fin} judged permuted-gold rate {rate:.3f} (n={len(res)})")
    json.dump(out, open(OUT_F, "w"), indent=1)
    print(f"wrote {OUT_F}")


if __name__ == "__main__":
    asyncio.run(main())
