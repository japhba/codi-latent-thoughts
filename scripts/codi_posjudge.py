"""Judge the per-position nz4-3200 posgen verbalizations, feeding fig_09's
position-resolved NLA baselines: for each reader (z, h24), fed position t=0..5 and
kind (own read vs +7-shifted donor control scored against the OWN problem's gold),
does the verbalization semantically express at least one gold NON-ANSWER chain
value? Judge and criterion identical to codi_inter_judge.py (imported, so run from
scripts/ with OPENROUTER_API_KEY set, e.g. after sourcing the repo .env).

judge() field mapping (reused verbatim): src=reader, step=position, kind=own|donor.

Inputs:  repo/artifacts/nz4_posgen_3200.jsonl (z),
         repo/artifacts/nz4_posgen_h24_3200.jsonl
Output:  repo/artifacts/nz4_posjudge_3200.jsonl (append, resumable), rows
         {"src": "z"|"h24", "t": 0-5, "kind": "own"|"donor", "uid",
          "matched", "justification"}
"""
import asyncio, json, os

import httpx
from codi_inter_judge import ART, META_F, canon, judge

OUT_F = os.path.join(ART, "nz4_posjudge_3200.jsonl")
SRC = {"z": os.path.join(ART, "nz4_posgen_3200.jsonl"),
       "h24": os.path.join(ART, "nz4_posgen_h24_3200.jsonl")}
CHUNK = 400


def collect():
    META = {m["uid"]: m for m in map(json.loads, open(META_F))}
    tasks = []
    for src, path in SRC.items():
        for r in map(json.loads, open(path)):
            m = META[r["uid"]]
            gold = sorted(set(canon(v) for v in m["intermediates"]) - {canon(m["answer"])})
            if gold:
                tasks.append(dict(kind=r["kind"], src=src, step=r["t"], uid=r["uid"],
                                  gold=gold, gen=r["gen"]))
    return tasks


async def main():
    tasks = collect()
    done = set()
    if os.path.exists(OUT_F):
        done = {(r["src"], r["t"], r["kind"], r["uid"])
                for r in map(json.loads, open(OUT_F))}
    todo = [t for t in tasks if (t["src"], t["step"], t["kind"], t["uid"]) not in done]
    print(f"{len(tasks)} rows total, {len(done)} already judged, {len(todo)} to judge")
    if todo:
        async with httpx.AsyncClient() as client:
            for i in range(0, len(todo), CHUNK):
                chunk = todo[i:i + CHUNK]
                results = await asyncio.gather(*[judge(client, t) for t in chunk])
                with open(OUT_F, "a") as f:
                    for r in results:
                        f.write(json.dumps(dict(src=r["src"], t=r["step"], kind=r["kind"],
                                                uid=r["uid"], matched=r["matched"],
                                                justification=r["justification"])) + "\n")
                print(f"judged {min(i + CHUNK, len(todo))}/{len(todo)}", flush=True)
    rows = [json.loads(l) for l in open(OUT_F)]
    for src in ("z", "h24"):
        for kind in ("own", "donor"):
            cells = []
            for t in range(6):
                sub = [r["matched"] for r in rows
                       if (r["src"], r["t"], r["kind"]) == (src, t, kind)]
                cells.append(f"t{t + 1} {sum(sub) / len(sub):.3f}(n={len(sub)})")
            print(f"{src:>4} {kind:>5}: " + "  ".join(cells))


if __name__ == "__main__":
    asyncio.run(main())
