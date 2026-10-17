"""Pick held-out VAL docs from mix_rl.parquet for the intermediate-swap test (pod).
- val split: doc-hash permille 215 (val_doc_permille(50000, 233028)), matching the
  trainer/monitor convention.
- 32 unique docs per stratum x 12 strata -> val_reads.parquet (same schema the
  read harness expects) + val_docs.jsonl (doc meta joined from gsm_meta.jsonl).
"""
import json, sys, zlib

sys.path.insert(0, "/root/nz2/EasyNLA")
import pyarrow as pa
import pyarrow.parquet as pq
from nla.val_split import is_val_doc, val_doc_permille

PQ = "/root/nz2/data/mix_rl.parquet"
META = "/root/nz2/data/gsm_meta.jsonl"
OUTP = "/root/lbeval/val_reads.parquet"
OUTD = "/root/lbeval/val_docs.jsonl"
PER_STRATUM = 32

pf = pq.ParquetFile(PQ)
total = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
pm = val_doc_permille(50000, total)
print(f"total rows {total}, val permille {pm}")

by_s = {}
rows = []
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg)
    for a, p, d, s in zip(t.column("activation_vector").to_pylist(),
                          t.column("prompt").to_pylist(),
                          t.column("doc_id").to_pylist(),
                          t.column("stratum").to_pylist()):
        if not is_val_doc(d, pm): continue
        got = by_s.setdefault(s, set())
        if len(got) >= PER_STRATUM or d in got: continue
        got.add(d)
        rows.append(dict(activation_vector=a, prompt=p, doc_id=d, stratum=s))
    if all(len(v) >= PER_STRATUM for v in by_s.values()) and len(by_s) == 12: break

print({s: len(v) for s, v in sorted(by_s.items())})
docs = sorted({r["doc_id"] for r in rows})
print(f"{len(rows)} reads, {len(docs)} unique docs")

meta = {}
for l in open(META):
    m = json.loads(l)
    meta[m["uid"]] = m
with open(OUTD, "w") as f:
    for d in docs:
        m = meta[d]
        f.write(json.dumps(dict(doc_id=d, question=m["question"], answer=m["answer"],
                                cot=m["cot"], intermediates=m["intermediates"])) + "\n")

pq.write_table(pa.Table.from_pylist(rows), OUTP)
print(f"wrote {OUTP} + {OUTD}")
