"""Reads parquet for the AR-roundtrip splice test (pod): 35 LB val docs x ALL
six z strata from mix_rl -> /root/lbeval/zlb_reads.parquet"""
import json
import pyarrow as pa
import pyarrow.parquet as pq

lbdocs = {r["doc_id"] for r in (json.loads(l) for l in open("/root/lbeval/val_lb.jsonl")) if r["lb"]}
want = {(d, f"z_p{p}") for d in lbdocs for p in range(1, 7)}
pf = pq.ParquetFile("/root/nz2/data/mix_rl.parquet")
rows = []
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg)
    for r in t.to_pylist():
        k = (r["doc_id"], r["stratum"])
        if k in want:
            rows.append(r)
            want.discard(k)
    if not want: break
print(f"{len(lbdocs)} LB docs -> {len(rows)} z reads (missing {len(want)})")
pq.write_table(pa.Table.from_pylist(rows), "/root/lbeval/zlb_reads.parquet")
