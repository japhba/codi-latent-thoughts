"""nz2: build the COMBINED tagged RL parquet for the unified recon-NLA.

One dataset, both read objects: z_t (fed-back thought) and h^L24 rows from the
nz1 parquets, interleaved+shuffled. Each row's AV prompt carries a source tag
(stream + thought position 1..6) so the actor knows what it is decoding, and a
`stratum` column ("z_p3", "h24_p5") drives strata-resolved loss/FVE logging in
the patched trainer. doc_id stays the problem uid — the crc32 doc-hash val
split keeps ALL 12 strata of a problem on the same side.

Positions are recovered from the nz1 builders' row order (per problem, t=0..5
consecutive — asserted here).

  python3 nz2_build_data.py
"""
import copy, json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

DATA = "/workspace-vast/jbauer/bt1/nz1/data"
OUT = f"{DATA}/mix_rl.parquet"
D = 4096

TAG = {"z": "the model's fed-back latent thought vector (the z-stream, post-projection)",
       "h24": "the model's residual stream at layer 24 (the h-stream)"}

vecs, doc_ids, strata, prompts = [], [], [], []
prompt_cache = {}
base_msgs = None
for src in ("z", "h24"):
    tbl = pq.read_table(f"{DATA}/{src}_rl.parquet")
    n = len(tbl)
    assert n % 6 == 0, (src, n)
    if base_msgs is None:
        base_msgs = tbl["prompt"][0].as_py()
    dids = tbl["doc_id"].to_pylist()
    acts = np.asarray(tbl["activation_vector"].combine_chunks().flatten(),
                      dtype=np.float32).reshape(n, D)
    for b in range(0, n, 6):
        assert len(set(dids[b:b + 6])) == 1, (src, b)   # 6-block per problem
    for i in range(n):
        p = i % 6 + 1
        s = f"{src}_p{p}"
        if s not in prompt_cache:
            msgs = copy.deepcopy(base_msgs)
            assert msgs[0]["content"].count("Here is the vector:") == 1
            msgs[0]["content"] = msgs[0]["content"].replace(
                "Here is the vector:",
                f"This vector was read from {TAG[src]}, at latent thought "
                f"position {p} of 6.\n\nHere is the vector:")
            prompt_cache[s] = msgs
        vecs.append(acts[i])
        doc_ids.append(dids[i])
        strata.append(s)
        prompts.append(prompt_cache[s])
    print(f"{src}: {n} rows ({n // 6} problems)")

N = len(vecs)
rng = np.random.default_rng(0)
perm = rng.permutation(N)
print(f"combined: {N} rows, shuffling with seed 0")

tbl = pa.table({
    "activation_vector": pa.FixedSizeListArray.from_arrays(
        pa.array(np.stack([vecs[i] for i in perm]).ravel()), D),
    "prompt": pa.array([prompts[i] for i in perm]),
    "doc_id": pa.array([doc_ids[i] for i in perm]),
    "stratum": pa.array([strata[i] for i in perm]),
})
pq.write_table(tbl, OUT, row_group_size=20000)

side = copy.deepcopy(yaml.safe_load(open(f"{DATA}/z_rl.parquet.nla_meta.yaml")))
side["dataset_id"] = "rl_Qwen3-8B_codi_mix_z_h24__tagged__shuf0"
side["row_count"] = N
side["extraction"]["corpus"] = ("fd1 captures, MIXED: CODI ckpt-7580 z_t (post-prj) + "
                                "residual h^L24, thought positions t1..t6, correct rows")
side["extraction"]["positions_per_doc"] = 12
side["note"] = ("nz2 unified NLA: per-row source tag (stream + position) in the AV prompt; "
                "`stratum` column = {z,h24}_p{1..6}; doc_id = problem uid (val split keeps "
                "all strata of a problem together); contract otherwise cloned from "
                "finefineweb_100k rl_shuf")
with open(OUT + ".nla_meta.yaml", "w") as f:
    yaml.safe_dump(side, f, allow_unicode=True, sort_keys=False)
print(f"WROTE {OUT} (+ sidecar)")
print(json.dumps(prompt_cache["z_p1"][0]["content"].split("\n\n")[2:4], indent=1))
