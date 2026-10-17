# Experiment scripts

The original experiment scripts, as they ran on the MATS cluster (2026-06 to 2026-09). They are kept verbatim as
the record of what produced the files in `artifacts/`; [../MANIFEST.md](../MANIFEST.md) maps each to its post section
and output.

They are not portable: they read and write the cluster's workspace layout (`/workspace-vast/<user>/...`,
`/root/...` on rented pods) and some expect inputs that are not bundled (raw activation captures, RL parquets,
intermediate checkpoints). To rerun a measurement on another machine, use the portable `codi_repro` modules
documented in [../docs/RERUNS.md](../docs/RERUNS.md) and [../docs/ALL_LAYER_LENSES.md](../docs/ALL_LAYER_LENSES.md),
which cover the controls, the lens readouts, the NLA reader evaluations and the interventions on checkpoint-7580.

`nz2/` holds the nz2-era splice campaign behind the NLA edit example (`figures/fig_15_nla_edit_example.py`).
`judge_protocol/` is the blinded two-judge protocol used for an earlier, cut systematicity analysis.
