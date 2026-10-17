# Method notes

Detailed notes written while the measurements were audited and rerun. Terms used throughout: *rv6* is the
corrected rerun batch on checkpoint-7580 whose outputs are hash-pinned in `artifacts/rv6/bundle.json`; the
*authoring checkout* and *review draft* are the working copy of the blog post and its analysis tree from which
the historical records were extracted.

| note | contents |
|---|---|
| [RERUNS.md](RERUNS.md) | GPU commands for the rv6 reruns: masked inference, all-thought reader scoring, matched NLA interventions, bundling |
| [ALL_LAYER_LENSES.md](ALL_LAYER_LENSES.md) | dedicated per-layer J-lens fitting, disjoint calibration, convergence, candidate semantics |
| [CURVE_BASELINES.md](CURVE_BASELINES.md) | the lens references drawn into the NLA and AO learning curves, and their comparability limits |
| [PROVENANCE.md](PROVENANCE.md) | where every bundled record comes from, source snapshots (CODI, EasyNLA), environment differences |
| [VALIDATION.md](VALIDATION.md) | dated validation log |
| [LENSES.md](LENSES.md) | the corrected historical 1200-rollout lens replay (behind `lens_dual.png`) |
| [AV_BASELINES.md](AV_BASELINES.md) | historical leading-digit LogitLens baselines for the readers (not in the post) |
