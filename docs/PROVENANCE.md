# Evidence and source provenance

The original measurement files are extracted from the recovered authoring checkout. Their SHA-256 hashes are in `manifest.json`. The extraction selected existing records and exact vector subsets without altering judge verdicts. The corrected lens readouts and AV baselines are new measurements; their inputs and run provenance are documented below.

| Bundled file | Historical source |
|---|---|
| `artifacts/controls.json` | `writeup/controls_instances.json`; unused lens payload omitted |
| `artifacts/twins.json` | `lbtwin_patch.json` raw z records, joined to `lb_twins.jsonl` for both questions |
| `artifacts/verified_twin_candidates.json` | All independently solver-verified candidates in `lb_twins.jsonl` |
| `artifacts/readouts.json` | `nz4_posgen_3200.jsonl`, `nz4_posgen_h24_3200.jsonl` |
| `artifacts/verdicts.json` | `nz4_posjudge_3200.jsonl` |
| `artifacts/problems.json` | Original capture question lists, selected by the readout UIDs |
| `artifacts/vectors_{z,h24}.npz` | Exact six-vector subsets of `nz1/data/{z,h24}_rl.parquet`, ordered by each reader's readout population |
| `artifacts/prompt_{z,h24}.json` | Actor prompt messages from the same parquets |
| `artifacts/reconstruction_history.json` | `nz4_fve_curves.json`, exported NLA evaluations |
| `artifacts/av_history.json.gz` | Complete `nz4_score_scan_{z,h24}.jsonl` descriptions joined to original `inter_match_judged.jsonl` verdicts and justifications; containment flags recomputed |

Recorded controls answer strings were originally saved with a 60-character cap. All retained recorded predictions can be re-parsed from those strings. New runs save complete strings. Non-numeric answers are represented by JSON `null`; historical infinities are not emitted into the new package.

The twin list contains 50 selected pairs from 35 originals. It follows numeric chain propagation, external solver verification, and selection for successful latent answers and failed zero-thought answers on both versions. This is a selected intervention population, not a random sample of GSM8k. The selected pairs and all 296 solver-verified candidates are bundled. Numeric twin mining and the external solver campaign are not needed to reproduce the fixed-pair measurement.

CODI source: [cywinski/codi](https://github.com/cywinski/codi), recovered checkout HEAD `41d509ecd9ff91a86bfcd7f4bd5907d37d515cd9` (local commits on upstream `f0bf2fa`), including uncommitted local edits. `training/codi_qwen3.patch` turns upstream `f0bf2fa` into exactly that recovered snapshot (verified file by file); it is **not** a claim that this source alone reproduces checkpoint training. The checkpoint's own reproduction notes specify the five-epoch Qwen3 recipe. The portable controls are derived from the historical battery implementation; only loading, paths, CLI, population input, and complete-output persistence changed.

NLA source: [asherps/EasyNLA](https://github.com/asherps/EasyNLA), recovered `EasyNLA_nz1pin` worktree at `ecb33c7`; the relevant package and license are vendored. The unrelated experimental `fromscratch` and data-generation trees are not included. The later main-tree trainer had rollout synchronization problems, so it is not used as the provenance source.

The CPU environment is locked for this reproduction package. It does not reconstruct the full original environment: the original CODI snapshot listed transformers 4.52.4 / torch 2.7.1, whereas the recovered NLA package pinned transformers 4.57.1. The tested portable inference environment uses transformers 4.57.6 / torch 2.10.0 / peft 0.19.1; original final reader configs identify peft 0.19.1. These are possible sources of fresh-run numerical differences.

The two reader adapter hashes pin recovered local files. Every recovered warmstart AV weight shard and tokenizer/config file was hashed and matched against the pinned public warmstart revision. Thus the public AV is the exact base used for these reader adapters.

Legacy full-column measurements are outside the audited reproduction because their last block was patched with an already normalized final state. Legacy NLA-mediated value edits used an older reader. The restored review draft preserves these original figures with explicit robot flags, rather than treating them as validated evidence. They do not contribute to the new lens baselines or the audited intervention counts.

## Corrected lens evidence

The full historical `codi_faithful_acts.npy` and CODI-specific `codi_thought_lens.pt` were recovered from the cluster backup. Their fingerprints and exact original fit metadata are in `artifacts/lens_provenance.json`. The 1,200 analysis rows are GSM8k-Aug-NL train 4000–5199; the 96-problem lens fit starts at 5200 and skipped 17 candidates. The current pinned dataset revision matches every saved gold and numeric reference chain in the capture. Complete questions and reference CoTs for these rows and the 113 fit candidates are bundled.

`artifacts/lens_readouts.npz` contains freshly recomputed float32 LogitLens/J-lens log probabilities, full-vocabulary digit ranks and top-five outputs using the fixed organism checkpoint, corrected normalization and exact captured input vectors. `lens_run.json`, `lens_head_audit.json`, `lens_normalization_audit.json` and `lens_recapture_validation.json` record the run and checks. `LENSES.md` describes the estimators, empirical label controls and limits. The large full capture and Jacobian files remain outside this publishable directory; offline reproduction needs only the included compact records.

## AV learning-curve baselines

`artifacts/av_best_lens_vectors.npz` and `artifacts/av_lens_readouts.json` are fresh LogitLens measurements on the exact reader evaluation questions, using the same checkpoint-7580. The strong location is selected on saved lens indices 4000–4999, excluding all reader evaluation problems. Matched-input baselines read the exact bundled z/h24 vectors; the h36 reference uses a fresh raw-question capture. The complete selection grid, capture environment and first-vector comparison are recorded. `AV_BASELINES.md` explains why these baselines share a leading-digit diagnostic with the AV curves rather than being overlaid on complete-value/judged metrics.

## Restored original curve figures

Every figure in the post is drawn by one script in `figures/` from the bundled artifacts (see `MANIFEST.md`); earlier static copies of the figures are no longer bundled.

`artifacts/ao_curves.json` includes the exact original nine AO training logs, validation rows, answer-score rows, all 2,016 intermediate descriptions and verdicts, and the 48 evaluation problems. `curve_reconstruction_history.json` is the exact original NLA figure input. The older `curve_lens_selection.json`, `curve_lens_ao_vectors.npz`, `curve_lens_ao_readouts.json`, `curve_lens_judge_inputs.json` and `curve_lens_verdicts.json` preserve the preceding single-location endpoint analysis; they are historical evidence, not the current curve references.

The preceding fixed-source references used `curve_lens_ensemble_readouts.json`: both methods' full top-five token alternatives and float32 log probabilities at all six thoughts from exact original NLA h24/z vectors and fresh AO all-stream vectors. `curve_lens_ao_ensemble_vectors.npz` and `curve_lens_ensemble_capture.json` preserve the full fresh AO capture and runtime protocol. Its first-thought h36 vectors agree byte for byte with the preceding AO capture. `curve_lens_ensemble_judge_inputs.json` preserves the pooled candidates and gold labels used for offline scoring; gold labels are not sent to the extraction model. `curve_lens_ensemble_extractions.json` preserves every gold-blind extraction, exact token citation and raw response. `curve_lens_ensemble_verdicts.json` contains the resulting deterministic answer/intermediate matches. A failed target-aware binary judging pass is archived outside the publication companion and is not used. The readouts record the historical Jacobian file and individual matrix hashes; h24 uses J[23], z uses J_emb. Nine h24 evaluation questions overlap the historical fit candidate range; accepted calibration IDs were not saved. The z and AO cohorts have no candidate overlap. `CURVE_BASELINES.md` documents the exact inputs, scoring, ensemble budget and comparison qualifications. The unchanged original reader series remain the figure inputs.

All pre-existing train, decode, validation and shuffled-control series were compared with the original figure scripts and agree to floating-point tolerance. The four original learning-curve figures add complete-value references without replacing their metrics with leading-digit recovery.

## Fresh all-layer calibration and current curve references

The current references use `all_layer_lens_readouts.json`: 37 states × six thoughts × two methods on the 156 unique evaluation questions. `layer_lens_fit.json` records the fresh dedicated matrix calibration, accepted question IDs, disjoint convergence-validation IDs, complete proxy curves, convergence decision, matrix-file fingerprint and W&B run. `all_layer_capture.json` records the raw-question capture and agreement with the exact historical h24/z vectors retained at those locations. `all_layer_{fit,validation,evaluation}_problems.json` preserve complete capture questions, gold/reference texts and fresh CODI answers. `all_layer_token_extractions.json` preserves the deduplicated candidate universe, independent gold-blind extraction batches and full responses; `all_layer_lens_verdicts.json` preserves deterministically derived matches. The original reader histories and layouts remain the figure inputs. `ALL_LAYER_LENSES.md` specifies the estimator, access and metric qualifications and commands. Earlier lens measurements are retained as historical evidence, rather than overwritten.

The compact figures restrict those fresh readouts to the reader source layers; `reader_layer_lens_verdicts.json` records the resulting complete-value matches. Appendix E instead uses the all-layer ensemble skylines. Both scopes retain all six thoughts and each reader's exact questions. `curve_evaluation_protocol.json` records the full UID sets, their 12-question NLA overlap, and the historical `nz4_score_scan.py` selection excerpt with its source fingerprint. That scan selected the first 60 held-out documents independently from each reader's own activation parquet, with the document-hash fraction derived from that file's length. The original reader curves consequently are not a paired h24-versus-z comparison. No new fitting, capture or API judging was needed to filter the saved all-layer evidence.
