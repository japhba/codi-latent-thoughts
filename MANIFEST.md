# Manifest: post → figure → script → data

Every figure in the post is drawn by exactly one script in `figures/` from files in `artifacts/`
(and, for `fig_07`, from the statistics `python -m codi_repro reproduce` writes to `results/`).
`python -m codi_repro reproduce` runs them all; each script also runs on its own from the repository root.
The experiment scripts that produced the artifacts are listed below the figure table.

## Figures

| post section | figure (`figs/`) | script (`figures/`) | inputs (`artifacts/` unless noted) |
|---|---|---|---|
| CODI | `codi_flow.png` | `fig_00_codi_flow.py` | `base_config.json`, `../manifest.json` (schematic) |
| Performance | `problem_types.png` | `fig_01_problem_types.py` | `rv6/controls/baseline.json`, `base_qwen_test.json` |
| Ablations | `ablations_lb.png` | `fig_16_ablations.py` | `rv6/controls/{baseline,predictions}.json`, `extra_predictions.json` |
| LogitLens and J-lens | `lens_dual.png` | `fig_02_lens_dual.py` | `lens_problems.json`, `lens_readouts.npz`, `lens_token_strings.json` |
| Natural Language Autoencoders | `nla_training_eval.png` | `fig_12_nla_training_eval.py` | `nz4_fve_curves.json`, `rv6/scores/all_thought_scores.json`, `reader_layer_lens_verdicts.json` |
| Qualitative analysis of NLA verbalizations | `qual_nla_duo.png` | `fig_05_qual_examples.py` | `nz4_qual_resample_{z,h24}.jsonl`, `nz4_qual_meta.jsonl`, `qual_nla_tldr.json` |
| Interventions | `patch_schematic.png` | `fig_10_patch_schematic.py` | none (schematic) |
| Transplant-patching z and h | `twin_patch.png` | `fig_09_twin_patch.py` | `lbtwin_patch.json`, `nz4_posjudge_3200.jsonl` |
| Intervening via the NLA | `nla_edit_example.png` | `fig_15_nla_edit_example.py` | `nz2/{zlb_gens.jsonl,val_docs.jsonl,zlb_splice_summary.json}` |
| Intervening via the NLA | `nla_intervention_z.png` | `fig_07_nla_intervention.py` | `results/twin_summary.json` (from `rv6/twins`) |
| App.: thought vectors are low-dimensional | `pca_dims.png` | `fig_14_pca_dims.py` | `pca_spectra.json` |
| App.: full LogitLens and J-lens top-5 tokens | `lens_dual_full.png` | `fig_02_lens_dual.py` | as `lens_dual.png` |
| App.: layer-resolved J-lens and LogitLens | `lens_topk_chance.png` | `fig_26_lens_topk_chance.py` | `lens_topk.npz`, `lens_topk_tokens.json`, `problems.json`, `ao_curves.json` |
| App.: composition of the GSM8k test set | `category_venn.png` | `fig_18_category_venn.py` | `controls_instances.json`, `base_qwen_test.json`, `heldout_instances.json` |
| App.: activation oracles | `fd1_performance.png` | `fig_03_fd1_performance.py` | `ao_curves.json`, `reader_layer_lens_verdicts.json` |
| App.: activation oracles | `ao_example.png` | `fig_23_ao_example.py` | `fd1_qual_resample.jsonl`, `nz2/lb41_capture.jsonl`, `qual_ao_tldr.json` |
| App.: scaling | `nla_training_eval_full.png` | `fig_12_nla_training_eval.py` | as `nla_training_eval.png` |
| App.: scaling | `fd1_performance_full.png` | `fig_03_fd1_performance.py` | as `fd1_performance.png` |
| App.: reusing vs regenerating later thoughts | `ablations_reuse_regen.png` | `fig_16_ablations.py` | as `ablations_lb.png` |

The HTML-rendered figures (`lens_dual*`, `qual_nla_duo`, `nla_edit_example`, `ao_example`) need playwright's
chromium (`playwright install chromium`); the rest are matplotlib. The italic one-line digests in the two
qualitative cards (`qual_nla_tldr.json`, `qual_ao_tldr.json`) are editorial, keyed by the SHA-1 of the verbatim
text they summarize, so a changed readout fails loudly.

`fig_15` (the NLA edit example) uses the nz2/nz3-era z reader and its splice campaign, not the final nz4 step-3200
readers used by `fig_07` and the rest of the NLA section; see its docstring.

## Measurements

GPU measurements come in two kinds. `codi_repro` holds the corrected, portable reruns (batch `rv6`) on
checkpoint-7580, with hash-pinned outputs in `artifacts/rv6/bundle.json`; [docs/RERUNS.md](docs/RERUNS.md) and
[docs/ALL_LAYER_LENSES.md](docs/ALL_LAYER_LENSES.md) give the commands. `scripts/` holds the original experiment
scripts as they ran on the MATS cluster ([scripts/README.md](scripts/README.md)).

| post section | measurement | code | output |
|---|---|---|---|
| Performance, Ablations | six vs zero thoughts and 61 recurrent ablations, masked decodes | `codi_repro.recurrent_controls`, `codi_repro.controlled_rollout` | `rv6/controls/` |
| Ablations | shuffled thoughts, another problem's thoughts, Gaussian thoughts | `codi_repro.extra_ablations` | `extra_predictions.json` |
| Performance, App. composition | original 28-condition controls battery | `scripts/codi_controls_battery.py` | `controls_instances.json`, `controls_summary.json` |
| Performance, App. composition | base Qwen3-8B with and without CoT on GSM8k test | `scripts/qwen_base_gsm8k_test.py` | `base_qwen_test.json` |
| App. composition | CODI's verbal and latent modes on held-out tasks | `scripts/codi_heldout_tasks.py` | `heldout_instances.json`, `heldout_summary.json` |
| Performance | answer-only and CoT SFT baselines on the CODI data | `scripts/sft_qwen8b_codi_data.py`, `scripts/sft_eval_heldout.py` | |
| LogitLens and J-lens | 1200-rollout lens readouts with corrected normalization | `codi_repro.lenses` (historical J-lens fit: `scripts/codi_jlens_fit_thought.py`) | `lens_readouts.npz`, `lens_*.json` |
| NLA curves, App. layer-resolved lenses | fresh all-layer J-lens fit and readouts on the reader cohorts | `codi_repro.all_layers`, `codi_repro.all_layer_analysis` | `all_layer_*.json`, `layer_lens_fit.json`, `reader_layer_lens_verdicts.json` |
| App. layer-resolved lenses | top-100 lens tokens at every layer | `codi_repro.lens_topk` | `lens_topk.npz`, `lens_topk_tokens.json` |
| NLA | reader training (EasyNLA, pinned at `ecb33c7`) | `vendor/EasyNLA`, `training/nla_{z,h24}_historical.yaml`, `scripts/nz2_build_data.py` | `nz4_fve_curves.json` (W&B export, project `nlarl`, runs `7goxj6kk` z, `25qqqwah` h24) |
| NLA | all-six-thought value recovery at nine reader checkpoints | `codi_repro.all_thought_readers`, `codi_repro.all_thought_scores` | `rv6/readers/`, `rv6/scores/` |
| NLA (earlier first-thought scans) | dense-ladder scan and judged intermediates | `scripts/nz4_score_scan.py`, `scripts/codi_inter_judge.py`, `scripts/codi_perm_judge.py` | `av_history.json.gz`, `inter_match_judged.jsonl`, `nz4_perm_control.json` |
| Qualitative NLA | resampled verbalizations at steps 0 and 3200 | `scripts/nz4_qual_resample.py` | `nz4_qual_resample_{z,h24}.jsonl` |
| Transplant-patching | load-bearing twin mining, LLM solve verification, z and column patches | `scripts/codi_lb_twins_mine.py`, `scripts/codi_lb_twins_verify.py`, `scripts/codi_twin_patch_lb.py` | `verified_twin_candidates.json`, `twins.json`, `lbtwin_patch.json` |
| Transplant-patching | per-position NLA reads at step 3200 and their judgments | `scripts/nz4_posgen.py`, `scripts/codi_posjudge.py` | `nz4_posgen{,_h24}_3200.jsonl`, `nz4_posjudge_3200.jsonl` |
| Intervening via the NLA | describe → edit → reconstruct → patch, with matched random displacements | `codi_repro.twin_controls` | `rv6/twins/` |
| Intervening via the NLA (example) | nz2-era splice campaign | `scripts/nz2/{val_pick,val_lb_mine,zlb_reads,zlb_arsplice}.py` | `nz2/` |
| App. low-dimensional | pooled PCA spectra of z and h24 | `scripts/codi_pca_spectra.py` | `pca_spectra.json` |
| App. activation oracles | AO capture, data, SFT; evaluation scans | `scripts/fd1_capture.py`, `fd1_data.py`, `fd1_sft.py`, `fd1_acc_scan.py`, `fd1_qonly6_scan.py`, `fd1_inter_gens.py` | `ao_curves.json` |
| App. activation oracles | resampled AO readouts for the example card | `scripts/fd1_qual_resample.py` | `fd1_qual_resample.jsonl` |

## Not in the post

Kept as evidence for analyses that were run but cut or superseded: thought-ordering
(`scripts/nx1_emission_order.py`, `nx2_pos_order.py`, `fd1_pos_order.py`, `codi_seq_ao.py`; `fd1_pos_order.jsonl`),
h cross-patch ladder and val-set twins (`codi_hpatch_ladder.py`, `codi_hpatch_twin.py`, `codi_zpatch_twin.py`;
`hpatch_ladder.json`, `hpatch_twin.json`, `zpatch_twin.json`), h-column corruption (`codi_hcorrupt_battery.py`;
`hcorrupt_battery.json`), NLA-mediated twin patch with the older reader (`codi_nlapatch_twin.py`; `nlapatch_twin.json`),
AO control cells (`fd1_battery.py`, `fd1_lbctl_eval.py`, `fd1_textcombine.py`, `fd1_wrongpred_scan.py`, `fd1_qonly5_scan.py`,
`fd1_qual_scan.py`), replications of earlier CODI interpretability papers (`codi_paper1_*.py`, `codi_paper2_forwardchain.py`,
`codi_patching_by_hops.py`), nz1/nz2 analyses (`nz1_score_scan.py`, `nz2_semfve_plot.py`) and the blinded judge
protocol (`scripts/judge_protocol/`). The historical lens and AV-baseline analyses are described in
[docs/LENSES.md](docs/LENSES.md) and [docs/AV_BASELINES.md](docs/AV_BASELINES.md).
