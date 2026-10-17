# Lens references in the original post curves

The four learning-curve figures retain the original reader, loss, reconstruction and control series. The compact NLA figure in the main text and AO figure in Appendix D use LogitLens/J-lens references restricted to **the corresponding reader's layers**. The full versions in Appendix E show the same reader curves with separate **all-layer ensemble skylines** instead. Both scopes use the fresh dedicated maps, checkpoint-7580, identical candidate semantics and the corresponding reader's exact evaluation questions. The final normalized state is a shared identity endpoint.

| Figure reference | Hidden states | Thought positions | Candidate slots per question and method |
|---|---|---|---|
| NLA h24 | h24 | 1–6 | 30 |
| NLA z | projected input h0 = z | 1–6 | 30 |
| AO all thought streams | z, h9, h18, h27, h35, h36 | 1–6 | 180 |
| AO z only | projected input h0 = z | 1–6 | 30 |
| Appendix E skyline | h0–h36 | 1–6 | 1,110 |

Each location contributes five independent token alternatives. Lenses retain the previously requested ensemble across positions. The original NLA scans read only thought one, so matching layers does not also match their position budget. No lens reference is added for the AO question-region-only feed because those question-region lens readouts were not captured. “Skyline” denotes the broader candidate-coverage reference, not a theoretical upper bound on all decoding methods.

[ALL_LAYER_LENSES.md](ALL_LAYER_LENSES.md) gives the complete calibration, resumption, convergence, extraction and reproduction protocol. `artifacts/layer_lens_fit.json` records the actual sample count and stop decision. `results/summary.json` (`reader_layer_lenses`, `all_layer_lens_skylines`) records each cohort's answer/intermediate counts and exact semantic label-permutation expectation. `results/layer_lens_fit.png` shows the calibration/validation curves; `results/all_layer_lens_profile.json` holds per-layer complete-value recovery before pooling, and `figs/lens_topk_chance.png` (Appendix B of the post) shows chance-controlled per-layer readability.

## Question matching

Every lens reference is evaluated on exactly the questions in the reader curve it accompanies. However, the original h24 and z readers were evaluated on **different 60-question sets**, with 12 shared questions; AO uses its separate 48-question set. This is inherited from the historical scan, which independently selected the first 60 unique document-hash-held-out questions from each reader's own activation parquet. The parquets and their lengths differed, and the validation hash fraction depended on each length. The selection code excerpt, source fingerprint, complete UID lists and overlap are bundled in `artifacts/curve_evaluation_protocol.json` and validated against every saved NLA evaluation step. The h24-versus-z curves therefore do not form a paired question comparison. A 12-question common-subset replot is possible from the saved histories; a larger common-set historical scan requires new reader inference.

Existing reader curves and answer parsers are unchanged. Semantic lens answer mentions differ from the reader's literal containment (NLA) or first-number answer accuracy (AO). Complete values must be expressed by individual candidates; unrelated digits are never concatenated. The semantic permutation expectations use the same extracted value sets, which is especially relevant as larger pools can cover many values by accident. Mention recovery alone does not establish faithful or causal intermediate computation.

The previous single-location and fixed-source six-position readouts, judge responses, audits and provenance remain in artifacts/ as historical evidence. In particular, the old h24 candidate-overlap qualification applies to the **old calibration**, and does not apply to these newly fitted all-layer maps. Older lens summaries and static historical figures are explicitly separate from the current curve references. Robot notes in the review draft distinguish them.

`python -m codi_repro reproduce` recomputes these statistics into `results/` and redraws every post figure, including the four learning-curve figures (`figures/fig_03_fd1_performance.py`, `figures/fig_12_nla_training_eval.py`), offline from compact bundled evidence. No model, large matrix file or judge key is needed for offline reproduction.

## Compact figures: reader-layer references

| Reader source | Method | Complete answer mention | Any non-answer intermediate | Semantic intermediate permutation expectation |
|---|---|---|---|---|
| NLA h24 | LogitLens | 1/60 | 5/55 | 3.1% |
| NLA h24 | J-lens | 0/60 | 1/55 | 0.3% |
| NLA z | LogitLens | 0/60 | 0/49 | 0.2% |
| NLA z | J-lens | 0/60 | 0/49 | 0.0% |
| AO selected streams | LogitLens | 16/48 | 35/48 | 26.2% |
| AO selected streams | J-lens | 10/48 | 31/48 | 19.3% |
| AO z only | LogitLens | 0/48 | 0/48 | 0.5% |
| AO z only | J-lens | 0/48 | 0/48 | 0.0% |

`artifacts/reader_layer_lens_verdicts.json` preserves every deterministic match and the candidate/value citations. These are subsets of the fresh all-layer readouts, rather than a return to the historical overlapping calibration. The uniform global candidate dictionary gives h24 LogitLens one more intermediate match than the preceding ensemble-by-ensemble extraction: the complete candidate `一个小时` explicitly expresses one hour.

## Appendix E: all-layer ensemble skylines

| Evaluation cohort | Method | Complete answer mention | Any non-answer intermediate | Semantic intermediate permutation expectation |
|---|---|---|---|---|
| NLA h24 cohort | LogitLens | 24/60 | 34/55 | 31.3% |
| NLA h24 cohort | J-lens | 21/60 | 31/55 | 16.6% |
| NLA z cohort | LogitLens | 26/60 | 31/49 | 27.3% |
| NLA z cohort | J-lens | 19/60 | 27/49 | 19.0% |
| AO cohort | LogitLens | 16/48 | 35/48 | 33.4% |
| AO cohort | J-lens | 14/48 | 32/48 | 22.7% |

The 36 matrices were calibrated on 72 correct rollouts; the CE convergence rule stopped the fit before its 96-example ceiling. Validation used 24 separate correct rollouts. The final h36 lists agree exactly. Across other states, 928–936 of the 936 problem-position top-five lists differ between methods.
