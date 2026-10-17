# LogitLens baselines for the activation verbalizers

*Historical diagnostic, not part of the post or of `reproduce`; `codi_repro.av_analysis.reproduce(out)` draws its two figures.*

`nla_digit_training.png` compares the NLA learning curves to LogitLens on a shared **answer-leading-digit** diagnostic. This separate diagnostic is not used as the baseline in the post; those complete-value baselines are documented in [CURVE_BASELINES.md](CURVE_BASELINES.md). An NLA prediction is the leading digit of the first numeric literal in its description. A lens prediction is its full-vocabulary top-1 token, which must be the gold digit token. Missing numbers and non-digit lens tokens count as failures. Only one prediction is scored per description; later number mentions do not increase this score. This is a literal mention diagnostic, not a test of the number's semantic role or faithful computation.

The complete-answer containment and judged intermediate-value metrics are retained in `nla_value_training.png`. They cannot be compared directly to the much easier first-digit lens score. Neither figure establishes faithful intermediate computation.

## One selection, two fixed baselines

The strong baseline selects a single layer and thought position globally. Selection uses the 807 CODI-correct saved lens rollouts at GSM8k-Aug-NL indices 4000–4999 and searches the 37 hidden-state readouts at the six thought positions. Every reader evaluation index is at least 5000, so none contributes to selection. The criterion is full-vocabulary top-1 answer-leading-digit recovery. It selects **post-final-norm hidden state 36, thought 1**, at 805/807. There is no per-example maximization, no selection over evaluation labels, and no reader-checkpoint selection. This is LogitLens only; the fitted J-lens is not used here.

The selected location is evaluated separately on each reader's 60 held-out questions (108 distinct questions in total). These cohorts differ and should not be compared as paired examples. The selection capture used the answer-only suffix; the matched-cohort capture uses the reader protocol, raw question + EOS + BOT, with explicit padding masks. Selection across the different prompt protocols does not guarantee that this location is optimal on the reader cohorts.

The **same-input** baseline applies the same checkpoint's final norm and output head to the exact bundled first-thought h24 or z vector. This gives LogitLens the same single-vector input as the corresponding NLA evaluation. The **best-location** baseline has access to h36 instead; it is a stronger reference with different input access, not an equal-input comparison. Its already-normalized h36 vector goes straight through the head without a second norm.

The best-location vectors are a fresh batch-2 recapture using the same pinned CODI checkpoint. `av_lens_readouts.json` records the capture environment and comparisons between the newly captured h24/z vectors and the original evaluation vectors. Small bf16/kernel/batch differences remain possible. The original reader capture batch size was 64. The same-input baseline uses the original saved vectors exactly.

The selected lens scores 57/60 in the z-reader cohort and 55/60 in the h24-reader cohort. Both same-input baselines score 0/60. Minimum fresh-versus-saved first-vector cosine similarities are 0.999977 for z and 0.999849 for h24. At step 3200, the NLA first-number digit rates are 4/60 and 12/60 respectively; those descriptions are free-form explanations, so the first number need not have the answer's role. These scores do not assess the NLA's ability to recover intermediate values elsewhere in its text.

## Included evidence

`artifacts/av_history.json.gz` includes all 96,120 complete descriptions at 801 reader steps per stream, with complete-answer containment flags and all 83,304 eligible saved judge verdicts and justifications. The offline command re-parses the answers and checks every UID, step, eligibility flag and cohort join. At step 3200 these histories' descriptions equal the existing per-position reader records at thought 1 byte for byte.

`artifacts/av_best_lens_vectors.npz` holds the 108 recaptured h36 vectors. `artifacts/av_lens_readouts.json` holds top-five token IDs, decoded tokens, unclipped float32 log probabilities, the full selection grid and capture checks. `av_summary.json` records all raw rates, baseline counts and the empirical majority-digit reference. The figures show faint raw measurements and a bias-corrected EMA with a 25-checkpoint half-life (100 reader steps). The reconstruction learning curves remain a separate figure because FVE has a different meaning and scale.

Offline, run `python -m codi_repro reproduce`. For a fresh GPU baseline:

```bash
python -m codi_repro av-lens --cache-dir /path/to/fast/cache --out /path/to/new/baseline --batch-size 2
```

This command checks the organism fingerprint, respects the existing CUDA environment, re-selects the location from the included disjoint records, captures the matched questions, and reads the exact original input vectors. It does not overwrite bundled evidence. All reader checkpoints contribute only to training diagnostics; the reported per-position descriptions and interventions still use the fixed step-3200 readers. The complete intermediate reader checkpoint ladder is not bundled, so a fresh rescan of that history requires recovering or retraining it.
