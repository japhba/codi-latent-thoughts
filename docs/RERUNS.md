# All-thought review reruns

Batch `rv6` uses organism checkpoint-7580 throughout. No new reader training is
performed: the two final NLA adapters are step 3200. Earlier recoverable adapters
are evaluated only to reconstruct learning curves. The eight available steps are
400, 800, 1200, 1600, 2000, 2400, 2800 and 3200; disabling the adapter supplies the
warmstart point. Placeholder directories do not contain additional recoverable
weights, so an 801-point all-thought curve cannot be recovered.

Install Python 3.13 dependencies with `uv sync --locked --extra gpu --extra judge
--extra training`. Authenticate to Hugging Face;
set cache locations outside the checkout and home directory. Preserve the CUDA
device allocation supplied by the scheduler. These modules contain no SLURM or
machine-specific inference requirements.

The reader weights are on Hugging Face. `hf download cds-jb/codi_qwen3-8b-nla-z --local-dir /path/to/z` (likewise
`-nla-h24`) gives the AV adapters under `/path/to/z/adapters/iter_*` and the step-3200 AR under `/path/to/z/critic`;
the warmstart AV is `asher577/nla-warmstart-2x` (subfolder `av`), the AO LoRAs are `cds-jb/codi_qwen3-8b-ao`.

## Six-description reader curves

For each source, use the exact bundled 60-question evaluation cohort and six
saved vectors per question. Readers receive one vector at a time. Pool the six
independent descriptions for scoring, without composing numbers across them.
Unrelated donor vectors use the original global shift of seven questions.

```bash
python -m codi_repro.all_thought_readers --stream z \
  --reader-root /path/to/z/adapters --reader-dir /path/to/z/adapters/iter_003200 \
  --reader-base-dir /path/to/warmstart/av --cache-dir /path/to/cache \
  --out /path/to/results/readers/z --group rv6 --resume
python -m codi_repro.all_thought_readers --stream h24 \
  --reader-root /path/to/h24/adapters --reader-dir /path/to/h24/adapters/iter_003200 \
  --reader-base-dir /path/to/warmstart/av --cache-dir /path/to/cache \
  --out /path/to/results/readers/h24 --group rv6 --resume
python -m codi_repro.all_thought_scores \
  --inputs /path/to/results/readers/z /path/to/results/readers/h24 \
  --out /path/to/results/scores --concurrency 6 --resume
```

Scoring requires OpenRouter. Extract complete numeric literals, spelled-out
numbers and unambiguous arithmetic from descriptions without questions or gold
labels, requiring exact evidence quotations. Parse complete numeric literals
deterministically, including the numeral in a percentage, as in the lens audit.
Reject a bare numeric quotation when the descriptions contain no complete
numeric occurrence with that canonical value; prefixes of different numbers and
identifier digits are insufficient. Preserve the boundary audit and raw responses.
Nonterminating arithmetic decimals use at most six significant digits, matching
the downstream canonical numeric representation. After extraction, intersect with the same answer and non-answer reference-value
sets used for the lens ensembles. Preserve raw judge responses and any explicit
gold-blind evidence corrections. Semantic permutation expectations compare the
same extracted sets against every label set in the eligible cohort. These scores
measure value coverage, not correct roles, precision, or faithful reasoning.
Also report reference values that are not numeric literals in the question,
with their own eligible denominator. This diagnostic removes literal given-value
echoes; it does not identify a number's role or exclude every spelled-out given.

The two source cohorts differ; each AV is paired with lenses on its own exact
questions and six vectors. An AV produces substantially more output text than a
top-five token lens. Equal input does not imply equal output capacity or training.
The main comparison restricts lenses to the AV source layers; the appendix uses
all-layer/six-position candidate-coverage skylines. AO histories already use all
six thought positions and need no position-aggregation rerun.

Historical directional FVE evaluates accepted descriptions only. Its curve is a
separate retained training diagnostic, not a new all-thought reconstruction curve
or an unconditional reconstruction metric. Historical AR checkpoints are not
available to regenerate that entire curve.

## Masked recurrent ablations

```bash
python -m codi_repro.recurrent_controls \
  --organism-dir /path/to/checkpoint-7580 --cache-dir /path/to/cache \
  --out /path/to/results/controls --group rv6 --resume
```

Every cached thought and answer pass has its full padding mask. Use all 1319
original GSM8k questions, batch size 32, and select six-only successes from fresh
six/zero decodes. Compare zero, mean, noise and unrelated-donor corruption at
each of six positions, either replaying later thought inputs or regenerating
them. Also evaluate native thought budgets, frozen single drops, reversal and
repetition. Deleting an iteration and regenerating the remaining autonomous
recurrence is a shorter rollout, not a uniquely identified original-thought drop.

## Matched twin interventions

```bash
python -m codi_repro.twin_controls --action capture \
  --organism-dir /path/to/checkpoint-7580 --cache-dir /path/to/cache \
  --out /path/to/results/twins --group rv6 --resume
python -m codi_repro.twin_controls --action raw \
  --organism-dir /path/to/checkpoint-7580 --cache-dir /path/to/cache \
  --out /path/to/results/twins --group rv6 --resume
```

Retain all 296 solver-verified candidates from 81 original problems. Recheck
both-correct and both-six-only membership with fresh masked decodes, without
discarding excluded records. Capture every raw block output before final
normalization. Exact own-column replay must preserve answers and all later
projected vectors. Compare own and original-donor transplants for z, h24 and full
residual columns at all six positions jointly, plus per-position diagnostics.
These interventions replace residual outputs and projected inputs, not complete
KV caches or the receiver's earlier question cache.
The old normalized-final-state substitution can affect a single-column patch
through downstream projection. When every next projected input is overwritten,
that final-output route is bypassed; do not attribute an all-six adoption change
to this normalization bug alone. The refreshed mask, population and protocol
also differ from the historical measurements.

For each `stream` (`z` and `h24`), run `--action describe`, `--action reconstruct`
and `--action language` with the same arguments above plus:

```bash
--stream z --reader-dir /path/to/z/adapters/iter_003200 \
--reader-base-dir /path/to/warmstart/av --critic-dir /path/to/z/critic
```

The final AV independently describes all six vectors of both problems. Edit
unambiguous complete numeric intermediate literals in receiver descriptions
toward the original donor's reference values; exclude changed givens and final
answers. Record edit coverage. This is a restricted literal-edit intervention,
not an LLM paraphrase or a semantic rewrite of all occurrences, units and roles.
Repeated reference values with conflicting destinations are excluded.

The matching final AR encodes the complete original and edited descriptions,
using its trained critic template, right padding, raw truncated residual output,
and FP32 value head. Predictions supply directions. All receiver-side language
conditions use the same true receiver vector norm. Compare own reconstruction,
targeted edits, donor descriptions, and random controls matching each edit's
angular displacement from the own reconstruction. Zero-size edits have zero-size
random controls. Save checkpoint hashes and unconditional directional fidelity.

Primary contrasts patch all six positions jointly. Report donor-answer adoption
and receiver-answer retention both before and after restricting to successful
own round trips, with explicit eligible denominators. Bootstrap original
problems, retaining their related twins together (5000 draws, seed 11). Failure
to transfer the expected answer does not establish absence of causal content.

## Resumption, plots and publication

All GPU modules atomically commit batch shards and support `--resume`. Long
inference reruns can restart after preemption without losing completed batches.
Run a separate small `--limit` validation directory before the full cohort;
never change the protocol or source code while resuming a full run. W&B runs use
group `rv6`. The older unmasked inference modules remain historical code; use
these commands for the corrected measurements.

```bash
python -m codi_repro.bundle_review --runs /path/to/results --out /path/to/bundle \
  --critic-z-dir /path/to/z/critic --critic-h24-dir /path/to/h24/critic
python -m codi_repro.review_results --controls /path/to/bundle/controls \
  --scores /path/to/bundle/scores/all_thought_scores.json \
  --twins /path/to/bundle/twins --out /path/to/figures
```

Compact generations, extraction audits, predictions, configuration, hashes and
scores can reproduce the plots offline. Raw captures and large checkpoints stay
in external storage. The final reader adapters and matching AR weights must be
made available before claiming a fully public end-to-end inference reproduction.
No command publishes or pushes them.
