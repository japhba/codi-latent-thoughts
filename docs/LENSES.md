# Corrected LogitLens and J-lens analysis

This document describes the corrected historical 1,200-rollout replay using the recovered maps. The current blog curve references use freshly fitted, disjoint all-layer maps; see [ALL_LAYER_LENSES.md](ALL_LAYER_LENSES.md).

All organism measurements use checkpoint-7580, with the weight fingerprint in `manifest.json`. This rerun reads the full recovered activation capture; it does not select a new model checkpoint or regenerate the historical population.

## Corrections

1. Qwen's `hidden_states[36]` is already normalized. It now goes directly through CODI's output head. Earlier states, and Jacobian transports into the pre-normalization final residual basis, receive final RMSNorm once.
2. The output head comes from the full CODI checkpoint. A byte-level audit finds that its first 151,669 ordinary-token rows and final norm are identical to the pinned base model. The base head has 151,936 rows; CODI has 151,671, including its two new delimiters. The head concern was about vocabulary rows, not changed ordinary-token weights.
3. The fed-vector row uses exact captured input embeddings at thoughts 1–6, instead of a float32 re-projection of saved final states.
4. Digit log probabilities remain in float32 without clipping. The older digit survey stored float16 probabilities and floored them at 1e-6 before ranking, which can erase differences between low-probability digits.
5. Population scores replace leading-digit tinting on an answer-selected example. Exact single-token values, leading-digit diagnostics, full-vocabulary ranks, and restricted ten-digit accuracy are separate outputs. Example tables select the first three eligible records without inspecting lens predictions.

## Population and lens

The capture contains all 1,200 GSM8k-Aug-NL training rows 4000–5199. The historical rollout adds the answer-only instruction, EOS and BOT, takes six projected thoughts, and greedily decodes four answer tokens. The 973 saved correct rollouts define the scored population. These are organism training examples, not GSM8k test problems. They are held out from the Jacobian fit and differ from the NLA reader cohort.

The recovered CODI-specific J-lens was fitted on 96 correct problems with nonempty reference chains, starting at row 5200 and skipping 17 candidates. Its matrices map block outputs 0–34 into the final block's pre-RMSNorm basis. The embedding matrix averages over thought-input positions; its application to BOT is outside that fitted source-position range and is not a headline result.

The estimator follows the [reference implementation](https://github.com/anthropics/jacobian-lens/blob/main/jlens/fitting.py): sum cotangents over valid target positions and average gradients over valid source positions. Here those positions are BOT and the six thoughts; the embedding-source mean uses only the thoughts. Recorded thought inputs are fixed during teacher-forced replay. The Jacobian excludes recursive projection/feedback, and targets stop before EOT and the verbal answer. It is a thought-replay lens, not a Jacobian of the complete free-running organism. The existing checkpoint-matched maps were reused; fixing the normalization/readout errors did not require refitting them.

## Results

| Readout | Population | Result |
|---|---|---|
| First thought, final layer: answer leading digit | 973 saved correct rollouts | 969/973 top-1; 973/973 top-5 |
| First thought, final layer: complete single-token answer | 100 eligible correct rollouts | 98/100 top-1; 100/100 top-5 |
| First thought, h33: answer leading digit | 973 saved correct rollouts | J-lens 83.2% top-5; LogitLens 57.8% |
| h24 or exact fed vectors: complete single-token answer | Same 100 rollouts, all six thoughts | 0/100 full-vocabulary top-5 for both lenses |
| Second thought, final layer: distinct reference value | 379 eligible correct rollouts | 182/379 top-5 (48.0%); conditional shuffle expectation 27.4% |
| Second thought, h24: distinct reference value, highest-ranked digit | Same 379 rollouts | LogitLens 24.3% versus 11.8% shuffled; J-lens 15.8% versus 9.1% shuffled |

The distinct-reference target is the first single-digit numeric mention in the reference CoT that is neither the final answer's leading digit nor among the question's numeric literals. It need not be a computed intermediate: a reference can write “2” where the question says “twice.” Recovery does not establish a verified arithmetic state or causal use.

Shuffle expectations preserve empirical label frequencies. Reference-value controls additionally shuffle within answer-leading-digit groups. They are exact permutation expectations, not assumed 10% chance levels or significance tests. The majority reference-digit prior is 28.8%, so the h24 ten-digit classifiers remain weaker than that prior despite exceeding matched shuffle controls. Restricted digit ranking is not full-vocabulary or whole-number decoding.

With the checkpoint head fixed, double normalization gives answer-leading-digit top-1 recovery of 798/973 at thought 1; the correction gives 969/973. Across 8,400 positions, the bug changes 4,762 top-1 predictions and 7,824 top-five sets; mean KL(correct || double-normalized) is 5.44 nats.

These results support answer readability near the first thought's upper layers, some reference-number signals, and poor literal readouts from fed vectors. They do not support calling the whole organism opaque, establish faithful intermediate reasoning, or establish greater opacity than the earlier Llama study without a matched comparison.

## Reproduction

`python -m codi_repro reproduce` checks all artifact hashes and regenerates the original five figures plus four corrected lens PNGs. `python -m codi_repro lens-analyze` runs just the lens analysis. Compact top-five outputs, unclipped digit log probabilities, digit ranks, complete questions/reference CoTs, and token strings are bundled. These commands need no model downloads.

Fresh GPU capture, fitting, and scoring use the pinned public CODI checkpoint:

```bash
python -m codi_repro lens capture --cache-dir /path/to/fast/cache --out /path/to/fast/capture
python -m codi_repro lens fit --cache-dir /path/to/fast/cache --out /path/to/fast/fit --dim-batch 1
python -m codi_repro lens readouts --cache-dir /path/to/fast/cache --acts /path/to/fast/capture/lens_activations.npy --problems /path/to/fast/capture/lens_problems.json --lens /path/to/fast/fit/jacobian_lens.pt --out /path/to/fast/readouts
python -m codi_repro lens-analyze --inputs /path/to/fast/readouts --out /path/to/figures
```

Fit resume state saves full matrix sums, selected IDs, data cursor, CPU/CUDA RNG states, and the checkpoint specification after every candidate. Use the same arguments plus `--resume`. Keep output on fast disk: matrix/resume files are about 2.4 GB, and a full fit is expensive. Larger GPUs permit increasing `--dim-batch`. The 113 supplied historical fit candidates may not yield 96 eligible problems under changed arithmetic or libraries; exhaustion fails loudly. Fresh captures/refits are approximate, and selected IDs are recorded.

The full historical activation array (2.55 GB) and Jacobian file (2.42 GB) are fingerprinted in `artifacts/lens_provenance.json` but are not bundled or publicly hosted. With them supplied locally:

```bash
python -m codi_repro lens readouts --recorded --cache-dir /path/to/fast/cache --acts /path/to/codi_faithful_acts.npy --lens /path/to/codi_thought_lens.pt --out /path/to/fast/readouts
```

`--organism-dir` accepts the full local checkpoint, always SHA-256 checked. `lens audit` additionally accepts `--base-dir` for the pinned base snapshot and `--acts` for the capture. Without a base override, it downloads only tokenizer metadata and the shards containing head/norm. Saved audits isolate normalization effects and check weight equality.

Nine numerical/harness tests cover final-norm fidelity, Jacobian orientation, the future-target gradient reduction, sample/target axis alignment, and cached rollout/patch behavior. Three fresh GPU recaptures preserve the recorded answers, with final-state cosines 0.9985–0.99998. This checks the loading/prompt path, not bitwise recovery of the complete historical capture.
