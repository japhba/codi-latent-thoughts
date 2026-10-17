# What's in CODI's latent thoughts?

Code, data and figures for the blog post *What's in CODI's latent recurrent thoughts?*. We try to read the
continuous thoughts of a Qwen3-8B CODI model
([`cds-jb/codi_qwen3-8b-answer_only`](https://huggingface.co/cds-jb/codi_qwen3-8b-answer_only), checkpoint-7580)
with LogitLens and J-lens, natural-language autoencoders (NLAs) and activation oracles (AOs), and intervene on them
by transplanting thoughts between twin problems and by editing NLA verbalizations.

## Reproduce every figure

```bash
uv sync --locked
uv run playwright install chromium
uv run python -m codi_repro reproduce
```

This needs Python 3.13 and [uv](https://docs.astral.sh/uv/), runs on a CPU in about a minute, and uses no GPU,
model weights or API keys. `reproduce` checks every bundled record against its SHA-256 (`manifest.json`,
`artifacts/rv6/bundle.json`), recomputes the statistics into `results/`, and redraws every figure in the post into
`figs/`. `python -m codi_repro validate` runs only the checks, and `python -m codi_repro figures` only redraws. The
five HTML-rendered figures use playwright's chromium, which needs the system libraries libnss3 and libnspr4.

[MANIFEST.md](MANIFEST.md) maps each figure to its script and inputs, and each input to the code that measured it.

## Layout

| path | contents |
|---|---|
| `codi_repro/` | measurement and statistics package: CODI loading and masked rollouts, controls, lenses, NLA reader evaluation and scoring, interventions |
| `figures/` | one script per post figure |
| `figs/` | the post's figures |
| `results/` | statistics computed by `reproduce`: bootstrap intervals, ablation, twin and lens summaries, the J-lens fit curve |
| `artifacts/` | measured records: decodes, readouts, verbalizations, extraction and judge outputs, vectors, training curves; `rv6/` holds the corrected reruns |
| `scripts/` | the original experiment scripts as they ran on the cluster ([scripts/README.md](scripts/README.md)) |
| `training/` | CODI and NLA training recipes, and our patch to the [cywinski/codi](https://github.com/cywinski/codi) training code ([training/README.md](training/README.md)) |
| `vendor/EasyNLA/` | the pinned [EasyNLA](https://github.com/asherps/EasyNLA) trainer (MIT) |
| `docs/` | method notes: GPU reruns, lens protocol, provenance, validation log ([docs/README.md](docs/README.md)) |
| `tests/` | numerical and API tests |

## What the records cover

| measurement | bundled evidence | interpretation |
|---|---|---|
| Six versus zero thoughts | all 1319 GSM8k test questions, complete masked decodes | paired, checkpoint-specific accuracy comparison |
| Recurrent ablations | 61 conditions on all 1319 questions | frozen versus regenerated later thoughts, on the problems that need their thoughts |
| NLA learning curves | 12960 descriptions, 2160 six-description extractions, nine recoverable checkpoints per reader | explicit answer and intermediate-value coverage, with unrelated-donor and semantic-permutation controls |
| AO learning curves | training histories, outputs and verdicts reading all six thought positions | training diagnostics with matching lens references |
| LogitLens and J-lens | all 37 hidden states and six positions on the readers' exact question cohorts | main figures use the readers' source layers, the appendix pools all layers |
| Twin transplants | all 296 verified twin candidates from 81 original problems, 98 conditions | common donor direction and population, bootstrap clustered by original problem |
| NLA edits | final verbalizer/reconstructor pair, complete original and edited descriptions, own and angularly matched random controls | literal intermediate edits, reported with round-trip retention and edit coverage |

Values are extracted from descriptions and lens tokens without access to the question or gold labels, then
compared with the reference solution. A value counts only if a single description or token expresses it
completely; numbers are never composed across candidates or positions. These scores measure value coverage, not
whether a value plays its computational role. NLA descriptions have more output capacity than five lens tokens.
The z and h24 reader cohorts have 60 questions each and share 12, so they are not a paired comparison; the
questions are held out from the readers but come from CODI's training corpus. Confidence intervals use 5000
bootstrap draws (seed 11), clustering twins by their original problem.

## Rerunning the measurements

```bash
uv sync --locked --extra gpu --extra judge --extra training
```

[docs/RERUNS.md](docs/RERUNS.md) and [docs/ALL_LAYER_LENSES.md](docs/ALL_LAYER_LENSES.md) give the GPU commands.
All weights are public. The organism, the base model (`Qwen/Qwen3-8B`) and the NLA warmstart
(`asher577/nla-warmstart-2x`) are pinned by revision in `manifest.json`. The trained readers and the AO are:

| model | contents |
|---|---|
| [`cds-jb/codi_qwen3-8b-nla-z`](https://huggingface.co/cds-jb/codi_qwen3-8b-nla-z) | NLA reading the fed-back thought vector z: verbalizer (AV) LoRA adapters at steps 400 to 3200 (`adapters/iter_*`), reconstructor (AR) at step 3200 (`critic/`) |
| [`cds-jb/codi_qwen3-8b-nla-h24`](https://huggingface.co/cds-jb/codi_qwen3-8b-nla-h24) | the same for the layer-24 residual stream h24 |
| [`cds-jb/codi_qwen3-8b-ao`](https://huggingface.co/cds-jb/codi_qwen3-8b-ao) | activation-oracle LoRAs on Qwen3-8B, every 1000 steps to 66000 (`loras/`) |

`manifest.json` and `artifacts/rv6/twins/critic_*_provenance.json` record the file hashes the measurements used. Value
extraction calls an LLM through OpenRouter (`OPENROUTER_API_KEY`), so reruns can drift with the service; all raw
responses are bundled. Model downloads honour `.env` and require an explicit cache directory.
`uv run python -m unittest discover -s tests -v` runs the tests (with the extras above).

[training/README.md](training/README.md) records the CODI and NLA training recipes. The original reader-training
captures and environment were not recovered, so exact retraining is not claimed. Source snapshots, environment
differences and the origin of every bundled record are in [docs/PROVENANCE.md](docs/PROVENANCE.md).

## License

MIT ([LICENSE](LICENSE)). `vendor/EasyNLA` keeps its own MIT license. `training/codi_qwen3.patch` modifies
[cywinski/codi](https://github.com/cywinski/codi); apply it to a checkout of that repository.
