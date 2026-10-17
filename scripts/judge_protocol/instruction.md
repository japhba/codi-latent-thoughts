# Blinded systematicity review

You are given `packet.json`. Two anonymous readers (A and B) were each trained to describe the internal state of a reasoning model while it solves grade-school math problems. For each problem the packet shows the question, the gold intermediate values, the gold answer, and a reader's free-text description at four checkpoints of the reader's training (step 0 = before training, then 100, 200, 300). You do not know what the two readers read or how they differ. Judge only from the text.

Write a structured report with these sections:

1. **Reader A systematicities.** Recurring properties of A's descriptions: genre and register of the text, whether descriptions are about the actual problem (topicality), whether numbers in the descriptions match gold intermediates or the answer (quote exact matches), per-problem stability across steps (does the same problem get the same kind of description at every step), and how descriptions change from step 0 to 300.
2. **Reader B systematicities.** Same dimensions.
3. **A vs B contrast.** The sharpest differences, each backed by at least two quoted examples (cite problem uid and step).
4. **Trajectory over training.** For each reader: does grounding in the actual problem improve, degrade, or stay flat across steps 0/100/200/300?
5. **Verdict scores.** For each reader give, on a 0.0 to 1.0 scale with 0.1 granularity: topicality (descriptions are about this problem at all), value recovery (gold intermediate or answer values appear), and stability (per-problem consistency across steps). Include a one-sentence justification per score.

Rules: quote evidence verbatim with uid and step for every claim. Do not speculate about what the readers are or how they were built. Full sentences, no tables of raw quotes without interpretation.
