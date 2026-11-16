# ProbeBench — Benchmark Card (draft v0.1, 2026-09-14)

## What this is

ProbeBench is a **behavioral-preservation verification instrument for
weight-edited models** (edited variants produced by weight surgery such as
activation-direction ablation or comparable targeted edits). It measures one
thing: **did refusal behavior change between a variant and its own base**, and
it reports that change as a per-category index (BDI) with full publication of
where independent judges disagree.

Framing statement (binding on all artifacts in this repo): this tooling exists
for **model-integrity verification** — helping anyone who publishes an
edited-variant demonstrate, with a third-party-runnable protocol, exactly how
its refusal behavior differs from its base. Required vocabulary: edited model /
edited-variant, weight surgery / weight edit, behavioral preservation, refusal
behavior, base model.

## What it measures

- 120 prompts, 4 categories × 30: `borderline-compliance`,
  `benign-adjacent`, `dual-use-framing`, `jailbreak-adjacent`.
- Per-item: three independent LLM judges apply an identical, pre-registered
  rubric and return a binary refusal verdict (plus a partial flag).
- Per-category: each judge's refusal rate, pairwise Cohen's κ, and the
  **Behavioral Drift Index** — a total-variation-form distance between the
  variant's and its own base's refusal-rate vectors (exact definition in
  `metrics.md`, pre-registered 2026-09-14 before any scoring run).
- 12 oracle anchors with known-safe ground truth calibrate the judges; a judge
  that scores factual compliance on an anchor as a refusal is reported as
  miscalibrated rather than silently trusted.
- The judge **disagreement table is a first-class output**, published per run.

## What it does NOT measure

- Capability, coherence, or factuality preservation (use a perplexity/KL
  floor for that; ProbeBench is refusal-behavior only).
- Safety or quality of any model in absolute terms. Refusal rates are only
  interpreted as variant-vs-own-base deltas; a single number for one model is
  descriptive, not evaluative.
- Robustness against determined adversaries. Categories sample *adjacent*
  request shapes at the published-benchmark level; this is a drift detector,
  not a security audit.
- Anything about model weights: ProbeBench is black-box (OpenAI-compatible
  endpoints only) and redistributes no weights.

## Intended use

1. A variant publisher (or third party) runs the suite against the variant and
   its own base with the same protocol, and publishes both BDI tables.
2. Reviewers re-run `probebench score` on the stored logs and check judge
   disagreement themselves.
3. Researchers study judge disagreement: the disagreement table is data, not
   noise to be tuned away.

## Known limitations

- **LLM judges are imperfect.** All judge-derived numbers carry the label
  `uncorrected — pending P8` until a published judge-corrections layer is
  applied. Three judges reduce single-judge bias but do not eliminate it; κ and
  the disagreement table quantify the residue.
- **Judge prompts can be format-sensitive.** The rubric template is fixed and
  versioned; changing it invalidates cross-run comparisons.
- **Category coverage is not exhaustive.** 30 items per category samples the
  space; a variant could pass at 120 items and still drift on unprobed shapes.
- **Subject decoding matters.** Temperature 0 is required; server-side
  nondeterminism (batching, quantized serving) adds noise that BDI can inherit.
- **Anchors calibrate, not certify.** 12 anchors catch gross judge
  miscalibration; they cannot prove a judge is correct on non-anchor items.

## Data provenance and licensing

- Every prompt carries a mandatory citation (see `SOURCES.md`); the validator
  rejects any prompt whose `source_id` does not resolve.
- Source instruments: AdvBench (MIT), XSTest (CC-BY-4.0), OR-Bench (CC-BY-4.0);
  JailbreakBench cited for methodology. All verified live 2026-09-14.
- Probe items are bare requests at the found-in-literature level; the suite
  contains no operational harmful content. It measures refusal of a request;
  it does not supply the means to fulfill one.

## Out of scope by design

This is not an attack toolkit and does not evaluate or improve
circumvention techniques. Categories containing refusal-expected items exist so
that a weight edit which **degrades** refusal behavior is detectable and
quantifiable — the mirror image of the over-refusal detection the benign
categories provide. Behavioral preservation means both directions are measured.
