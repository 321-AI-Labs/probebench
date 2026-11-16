# ProbeBench

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Did a weight edit change how a model refuses? ProbeBench answers with a number, not a vibe.**

ProbeBench is a behavioral-drift verification protocol for weight-edited
language models: 120 pre-registered prompts in four categories, three
independent LLM judges applying one identical rubric, per-category refusal
rates, pairwise judge agreement, and a Behavioral Drift Index (BDI) computed
variant-against-own-base. Judge biases are measured separately and backed out
of the tables before any headline number, under binding rules that refuse
corrections without a mechanism surface or significance.

Study page and paper: [ProbeBench: Measuring Behavior Drift in Knowledge-Edited
Language Models with Bias-Corrected AI Judge Panels (321 AI Labs
TR#2)](https://321ai.xyz/studies/probebench/)
| [PDF](https://321ai.xyz/studies/probebench/P2-probebench-report.pdf)

## Run it

Requirements: Python 3.9+ (standard library only, no dependencies).

```bash
pip install -e .

# offline self-check: schema, parser, metrics, determinism proof (mock judge)
probebench selftest

# one-command run: subject answers all 120 prompts, three judges score
probebench run http://localhost:8000/v1 --subject-model my-edited-variant

# re-score stored logs; add --base-results for the drift index vs the base
probebench score --results results/<variant-run> --base-results results/<base-run>
```

`<model>` is any OpenAI-compatible base URL. A live-judge interlock refuses
contact with the lab judge endpoints unless `--live-judges-acknowledged` is
passed, so offline tests and live scoring runs differ in the CLI surface, not
just in procedure. A deterministic mock judge (`probebench mock-judge`) covers
offline demos and CI.

## Headline results (P2 study, 321 AI Labs, Nov 2026)

Measured on three edited-model pairs, six scoring arms, 120/120 answers ok in
every arm, zero transport errors in final state:

| Pair | Suite BDI (uncorrected) | Suite BDI (corrected) | Delta |
|---|---|---|---|
| p2 (Qwen2.5-7B edit) | 0.0639 | 0.0709 | +0.0070 |
| p3 (FuseChat-Qwen-2.5-7B edit) | 0.0597 | 0.0622 | +0.0024 |
| demo (quantization regimes) | 0.0153 | 0.0189 | +0.0036 |

- Correction **raises** measured drift on all three pairs. The mechanism is
  coverage asymmetry: per-judge constant biases cancel in the drift index only
  when every judge covers every cell, and one judge's reduced coverage broke
  that symmetry.
- Drift concentrates where the protocol was designed to look:
  jailbreak-adjacent corrected BDI 0.2333 (p2) and 0.2234 (p3), against
  near-zero values in the three compliance categories.
- Out-of-range corrected values are reported as-is and flagged, never clipped
  (two corrected kappa cells, -0.5991 and -0.5736).
- The derivation is deterministic: the correction pipeline executed twice
  produces byte-identical outputs.

Coverage disclosure, carried verbatim from the report: one judge's per-category
n_valid is reduced (15/17/23/30 across categories on the p2 pair vs 30/30/30/30
for the other two judges); the affected cells carry their own n_valid in the
source artifacts and no values are imputed.

## The suite

| file | role |
|---|---|
| `suite/prompts.jsonl` | 120 prompts, 4 categories x 30, every row with rubric + provenance |
| `suite/metrics.md` | pre-registered metric definitions (written before any scoring run) |
| `suite/SOURCES.md` | per-source citations: title, authors, URL, license, what was taken, access date |
| `suite/benchmark-card.md` | what it measures, what it does not, limitations, intended use |

Categories: `borderline-compliance` (lawful but edgy; compliance expected),
`benign-adjacent` (over-refusal traps, XSTest/OR-Bench style),
`dual-use-framing` (legitimate purpose, misuse-capable topic), and
`jailbreak-adjacent` (refusal-expected items at the published-benchmark level,
the direction where a refusal-degrading edit shows up). Twelve oracle anchors
with known-safe ground truth calibrate the judges.

## Limits (honest ones)

Three judges reduce single-judge bias without eliminating it. 120 prompts
sample the space; passing ProbeBench is a bounded drift measurement, not proof
of preservation, and BDI says nothing about capability preservation. The demo
quantization claim test rests on one model pair. See the paper's threats-to-
validity section for the full list.

## Citation

See `CITATION.cff`. Zenodo DOI backfills here after the v1.0.0 GitHub release
mints it.

```bibtex
@software{casey2026probebench,
  author = {Casey, Kiyoshi},
  organization = {321 AI Labs},
  title = {ProbeBench: Measuring Behavior Drift in Knowledge-Edited Language Models with Bias-Corrected AI Judge Panels},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/321-AI-Labs/probebench}
}
```

## License

Code: MIT, see `LICENSE`. Redistributed prompt data keeps its original
license: XSTest and OR-Bench items are CC-BY-4.0, AdvBench strings MIT; full
attribution and access dates in `suite/SOURCES.md`. All artifacts are research
use only.
