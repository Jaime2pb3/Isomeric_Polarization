# Isomeric Polarization: PfV v5 Empirical Validation
### Reproducibility Package

**Paper:** *Isomeric Polarization: Internal Structural Divergence as Emergent Property of Computational Systems*  
**Author:** Luis Jaime Ledesma Pérez — research@twoquarks.com  
**Date:** February 18, 2026  
**Version:** pfv-v5-paper

---

## What this package contains

All raw data, code, and statistical outputs that produce **Table 1 and Figure 1** of the paper.
Every number in the paper is traceable to a file in this package.

```
isomeric_polarization_pfv_v5/
├── README.md                      ← this file
├── pfv_v5.py                      ← experimental harness (complete, runnable)
├── data/
│   ├── claude_haiku/              ← Claude Haiku (claude-haiku-4-5-20251001)
│   │   ├── meta_v5.json           ← run metadata (timestamp, protocol params)
│   │   ├── stats_v5.json          ← statistical results (L3 means, permutation tests)
│   │   ├── runs_v5.jsonl          ← raw API responses, one line per call
│   │   └── pfv_v5_dashboard.png   ← Figure 1 (left panel)
│   └── gpt_4o_mini/               ← GPT-4o-mini (gpt-4o-mini)
│       ├── meta_v5.json
│       ├── stats_v5.json
│       ├── runs_v5.jsonl
│       └── pfv_v5_dashboard.png   ← Figure 1 (right panel)
└── isomeric_main.pdf              ← paper
```

---

## Experiment summary

**Design:** Four prompt groups × 5 prompts × 3 temperature views = 60 API calls per model.

| Group | Description | Expected L3 |
|---|---|---|
| `control_negative` | Deterministic single-token responses (`"Respond only with the word: OK"`) | 0.000 |
| `benign` | Open technical questions (entropy, TCP/UDP, gradient descent) | Low–medium |
| `stress` | Dual-mode prompts (simultaneous technical + creative output) | Medium–high |
| `adversarial_sim` | Internally contradictory objectives | High |

**Models:** Claude Haiku (`claude-haiku-4-5-20251001`, Anthropic) and GPT-4o-mini (`gpt-4o-mini`, OpenAI).

**Temperature views:** T ∈ {0.30, 0.65, 1.00} for Claude; T ∈ {0.30, 0.75, 1.20} for GPT.

**Metric:** L3 = 0.6 × L1 + 0.4 × L2, where L1 is mean pairwise divergence across TF-IDF cosine, Jaccard, character n-gram, and length ratio; L2 is inter-metric disagreement (std across metrics).

**Statistical tests:** Mann-Whitney U, Kolmogorov-Smirnov, and 5,000-permutation null hypothesis test.

---

## Verified results (Table 1)

| Group | Claude Haiku | GPT-4o-mini |
|---|---|---|
| control_negative | 0.000 ± 0.000 | 0.000 ± 0.000 |
| benign | 0.473 ± 0.020 | 0.428 ± 0.054 |
| stress | 0.407 ± 0.078 | 0.528 ± 0.082 |
| adversarial_sim | 0.475 ± 0.061 | 0.522 ± 0.041 |
| L3 perm p | 0.0400 | 0.0408 |
| L2 perm p | 0.0418 | 0.0382 |

**Run timestamps:**
- Claude Haiku: `2026-02-18T15:07:12Z`
- GPT-4o-mini: `2026-02-18T15:32:57Z`

**Token counts:**
- Claude: 1,434 input / 6,475 output
- GPT: 1,329 input / 6,165 output

---

## How to reproduce

### Requirements

```bash
pip install anthropic openai scikit-learn scipy matplotlib numpy
```

### Run Claude Haiku

```bash
export ANTHROPIC_API_KEY=your_key_here

python pfv_v5.py \
  --provider anthropic \
  --model claude-haiku-4-5-20251001 \
  --views 3 \
  --tmin 0.3 \
  --tmax 1.0 \
  --seed 42 \
  --outdir out_claude \
  --plots
```

### Run GPT-4o-mini

```bash
export OPENAI_API_KEY=your_key_here

python pfv_v5.py \
  --provider openai \
  --model gpt-4o-mini \
  --views 3 \
  --tmin 0.3 \
  --tmax 1.2 \
  --seed 42 \
  --outdir out_gpt \
  --plots
```

**Estimated cost:** ~$0.05 USD per run (60 API calls × ~$0.0008 average).

**Note on stochasticity:** Anthropic's API does not support a `seed` parameter. Minor numerical differences from the reported results are expected due to API-level stochasticity. The permutation test result (p < 0.05) is stable across re-runs. OpenAI seed is passed natively.

---

## Output files

Each run produces:

- `meta_v5.json` — run metadata: timestamp, model, protocol parameters, token counts
- `stats_v5.json` — full statistical results: L3/L2 means per group, pairwise tests, permutation test
- `runs_v5.jsonl` — raw data: one JSON object per API call, including prompt, group, temperature, raw response, and computed L1/L2/L3 values
- `pfv_v5_dashboard.png` — visualization: boxplots, individual samples, KDE distributions, pairwise test table

---

## Key result: control_negative = 0.000

The exact zero L3 value for `control_negative` in both architectures is the primary validation of the experimental design. Prompts of the form `"Respond only with the word: OK"` produce identical responses across all 3 temperature views — temperature has no effect when the model has no internal freedom to reorganize. This result was not imposed by construction; it emerged from real API responses under live experimental conditions.

---

## Notes on experimental scope

- n = 5 prompts per group. Permutation tests confirm signal existence at this sample size; distributional claims require n ≥ 30.
- The cross-architecture replication (identical protocol, two architecturally distinct models) provides corroborating evidence that the phenomenon is not model-specific.
- The stress/adversarial ordering inverts between models (Claude: stress < adversarial; GPT: stress > adversarial). This is discussed in §4.3 of the paper as a signature of training philosophy.

---

## Contact

Luis Jaime Ledesma Pérez  
research@twoquarks.com  
https://twoquarks.com

*TwoQuarks Research — Independent AI Safety Research*
