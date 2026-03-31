# Isomeric Polarization

**Detecting Semantic Regime Transitions in Large Language Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

→ **[twoquarks.com](https://twoquarks.com)** | **[Preprint](https://twoquarks.com/isomeric_polarization.pdf)**

---

## Overview

Isomeric Polarization is a framework for detecting behavioral instability in language models at inference time — without modifying model parameters, weights, or training objectives.

The core idea: model collapse is not a capacity failure. It is a **stability failure** — observable, measurable, and detectable before it happens.

### Polarization Tensor

Given a set of responses `{Φᵢ(t)}` from isomeric configurations under contextual pressure:

```
Pt = Agg(Varᵢ∈It[Φᵢ(t)])
```

- **L1** — Mean pairwise semantic distance (aggregate drift)
- **L2** — Variance across distance metrics (disagreement)
- **ΔL3 = 0.6·L1 + 0.4·L2** — Composite polarization score

### Metrics (fixed, non-adaptive)

- TF-IDF cosine distance
- Jaccard token distance
- Character n-gram distance
- Length ratio

> A quark that learns can be trained to tolerate drift instead of detecting it. The operators are invariant by design.

---

## Probe Cases

| Case | Name | Strongest Signal |
|------|------|-----------------|
| C1 | Sycophancy Induction | Social validation pressure |
| C2 | Refusal Erosion | Boundary dissolution — **ρ=+0.713** |
| C3 | Anchor Displacement | Context replacement — **p=0.054** (strongest cross-arch) |
| C4 | Narrative Rule Override | Character/role injection |
| C5 | Reasoning Drift | Chain-of-thought steering |

---

## Empirical Results

Cross-architecture validation across Claude Haiku, GPT-4o-mini, and Mistral 7B.

- Statistically significant regime separation (p < 0.05, 5,000-permutation null)
- Control negative confirmed at L₃ = 0.000 in both architectures
- C2 Refusal Erosion: ρ = +0.713 confirmed
- C3 Anchor Displacement: strongest cross-architectural signal (p = 0.054)

---

## Implementation

Available as a Python package:

```bash
pip install twoquarks
```

→ [github.com/TwoQuarks/molecule](https://github.com/TwoQuarks/molecule)

---

## Citation

```bibtex
@techreport{ledesma2026isomeric,
  author      = {Ledesma, Luis Jaime},
  title       = {Isomeric Polarization: Detecting Semantic Regime Transitions in Large Language Models},
  institution = {TwoQuarks Research},
  year        = {2026},
  url         = {https://twoquarks.com/isomeric_polarization.pdf}
}
```

---

**TwoQuarks Research** · [twoquarks.com](https://twoquarks.com) · research@twoquarks.com
