# DiEP with 80 search steps — run card

**What this is.** A second DiEP study that changes one thing: the number of
search epochs over the calibration set. It answers whether the first study
under-trained DiEP, and whether DPMA still beats it once DiEP gets the paper's
number of optimisation steps.

- Results: [`ANALYSIS.md`](ANALYSIS.md); **section 4a** is the 20-step vs 80-step
  comparison and whether DPMA still wins. Figures are in [`figures/`](figures/).
  Both are rebuilt after every finished job.
- The first study is untouched in [`../`](../) (`results/`), label `diep`.
  `study.json` in each folder records its settings, and the runner refuses to
  write into a folder that holds a different study.
- Command: `./scripts/run_epochs40.sh` (one command; restartable).

---

## The one change

| | first study (`results/`, `diep`) | this study (`results/epochs40/`, `diep-80steps`) |
|---|---|---|
| search epochs | 10 (the spec's value) | **40** |
| optimiser steps | 20: 15 α / 5 β | **80: 60 α / 20 β** — the paper's exact count |
| cosine schedules | α over 15 steps, β over 5 | α over 60 steps, β over 20 |

**Why.** Steps = epochs × (sequences / batch size). The paper's 128 calibration
sequences at batch 16 make one epoch 8 steps, so 10 epochs is 80 steps. This
study's calibration set is 32 sequences, so one epoch is 2 steps and 10 epochs is
only 20. In the first study the search barely moved from its initialisation: the
largest |α| was about 0.03, β stayed within 0.993–1.005, and softmax(α) stayed
uniform to four decimals. The pruning ranking was therefore set by very small
differences. 40 epochs restores the paper's number of steps **on the same
tokens**: each of the 32 sequences is simply visited 4× more often.

## What stays identical

| | value |
|---|---|
| **calibration** | **`GA/MoE_compression_GA/scripts/results/calibration/c4-32x1024-seed0.pt`**, md5 `d0b30c732b0c` — the same file as the GA, DPMA and the first DiEP study |
| model | `allenai/OLMoE-1B-7B-0924`, bf16 |
| search | routed forward; λ = 0.01; batch 16 sequences per step; Adam lr 5e-3 for α and β, cosine to 0; 3:1 alternation; β ≥ 1e-3; H cached through the identity forward |
| pruning | score softmax(α)·β; global walk with per-layer floor 8; physical pruning |
| merge | Appendix A.5 exactly as in the first study (`diep/merge.py`) |
| evaluation | the GA's lm-eval protocol: 1-shot, `all` tasks, `mean_mc` over the 8 MC tasks |
| seeds | 0, 1, 2 (the seed changes the batch order each epoch) |

## The arms

| label | what | seeds | keep ratios |
|---|---|---|---|
| **`diep-80steps`** | plain DiEP, 80-step search | 0, 1, 2 | 0.75, 0.5, 0.25 |
| `diep-80steps-uniform` | same scores, `N·r` pruned per layer (spec §12's baseline) | 0 | 0.75, 0.5, 0.25 |
| `diep-80steps-merge` | + Appendix A.5 merge, same scores as seed 0 | 0 | 0.75, 0.5, 0.25 |

Adaptive skipping and the throughput benchmark are not rerun. Skipping lost
accuracy in the first study, and memory and speed depend only on how many
experts are kept, which is the same.

## How to read the comparison

- **`diep-80steps` vs `diep`** (ANALYSIS §4a): same everything except the search
  length. With 3 seeds each, the difference counts only if the two seed ranges
  do not overlap.
- **`diep-80steps` vs DPMA v3** (ANALYSIS §4a, last table): the question this
  study exists for. `dpma-v3-hard` is the compute-matched comparison (≤ 8 experts
  per token, like DiEP); `dpma-v3` uses the soft export (more experts per token).
- **Uniform and merge** (ANALYSIS §4 and §4b): the same checks as the first study,
  now on the longer search.

## Timeline

Started 2026-09-28 at 09:38 UTC on GPUs 1, 3, 5 and 7. There are 3 searches
(about 20 min each) and 15 evaluations (about 75 min each), so complete results
are expected around **15:00–15:30 UTC**.

## Layout

```
results/epochs40/
├── card.md              this file
├── study.json           {"epochs": 40, "method": "diep-80steps", "calibration": "c4-32x1024-seed0.pt"}
├── ANALYSIS.md, figures/
├── runs/seed{0,1,2}/    scores.pt (80-step search), pruned-r{r}-{global,uniform,merge}/
├── harness/             diep-80steps*.json, the GA schema
└── logs/, queue/
```
