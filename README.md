# DiEP — Differentiable Expert Pruning, on OLMoE

An implementation of **DiEP** (Bai et al., *Adaptive Mixture-of-Experts Compression
through Differentiable Expert Pruning*, NeurIPS 2025) following
[`specs.md`](specs.md). It is set up to be compared with the two other methods in
this repository, the genetic algorithm (`../GA/MoE_compression_GA`) and DPMA
(`../DPMA`), on the same terms:

- **the same model**: `allenai/OLMoE-1B-7B-0924`, 16 MoE layers × 64 routed experts, top-8;
- **the same calibration tokens**: the GA's `c4-32x1024-seed0.pt`, byte for byte;
- **the same evaluation**: the GA's lm-eval protocol (1-shot, 8 MC + 2 generative
  tasks), with the same result schema.

Results: [`results/ANALYSIS.md`](results/ANALYSIS.md) and [`results/figures/`](results/figures/). What was run, and the Appendix A.5 merge in detail: [`results/card.md`](results/card.md).

---

## The method, in three stages

| stage | what happens | spec |
|---|---|---|
| **search** | Model frozen, in `eval()`. Learn `alpha [L, N]` (intra-layer expert logits, init 0) and `beta [L]` (layer scales, init 1) by minimising `CE + 0.01 · ‖H' − H‖_F` on the calibration set, alternating 3 alpha-steps : 1 beta-step, two Adam optimisers with cosine schedules. The routed forward is `beta_l · Σ_{i∈S(x)} w_i · (N · softmax(alpha_l)_i) · FFN_i(x)`: the router is unchanged, and at init the model is exactly the original. | §4 |
| **prune** | Score `softmax(alpha_l)_i · beta_l`, walk all experts from the lowest score and remove the bottom `K = round(N·L·r)`, never taking a layer below `m_min = k = 8`. Then physically slice the router rows and expert weights. | §5 |
| **skip** (optional) | At inference, skip a lower-ranked selected expert when `w_ej < min(1, gamma1_j · gamma2(e0, ej)) · w_e0`; `gamma1` is the median weight ratio, `gamma2` the RBF-CKA similarity relative to the layer mean. | §7 |

**merge** (optional, off by default, §8): before pruning, each pruned expert is
folded into its most CKA-similar retained expert with softmax(CKA) weights.

The search does not depend on the sparsity `r`; only pruning does. One search per
seed therefore serves every ratio.

---

## What differs from the paper, and why

Every item below is forced by the comparison or by the model. The spec's own
**[DECISION]** choices (§13) are all kept.

| | paper / spec | here | why |
|---|---|---|---|
| model | Mixtral-8x7B, DeepSeek-MoE-16B, Qwen2-57B | **OLMoE-1B-7B** | the model the GA and DPMA compress |
| calibration | 128 × 2048 C4 tokens | **32 × 1024** (`c4-32x1024-seed0.pt`) | must be byte-identical to the GA's and DPMA's. `diep/data.py` refuses any other file |
| steps | 10 epochs × 8 = 80 (60 α / 20 β) | 10 epochs × **2 = 20 (15 α / 5 β)** | same epochs and batch (16); the smaller calibration set makes each epoch 2 steps |
| evaluation | 0-shot `acc`, MMLU/BoolQ/OBQA/RTE | **GA protocol**: 1-shot, 8 MC + gsm8k + mbpp, `acc_norm` first, `mean_mc` | identical scoring across GA, DPMA and DiEP; `tests/test_extras.py` checks the constants against the GA's file |
| sparsities | 25%, 50% | **25%, 50%, 75%** (keep 0.75 / 0.5 / 0.25) | the GA's and DPMA's ratios |
| floor | `m_min = k` | `m_min = k = 8` | as specified. DPMA's floor is 10 (top-k + 2); the GA's is its own |

Two implementation choices worth knowing:

- **H is cached through the DiEP forward at the identity** (`alpha = 0, beta = 1`),
  so `Phi = 0` at step 0 exactly, as spec §4.3 states. In transformers 5.x OLMoE's
  experts default to `grouped_mm` kernels, which compute the same function with
  different rounding; caching through them would put kernel noise into `Phi` from
  the start. Measured on the real model: the DiEP forward at the identity equals
  the model's eager expert loop exactly (max |Δlogit| = 0), and differs from
  `grouped_mm` by 1.875 — the same as the model differs from *itself* when the
  batch changes.
- **Evaluation runs the physically pruned model with its native kernels**, loaded
  back from disk, which is how the pruned model would be used. Masking and physical
  pruning are equivalent up to that same kernel noise. A 50%-pruned OLMoE is
  numerically fragile: the same pruned model agrees with itself on 92% of top-1
  tokens across the two kernel implementations, against 98% for the dense model.

---

## Layout

```
DiEP/
├── specs.md                    the specification
├── diep/
│   ├── adapters/               the model interface (spec §2.1): base.py, olmoe.py
│   │                           (both transformers layouts: fused 5.x, legacy 4.x)
│   ├── moe_forward.py          the one DiEP forward: modes search / dense / mask / skip
│   ├── data.py                 calibration loading (refuses any file but the study's)
│   ├── search.py               stage 1: alpha/beta, CE + λΦ, 3:1 alternation, ablations
│   ├── prune.py                stage 2: scores, global + uniform selection, physical
│   │                           pruning, save/load of checkpoints with uneven expert counts
│   ├── cka.py                  RBF-CKA
│   ├── skip.py                 stage 3: gamma calibration and the skipping rule's tables
│   ├── merge.py                optional merge of pruned experts (spec §8)
│   ├── evaluate.py             the GA/DPMA lm-eval protocol
│   ├── bench.py                memory ratio and generation throughput (spec §6.1)
│   ├── provenance.py           versions, commit, command line
│   └── cli.py                  `python -m diep search|prune|calib-skip|eval|bench`
├── tests/                      the 6 tests spec §11 requires, and more (25)
├── scripts/run_all.sh          the whole study as one GPU job queue
├── examples/report.py          figures and ANALYSIS.md, against the GA and DPMA
└── results/                    runs/, harness/, figures/, logs/, ANALYSIS.md
```

## Usage

The whole study, one command (restartable; finished jobs are skipped):

```bash
cd DiEP
GPUS="1 3 5 7" ./scripts/run_all.sh
```

It runs, queued over the GPUs, longest first:

| job | what | seeds |
|---|---|---|
| `search` | alpha/beta search → `results/runs/seed{s}/scores.pt` | 0, 1, 2 |
| `diep` | global selection, no skipping — the paper's main setting | 0, 1, 2 × r ∈ {0.25, 0.5, 0.75} |
| `uniform` | same scores, N·r pruned per layer — the baseline spec §12 requires DiEP to beat | 0 |
| `merge` | with spec §8's merge | 0 |
| `skip` | the global model with adaptive skipping | 0 |
| `bench` | memory ratio, tokens/s (full, pruned, pruned + skip) | 0 |

One step at a time:

```bash
CAL=../GA/MoE_compression_GA/scripts/results/calibration/c4-32x1024-seed0.pt
python -m diep search     --calibration-file $CAL --out results/runs/seed0 --seed 0
python -m diep prune      --scores results/runs/seed0/scores.pt --sparsity 0.5 \
                          --out results/runs/seed0/pruned-r0.5-global      # [--selection uniform] [--merge]
python -m diep calib-skip --model results/runs/seed0/pruned-r0.5-global --calibration-file $CAL \
                          --out results/runs/seed0/pruned-r0.5-global/skip_config.json
python -m diep eval       --model results/runs/seed0/pruned-r0.5-global --method diep \
                          --tasks all --allow-unsafe-code --output results/harness   # [--skip ...]
python -m diep bench      --model results/runs/seed0/pruned-r0.5-global --skip ... \
                          --baseline allenai/OLMoE-1B-7B-0924 --calibration-file $CAL --out bench.json
```

Search options mirror spec §10: `--search-forward routed|dense`, `--epochs`,
`--batch-size`, `--lr-alpha/--lr-beta`, `--update-ratio`, `--lam`,
`--recon-target hidden|logits`, `--score-mode softmax|raw`, and the §9 ablations
`--ablation alpha_only|beta_only`, `--fixed-beta a:b|random`.

Environment: `/home/jupyter/workspace/llm_evaluation/llm-eval/bin/python`
(transformers 5.12, lm-eval 0.4.12) with `DS_IGNORE_CUDA_DETECTION=1`, both set by
`scripts/config.sh`.

## Outputs

```
results/runs/seed0/
├── scores.pt                   alpha, beta, scores, config, per-step logs (spec §11)
├── search_log.json, run_card.json
├── pruned-r0.5-global/         a pruned checkpoint: model.safetensors, config.json with
│   ├── prune_card.json         num_experts_per_layer and the original indices, tokenizer
│   └── skip_config.json        gamma1 per rank, CKA matrix, CKA mean, per layer (spec §7.2)
├── pruned-r0.5-uniform/, pruned-r0.5-merge/
└── bench-r0.5.json
results/harness/diep-keep0.500-rep0-1shot.json    the GA schema: tasks, headline, mean_mc,
                                                  layer_budgets, parameter_ratio, provenance
```

A pruned checkpoint has a different expert count in every layer, which a single
`config.num_experts` cannot express. `diep.prune.load_pruned(path)` builds the
architecture on the meta device, cuts every block to its saved size and loads the
tensors. The loaded model reproduces the in-memory pruned model exactly (checked
on the real model).

## Tests

```bash
python -m pytest tests                                   # transformers 5.x (fused layout)
/home/jupyter/workspace/reap/.venv/bin/python -m pytest tests   # transformers 4.x (legacy layout)
```

The six required by spec §11 (`tests/test_spec_required.py`): (1) the routed search
at init equals the original, (2) masking equals physical pruning, (3) selection
respects `K` and `m_min`, (4) the gradient is finite at step 0 where `Phi = 0`, (5)
skipping with `gamma = 0` is no skipping, (6) CKA(Y, Y) = 1 and CKA is invariant to
isotropic scaling. `tests/test_extras.py` adds the 3:1 schedule, a full search on a
tiny model, the dense forward, score modes, uniform selection, the skip rule,
save/load, merging, the calibration guard, and the protocol-identity check against
the GA.

## Known gaps

- **Only OLMoE has an adapter.** Mixtral, DeepSeek-MoE (dense layer 0, 2 shared
  experts) and Qwen2-MoE (a sigmoid-gated shared expert) each need a subclass of
  `diep.adapters.MoEAdapter`; the rest of the code is architecture-agnostic.
  `shared_out` is already added unscaled in every mode.
- **The paper's own evaluation protocol** (0-shot `acc`, RTE, the MMLU categories)
  is not run. This study uses the GA/DPMA protocol so all three methods are scored
  identically.
- **Skipping speed.** Skipping needs a custom forward, a Python loop over experts
  (as OLMoE's own eager loop is). Its speedup is reported against that same
  forward. The native `grouped_mm` kernels are faster than either, and a fused
  kernel that honours skipped pairs is not implemented.
- **The dense search forward and the §9 ablations** are implemented and tested but
  not part of `run_all.sh`.

## What this repository stores

**Kept:**
- the code, tests and `specs.md`;
- for every run under `results/`: the run card, search log and learned expert scores (`scores.pt`), each pruned model's config and skip config, the benchmark and lm-eval records, logs and figures.

**Not kept** (see `.gitignore`, about 222 GB): the pruned weights (`model.safetensors`) and the tokenizer copies saved next to them. Each pruned model can be rebuilt from its run's `scores.pt` and the public OLMoE weights.

The DPMA study (separate repository, `DPMA`) imports `diep.evaluate` for scoring, so keep the two together.
