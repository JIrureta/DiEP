# DiEP study — run card

**What this is.** DiEP (Bai et al., NeurIPS 2025), implemented from
[`../specs.md`](../specs.md) and run on OLMoE-1B-7B under exactly the conditions of
the GA and DPMA studies, so the three methods can be compared directly. The study
covers **plain DiEP** (the paper's main method) and **DiEP + the Appendix A.5
merge**, plus the uniform baseline, adaptive skipping and efficiency.

- Results: [`ANALYSIS.md`](ANALYSIS.md). Plain vs merge is **section 4b**; figures
  are in [`figures/`](figures/) (plain vs merge in the right panel of `comparison.png`).
  Both are rebuilt after every finished job.
- Started 2026-09-26 09:51 UTC with `scripts/run_all.sh` (24 jobs, GPUs 1, 3, 5, 7);
  expected complete around 19:30–20:30 UTC.
- Code and design: [`../README.md`](../README.md).

---

## Same calibration data, same evaluation

| | calibration file | md5 |
|---|---|---|
| GA (frequency baseline) | `GA/MoE_compression_GA/scripts/results/calibration/c4-32x1024-seed0.pt` | `d0b30c732b0c` |
| DPMA v3 | same file | `d0b30c732b0c` |
| DiEP search, seeds 0, 1, 2 | same file | `d0b30c732b0c` |
| DiEP merge (CKA) and skip calibration | same file | `d0b30c732b0c` |

`diep/data.py` refuses any other file. Evaluation is the GA's lm-eval protocol:
1-shot, `all` tasks (8 multiple-choice + gsm8k + mbpp), `acc_norm` first, headline
`mean_mc` over the 8 multiple-choice tasks, `bootstrap_iters=0`, the same result
schema. `tests/test_extras.py` checks these constants against the GA's own file.

---

## The arms

| label in the results | what | seeds | ratios (keep) | pruned checkpoint |
|---|---|---|---|---|
| **`diep`** | **plain DiEP**: search → global ranking → physical pruning | 0, 1, 2 | 0.75, 0.5, 0.25 | `runs/seed{s}/pruned-r{r}-global/` |
| **`diep-merge`** | **plain DiEP + Appendix A.5 merge**, same scores as `diep` seed 0 | 0 | 0.75, 0.5, 0.25 | `runs/seed0/pruned-r{r}-merge/` |
| `diep-uniform` | same scores, `N·r` experts pruned in every layer (the baseline spec §12 requires DiEP to beat) | 0 | 0.75, 0.5, 0.25 | `runs/seed0/pruned-r{r}-uniform/` |
| `diep-skip` | the `diep` seed-0 model with adaptive skipping (spec §7) | 0 | 0.75, 0.5, 0.25 | uses `pruned-r{r}-global/skip_config.json` |
| bench | memory ratio and tokens/s (spec §6.1) | 0 | 0.75, 0.5, 0.25 | `runs/seed0/bench-r{r}.json` |

`r` in folder names is the fraction **removed** (0.25, 0.5, 0.75), so keep = 1 − r.

---

## Plain DiEP

1. **Search** (spec §4). Model frozen, in `eval()`. Learn `alpha [16, 64]` (init 0)
   and `beta [16]` (init 1): 1,040 numbers. The objective is next-token CE +
   0.01 · ‖H′ − H‖_F, with H the original model's final hidden state. Three
   α-steps then one β-step, 16 sequences per step, two Adam optimisers at lr
   5e-3 each with its own cosine schedule, β clamped ≥ 1e-3. The search forward
   keeps the router: `beta_l · Σ_{i∈S(x)} w_i · 64·softmax(alpha_l)_i · FFN_i(x)`,
   which at init *is* the original model (Φ = 1e-4 = √1e-8 at step 0 in every
   seed's log).
2. **Prune** (spec §5). Score `softmax(alpha_l)_i · beta_l`. Walk every expert from
   the lowest score and remove the bottom `K = round(64 · 16 · r)`, never taking a
   layer below 8 experts (the top-k). Then physically slice the router rows and the
   expert weights, and save the smaller model.

On this calibration set, the spec's 10 epochs are **20 steps (15 α / 5 β)**. The
paper, with 4× more sequences, ran 80. The data is not the issue; only the step
count follows from it.

---

## The merge (Appendix A.5, spec §8)

**Where it sits.** After choosing which experts to prune (§5.2), before physically
removing them (§5.3). It runs on the **unpruned** model, separately in each of
the 16 layers. Code: `diep/merge.py`; enabled with `diep prune --merge`.

**What it does, step by step:**

1. **Similarity.** Run the 32 calibration sequences through the original model and
   sample 2048 token positions (seed 0) of every layer's input. Run all 64 experts
   of the layer on those tokens and compute RBF-kernel CKA between every pair of
   experts (`diep/cka.py`). The kernel width σ is the median non-zero pairwise
   distance of each expert's outputs, and HSIC uses centred kernels, in fp32.
2. **Assignment.** Each pruned expert `p` goes to the retained expert it is most
   similar to: `r* = argmax_{retained r} CKA(p, r)`.
3. **Groups.** For each retained expert `r`: `G_r = {r} ∪ {pruned experts assigned to r}`,
   with `CKA(r, r) = 1`.
4. **Weights.** `c_e = softmax_{e ∈ G_r} CKA(e, r)`, so the weights in a group sum to 1.
5. **Merge.** Every weight tensor of expert `r` (OLMoE: `gate_up_proj`, `down_proj`)
   becomes `Σ_{e∈G_r} c_e · W_e`, accumulated in fp32 and cast back to bf16.
   **The router row of `r` is unchanged.**

**What changes, and what does not:**

| | plain `diep` | `diep-merge` |
|---|---|---|
| which experts survive | chosen by the scores | **identical** (same scores, same selection) |
| router | original rows of the survivors | **identical** |
| experts per layer, memory, compute | — | **identical** |
| weights of a survivor with no donors | original | **identical** |
| weights of a survivor with donors | original | softmax(CKA)-weighted average with its donors |

So `diep-merge` vs `diep` (seed 0) is a **paired comparison**. The only thing that
differs is the merged weights, and plain DiEP's own spread over 3 seeds is the
yardstick for whether the difference means anything (ANALYSIS §4b).

**A property to expect.** CKA lies in [0, 1], so `softmax(CKA)` is close to
uniform. A retained expert that absorbs several donors ends up mostly made of
them. As an illustration only, from the smoke test (a 1-epoch search at r = 0.5,
not the study's runs): 220 of the 512 retained experts received donors, 2.3 on
average and up to 12. The expert with 12 donors kept only 10% of its own weights;
the average receiver kept 51%. The study's own numbers are in ANALYSIS §4b ("what
the merge did"), read from each run's `prune_card.json`.

**Tested** (`tests/test_extras.py`): every pruned expert is assigned to exactly one
receiver, each group's weights sum to 1, the router is untouched, and merging
experts that are already identical changes nothing.

**How this differs from DPMA's merge.** DPMA *learns* its merge coefficients and a
top-3 assignment by gradient descent against the dense model's KL, jointly with
absorbing (re-routing). DiEP's merge is training-free: fixed weights from a
similarity statistic, one target per donor, no re-routing.

---

## What is not averaged over seeds

`diep` has 3 seeds at every ratio. The merge, uniform and skip arms run on seed 0
only, and are compared with `diep` seed 0 (the same search). If the merge
comparison needs its own spread, `diep-merge` for seeds 1 and 2 is six more
evaluations (about 3 h on 4 GPUs) after the main study finishes.

---

## Reproduce

```bash
cd DiEP
GPUS="1 3 5 7" ./scripts/run_all.sh            # the whole study; restartable

# just the merge arm for one ratio, by hand:
CAL=../GA/MoE_compression_GA/scripts/results/calibration/c4-32x1024-seed0.pt
python -m diep prune --scores results/runs/seed0/scores.pt --sparsity 0.5 --merge \
                     --calibration-file $CAL --out results/runs/seed0/pruned-r0.5-merge
python -m diep eval  --model results/runs/seed0/pruned-r0.5-merge --method diep-merge \
                     --tasks all --allow-unsafe-code --output results/harness
```
