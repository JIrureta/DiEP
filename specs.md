# DiEP: Differentiable Expert Pruning. Implementation Spec

Source: Bai et al., *DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning*, NeurIPS 2025.
This document is self-contained. Where the paper is ambiguous or inconsistent, this spec makes a decision, marked **[DECISION]**. §13 summarizes these decisions.

---

## 1. Goal

Given a pretrained sparse MoE LLM, remove a fraction `r` of all **routed experts** across the whole model. The per-layer pruning rate is non-uniform and is learned, not fixed. The method has three parts:

1. **Search.** The model weights stay frozen. Two sets of parameters are learned on a small calibration set: intra-layer expert logits `α ∈ R^{L×N}` and inter-layer layer scales `β ∈ R^L`. They are updated alternately by gradient descent.
2. **Prune.** Each expert gets a score `s = importance(α) · β`. All experts are ranked globally, and the bottom `K = round(N·L·r)` are removed physically.
3. **Adaptive skipping (inference, optional).** Per token, a lower-weighted selected expert is skipped when its routing weight is small relative to the top expert. The threshold is calibrated from routing-weight statistics and CKA similarity between experts.

An optional extension (§8) merges pruned experts into retained ones.

---

## 2. Supported models

| Model (HF id) | MoE layers `L` | Routed experts `N` | top-k `k` | Renormalize top-k weights | Shared experts |
|---|---|---|---|---|---|
| `mistralai/Mixtral-8x7B-v0.1` | 32 | 8 | 2 | yes | none |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 32 | 8 | 2 | yes | none |
| `mistralai/Mixtral-8x22B-v0.1` | 56 | 8 | 2 | yes | none |
| `deepseek-ai/deepseek-moe-16b-base` (trust_remote_code) | 27 (layer 0 is dense) | 64 | 6 | no | 2 (always on) |
| `Qwen/Qwen2-57B-A14B` | 28 | 64 | 8 | no | 1 (sigmoid-gated) |

Rules:
- Only **routed** experts are ever pruned, scored, or skipped. Shared experts, dense FFN layers, attention, and embeddings are untouched.
- `L` counts **MoE layers only**. All indexing below is over MoE layers.
- Read `k`, `N`, and top-k renormalization from the model config. Do not hard-code them.

### 2.1 Model adapter interface

Isolate all model-specific code behind this interface so the rest of the code is architecture-agnostic:

```python
class MoEAdapter:
    def moe_blocks(model) -> list[Block]          # ordered routed-MoE blocks, length L
    def router_logits(block, x) -> Tensor[T, N]   # pre-softmax router logits
    def expert(block, i) -> nn.Module             # FFN_i: [T, d] -> [T, d]
    def shared_out(block, x) -> Tensor[T, d] | None   # shared-expert output incl. its own gate, else None
    num_experts(block) -> int; top_k -> int; renormalize -> bool
    def prune(block, keep: list[int]) -> None     # slice router rows, drop experts, set per-layer N_l and k_l
```

Each MoE block's `forward` is replaced with a single DiEP forward that is controlled by a `mode` flag (§4.2, §6.3, §7). The original module forward is never called during search.

---

## 3. Notation

- `x ∈ R^{T×d}`: tokens entering MoE layer `l`.
- `p = softmax(router_logits(x))`: original routing probabilities over the `N` experts.
- `S(x) = TopK(p, k)`. The routing weight `w_i` equals `p_i`, renormalized over `S` if the model renormalizes.
- `ᾱ^(l) = softmax(α^(l))` over the experts of layer `l` (Eq. 4).
- `FFN_i`: routed expert `i`.
- `F(x)`: output of the original model. `F'(x; α, β)`: output of the model with DiEP gating.

---

## 4. Stage 1: Differentiable search

### 4.1 Parameters

- `α`: fp32, shape `[L, N]`, initialized to **0** (uniform ᾱ = 1/N). Each row lives on the device of its layer.
- `β`: fp32, shape `[L]`, initialized to **1**.
- Freeze all model weights (`requires_grad=False`) and put the model in `eval()` mode. Only `α` and `β` are trainable, which is about 0.01% extra parameters.
- Disable router auxiliary / load-balancing losses.

### 4.2 Search forward (mode `search`)

**[DECISION]** The paper's Eq. 5, `y = β Σ_{i=1..N} ᾱ_i FFN_i(x)`, omits the router, just as its Eq. 3 omits the router for a pruned layer. This spec treats Eq. 5 as a relaxation of the **routed** layer and keeps the original router. This is the default, `search_forward = "routed"`:

```
y^(l) = β^(l) · Σ_{i ∈ S(x)} w_i · g_i^(l) · FFN_i(x)  +  shared_out(x)
g_i^(l) = N · ᾱ_i^(l)          # equals 1 at init, so the model is exactly the original at init
```

- Top-k selection `S(x)` and weights `w_i` come from the **unmodified** router. `α` does not change which experts are selected.
- `β` scales only the routed part. The shared-expert output is added unscaled.
- Compute `w_i · g_i · β` in fp32 and cast to the expert output dtype before the weighted sum.

The paper-literal alternative, `search_forward = "dense"`, must also be implemented:

```
y^(l) = β^(l) · Σ_{i=1..N} ᾱ_i^(l) · FFN_i(x)  +  shared_out(x)
```

Every token passes through all `N` experts, so compute is about `N/k` times higher.

### 4.3 Objective (Eq. 6–7)

```
L(α, β) = CE(y_next_token, F'(x; α, β))  +  λ · Φ(α, β)
Φ       = sqrt( Σ_{tokens, dims} (H'(x) − H(x))² + 1e-8 )
```

- `CE` is the standard causal-LM next-token loss, averaged over tokens.
- **[DECISION]** `H` is the final hidden state, meaning the input to `lm_head` after the final norm. `H(x)` comes from the original model and is **precomputed once** for all calibration sequences, then cached on CPU in bf16. Config option `recon_target: hidden | logits`, default `hidden`.
- `Φ` is the **non-squared** Frobenius norm over the full micro-batch, computed in fp32 (paper Eq. 6). The `+1e-8` inside the sqrt is mandatory. In routed mode `Φ = 0` at initialization, and without the epsilon the gradient would be NaN.
- `λ = 0.01`.

### 4.4 Optimization (Eq. 8–9, Algorithm 1)

- **Calibration data:** 128 sequences of **2048** tokens from C4 (`allenai/c4`, `en`, train split). Sample random documents with at least 2048 tokens and take a random 2048-token window. Seed = 0. **[DECISION]** The sequence length is not stated in the paper; 2048 follows NAEE.
- **Batch:** effective batch 16 sequences. Gradient accumulation over micro-batches is allowed. One optimizer step consumes 16 sequences, so an epoch has 8 steps.
- **Epochs:** 10, which gives 80 steps in total. Reshuffle every epoch.
- **Alternation (3:1):** global step `c = 0, 1, 2, …`. If `c mod 4 < 3`, the step updates `α` only (β fixed). Otherwise it updates `β` only, using the latest `α`. That gives 60 α-steps and 20 β-steps. The ratio `p:q` is configurable.
- **Optimizers [DECISION]:** two separate Adam optimizers (betas 0.9/0.999, eps 1e-8, no weight decay). Both use lr `5e-3` and cosine decay to 0 with no warmup. Each schedule runs over its own number of steps (60 for α, 20 for β).
- After each β-step, clamp `β ≥ 1e-3` **[DECISION]** so that the ranking stays well-defined.
- **Memory:** load the model in bf16 with `device_map="auto"` across GPUs. Enable gradient checkpointing on the decoder layers. Because the weights are frozen, call `enable_input_require_grads()`.
- Log per step: CE, Φ, the entropy of each ᾱ row, and β.

Reference budget: about 0.23 h for Mixtral-8x7B on 4×A800 (80 GB).

```python
for epoch in range(10):
    for c, batch in enumerate(loader, start=epoch*8):
        target = "alpha" if c % 4 < 3 else "beta"
        out = model(batch.input_ids, output_hidden_states=True)   # DiEP search forward
        loss = ce(out.logits, batch.input_ids) + lam * frob(out.hidden_states[-1], H_cache[batch.idx])
        loss.backward()
        opt[target].step(); sched[target].step(); zero_grad(alpha, beta)
        if target == "beta": beta.data.clamp_(min=1e-3)
```

---

## 5. Stage 2: Global pruning (Eq. 10–11)

### 5.1 Scores

```
s_i^(l) = ᾱ_i^(l) · β^(l)          # score_mode = "softmax" (default)
s_i^(l) = α_i^(l) · β^(l)          # score_mode = "raw" (paper-literal Eq. 10; requires α init = 1)
```

**[DECISION]** The default uses `ᾱ`. Softmax is shift-invariant, so the raw logit level of each layer is not identified by the loss, which makes raw-logit products unreliable across layers.

### 5.2 Selection

- Target total pruned count: `K = round(N · L · r)`. For example, Mixtral-8x7B with r=25% gives K=64.
- **[DECISION]** Per-layer floor: every layer keeps at least `m_min` experts. The default is `m_min = k`, and it is configurable (min 1). Assert `K ≤ Σ_l (N − m_min)`.
- Algorithm: sort all `(l, i)` by `s` in ascending order, breaking ties by `(l, i)`. Walk the list and prune `(l, i)` if layer `l` currently keeps more than `m_min` experts. Stop at `K` pruned.
- Output: a binary mask `m ∈ {0,1}^{L×N}` and a per-layer keep list `keep_l` (sorted original indices).

For architectures specified by an average number of experts per layer (Fig. 3: 64 → 62, 60, …, 48), use `r = 1 − target_avg / N`.

### 5.3 Physical pruning

For each layer, `adapter.prune(block, keep_l)`:
- Delete the pruned expert modules. Slice the router weight rows (and bias, if present) to `keep_l`. Store `N_l = |keep_l|`.
- Set `k_l = min(k, N_l)`. Routing is softmax over the **retained** logits only, then top-`k_l`, then renormalization if the model renormalizes.
- Save the model with a per-layer expert count in its config (a custom `num_experts_per_layer` list) plus a loader that rebuilds the blocks. Save `keep_l` with the original indices.

For quick evaluation without saving, a masking implementation may be used instead. It sets pruned router logits to `−inf` before the softmax. This must be numerically equivalent to physical pruning, but it gives no memory saving.

---

## 6. Evaluation (paper Table 1 protocol)

- lm-evaluation-harness, zero-shot, metric `acc` (not `acc_norm`). Main tasks: `mmlu`, `boolq`, `openbookqa`, `rte`. Extra tasks: `arc_challenge`, `arc_easy`, `hellaswag`, `winogrande`, `gsm8k`.
- Report the 4 MMLU category scores (humanities, social sciences, STEM, other), the MMLU overall score, and **Average** = mean of (MMLU, BoolQ, OpenBookQA, RTE).
- Table 1 numbers use pruning **without** skipping.
- Run 3 search seeds (0, 1, 2) and report the mean.

### 6.1 Efficiency metrics (Table 4)

- **GPU memory ratio** = parameter bytes of the pruned model divided by those of the full model. For Mixtral-8x7B this is 0.76 at r=25% and 0.52 at r=50%.
- **Speedup** = generation throughput (tokens/s) relative to the full model under an identical setup: bf16, the same GPUs and `device_map`, batch size 1, a 128-token prompt, and 256 new tokens with greedy decoding. Average over 20 prompts after 3 warm-up runs.

---

## 7. Stage 3: Adaptive expert skipping (inference, §4.3)

This stage is applied to the **pruned** model. It needs a one-time calibration pass on the same C4 calibration set.

### 7.1 Rule

For each token in layer `l`, sort the selected experts by weight: `w_{e0} ≥ w_{e1} ≥ … ≥ w_{e(k_l−1)}`. For each rank `j ≥ 1`:

```
skip e_j  if  w_{ej} < γ^(l)_j(e0, ej) · w_{e0}
γ^(l)_j(a, b) = min(1, γ1^(l)_j · γ2^(l)(a, b))
```

- The top expert `e0` and any shared experts are never skipped.
- When an expert is skipped, its term is dropped and the remaining weights are **not** renormalized (config `renormalize_on_skip=false`).
- The rule applies at every forward call, both prefill and each decode step. For k=2 it reduces exactly to the paper's rule. **[DECISION]** For k>2 each rank is thresholded against `e0`.

### 7.2 Calibration of γ

- **γ1^(l)_j:** run the pruned model on the calibration set and collect `w_{ej} / w_{e0}` for every token at rank `j` in layer `l`. `γ1^(l)_j` is the median of these ratios.
- **γ2^(l)(a, b) = CKA^(l)(a, b) / mean_{i<j} CKA^(l)(i, j)**, where the mean is over all pairs of retained experts in layer `l`.
- **CKA computation, per layer:**
  1. Sample `n = 2048` tokens of the layer input `x` from the calibration forward.
  2. Run every retained expert on these tokens to get `Y_i ∈ R^{n×d}`.
  3. Compute RBF-kernel CKA in fp32:
     ```
     K_i[a,b] = exp(−‖Y_i[a] − Y_i[b]‖² / (2σ_i²)),  σ_i = median of nonzero pairwise distances in Y_i
     HSIC(K, M) = tr(K H M H) / (n−1)²,  H = I − 11ᵀ/n
     CKA(i, j) = HSIC(K_i, K_j) / sqrt(HSIC(K_i, K_i) · HSIC(K_j, K_j))
     ```
- Store the results in `skip_config.json` as `{layer: {"gamma1": [per rank], "cka": NxN matrix, "cka_mean": float}}`. At runtime, use a precomputed `γ` table indexed by `[l, j, a, b]`.

Expected effect (Mixtral-8x7B): an extra speedup of about 1.02–1.07× for a drop of ≤1 Average point. At r=50% the combined result is about 1.28× speedup with 59.6 Average.

---

## 8. Optional: merging pruned experts (Appendix A.5)

This is disabled by default (`merge=false`). If enabled, it runs after §5.2 and before §5.3, separately in each layer:
1. Compute the RBF-CKA between all experts (as in §7.2) on the unpruned layer.
2. Assign each pruned expert `p` to `r* = argmax_{retained r} CKA(p, r)`.
3. For each retained `r`, form the group `G_r = {r} ∪ {pruned p assigned to r}`, with `CKA(r, r) = 1`.
4. Compute merging weights `c_e = softmax_{e∈G_r}(CKA(e, r))`.
5. Replace every weight tensor of `FFN_r` (for example `w1, w2, w3`) with `Σ_{e∈G_r} c_e · W_e`. The router row of `r` is unchanged.

---

## 9. Ablation hooks (Table 2, Table 10)

| Flag | Behavior |
|---|---|
| `ablation=alpha_only` | β frozen at 1; score = ᾱ |
| `ablation=beta_only` | α frozen; score = `β^(l) · f_i^(l)`, where `f` = fraction of calibration tokens routed to expert `i` in the original model |
| `fixed_beta="a:b"` | β frozen at `a` for the first half of MoE layers and `b` for the second half; only α is trained |
| `fixed_beta=random` | β sampled once from U(0.5, 2.0) with the seed, then frozen |
| `update_ratio="p:q"` | alternation ratio (default 3:1) |

---

## 10. Configuration defaults

| Key | Default | Notes |
|---|---|---|
| `sparsity r` | 0.25 / 0.5 | fraction of routed experts removed |
| `search_forward` | `routed` | or `dense` (paper-literal Eq. 5) |
| `calib` | C4, 128 × 2048 tokens | sizes {32…1024} tested; 128 is best |
| `batch_size` | 16 | effective |
| `epochs` | 10 | tested {5,10,15,20,30} |
| `lr_alpha`, `lr_beta` | 5e-3, cosine | Adam |
| `update_ratio` | 3:1 | |
| `lambda` | 0.01 | tested {0, 0.001, 0.01, 0.1, 0.5} |
| `recon_target` | `hidden` | or `logits` |
| `score_mode` | `softmax` | or `raw` |
| `m_min` | `k` | per-layer floor on retained experts |
| `alpha_init`, `beta_init` | 0, 1 | `alpha_init=1` if `score_mode=raw` |
| `beta_min` | 1e-3 | clamp |
| `skip` | off for Table 1; on for Table 4 | |
| `cka_tokens` | 2048 | per layer |
| `merge` | false | |
| `seeds` | 0, 1, 2 | |

---

## 11. Deliverables

**Code layout (suggested):** `adapters/` (one adapter per architecture), `data.py`, `search.py`, `prune.py`, `skip.py`, `merge.py`, `evaluate.py`, `bench.py`. A single CLI provides these subcommands:

```
diep search     --model M --sparsity R --out RUN      # writes RUN/scores.pt
diep prune      --model M --scores RUN/scores.pt --out RUN/pruned [--merge]
diep calib-skip --model RUN/pruned --out RUN/skip_config.json
diep eval       --model RUN/pruned [--skip RUN/skip_config.json] --tasks mmlu,boolq,openbookqa,rte
diep bench      --model RUN/pruned [--skip ...]
```

**Files written to `RUN/`:**
- `scores.pt` contains `alpha [L,N]`, `beta [L]`, `scores [L,N]`, `mask [L,N]`, `keep` (list of lists), `config`, and training logs.
- `pruned/` is the HF-format pruned checkpoint.
- `skip_config.json` holds the skipping calibration.
- `results.json` holds the metrics.

**Unit tests (required):**
1. In routed mode with α=0 and β=1, logits equal those of the original model (max abs diff ≤ 1e-3 in bf16).
2. The masking implementation and physical pruning give identical logits.
3. The selection respects `K` and `m_min` exactly.
4. The gradient is finite at step 0, when Φ=0.
5. Skipping with γ≡0 reproduces the no-skip outputs exactly.
6. CKA(Y, Y) = 1, and CKA is invariant to isotropic scaling of `Y`.

---

## 12. Acceptance targets (Mixtral-8x7B, no skipping, mean of 3 seeds)

| Setting | MMLU | BoolQ | OBQA | RTE | Average |
|---|---|---|---|---|---|
| Full model | 67.9 | 85.3 | 35.4 | 71.5 | 65.1 |
| DiEP r=25% | 64.9 | 86.6 | 33.1 | 70.7 | 63.8 |
| DiEP r=50% | 57.9 | 84.0 | 29.6 | 68.2 | 59.9 |

- A reproduction passes if Average is within ±1.5 of these targets and DiEP beats the uniform-per-layer baseline (same score, pruning `N·r` experts per layer) at both sparsities.
- Search time should be ≤ 0.5 h on 4×A800-80GB.
- Qualitative check: the learned pruning is non-uniform, with more experts pruned in deeper layers of Mixtral (paper Fig. 4/5).

---

## 13. Ambiguity log

| # | Paper | Decision |
|---|---|---|
| 1 | Eq. 5 sums over all experts without the router | Default keeps the router; the gate is `β·N·ᾱ_i` (identity at init). `dense` mode is kept as the literal option. |
| 2 | Eq. 10 multiplies raw logits α by β | Default uses softmax ᾱ·β; `raw` is available as an option. |
| 3 | "F(x)" / "features" undefined | Final hidden state (pre-`lm_head`), cached. |
| 4 | Φ is a non-squared norm | Kept as written, with `+1e-8` inside the sqrt so the gradient is defined at Φ=0. |
| 5 | Optimizer, sequence length, init values not stated | Adam, 2048 tokens, α=0, β=1. |
| 6 | Meaning of "3:1" | 3 α-steps then 1 β-step, each on a new batch of 16. |
| 7 | Layers may lose all or most experts under global ranking | Per-layer floor `m_min = k`. |
| 8 | Skipping defined only for k=2; "decayed weights" unspecified | Rank-wise rule against `e0`; skipped term dropped without renormalization; γ clipped to ≤1. |
| 9 | CKA variant for γ2 | RBF-kernel CKA (as in Fig. 1), 2048 tokens, σ = median distance. |
| 10 | λ grid in the text ({5…30}) contradicts Fig. 8 | Use the Fig. 8 grid {0, 0.001, 0.01, 0.1, 0.5}; {5…30} is the epoch grid. |
| 11 | Treatment of shared experts | Never pruned or skipped; not scaled by β. |
| 12 | MMLU aggregation inconsistent across Table 1 rows | Report lm-eval's overall MMLU accuracy plus the 4 categories. |
