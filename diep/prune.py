"""Stage 2: global pruning (spec §5, Eq. 10-11), and the pruned checkpoint.

Scores::

    s = softmax(alpha_l)_i * beta_l      score_mode = "softmax" (default)
    s = alpha_i * beta_l                 score_mode = "raw" (paper-literal Eq. 10)
    s = beta_l * f_i                     ablation = "beta_only" (f: routing frequency)

Selection: ``K = round(N L r)`` experts are removed. Walk all ``(l, i)`` in
ascending score (ties by ``(l, i)``) and prune ``(l, i)`` while layer ``l``
still keeps more than ``m_min`` experts, until ``K`` are pruned. ``m_min``
defaults to ``k``, so no layer drops below its top-k.

The **uniform** selection of spec §12 -- the same scores, but exactly ``N r``
experts removed from every layer -- is the baseline DiEP must beat.

A pruned OLMoE has a different expert count in every layer, which a single
``config.num_experts`` cannot express, so :func:`save_pruned` writes the full
pruned state dict plus ``num_experts_per_layer`` and the original indices, and
:func:`load_pruned` rebuilds the blocks at those sizes before loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

__all__ = ["compute_scores", "select_global", "select_uniform", "keep_lists",
           "prune_model", "masked_state", "save_pruned", "load_pruned"]


def compute_scores(alpha, beta, *, score_mode="softmax", ablation="none", frequency=None):
    """``[L, N]`` expert scores (spec §5.1, §9)."""
    alpha, beta = alpha.float(), beta.float()
    if ablation == "beta_only":
        if frequency is None:
            raise ValueError("beta_only needs the routing frequencies")
        return frequency.float() * beta[:, None]
    if ablation == "alpha_only":
        return torch.softmax(alpha, dim=-1) if score_mode == "softmax" else alpha
    base = torch.softmax(alpha, dim=-1) if score_mode == "softmax" else alpha
    return base * beta[:, None]


def select_global(scores: torch.Tensor, sparsity: float, m_min: int) -> torch.Tensor:
    """Spec §5.2. Returns a ``{0,1}`` keep mask ``[L, N]`` with exactly ``K`` zeros."""
    layers, experts = scores.shape
    k_prune = round(experts * layers * sparsity)
    capacity = layers * (experts - m_min)
    if k_prune > capacity:
        raise ValueError(f"K={k_prune} > sum_l (N - m_min) = {capacity}: infeasible at "
                         f"m_min={m_min}")
    # Ascending score, ties broken by (l, i): a stable sort of the row-major
    # flattening does exactly that.
    order = torch.sort(scores.flatten().float(), stable=True).indices.tolist()
    mask = torch.ones(layers, experts, dtype=torch.long)
    kept = [experts] * layers
    pruned = 0
    for flat in order:
        if pruned == k_prune:
            break
        layer, expert = divmod(flat, experts)
        if kept[layer] > m_min:
            mask[layer, expert] = 0
            kept[layer] -= 1
            pruned += 1
    assert pruned == k_prune and int((mask == 0).sum()) == k_prune
    return mask


def select_uniform(scores: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Spec §12's baseline: the same scores, ``N r`` experts pruned in every layer."""
    layers, experts = scores.shape
    per_layer = round(experts * sparsity)
    mask = torch.ones(layers, experts, dtype=torch.long)
    for layer in range(layers):
        order = torch.sort(scores[layer].float(), stable=True).indices[:per_layer]
        mask[layer, order] = 0
    return mask


def keep_lists(mask: torch.Tensor) -> list[list[int]]:
    """Per layer, the sorted original indices of the retained experts."""
    return [row.nonzero().flatten().tolist() for row in mask]


def prune_model(model, adapter, keep: list[list[int]]) -> None:
    """Physically remove the pruned experts (spec §5.3), in place."""
    blocks = adapter.moe_blocks(model)
    if len(blocks) != len(keep):
        raise ValueError(f"{len(keep)} keep lists for {len(blocks)} MoE blocks")
    for block, indices in zip(blocks, keep):
        adapter.prune(block, indices)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def masked_state(mask: torch.Tensor, device):
    """The ``mask`` mode of the DiEP forward: same model, pruned logits at -inf."""
    from .moe_forward import DiEPState
    return DiEPState(mode="mask", keep_mask=[row.bool().to(device) for row in mask])


# --------------------------------------------------------------------------
# The pruned checkpoint
# --------------------------------------------------------------------------
def save_pruned(model, adapter, keep, out_dir, *, tokenizer=None, meta: dict | None = None):
    """Write the pruned model: every tensor, the per-layer expert counts, the
    original indices, and whatever provenance ``meta`` carries."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().contiguous().cpu() for name, tensor in model.state_dict().items()}
    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    config = model.config.to_dict()
    config["num_experts_per_layer"] = [len(k) for k in keep]
    config["diep"] = {"keep": keep, **(meta or {})}
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    return out_dir


def load_pruned(path, *, device="cuda", dtype=torch.bfloat16):
    """Rebuild a pruned model saved by :func:`save_pruned`.

    The architecture is instantiated on the meta device (no 7B random init),
    each MoE block is cut to its saved size, the tensors are loaded in place,
    and the buffers the state dict does not carry (rotary frequencies) are
    rebuilt on the target device.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    from .adapters import get_adapter

    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    keep = raw["diep"]["keep"]
    config = AutoConfig.from_pretrained(path)
    import transformers
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, **{key: dtype})
    adapter = get_adapter(model)
    for block, indices in zip(adapter.moe_blocks(model), keep):
        adapter.prune(block, list(range(len(indices))))      # shapes only; values follow
    state = load_file(str(path / "model.safetensors"), device=str(device))
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected or [m for m in missing if "rotary" not in m]:
        raise RuntimeError(f"pruned checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    _rebuild_buffers(model, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, raw


def _rebuild_buffers(model, device):
    """Non-persistent buffers (rotary ``inv_freq``) are not in the state dict and
    were created on the meta device; re-create those modules.

    On the **CPU**, then moved -- as ``from_pretrained`` does. Computing
    ``inv_freq`` on the GPU differs in the last bits (3.7e-9 for OLMoE), and a
    heavily pruned model amplifies that into different routing on a few percent
    of tokens: the loaded checkpoint would not be the model that was saved.
    """
    for name, module in list(model.named_modules()):
        if any(buffer.is_meta for buffer in module.buffers(recurse=False)):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            rebuilt = type(module)(config=model.config, device="cpu").to(device)
            setattr(parent, child, rebuilt)
    leftover = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if leftover:
        raise RuntimeError(f"tensors still on the meta device after loading: {leftover[:5]}")
