"""Stage 3: adaptive expert skipping at inference (spec §7), on the pruned model.

Rule, per token and layer, with the selected experts sorted by weight
``w_e0 >= w_e1 >= ...``: for every rank ``j >= 1``::

    skip e_j   if   w_ej < gamma_j(e0, ej) * w_e0
    gamma_j(a, b) = min(1, gamma1_j * gamma2(a, b))

``e0`` is never skipped; a skipped term is dropped and the rest are not
renormalised. For OLMoE ``k = 8``, and each rank is thresholded against ``e0``
([DECISION] in the spec; for ``k = 2`` it is exactly the paper's rule).

Calibration, one pass over the calibration set through the *pruned* model:

``gamma1_j``   median over tokens of ``w_ej / w_e0`` at rank ``j``, per layer
``gamma2(a,b)`` ``CKA(a, b) / mean_{i<j} CKA(i, j)`` over the retained experts,
               RBF-CKA on 2048 sampled layer inputs (:mod:`diep.cka`)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .cka import cka_matrix
from .moe_forward import route

__all__ = ["collect_layer_inputs", "calibrate_skip", "build_gamma", "save_skip_config",
           "load_skip_config"]


def _sample_positions(calibration, n_tokens: int, seed: int) -> dict[int, torch.Tensor]:
    """``n_tokens`` (sequence, position) pairs, grouped by sequence."""
    total = len(calibration) * calibration.seq_len
    flat = torch.randperm(total, generator=torch.Generator().manual_seed(seed))[:n_tokens]
    flat = flat.sort().values
    grouped: dict[int, list[int]] = {}
    for f in flat.tolist():
        grouped.setdefault(f // calibration.seq_len, []).append(f % calibration.seq_len)
    return {s: torch.tensor(p) for s, p in grouped.items()}


@torch.no_grad()
def collect_layer_inputs(model, adapter, calibration, device, *, n_tokens=2048, seed=0,
                         ratios=False):
    """Run the calibration set once; per MoE layer, keep the inputs at
    ``n_tokens`` sampled positions and (optionally) every token's ``w_ej / w_e0``."""
    blocks = adapter.moe_blocks(model)
    positions = _sample_positions(calibration, n_tokens, seed)
    inputs = [[] for _ in blocks]
    weight_ratios = [[] for _ in blocks]
    current = {"seq": 0}
    handles = []
    for layer, block in enumerate(blocks):
        def hook(module, args, _layer=layer):
            x = args[0].reshape(-1, args[0].shape[-1])
            picked = positions.get(current["seq"])
            if picked is not None:
                inputs[_layer].append(x[picked.to(x.device)].clone())
            if ratios:
                w, _, _ = route(adapter, module, x)
                weight_ratios[_layer].append((w[:, 1:] / w[:, :1]).float().cpu())
        handles.append(block.register_forward_pre_hook(hook))
    try:
        for s in range(len(calibration)):
            current["seq"] = s
            model(input_ids=calibration.input_ids[s:s + 1].to(device),
                  attention_mask=calibration.attention_mask[s:s + 1].to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    inputs = [torch.cat(chunks) for chunks in inputs]
    weight_ratios = [torch.cat(r) for r in weight_ratios] if ratios else None
    return inputs, weight_ratios


@torch.no_grad()
def layer_cka(adapter, block, x):
    """CKA between every expert of ``block`` on the inputs ``x``."""
    outputs = [adapter.expert(block, i, x).float() for i in range(adapter.num_experts(block))]
    return cka_matrix(outputs)


@torch.no_grad()
def calibrate_skip(model, adapter, calibration, device, *, n_tokens=2048, seed=0) -> dict:
    blocks = adapter.moe_blocks(model)
    inputs, ratios = collect_layer_inputs(model, adapter, calibration, device,
                                          n_tokens=n_tokens, seed=seed, ratios=True)
    config = {}
    for layer, block in enumerate(blocks):
        gamma1 = ratios[layer].median(dim=0).values.tolist()        # ranks 1..k_l-1
        cka = layer_cka(adapter, block, inputs[layer])
        n = cka.shape[0]
        upper = cka[torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)]
        config[str(layer)] = {"gamma1": gamma1, "cka": cka.cpu().tolist(),
                              "cka_mean": float(upper.mean()), "num_experts": n,
                              "top_k": adapter.top_k(block)}
    return config


def build_gamma(config: dict, device="cpu") -> list[torch.Tensor]:
    """Per layer ``gamma[j, a, b] = min(1, gamma1_j * CKA(a, b) / mean CKA)``;
    row ``j = 0`` is zero (the top expert is never skipped)."""
    tables = []
    for layer in sorted(config, key=int):
        entry = config[layer]
        cka = torch.tensor(entry["cka"], dtype=torch.float32)
        gamma2 = cka / max(entry["cka_mean"], 1e-12)
        gamma1 = torch.tensor([0.0] + list(entry["gamma1"]), dtype=torch.float32)
        tables.append(torch.clamp(gamma1[:, None, None] * gamma2[None], max=1.0).to(device))
    return tables


def save_skip_config(config: dict, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(config))
    return path


def load_skip_config(path) -> dict:
    return json.loads(Path(path).read_text())
