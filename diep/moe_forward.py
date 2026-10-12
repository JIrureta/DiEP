"""The one DiEP forward that replaces every routed-MoE block's forward (spec §2.1).

A single function, driven by ``DiEPState.mode``:

=============== ===============================================================
mode            what a block computes (``S`` = top-k of the unmodified router)
=============== ===============================================================
``original``    the model's own forward, untouched
``search``      ``beta_l * sum_{i in S} w_i * g_i * FFN_i(x)``, ``g_i = N * softmax(alpha_l)_i``
                -- spec §4.2 default (``search_forward = "routed"``)
``dense``       ``beta_l * sum_{i=1..N} softmax(alpha_l)_i * FFN_i(x)`` -- the
                paper-literal Eq. 5 (``search_forward = "dense"``), ~N/k x compute
``mask``        pruned experts' router logits set to -inf before the softmax;
                numerically equivalent to physical pruning (spec §5.3)
``skip``        the routing of the block as it is (pruned or masked), then the
                adaptive-skipping rule of spec §7.1
=============== ===============================================================

The shared-expert output, when the architecture has one, is added unscaled in
every mode. The original module forward is never called during search.

Numerics are copied from the model's own expert loop, not reinvented: routing
probabilities in fp32, the combination coefficient ``w * g * beta`` in fp32 and
then cast to the expert output dtype (spec §4.2), experts visited in ascending
index order, and within an expert the tokens in the (rank, token) order the
model uses. At ``alpha = 0, beta = 1`` the coefficient is exactly ``w`` (``g =
64 * (1/64) = 1.0`` is exact in fp32), so the search forward reproduces the
original bit for bit -- unit test 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

MODES = ("original", "search", "dense", "mask", "skip")


@dataclass
class DiEPState:
    """Everything the replaced forwards read. One instance per model."""

    mode: str = "original"
    #: ``[L, N]`` fp32 intra-layer logits, and ``[L]`` fp32 layer scales.
    alpha: torch.Tensor | None = None
    beta: torch.Tensor | None = None
    #: Per layer, a bool ``[N_l]`` mask of retained experts (``mask`` mode).
    keep_mask: list[torch.Tensor] | None = None
    #: Per layer, ``gamma[j, a, b]`` (``[k_l, N_l, N_l]``) for ``skip`` mode.
    gamma: list[torch.Tensor] | None = None
    #: Counters filled in ``skip`` mode: evaluated and skipped (token, expert) pairs.
    skip_stats: dict = field(default_factory=lambda: {"pairs": 0, "skipped": 0})

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")


# --------------------------------------------------------------------------
# Routing and dispatch
# --------------------------------------------------------------------------
def route(adapter, block, x, keep_mask=None):
    """The block's own routing rule: softmax (fp32) -> top-k -> optional renorm.

    Returns fp32 weights ``[T, k]`` (sorted, largest first), indices ``[T, k]``
    and the raw logits. With ``keep_mask``, pruned logits are -inf, so the
    softmax runs over the retained experts only and ``k`` becomes ``min(k, N_l)``.
    """
    logits = adapter.router_logits(block, x)
    k = adapter.top_k(block)
    if keep_mask is not None:
        logits_eff = logits.masked_fill(~keep_mask.to(logits.device), float("-inf"))
        k = min(k, int(keep_mask.sum()))
    else:
        logits_eff = logits
    probs = F.softmax(logits_eff, dim=-1, dtype=torch.float32)
    weights, index = torch.topk(probs, k, dim=-1)
    if adapter.renormalize(block):
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, index, logits


def dispatch(adapter, block, x, index, coef, active=None):
    """``sum_j coef[t, j] * FFN_{index[t, j]}(x_t)``, evaluated once per expert.

    ``coef`` is already in the activation dtype. ``active`` (bool ``[T, k]``)
    drops (token, expert) pairs without computing them -- adaptive skipping.
    """
    out = torch.zeros_like(x)
    hit = index if active is None else index.masked_fill(~active, -1)
    for e in torch.unique(hit[hit >= 0]).tolist():
        # Transposed so the tokens come out in (rank, token) order -- the order
        # the model's own loop uses, which keeps the GEMM inputs identical.
        pos, tok = torch.where((hit == e).T)
        y = adapter.expert(block, e, x[tok]) * coef[tok, pos, None]
        out.index_add_(0, tok, y.to(out.dtype))
    return out


def dense_dispatch(adapter, block, x, coef_per_expert):
    """Every token through every expert: ``sum_i coef_i * FFN_i(x)``."""
    out = torch.zeros_like(x)
    for e in range(coef_per_expert.shape[0]):
        out = out + adapter.expert(block, e, x) * coef_per_expert[e].to(x.dtype)
    return out


def diep_forward(adapter, block, layer, x, state: DiEPState):
    """One routed-MoE layer under DiEP. ``x`` is ``[T, d]``."""
    mode = state.mode
    if mode == "search":
        weights, index, logits = route(adapter, block, x)
        n = adapter.num_experts(block)
        g = n * torch.softmax(state.alpha[layer], dim=-1)          # 1 at init
        scale = g * state.beta[layer]                                # fp32 [N]
        coef = (weights * scale[index]).to(x.dtype)                 # fp32 product, then cast
        out = dispatch(adapter, block, x, index, coef)
    elif mode == "dense":
        logits = adapter.router_logits(block, x)
        abar = torch.softmax(state.alpha[layer], dim=-1)
        out = dense_dispatch(adapter, block, x, abar * state.beta[layer])
    elif mode == "mask":
        weights, index, logits = route(adapter, block, x, state.keep_mask[layer])
        out = dispatch(adapter, block, x, index, weights.to(x.dtype))
    elif mode == "skip":
        mask = state.keep_mask[layer] if state.keep_mask is not None else None
        weights, index, logits = route(adapter, block, x, mask)
        gamma = state.gamma[layer] if state.gamma is not None else None   # None: skip nothing
        active = skip_pairs(weights, index, gamma)
        coef = torch.where(active, weights, torch.zeros_like(weights)).to(x.dtype)
        state.skip_stats["pairs"] += active.numel()
        state.skip_stats["skipped"] += int((~active).sum())
        # Skipped terms are dropped and the rest are NOT renormalised (spec §7.1).
        out = dispatch(adapter, block, x, index, coef, active)
    else:
        raise ValueError(f"diep_forward cannot run mode {mode!r}")

    shared = adapter.shared_out(block, x)
    if shared is not None:
        out = out + shared                       # never scaled by beta (spec §4.2)
    return out, logits


def skip_pairs(weights, index, gamma):
    """Spec §7.1: keep ``e_j`` (j >= 1) unless ``w_ej < gamma[j, e0, ej] * w_e0``.

    ``weights`` is sorted largest-first (``torch.topk``), so column 0 is ``e0``,
    which is never skipped. ``gamma`` is indexed in the block's own expert
    numbering (the retained experts of a physically pruned block).
    """
    active = torch.ones_like(weights, dtype=torch.bool)
    if gamma is None or weights.shape[1] < 2:
        return active
    e0 = index[:, :1].expand_as(index)
    ranks = torch.arange(index.shape[1], device=index.device).expand_as(index)
    threshold = gamma.to(weights.device)[ranks, e0, index] * weights[:, :1]
    active[:, 1:] = ~(weights[:, 1:] < threshold[:, 1:])
    return active


# --------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------
class DiEPModel:
    """A model whose MoE blocks run :func:`diep_forward`, plus the handles to undo it."""

    def __init__(self, model, adapter, state, originals):
        self.model, self.adapter, self.state = model, adapter, state
        self._originals = originals
        self._checkpointed = []

    @property
    def blocks(self):
        return self.adapter.moe_blocks(self.model)

    def restore(self):
        for block, forward in self._originals:
            block.forward = forward
        for layer, forward in self._checkpointed:
            layer.forward = forward
        self._originals, self._checkpointed = [], []
        return self.model

    def enable_checkpointing(self):
        """Recompute each decoder layer in the backward (spec §4.4, memory).

        Non-reentrant checkpointing works with frozen weights and parameters that
        live outside the module (``alpha``, ``beta``), and keeps the model in
        ``eval()`` -- HF's own checkpointing only runs in training mode. The DiEP
        forward is deterministic, so the recompute is exact.
        """
        for layer in self.adapter.decoder_layers(self.model):
            original = layer.forward

            def forward(*args, _original=original, **kwargs):
                if self.state.mode in ("search", "dense") and torch.is_grad_enabled():
                    return torch.utils.checkpoint.checkpoint(
                        _original, *args, use_reentrant=False, **kwargs)
                return _original(*args, **kwargs)

            layer.forward = forward
            self._checkpointed.append((layer, original))
        return len(self._checkpointed)


def install(model: nn.Module, adapter, state: DiEPState) -> DiEPModel:
    """Replace every routed-MoE block's forward with the DiEP one."""
    originals = []
    for layer, block in enumerate(adapter.moe_blocks(model)):
        original = block.forward
        originals.append((block, original))
        returns_tuple = adapter.returns_router_logits(block)

        def forward(hidden_states, *args, _block=block, _layer=layer, _original=original,
                    _tuple=returns_tuple, **kwargs):
            if state.mode == "original":
                return _original(hidden_states, *args, **kwargs)
            batch, seq, hidden = hidden_states.shape
            out, logits = diep_forward(adapter, _block, _layer,
                                       hidden_states.reshape(-1, hidden), state)
            out = out.reshape(batch, seq, hidden)
            return (out, logits) if _tuple else out

        block.forward = forward
    return DiEPModel(model, adapter, state, originals)
