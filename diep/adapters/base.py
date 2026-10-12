"""The architecture-facing interface of spec §2.1."""

from __future__ import annotations

import torch
from torch import nn


class MoEAdapter:
    """What DiEP needs to know about one model family.

    ``L`` counts routed-MoE layers only (spec §2): a dense first layer, as in
    DeepSeek-MoE, is simply not returned by :meth:`moe_blocks`.
    """

    name = "base"

    def matches(self, model: nn.Module) -> bool:
        raise NotImplementedError

    # -- structure -------------------------------------------------------
    def moe_blocks(self, model: nn.Module) -> list[nn.Module]:
        """Ordered routed-MoE blocks, length ``L``."""
        raise NotImplementedError

    def decoder_layers(self, model: nn.Module) -> list[nn.Module]:
        """The decoder layers, for gradient checkpointing."""
        raise NotImplementedError

    def num_experts(self, block: nn.Module) -> int:
        raise NotImplementedError

    def top_k(self, block: nn.Module) -> int:
        raise NotImplementedError

    def renormalize(self, block: nn.Module) -> bool:
        raise NotImplementedError

    # -- computation -----------------------------------------------------
    def router_logits(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Pre-softmax router logits ``[T, N]``."""
        raise NotImplementedError

    def expert(self, block: nn.Module, i: int, x: torch.Tensor) -> torch.Tensor:
        """``FFN_i(x)``: ``[T, d] -> [T, d]``, exactly as the model computes it."""
        raise NotImplementedError

    def shared_out(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
        """Shared-expert output including its own gate, or ``None``."""
        return None

    def returns_router_logits(self, block: nn.Module) -> bool:
        """Whether the block's forward returns ``(hidden, router_logits)``."""
        return False

    # -- surgery ---------------------------------------------------------
    def prune(self, block: nn.Module, keep: list[int]) -> None:
        """Keep only ``keep`` (sorted original indices): slice router rows,
        drop expert weights, set per-layer ``N_l`` and ``k_l = min(k, N_l)``."""
        raise NotImplementedError

    def expert_params(self, block: nn.Module, i: int) -> dict[str, torch.Tensor]:
        """Every weight tensor of ``FFN_i``, by name (for merging, spec §8)."""
        raise NotImplementedError

    def set_expert_params(self, block: nn.Module, i: int, params: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError
