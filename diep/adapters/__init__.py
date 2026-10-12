"""The model adapter interface (spec §2.1), and the registry.

Everything outside this package is architecture-agnostic: it asks an adapter
for the routed MoE blocks, their router logits, one expert's output, the
shared-expert output (``None`` when there is none), ``N``, ``k``, whether top-k
weights are renormalised, and how to physically remove experts.
"""

from __future__ import annotations

from torch import nn

from .base import MoEAdapter
from .olmoe import OlmoeAdapter

__all__ = ["MoEAdapter", "OlmoeAdapter", "get_adapter", "ADAPTERS"]

#: One entry per supported architecture. The paper's Mixtral, DeepSeek-MoE and
#: Qwen2-MoE need an adapter each; this study compresses OLMoE.
ADAPTERS: list[MoEAdapter] = [OlmoeAdapter()]


def get_adapter(model: nn.Module) -> MoEAdapter:
    for adapter in ADAPTERS:
        if adapter.matches(model):
            return adapter
    model_type = getattr(getattr(model, "config", None), "model_type", type(model).__name__)
    raise LookupError(f"no DiEP adapter for {model_type!r}; registered: "
                      f"{[a.name for a in ADAPTERS]}")
