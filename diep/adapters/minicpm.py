"""MiniCPM-MoE (``openbmb/MiniCPM-MoE-8x2B``, through the port in DPMA/models/minicpm_moe).

Each decoder layer's ``.mlp`` is a ``MiniCPMMoE``: a bias-free ``nn.Linear``
router (``gate``), an ``nn.ModuleList`` of SwiGLU experts with ``w1`` (gate),
``w3`` (up) and ``w2`` (down), softmax over all experts, top-k and an
unconditional renormalisation of the top-k weights. The block returns the hidden
states alone. Its forward reads ``num_experts_per_tok`` and indexes
``experts`` by position, so pruning only has to slice the router, keep the
surviving modules, and shrink the two counts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import MoEAdapter


class MiniCPMAdapter(MoEAdapter):
    name = "minicpm"

    def matches(self, model: nn.Module) -> bool:
        config = getattr(model, "config", None)
        return (getattr(config, "model_type", None) == "minicpm"
                and getattr(config, "num_experts", None) is not None)

    # -- structure -------------------------------------------------------
    def decoder_layers(self, model):
        return list(model.model.layers)

    def moe_blocks(self, model):
        return [layer.mlp for layer in model.model.layers]

    def num_experts(self, block):
        return len(block.experts)

    def top_k(self, block):
        return block.num_experts_per_tok

    def renormalize(self, block):
        return True                      # MiniCPMMoE.forward renormalises unconditionally

    def returns_router_logits(self, block):
        return False

    # -- computation -----------------------------------------------------
    def router_logits(self, block, x):
        return F.linear(x, block.gate.weight)

    def expert(self, block, i, x):
        return block.experts[i](x)

    # -- surgery ---------------------------------------------------------
    @torch.no_grad()
    def prune(self, block, keep):
        keep = sorted(int(i) for i in keep)
        n_l = len(keep)
        index = torch.tensor(keep, device=block.gate.weight.device)
        k = self.top_k(block)
        block.gate.weight = nn.Parameter(block.gate.weight.data.index_select(0, index).clone(),
                                         requires_grad=False)
        block.gate.out_features = n_l
        block.experts = nn.ModuleList([block.experts[i] for i in keep])
        block.num_experts = n_l
        block.num_experts_per_tok = min(k, n_l)

    def expert_params(self, block, i):
        mlp = block.experts[i]
        return {"w1": mlp.w1.weight.data, "w3": mlp.w3.weight.data, "w2": mlp.w2.weight.data}

    @torch.no_grad()
    def set_expert_params(self, block, i, params):
        mlp = block.experts[i]
        for name in ("w1", "w3", "w2"):
            getattr(mlp, name).weight.data.copy_(params[name])
