"""OLMoE (``allenai/OLMoE-1B-7B-0924``) in both ``transformers`` layouts.

=========== ================================== ======================================
layout      transformers                        expert weights
=========== ================================== ======================================
fused       5.x (``OlmoeExperts``)             ``gate_up_proj [N, 2*d_ff, d]``,
                                                ``down_proj [N, d, d_ff]``
legacy      4.x (``nn.ModuleList`` of MLPs)    ``gate_proj``, ``up_proj``, ``down_proj``
=========== ================================== ======================================

OLMoE's routing, from the model's own source: softmax over all ``N`` logits in
fp32, top-``k`` (k = 8), **no** renormalisation (``norm_topk_prob = False``),
weights cast to the activation dtype. 16 MoE layers, 64 routed experts, no
shared experts, no dense layers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import MoEAdapter


def _fused(block: nn.Module) -> bool:
    return hasattr(block.experts, "gate_up_proj")


class OlmoeAdapter(MoEAdapter):
    name = "olmoe"

    def matches(self, model: nn.Module) -> bool:
        return getattr(getattr(model, "config", None), "model_type", None) == "olmoe"

    # -- structure -------------------------------------------------------
    def decoder_layers(self, model):
        return list(model.model.layers)

    def moe_blocks(self, model):
        return [layer.mlp for layer in model.model.layers]

    def num_experts(self, block):
        return block.experts.num_experts if _fused(block) else len(block.experts)

    def top_k(self, block):
        return block.gate.top_k if _fused(block) else block.top_k

    def renormalize(self, block):
        return bool(block.gate.norm_topk_prob if _fused(block) else block.norm_topk_prob)

    def returns_router_logits(self, block):
        return not _fused(block)

    # -- computation -----------------------------------------------------
    def router_logits(self, block, x):
        return F.linear(x, block.gate.weight)

    def expert(self, block, i, x):
        if _fused(block):
            experts = block.experts
            gate, up = F.linear(x, experts.gate_up_proj[i]).chunk(2, dim=-1)
            return F.linear(experts.act_fn(gate) * up, experts.down_proj[i])
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
        if _fused(block):
            experts = block.experts
            experts.gate_up_proj = nn.Parameter(
                experts.gate_up_proj.data.index_select(0, index).clone(), requires_grad=False)
            experts.down_proj = nn.Parameter(
                experts.down_proj.data.index_select(0, index).clone(), requires_grad=False)
            experts.num_experts = n_l
            block.gate.num_experts = n_l
            block.gate.top_k = min(k, n_l)
        else:
            block.gate.out_features = n_l
            block.experts = nn.ModuleList([block.experts[i] for i in keep])
            block.num_experts = n_l
            block.top_k = min(k, n_l)

    def expert_params(self, block, i):
        if _fused(block):
            return {"gate_up_proj": block.experts.gate_up_proj.data[i],
                    "down_proj": block.experts.down_proj.data[i]}
        mlp = block.experts[i]
        return {"gate_proj": mlp.gate_proj.weight.data, "up_proj": mlp.up_proj.weight.data,
                "down_proj": mlp.down_proj.weight.data}

    @torch.no_grad()
    def set_expert_params(self, block, i, params):
        if _fused(block):
            block.experts.gate_up_proj.data[i].copy_(params["gate_up_proj"])
            block.experts.down_proj.data[i].copy_(params["down_proj"])
        else:
            mlp = block.experts[i]
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(mlp, name).weight.data.copy_(params[name])
