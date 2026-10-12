"""Optional: merge pruned experts into retained ones (spec §8, Appendix A.5).

Off by default. When enabled it runs after selection (§5.2) and before physical
pruning (§5.3), independently per layer, on the **unpruned** model:

1. RBF-CKA between all experts on 2048 sampled layer inputs (as in §7.2);
2. each pruned expert ``p`` goes to ``r* = argmax_{retained r} CKA(p, r)``;
3. for each retained ``r``: ``G_r = {r} + {p assigned to r}``, ``CKA(r, r) = 1``;
4. merging weights ``c_e = softmax_{e in G_r} CKA(e, r)``;
5. every weight tensor of ``FFN_r`` becomes ``sum_e c_e W_e``; the router row of
   ``r`` is unchanged.

Accumulation is in fp32 and the result is cast back to the model dtype. Pruned
experts are only ever read, and each retained expert belongs to exactly one
group, so writing the merged weights in place is order-independent.
"""

from __future__ import annotations

import torch

from .skip import collect_layer_inputs, layer_cka

__all__ = ["merge_pruned_experts"]


@torch.no_grad()
def merge_pruned_experts(model, adapter, calibration, mask, device, *, n_tokens=2048,
                         seed=0) -> list[dict]:
    """Merge in place. Returns, per layer, the assignment and weights used."""
    inputs, _ = collect_layer_inputs(model, adapter, calibration, device,
                                     n_tokens=n_tokens, seed=seed)
    report = []
    for layer, block in enumerate(adapter.moe_blocks(model)):
        cka = layer_cka(adapter, block, inputs[layer]).cpu()
        retained = mask[layer].nonzero().flatten().tolist()
        pruned = (mask[layer] == 0).nonzero().flatten().tolist()
        groups = {r: [r] for r in retained}
        for p in pruned:
            target = max(retained, key=lambda r: float(cka[p, r]))
            groups[target].append(p)
        merged_layer = {}
        for r, members in groups.items():
            if len(members) == 1:
                continue
            similarity = torch.tensor([1.0 if e == r else float(cka[e, r]) for e in members])
            weights = torch.softmax(similarity, dim=0)
            params = {e: adapter.expert_params(block, e) for e in members}
            new = {name: sum(float(w) * params[e][name].float() for e, w in zip(members, weights))
                   for name in params[r]}
            adapter.set_expert_params(block, r, {name: t.to(params[r][name].dtype)
                                                 for name, t in new.items()})
            merged_layer[r] = {"members": members, "weights": weights.tolist()}
        report.append(merged_layer)
    return report
