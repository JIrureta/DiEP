"""Efficiency metrics (spec §6.1, Table 4).

``memory ratio``  parameter bytes of the pruned model / the full model's
``speedup``       generation throughput relative to the full model: bf16, one
                  GPU, batch 1, a 128-token prompt, 256 new tokens, greedy;
                  mean over 20 prompts after 3 warm-up runs

Prompts are the first 128 tokens of the first 20 calibration sequences.
``min_new_tokens`` equals ``max_new_tokens`` so every run generates exactly 256
tokens. The DiEP forward (needed for skipping) is a Python loop over experts, as
is OLMoE's own eager expert loop in transformers 5.x; the pruned model is also
timed through the DiEP forward *without* skipping, so the skip speedup is
measured against the same implementation.
"""

from __future__ import annotations

import time

import torch

__all__ = ["param_bytes", "throughput"]


def param_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


@torch.no_grad()
def throughput(model, prompts: list[torch.Tensor], *, new_tokens=256, warmup=3,
               pad_token_id=None) -> dict:
    def generate(ids):
        return model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                              do_sample=False, use_cache=True, pad_token_id=pad_token_id,
                              attention_mask=torch.ones_like(ids))
    for i in range(warmup):
        generate(prompts[i % len(prompts)])
    elapsed, generated = 0.0, 0
    for ids in prompts:
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = generate(ids)
        torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        generated += out.shape[1] - ids.shape[1]
    return {"tokens_per_s": generated / elapsed, "seconds": elapsed, "tokens": generated,
            "prompts": len(prompts), "new_tokens": new_tokens}
