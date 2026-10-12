import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def set_experts_impl(model, name):
    """transformers 5.x picks OLMoE's expert kernels from the config at forward
    time ("grouped_mm" by default); 4.x has only its loop. No-op there."""
    if hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = name
        return True
    return False


def make_tiny_olmoe(dtype=torch.float32, seed=0, experts=8, top_k=2):
    """A 2-layer OLMoE small enough for exact checks, in whichever layout the
    installed transformers builds (fused on 5.x, legacy on 4.x)."""
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    torch.manual_seed(seed)
    model = OlmoeForCausalLM(OlmoeConfig(
        vocab_size=64, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, num_experts=experts,
        num_experts_per_tok=top_k, norm_topk_prob=False, bos_token_id=0, eos_token_id=1,
        pad_token_id=2)).eval()
    # A randomly initialised router is near-uniform; spread it so routing is
    # decisive and the checks exercise real top-k choices.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(20.0)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(dtype)


@pytest.fixture
def tiny():
    return make_tiny_olmoe()


@pytest.fixture
def ids():
    return torch.randint(3, 64, (2, 12), generator=torch.Generator().manual_seed(1))


def tiny_calibration(n=4, seq=12):
    from diep.data import Calibration
    ids = torch.randint(3, 64, (n, seq), generator=torch.Generator().manual_seed(2))
    return Calibration(ids, torch.ones_like(ids), "synthetic")
