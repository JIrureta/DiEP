"""The six unit tests spec §11 requires."""

import copy

import pytest
import torch

from conftest import make_tiny_olmoe, set_experts_impl
from diep.adapters import get_adapter
from diep.cka import cka, cka_matrix
from diep.moe_forward import DiEPState, install
from diep.prune import masked_state, prune_model, select_global, keep_lists


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits.float()


# 1 ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_1_routed_search_at_init_is_the_original(ids, dtype):
    """alpha = 0, beta = 1: g = N * (1/N) = 1 exactly. Against the model's eager
    expert loop -- the one DiEP replicates -- the logits are identical; against
    transformers 5.x's default grouped_mm kernels they agree to rounding, well
    inside the spec's 1e-3 (bf16)."""
    model = make_tiny_olmoe(dtype)
    adapter = get_adapter(model)
    default = _logits(model, ids)
    has_impl = set_experts_impl(model, "eager")
    eager = _logits(model, ids)
    state = DiEPState(mode="search", alpha=torch.zeros(2, 8), beta=torch.ones(2))
    dmodel = install(model, adapter, state)
    ours = _logits(model, ids)
    assert torch.equal(ours, eager)
    assert (ours - default).abs().max().item() <= (1e-3 if dtype == torch.bfloat16 else 1e-6)
    dmodel.restore()
    assert torch.equal(_logits(model, ids), eager)
    if has_impl:
        set_experts_impl(model, "grouped_mm")
        assert torch.equal(_logits(model, ids), default)


# 2 ---------------------------------------------------------------------------
@pytest.mark.parametrize("impl", ["eager", "grouped_mm"])
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 1e-2)])
def test_2_masking_equals_physical_pruning(ids, dtype, tol, impl):
    """Also checks that the model's native kernels cope with a block whose
    expert count is no longer the config's -- which is how DiEP is evaluated."""
    model = make_tiny_olmoe(dtype)
    if not set_experts_impl(model, impl) and impl != "eager":
        pytest.skip("this transformers has a single expert implementation")
    adapter = get_adapter(model)
    scores = torch.rand(2, 8, generator=torch.Generator().manual_seed(3))
    mask = select_global(scores, 0.375, m_min=2)            # 6 of 16 pruned
    dmodel = install(model, adapter, masked_state(mask, "cpu"))
    masked = _logits(model, ids)
    dmodel.restore()
    physical = copy.deepcopy(model)
    prune_model(physical, adapter, keep_lists(mask))
    # Not bit-exact even with the same loop: a softmax over 8 logits with two at
    # -inf and one over the 6 retained ones reduce in a different order.
    assert (masked - _logits(physical, ids)).abs().max().item() <= tol


# 3 ---------------------------------------------------------------------------
@pytest.mark.parametrize("r", [0.25, 0.5, 0.75])
def test_3_selection_respects_K_and_m_min(r):
    scores = torch.rand(16, 64, generator=torch.Generator().manual_seed(4))
    scores[3] = 0.0                    # one layer is uniformly the least useful
    mask = select_global(scores, r, m_min=8)
    assert int((mask == 0).sum()) == round(64 * 16 * r)
    assert (mask.sum(1) >= 8).all()
    assert int(mask[3].sum()) == 8     # it is pruned down to the floor, never below


def test_3b_infeasible_K_is_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        select_global(torch.rand(16, 64), 0.9, m_min=8)


def test_3c_ties_are_broken_by_layer_then_expert():
    mask = select_global(torch.zeros(2, 4), 0.25, m_min=1)   # K = 2, all tied
    assert mask[0].tolist() == [0, 0, 1, 1] and mask[1].tolist() == [1, 1, 1, 1]


# 4 ---------------------------------------------------------------------------
def test_4_gradient_is_finite_at_step_0_where_phi_is_0(ids):
    """The +1e-8 inside the sqrt: without it d sqrt(x)/dx at x = 0 is inf."""
    model = make_tiny_olmoe()
    set_experts_impl(model, "eager")
    adapter = get_adapter(model)
    reference = model.model(input_ids=ids).last_hidden_state.detach()
    alpha = torch.zeros(2, 8, requires_grad=True)
    beta = torch.ones(2, requires_grad=True)
    dmodel = install(model, adapter, DiEPState(mode="search", alpha=alpha, beta=beta))
    out = model.model(input_ids=ids).last_hidden_state
    phi = torch.sqrt(((out - reference) ** 2).sum() + 1e-8)
    assert phi.item() == pytest.approx(1e-4)                 # Phi = 0 at init
    logits = model.lm_head(out)
    ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 64), ids[:, 1:].reshape(-1))
    (ce + 0.01 * phi).backward()
    assert torch.isfinite(alpha.grad).all() and torch.isfinite(beta.grad).all()
    assert alpha.grad.abs().sum() > 0 and beta.grad.abs().sum() > 0
    dmodel.restore()


# 5 ---------------------------------------------------------------------------
def test_5_skipping_with_gamma_zero_is_no_skipping(ids):
    model = make_tiny_olmoe()
    set_experts_impl(model, "eager")
    adapter = get_adapter(model)
    prune_model(model, adapter, [list(range(6)), [0, 2, 3, 5, 6, 7]])
    native = _logits(model, ids)
    state = DiEPState(mode="skip", gamma=None)               # the same forward, no skipping
    dmodel = install(model, adapter, state)
    no_skip = _logits(model, ids)
    state.gamma = [torch.zeros(adapter.top_k(b), 6, 6) for b in adapter.moe_blocks(model)]
    assert torch.equal(_logits(model, ids), no_skip)
    assert torch.equal(no_skip, native)
    assert state.skip_stats["skipped"] == 0
    dmodel.restore()


# 6 ---------------------------------------------------------------------------
def test_6_cka_is_one_on_itself_and_scale_invariant():
    y = torch.randn(64, 16, generator=torch.Generator().manual_seed(5))
    assert cka(y, y) == pytest.approx(1.0, abs=1e-6)
    z = torch.randn(64, 16, generator=torch.Generator().manual_seed(6))
    assert cka(3.7 * y, z) == pytest.approx(cka(y, z), abs=1e-6)
    matrix = cka_matrix([y, z, 0.5 * y])
    assert torch.allclose(matrix, matrix.T, atol=1e-6)
    assert matrix[0, 2].item() == pytest.approx(1.0, abs=1e-6)
