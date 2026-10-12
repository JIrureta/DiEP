"""Beyond the spec's list: the parts the study depends on."""

import ast
import copy
from pathlib import Path

import pytest
import torch

from conftest import make_tiny_olmoe, tiny_calibration
from diep.adapters import get_adapter
from diep.data import load_calibration
from diep.moe_forward import DiEPState, install, skip_pairs
from diep.prune import (compute_scores, keep_lists, load_pruned, prune_model, save_pruned,
                        select_global, select_uniform)
from diep.search import SearchConfig, search, step_targets


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits.float()


def test_alternation_is_three_alpha_then_one_beta():
    targets = step_targets(SearchConfig(), 20)          # 10 epochs x 2 steps on 32 seqs
    assert targets[:8] == ["alpha"] * 3 + ["beta"] + ["alpha"] * 3 + ["beta"]
    assert targets.count("alpha") == 15 and targets.count("beta") == 5
    assert set(step_targets(SearchConfig(ablation="alpha_only"), 8)) == {"alpha"}
    assert set(step_targets(SearchConfig(fixed_beta="1:2"), 8)) == {"alpha"}
    assert set(step_targets(SearchConfig(ablation="beta_only"), 8)) == {"beta"}


def test_search_runs_and_moves_both_parameter_sets():
    # bf16, as the real model: H is cached in bf16 (spec §4.3), so Phi = 0 at
    # init holds exactly only when the model computes in bf16 too.
    model = make_tiny_olmoe(torch.bfloat16)
    adapter = get_adapter(model)
    cfg = SearchConfig(epochs=2, batch_size=2, micro_batch=1, lr_alpha=0.05, lr_beta=0.05,
                       update_ratio="1:1")
    result = search(model, adapter, tiny_calibration(), cfg, device="cpu", console=lambda *_: None)
    assert result["steps"] == {"total": 4, "alpha": 2, "beta": 2, "steps_per_epoch": 2}
    assert result["alpha"].abs().sum() > 0 and (result["beta"] != 1).any()
    assert (result["beta"] >= 1e-3).all()
    assert result["logs"][0]["phi"] == pytest.approx(1e-4, rel=1e-3)   # Phi = 0 at init
    # The model is handed back untouched.
    ids = torch.randint(3, 64, (1, 8))
    assert torch.equal(_logits(model, ids), _logits(make_tiny_olmoe(torch.bfloat16), ids))


def test_dense_search_forward_passes_gradient():
    model = make_tiny_olmoe()
    alpha = torch.zeros(2, 8, requires_grad=True)
    beta = torch.ones(2, requires_grad=True)
    install(model, get_adapter(model), DiEPState(mode="dense", alpha=alpha, beta=beta))
    model(input_ids=torch.randint(3, 64, (1, 8))).logits.sum().backward()
    assert alpha.grad.abs().sum() > 0 and beta.grad.abs().sum() > 0


def test_scores_modes():
    alpha, beta = torch.randn(3, 5), torch.tensor([1.0, 2.0, 0.5])
    s = compute_scores(alpha, beta)
    assert torch.allclose(s, torch.softmax(alpha, -1) * beta[:, None])
    assert torch.allclose(compute_scores(alpha, beta, score_mode="raw"), alpha * beta[:, None])
    f = torch.rand(3, 5)
    assert torch.allclose(compute_scores(alpha, beta, ablation="beta_only", frequency=f),
                          f * beta[:, None])


def test_uniform_selection_prunes_N_r_per_layer():
    mask = select_uniform(torch.rand(16, 64), 0.75)
    assert (mask.sum(1) == 16).all()


def test_skip_rule():
    weights = torch.tensor([[0.5, 0.3, 0.1], [0.4, 0.39, 0.2]])
    index = torch.tensor([[0, 1, 2], [2, 0, 1]])
    gamma = torch.full((3, 3, 3), 0.5)
    active = skip_pairs(weights, index, gamma)
    # rank 1: 0.3 < 0.25? no;  rank 2: 0.1 < 0.25 -> skip.  Row 2: 0.39 < 0.2? no; 0.2 < 0.2? no.
    assert active.tolist() == [[True, True, False], [True, True, True]]


def test_save_and_load_roundtrip(tmp_path, ids):
    model = make_tiny_olmoe()
    adapter = get_adapter(model)
    keep = [[0, 1, 4, 5, 7], [1, 2, 3, 6, 7]]
    prune_model(model, adapter, keep)
    reference = _logits(model, ids)
    save_pruned(model, adapter, keep, tmp_path / "pruned", meta={"sparsity": 0.375})
    loaded, raw = load_pruned(tmp_path / "pruned", device="cpu", dtype=torch.float32)
    assert raw["num_experts_per_layer"] == [5, 5] and raw["diep"]["keep"] == keep
    assert torch.equal(_logits(loaded, ids), reference)


def test_merge_weights_and_router(tmp_path):
    from diep.merge import merge_pruned_experts
    model = make_tiny_olmoe()
    adapter = get_adapter(model)
    router_before = [b.gate.weight.clone() for b in adapter.moe_blocks(model)]
    mask = select_global(torch.rand(2, 8, generator=torch.Generator().manual_seed(7)), 0.5, 2)
    report = merge_pruned_experts(model, adapter, tiny_calibration(8, 16), mask, "cpu", n_tokens=64)
    for layer, groups in enumerate(report):
        assert sum(len(g["members"]) - 1 for g in groups.values()) == int((mask[layer] == 0).sum())
        for r, g in groups.items():
            assert g["members"][0] == r and sum(g["weights"]) == pytest.approx(1.0)
    for block, before in zip(adapter.moe_blocks(model), router_before):
        assert torch.equal(block.gate.weight, before)       # router rows unchanged


def test_merge_of_identical_experts_changes_nothing():
    from diep.merge import merge_pruned_experts
    model = make_tiny_olmoe()
    adapter = get_adapter(model)
    for block in adapter.moe_blocks(model):
        first = adapter.expert_params(block, 0)
        for e in range(1, 8):
            adapter.set_expert_params(block, e, {k: v.clone() for k, v in first.items()})
    before = copy.deepcopy(model)
    mask = select_global(torch.rand(2, 8), 0.5, 2)
    merge_pruned_experts(model, adapter, tiny_calibration(8, 16), mask, "cpu", n_tokens=64)
    for a, b in zip(model.parameters(), before.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_calibration_file_must_be_the_studys(tmp_path):
    other = tmp_path / "c4-128x2048-seed0.pt"
    torch.save({"input_ids": torch.zeros(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long)}, other)
    with pytest.raises(ValueError, match="refusing"):
        load_calibration(other)
    assert len(load_calibration(other, allow_other_file=True)) == 1


def test_protocol_is_identical_to_the_ga_harness():
    """Parse the GA's run_harness.py and compare the constants that define the
    evaluation, so GA, DPMA and DiEP numbers mean the same thing."""
    from diep import evaluate
    ga = Path(__file__).resolve().parents[2] / "GA/MoE_compression_GA/examples/run_harness.py"
    if not ga.is_file():
        pytest.skip("GA project not present")
    tree = ast.parse(ga.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id in (
                    "MC_TASKS", "GEN_TASKS", "CODE_EXECUTION_TASKS", "PREFERRED_METRICS"):
                values[target.id] = ast.literal_eval(node.value)
    assert values["MC_TASKS"] == evaluate.MC_TASKS
    assert values["GEN_TASKS"] == evaluate.GEN_TASKS
    assert set(values["CODE_EXECUTION_TASKS"]) == evaluate.CODE_EXECUTION_TASKS
    assert tuple(values["PREFERRED_METRICS"]) == evaluate.PREFERRED_METRICS
