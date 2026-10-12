"""Stage 1: the differentiable search (spec §4, Algorithm 1).

Model weights are frozen and the model stays in ``eval()``. Two parameter sets
are learned on the calibration set, alternately:

``alpha [L, N]``  intra-layer expert logits, fp32, init 0 (so ``softmax`` is 1/N)
``beta  [L]``     inter-layer scales, fp32, init 1

Objective (Eq. 6-7), per micro-batch::

    L = CE(next token) + lambda * Phi,   Phi = sqrt(sum (H' - H)^2 + 1e-8)

``H`` is the original model's final hidden state -- the input to ``lm_head``,
after the final norm -- precomputed once for every calibration sequence and
cached on CPU in bf16. ``Phi`` is the *non-squared* Frobenius norm in fp32; the
``+1e-8`` is what keeps the gradient finite at step 0, where ``Phi = 0``.

Alternation (3:1): at global step ``c``, ``c mod 4 < 3`` updates alpha only,
otherwise beta only, each with its own Adam and its own cosine schedule to 0
over its own number of steps. ``beta`` is clamped to ``>= 1e-3`` after each
beta-step. Nothing here depends on the sparsity ``r``: the scores serve every
ratio, and ``r`` only enters at pruning.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .moe_forward import DiEPState, install, route

__all__ = ["SearchConfig", "search", "routing_frequency", "step_targets"]


@dataclass
class SearchConfig:
    """Spec §10 defaults. Only ``epochs`` x ``len(calibration)`` differs in effect."""

    search_forward: str = "routed"        #: "routed" (default) or "dense" (paper Eq. 5)
    epochs: int = 10
    batch_size: int = 16                  #: sequences per optimiser step (effective)
    micro_batch: int = 1                  #: sequences per forward; Phi is per micro-batch
    lr_alpha: float = 5e-3
    lr_beta: float = 5e-3
    update_ratio: str = "3:1"
    lam: float = 0.01
    recon_target: str = "hidden"          #: "hidden" (pre-lm_head) or "logits"
    score_mode: str = "softmax"           #: "softmax" (default) or "raw" (Eq. 10)
    alpha_init: float | None = None       #: default 0, or 1 when score_mode == "raw"
    beta_init: float = 1.0
    beta_min: float = 1e-3
    seed: int = 0
    ablation: str = "none"                #: "none" | "alpha_only" | "beta_only"
    fixed_beta: str | None = None         #: None | "a:b" | "random"

    def __post_init__(self):
        if self.search_forward not in ("routed", "dense"):
            raise ValueError("search_forward must be 'routed' or 'dense'")
        if self.recon_target not in ("hidden", "logits"):
            raise ValueError("recon_target must be 'hidden' or 'logits'")
        if self.score_mode not in ("softmax", "raw"):
            raise ValueError("score_mode must be 'softmax' or 'raw'")
        if self.ablation not in ("none", "alpha_only", "beta_only"):
            raise ValueError("ablation must be none, alpha_only or beta_only")
        if self.batch_size % self.micro_batch:
            raise ValueError("batch_size must be a multiple of micro_batch")
        if self.alpha_init is None:
            self.alpha_init = 1.0 if self.score_mode == "raw" else 0.0

    @property
    def ratio(self) -> tuple[int, int]:
        p, q = (int(v) for v in self.update_ratio.split(":"))
        return p, q


def step_targets(cfg: SearchConfig, total_steps: int) -> list[str]:
    """Which parameter set each global step updates.

    Default: 3 alpha-steps then 1 beta-step, repeating. [DECISION] When beta is
    frozen (``alpha_only``, ``fixed_beta``) every step updates alpha; when alpha
    is frozen (``beta_only``) every step updates beta -- the alternation has
    nothing to alternate with.
    """
    if cfg.ablation == "alpha_only" or cfg.fixed_beta:
        return ["alpha"] * total_steps
    if cfg.ablation == "beta_only":
        return ["beta"] * total_steps
    p, q = cfg.ratio
    return ["alpha" if c % (p + q) < p else "beta" for c in range(total_steps)]


def _initial_beta(cfg: SearchConfig, layers: int, device) -> torch.Tensor:
    """``beta_init``, or the frozen values of the ``fixed_beta`` ablations (spec §9)."""
    if cfg.fixed_beta is None:
        return torch.full((layers,), float(cfg.beta_init), device=device)
    if cfg.fixed_beta == "random":
        generator = torch.Generator().manual_seed(cfg.seed)
        return (0.5 + 1.5 * torch.rand(layers, generator=generator)).to(device)
    a, b = (float(v) for v in cfg.fixed_beta.split(":"))
    half = layers // 2
    return torch.tensor([a] * half + [b] * (layers - half), device=device)


class _HeadInput:
    """Captures the input to the output head: the final hidden state ``H``."""

    def __init__(self, model):
        self.value = None
        self._handle = model.get_output_embeddings().register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        self.value = args[0]

    def close(self):
        self._handle.remove()


@torch.no_grad()
def _cache_targets(model, dmodel, calibration, cfg, device, head):
    """The original model's ``H`` (or logits) for every sequence, bf16 on CPU.

    Computed through the DiEP forward at the identity (``alpha = 0``,
    ``beta = 1``), which *is* the original model -- but through the same code
    path the search runs, so ``Phi = 0`` at initialisation exactly, as spec §4.3
    states. The model's native forward would give the same function with
    different kernels (transformers 5.x defaults OLMoE's experts to
    ``grouped_mm``), leaving rounding noise in ``Phi`` from step 0.
    """
    state = dmodel.state
    saved = (state.mode, state.alpha, state.beta)
    state.mode = "search"
    state.alpha = torch.zeros_like(saved[1])
    state.beta = torch.ones_like(saved[2])
    cache = []
    for i in range(len(calibration)):
        ids = calibration.input_ids[i:i + 1].to(device)
        mask = calibration.attention_mask[i:i + 1].to(device)
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        target = head.value if cfg.recon_target == "hidden" else out.logits
        cache.append(target.to(torch.bfloat16).cpu())
    state.mode, state.alpha, state.beta = saved
    return torch.cat(cache)


@torch.no_grad()
def routing_frequency(model, adapter, calibration, device) -> torch.Tensor:
    """``f [L, N]``: fraction of calibration tokens routed to each expert by the
    original model (the ``beta_only`` ablation's score, spec §9)."""
    blocks = adapter.moe_blocks(model)
    counts = torch.zeros(len(blocks), adapter.num_experts(blocks[0]), dtype=torch.float64)
    tokens = [0]
    handles = []
    for layer, block in enumerate(blocks):
        def hook(module, args, _layer=layer):
            x = args[0].reshape(-1, args[0].shape[-1])
            _, index, _ = route(adapter, module, x)
            counts[_layer] += torch.bincount(index.flatten().cpu(),
                                             minlength=counts.shape[1]).double()
            if _layer == 0:
                tokens[0] += x.shape[0]
        handles.append(block.register_forward_pre_hook(hook))
    try:
        for i in range(len(calibration)):
            model(input_ids=calibration.input_ids[i:i + 1].to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return (counts / max(tokens[0], 1)).float()


def _entropy(alpha: torch.Tensor) -> list[float]:
    p = torch.softmax(alpha.detach().float(), dim=-1)
    return (-(p * p.clamp_min(1e-30).log()).sum(-1)).tolist()


def search(model, adapter, calibration, cfg: SearchConfig, *, device, log_every: int = 1,
           console=print) -> dict:
    """Run Algorithm 1. Returns alpha, beta, logs and (for beta_only) frequencies."""
    torch.manual_seed(cfg.seed)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    blocks = adapter.moe_blocks(model)
    layers, experts = len(blocks), adapter.num_experts(blocks[0])
    alpha = torch.full((layers, experts), float(cfg.alpha_init), device=device, requires_grad=True)
    beta = _initial_beta(cfg, layers, device).requires_grad_(True)

    state = DiEPState(mode="original", alpha=alpha, beta=beta)
    dmodel = install(model, adapter, state)
    dmodel.enable_checkpointing()
    head = _HeadInput(model)
    frequency = None
    try:
        started = time.time()
        console(f"      caching the original model's {cfg.recon_target} for "
                f"{len(calibration)} sequences ...")
        cached = _cache_targets(model, dmodel, calibration, cfg, device, head)
        if cfg.ablation == "beta_only":
            frequency = routing_frequency(model, adapter, calibration, device)

        steps_per_epoch = math.ceil(len(calibration) / cfg.batch_size)
        total = cfg.epochs * steps_per_epoch
        targets = step_targets(cfg, total)
        counts = {"alpha": targets.count("alpha"), "beta": targets.count("beta")}
        optimisers, schedulers = {}, {}
        for name, tensor, lr in (("alpha", alpha, cfg.lr_alpha), ("beta", beta, cfg.lr_beta)):
            if counts[name]:
                optimisers[name] = torch.optim.Adam([tensor], lr=lr, betas=(0.9, 0.999),
                                                    eps=1e-8, weight_decay=0.0)
                schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimisers[name], T_max=counts[name], eta_min=0.0)
        console(f"      {cfg.epochs} epochs x {steps_per_epoch} steps = {total} steps "
                f"({counts['alpha']} alpha, {counts['beta']} beta), batch {cfg.batch_size} "
                f"in micro-batches of {cfg.micro_batch}, search_forward={cfg.search_forward}")

        state.mode = "search" if cfg.search_forward == "routed" else "dense"
        logs = []
        c = 0
        for epoch in range(cfg.epochs):
            for batch in calibration.batches(epoch, cfg.seed, cfg.batch_size):
                target = targets[c]
                alpha.grad = None
                beta.grad = None
                micro = [batch[i:i + cfg.micro_batch] for i in range(0, len(batch), cfg.micro_batch)]
                ce_sum = phi_sum = 0.0
                for chunk in micro:
                    ids = calibration.input_ids[chunk].to(device)
                    mask = calibration.attention_mask[chunk].to(device)
                    out = model(input_ids=ids, attention_mask=mask, use_cache=False)
                    logits = out.logits
                    # Causal-LM next-token loss, averaged over (real) tokens, fp32.
                    shift_mask = mask[:, 1:].reshape(-1).bool()
                    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float()[shift_mask],
                                         ids[:, 1:].reshape(-1)[shift_mask])
                    current = head.value if cfg.recon_target == "hidden" else logits
                    diff = current.float() - cached[chunk].to(device).float()
                    phi = torch.sqrt((diff ** 2).sum() + 1e-8)
                    loss = ce + cfg.lam * phi
                    (loss / len(micro)).backward()
                    ce_sum += ce.item()
                    phi_sum += phi.item()
                grad_norm = {"alpha": alpha.grad.norm().item() if alpha.grad is not None else 0.0,
                             "beta": beta.grad.norm().item() if beta.grad is not None else 0.0}
                if not all(math.isfinite(v) for v in grad_norm.values()):
                    raise RuntimeError(f"step {c}: non-finite gradient {grad_norm}")
                lr = optimisers[target].param_groups[0]["lr"]
                optimisers[target].step()
                schedulers[target].step()
                if target == "beta":
                    with torch.no_grad():
                        beta.clamp_(min=cfg.beta_min)
                record = {"step": c, "epoch": epoch, "target": target, "lr": lr,
                          "ce": ce_sum / len(micro), "phi": phi_sum / len(micro),
                          "grad_norm": grad_norm, "beta": beta.detach().tolist(),
                          "alpha_entropy": _entropy(alpha),
                          "alpha_range": [alpha.min().item(), alpha.max().item()],
                          "elapsed": time.time() - started}
                logs.append(record)
                if c % log_every == 0 or c == total - 1:
                    console(f"      step {c:3d} [{target:5s}] ce {record['ce']:.4f} "
                            f"phi {record['phi']:9.4f} lr {lr:.2e} | beta "
                            f"{min(record['beta']):.4f}..{max(record['beta']):.4f} | "
                            f"H(alpha) {min(record['alpha_entropy']):.4f}..."
                            f"{max(record['alpha_entropy']):.4f}")
                c += 1
    finally:
        head.close()
        dmodel.restore()

    return {"alpha": alpha.detach().cpu(), "beta": beta.detach().cpu(), "logs": logs,
            "frequency": frequency, "config": asdict(cfg),
            "steps": {"total": total, **counts, "steps_per_epoch": steps_per_epoch}}
