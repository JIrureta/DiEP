"""RBF-kernel CKA between expert outputs (spec §7.2), in fp32.

For expert ``i`` with outputs ``Y_i [n, d]`` on the same ``n`` tokens::

    K_i[a, b] = exp(-||Y_i[a] - Y_i[b]||^2 / (2 sigma_i^2)),
    sigma_i   = median of the nonzero pairwise distances in Y_i
    HSIC(K, M) = tr(K H M H) / (n - 1)^2,   H = I - 11^T / n
    CKA(i, j)  = HSIC(K_i, K_j) / sqrt(HSIC(K_i, K_i) HSIC(K_j, K_j))

``tr(K H M H) = <H K H, H M H>`` for symmetric kernels (H is idempotent), so the
centred kernels are formed once and every pair is one inner product: the whole
``[E, E]`` matrix is a single matrix product. Scaling ``Y`` by a constant scales
every distance and ``sigma`` alike, so ``K`` -- and CKA -- are unchanged.
"""

from __future__ import annotations

import torch

__all__ = ["rbf_kernel", "centred_kernel", "cka_matrix", "cka"]


def rbf_kernel(y: torch.Tensor) -> torch.Tensor:
    y = y.float()
    sq = (y * y).sum(-1)
    d2 = (sq[:, None] + sq[None, :] - 2.0 * (y @ y.T)).clamp_min_(0.0)
    d2.fill_diagonal_(0.0)
    dist = d2.sqrt()
    off = dist[~torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device)]
    nonzero = off[off > 0]
    sigma = nonzero.median() if nonzero.numel() else torch.tensor(1.0, device=y.device)
    return torch.exp(-d2 / (2.0 * sigma ** 2))


def centred_kernel(y: torch.Tensor) -> torch.Tensor:
    k = rbf_kernel(y)
    return k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()   # H K H


def cka_matrix(outputs: list[torch.Tensor]) -> torch.Tensor:
    """``[E, E]`` CKA between every pair of expert outputs (each ``[n, d]``)."""
    n = outputs[0].shape[0]
    flat = torch.stack([centred_kernel(y).flatten() for y in outputs])   # [E, n^2]
    hsic = (flat @ flat.T) / float((n - 1) ** 2)
    diag = hsic.diagonal().clamp_min(1e-30)
    return (hsic / torch.sqrt(diag[:, None] * diag[None, :])).clamp(-1.0, 1.0)


def cka(y1: torch.Tensor, y2: torch.Tensor) -> float:
    return float(cka_matrix([y1, y2])[0, 1])
