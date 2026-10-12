"""DiEP -- Differentiable Expert Pruning (Bai et al., NeurIPS 2025), per ../specs.md.

Three stages, one module each:

``search``  learn intra-layer expert logits ``alpha [L, N]`` and inter-layer
            scales ``beta [L]`` on frozen weights, alternating 3:1   (spec §4)
``prune``   score ``softmax(alpha) * beta``, rank globally, remove the bottom
            ``K = round(N L r)`` subject to a per-layer floor          (spec §5)
``skip``    optional inference-time skipping of low-weight experts,
            thresholds from routing statistics and RBF-CKA             (spec §7)

plus the optional merge of pruned experts into retained ones (spec §8, off by
default). All model-specific code is behind ``diep.adapters``.

Deviations forced by this study, all in the README: OLMoE instead of the
paper's Mixtral/DeepSeek/Qwen, the GA's fixed calibration set (32 x 1024 C4
tokens, not 128 x 2048), and the GA/DPMA evaluation protocol so the three
methods are scored identically.
"""

__version__ = "0.1.0"
