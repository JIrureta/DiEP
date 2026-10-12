"""Calibration data: the GA's file, byte for byte (spec §4.4, adapted).

The paper calibrates on 128 C4 sequences of 2048 tokens. This study compares
DiEP with the genetic algorithm and DPMA, which were both calibrated on
``c4-32x1024-seed0.pt``: 32 random 1024-token windows of C4 (``allenai/c4``,
en, seed 0). The comparison is only fair on identical tokens, so DiEP reads
that same file and nothing else -- :func:`load_calibration` refuses any other
file name unless explicitly overridden.

Consequences for the schedule (README): with 32 sequences and an effective
batch of 16, an epoch is 2 optimiser steps, so the spec's 10 epochs are 20
steps -- 15 alpha-steps and 5 beta-steps at 3:1 -- instead of 80.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

__all__ = ["Calibration", "load_calibration", "REQUIRED_CALIBRATION"]

#: The one calibration file every method in this study uses.
REQUIRED_CALIBRATION = "c4-32x1024-seed0.pt"


@dataclass
class Calibration:
    input_ids: torch.Tensor           # [n, T] int64
    attention_mask: torch.Tensor      # [n, T] int64
    path: str
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    def epoch_order(self, epoch: int, seed: int) -> torch.Tensor:
        """A fresh permutation per epoch ("reshuffle every epoch"), reproducible
        from ``(seed, epoch)`` alone."""
        generator = torch.Generator().manual_seed(seed * 100_003 + epoch)
        return torch.randperm(len(self), generator=generator)

    def batches(self, epoch: int, seed: int, batch_size: int):
        """The optimiser-step batches of one epoch, as index tensors."""
        order = self.epoch_order(epoch, seed)
        return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    def describe(self) -> str:
        return (f"{Path(self.path).name}: {len(self)} x {self.seq_len} tokens "
                f"({int(self.attention_mask.sum())} real), tokenizer "
                f"{self.metadata.get('tokenizer', '?')}")


def load_calibration(path: str | Path, *, expect_tokenizer: str | None = None,
                     allow_other_file: bool = False) -> Calibration:
    """Load a calibration set written by the GA's ``prepare_calibration.py``."""
    path = Path(path)
    if path.name != REQUIRED_CALIBRATION and not allow_other_file:
        raise ValueError(
            f"refusing {path.name}: every method in this study is calibrated on "
            f"{REQUIRED_CALIBRATION}, and a comparison on different tokens is not a "
            "comparison. Pass allow_other_file=True (--allow-other-calibration) only "
            "for tests or a deliberately separate study.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    recorded = metadata.get("tokenizer")
    if expect_tokenizer and recorded and recorded != expect_tokenizer:
        raise ValueError(f"{path} was tokenised with {recorded!r}, not {expect_tokenizer!r}")
    return Calibration(payload["input_ids"].long(), payload["attention_mask"].long(),
                       str(path.resolve()), metadata)
