"""What produced a result: versions, commit, command line, time."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "lm_eval", "numpy", "safetensors"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["cuda"] = torch.version.cuda
    except Exception:
        pass
    return out


def git_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                         "DiEP"], text=True, stderr=subprocess.DEVNULL).strip()
        return commit + (" (DiEP uncommitted)" if dirty else "")
    except Exception:
        return None


def card(**extra) -> dict:
    return {"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv), "git_commit": git_commit(),
            "versions": versions(), **extra}
