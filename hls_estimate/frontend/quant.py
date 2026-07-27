"""Per-tensor symmetric quantization helpers (SPEC §2)."""
from __future__ import annotations

import numpy as np
import torch


def qmax_for(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def quantize_tensor(t: torch.Tensor, bits: int) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor quantization. Returns (int array, scale).

    scale = max|t| / qmax, so `t ≈ q * scale`. An all-zero tensor gets scale 1.0.
    """
    qmax = qmax_for(bits)
    amax = float(t.abs().max()) if t.numel() else 0.0
    scale = (amax / qmax) if amax > 0 else 1.0
    q = torch.clamp(torch.round(t / scale), -qmax - 1, qmax)
    return q.to(torch.int64).numpy(), scale
