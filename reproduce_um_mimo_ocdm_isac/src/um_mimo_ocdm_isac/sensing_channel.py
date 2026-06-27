from __future__ import annotations

import numpy as np

from .config import SystemConfig


def complex_gains(cfg: SystemConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate alpha_{n,j,l} for correlated, SUC, and SNS channels."""
    if cfg.channel_model == "correlated":
        base = (rng.normal(size=cfg.L) + 1j * rng.normal(size=cfg.L)) / np.sqrt(2.0)
        alpha = np.tile(base[None, None, :], (cfg.N_dss, cfg.N_rx, 1))
    else:
        alpha = (rng.normal(size=(cfg.N_dss, cfg.N_rx, cfg.L)) + 1j * rng.normal(size=(cfg.N_dss, cfg.N_rx, cfg.L))) / np.sqrt(2.0)

    if cfg.channel_model == "sns" and cfg.sns_null_count > 0:
        flat = alpha.reshape(-1)
        count = min(cfg.sns_null_count, flat.size)
        idx = rng.choice(flat.size, size=count, replace=False)
        flat[idx] = 0.0
        alpha = flat.reshape(alpha.shape)
    return alpha
