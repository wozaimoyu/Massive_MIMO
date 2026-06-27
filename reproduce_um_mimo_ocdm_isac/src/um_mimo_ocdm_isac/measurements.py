from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SystemConfig
from .geometry import Scene, all_pair_parameters
from .ocdm import b_response, velocity_response
from .sensing_channel import complex_gains


@dataclass
class MeasurementSet:
    R: np.ndarray       # (N_dss, N_rx, Q, M)
    mu_true: np.ndarray # (N_dss, N_rx, L)
    nu_true: np.ndarray # (N_dss, N_rx, L)
    alpha: np.ndarray   # (N_dss, N_rx, L)


def synthesize_measurements(cfg: SystemConfig, scene: Scene, snr_db: float) -> MeasurementSet:
    rng = np.random.default_rng(cfg.seed + 101)
    mu_true, nu_true = all_pair_parameters(cfg, scene)
    alpha = complex_gains(cfg, rng)

    R = np.zeros((cfg.N_dss, cfg.N_rx, cfg.Q, cfg.M), dtype=np.complex128)
    scale = cfg.P_tx_watt / cfg.N_dss
    for n, k_n in enumerate(scene.dss_indices):
        for j in range(cfg.N_rx):
            mat = np.zeros((cfg.Q, cfg.M), dtype=np.complex128)
            for l in range(cfg.L):
                b = b_response(cfg, int(k_n), mu_true[n, j, l])[:, None]
                a = velocity_response(cfg, nu_true[n, j, l])[None, :]
                mat += scale * alpha[n, j, l] * b * np.conj(a)
            sig_power = float(np.mean(np.abs(mat) ** 2))
            noise_power = sig_power / (10.0 ** (snr_db / 10.0)) if sig_power > 0 else 1e-12
            noise = np.sqrt(noise_power / 2.0) * (rng.normal(size=mat.shape) + 1j * rng.normal(size=mat.shape))
            R[n, j] = mat + noise
    return MeasurementSet(R=R, mu_true=mu_true, nu_true=nu_true, alpha=alpha)
