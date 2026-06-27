from __future__ import annotations

import numpy as np

from .config import SystemConfig


def check_dss_lpf_condition(cfg: SystemConfig, dss_indices: np.ndarray) -> bool:
    """Check sufficient DSS/LPF spacing rule in Eq. (23)."""
    min_gap = cfg.B * cfg.T_GI + cfg.f_LPF * cfg.T
    gaps = []
    for a in dss_indices:
        for b in dss_indices:
            if a == b:
                continue
            diff = abs(int(a) - int(b))
            gaps.append(diff >= min_gap and diff <= cfg.K - min_gap)
    return bool(np.all(gaps)) if gaps else True


def omega_sets(cfg: SystemConfig, k_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.arange(1, cfg.Q + 1)
    t_rel = cfg.T_GI + (q - 1) / cfg.f_ADC
    omega_i = np.where((cfg.T_GI <= t_rel) & (t_rel < k_n / cfg.B))[0]
    omega_iii = np.where(((k_n / cfg.B + cfg.T_GI) <= t_rel) & (t_rel < cfg.T))[0]
    omega_ini = omega_i if len(omega_i) >= len(omega_iii) else omega_iii
    return omega_i, omega_iii, omega_ini


def b_response(cfg: SystemConfig, k_n: int, mu: float) -> np.ndarray:
    """Digitized IF response b_n(mu) in Eq. (27).

    This implementation computes the dual-discontinuity windows from the
    equivalent delay tau = mu * f_ADC * T / B.
    """
    q = np.arange(1, cfg.Q + 1)
    t_rel = cfg.T_GI + (q - 1) / cfg.f_ADC
    tau = mu * cfg.f_ADC * cfg.T / cfg.B
    t_i_end = max(tau, min(k_n / cfg.B, k_n / cfg.B + tau))
    t_ii_start = t_i_end
    t_ii_end = min(cfg.T, max(k_n / cfg.B, k_n / cfg.B + tau))

    out = np.exp(1j * 2.0 * np.pi * mu * (q - 1))
    mask_ii = (t_rel >= t_ii_start) & (t_rel < t_ii_end)
    mask_iii = t_rel >= t_ii_end
    out[mask_ii] = 0.0
    phi = 2.0 * np.pi * cfg.f_ADC * cfg.T
    out[mask_iii] *= np.exp(-1j * mu * phi)
    return out


def velocity_response(cfg: SystemConfig, nu: float) -> np.ndarray:
    m = np.arange(cfg.M)
    return np.exp(1j * 2.0 * np.pi * nu * m)
