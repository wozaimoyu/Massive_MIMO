from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import SystemConfig


@dataclass
class Scene:
    tx_pos: np.ndarray        # (N_tx, 3)
    rx_pos: np.ndarray        # (N_rx, 3)
    target_pos: np.ndarray    # (L, 3)
    target_vel: np.ndarray    # (L, 3), m/s
    dsa_indices: np.ndarray   # (N_dss,), zero-based Tx indices
    dss_indices: np.ndarray   # (N_dss,), zero-based subcarrier indices


def make_tx_positions(cfg: SystemConfig) -> np.ndarray:
    x = np.arange(cfg.N_tx, dtype=float) * cfg.wavelength / 2.0
    return np.column_stack([x, np.zeros(cfg.N_tx), np.zeros(cfg.N_tx)])


def make_rx_positions(cfg: SystemConfig) -> np.ndarray:
    d = cfg.D_rx / 2.0
    return np.array([[d, d, 0.0], [d, -d, 0.0], [-d, d, 0.0], [-d, -d, 0.0]], dtype=float)[: cfg.N_rx]


def sample_targets(cfg: SystemConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    r = rng.uniform(cfg.range_min, cfg.range_max, cfg.L)
    theta = rng.uniform(cfg.elevation_min, cfg.elevation_max, cfg.L)
    phi = rng.uniform(cfg.azimuth_min, cfg.azimuth_max, cfg.L)

    # Spherical convention used here: theta is measured from the positive z-axis.
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pos = np.column_stack([x, y, z])

    v_min = cfg.velocity_min_kmh / 3.6
    v_max = cfg.velocity_max_kmh / 3.6
    vel = rng.uniform(v_min, v_max, size=(cfg.L, 3))
    return pos, vel


def select_dsa_indices(cfg: SystemConfig) -> np.ndarray:
    return np.linspace(0, cfg.N_tx - 1, cfg.N_dss, dtype=int)


def select_dss_indices(cfg: SystemConfig) -> np.ndarray:
    # Satisfies the spacing rule in Eq. (23) for the default setup.
    guard = int(np.ceil(cfg.B * cfg.T_GI + cfg.f_LPF * cfg.T))
    if cfg.N_dss == 1:
        return np.array([cfg.K // 2], dtype=int)
    candidates = np.linspace(guard, cfg.K - guard - 1, cfg.N_dss, dtype=int)
    return np.unique(candidates)[: cfg.N_dss]


def make_scene(cfg: SystemConfig) -> Scene:
    rng = np.random.default_rng(cfg.seed)
    tx = make_tx_positions(cfg)
    rx = make_rx_positions(cfg)
    pos, vel = sample_targets(cfg, rng)
    return Scene(tx, rx, pos, vel, select_dsa_indices(cfg), select_dss_indices(cfg))


def bistatic_distance_velocity(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    tx_point: np.ndarray,
    rx_point: np.ndarray,
) -> Tuple[float, float]:
    """Equations (13) and (14): bistatic distance and its time derivative."""
    u_tx = target_pos - tx_point
    u_rx = target_pos - rx_point
    d_tx = np.linalg.norm(u_tx)
    d_rx = np.linalg.norm(u_rx)
    distance = d_tx + d_rx
    direction_sum = u_tx / d_tx + u_rx / d_rx
    velocity = float(direction_sum @ target_vel)
    return float(distance), velocity


def all_pair_parameters(cfg: SystemConfig, scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    """Return true normalized mu and nu for every DSA/Rx/target."""
    mu = np.zeros((cfg.N_dss, cfg.N_rx, cfg.L), dtype=float)
    nu = np.zeros_like(mu)
    for n, tx_idx in enumerate(scene.dsa_indices):
        for j in range(cfg.N_rx):
            for l in range(cfg.L):
                d, v = bistatic_distance_velocity(scene.target_pos[l], scene.target_vel[l], scene.tx_pos[tx_idx], scene.rx_pos[j])
                tau = d / cfg.c
                mu[n, j, l] = cfg.B * tau / (cfg.f_ADC * cfg.T)
                nu[n, j, l] = v * cfg.frame_step / cfg.wavelength
    return mu, nu
