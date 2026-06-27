from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import KMeans

from .config import SystemConfig
from .geometry import Scene
from .parameter_estimation import PairEstimate


@dataclass
class VIBSResult:
    position: np.ndarray
    velocity: np.ndarray
    labels: np.ndarray
    measurements: np.ndarray


def collect_bistatic_measurements(
    cfg: SystemConfig,
    scene: Scene,
    estimates: dict[tuple[int, int], PairEstimate],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rows = []
    pair_ids = []
    for (n, j), est in estimates.items():
        for l in range(len(est.mu)):
            d_hat = cfg.c * cfg.f_ADC * cfg.T * est.mu[l] / cfg.B
            v_hat = cfg.wavelength * est.nu[l] / cfg.frame_step
            rows.append([d_hat, v_hat])
            pair_ids.append((n, j))
    return np.asarray(rows, dtype=float), pair_ids


def _range_residual(p: np.ndarray, d_meas: np.ndarray, tx_points: np.ndarray, rx_points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(p[None, :] - tx_points, axis=1) + np.linalg.norm(p[None, :] - rx_points, axis=1) - d_meas


def estimate_position(d_meas: np.ndarray, tx_points: np.ndarray, rx_points: np.ndarray, seed_pos: np.ndarray | None = None) -> np.ndarray:
    if seed_pos is None:
        # Coarse but robust: start above the array center.
        xy = 0.5 * np.mean(np.vstack([tx_points[:, :2], rx_points[:, :2]]), axis=0)
        z = max(0.5, np.median(d_meas) / 2.0)
        seed_pos = np.array([xy[0], xy[1], z])
    res = least_squares(lambda p: _range_residual(p, d_meas, tx_points, rx_points), seed_pos, bounds=([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, np.inf]), max_nfev=200)
    return res.x


def estimate_velocity(p: np.ndarray, v_meas: np.ndarray, tx_points: np.ndarray, rx_points: np.ndarray) -> np.ndarray:
    u_tx = p[None, :] - tx_points
    u_rx = p[None, :] - rx_points
    A = u_tx / np.linalg.norm(u_tx, axis=1, keepdims=True) + u_rx / np.linalg.norm(u_rx, axis=1, keepdims=True)
    return np.linalg.pinv(A, rcond=1e-8) @ v_meas


def run_vibs(cfg: SystemConfig, scene: Scene, estimates: dict[tuple[int, int], PairEstimate]) -> VIBSResult:
    meas, pair_ids = collect_bistatic_measurements(cfg, scene, estimates)
    km = KMeans(n_clusters=cfg.L, n_init=20, random_state=cfg.seed)
    labels = km.fit_predict(meas)

    pos_hat = np.zeros((cfg.L, 3), dtype=float)
    vel_hat = np.zeros((cfg.L, 3), dtype=float)
    for l in range(cfg.L):
        idx = np.where(labels == l)[0]
        tx_points = []
        rx_points = []
        for row_idx in idx:
            n, j = pair_ids[row_idx]
            tx_points.append(scene.tx_pos[scene.dsa_indices[n]])
            rx_points.append(scene.rx_pos[j])
        tx_points = np.asarray(tx_points)
        rx_points = np.asarray(rx_points)
        pos_hat[l] = estimate_position(meas[idx, 0], tx_points, rx_points)
        vel_hat[l] = estimate_velocity(pos_hat[l], meas[idx, 1], tx_points, rx_points)
    return VIBSResult(position=pos_hat, velocity=vel_hat, labels=labels, measurements=meas)


def match_targets(true_pos: np.ndarray, est_pos: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    cost = np.linalg.norm(true_pos[:, None, :] - est_pos[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    out = np.empty_like(col)
    out[row] = col
    return out
