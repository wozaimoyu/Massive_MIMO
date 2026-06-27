from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .config import SystemConfig
from .ocdm import b_response, velocity_response


@dataclass
class PairEstimate:
    mu: np.ndarray
    nu: np.ndarray
    alpha: np.ndarray
    residual_norm: float


def _atom(cfg: SystemConfig, k_n: int, mu: float, nu: float) -> np.ndarray:
    b = b_response(cfg, k_n, mu)[:, None]
    a = velocity_response(cfg, nu)[None, :]
    return (b * np.conj(a)).reshape(-1, order="F")


def _least_squares_alpha(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(A, rcond=1e-8) @ y


def _omp_grid_init(
    cfg: SystemConfig,
    k_n: int,
    y: np.ndarray,
    n_paths: int,
    mu_bounds: tuple[float, float],
    nu_bounds: tuple[float, float],
    grid_mu: int = 80,
    grid_nu: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu_grid = np.linspace(mu_bounds[0], mu_bounds[1], grid_mu)
    nu_grid = np.linspace(nu_bounds[0], nu_bounds[1], grid_nu)
    atoms = []
    params = []
    for mu in mu_grid:
        b = b_response(cfg, k_n, mu)[:, None]
        for nu in nu_grid:
            a = velocity_response(cfg, nu)[None, :]
            atom = (b * np.conj(a)).reshape(-1, order="F")
            norm = np.linalg.norm(atom) + 1e-12
            atoms.append(atom / norm)
            params.append((mu, nu))
    D = np.column_stack(atoms)
    residual = y.copy()
    chosen: list[int] = []
    for _ in range(n_paths):
        corr = np.abs(D.conj().T @ residual)
        idx = int(np.argmax(corr))
        if idx in chosen:
            break
        chosen.append(idx)
        A = np.column_stack([_atom(cfg, k_n, *params[i]) for i in chosen])
        alpha = _least_squares_alpha(A, y)
        residual = y - A @ alpha
    if not chosen:
        chosen = [0]
    mu0 = np.array([params[i][0] for i in chosen])
    nu0 = np.array([params[i][1] for i in chosen])
    A = np.column_stack([_atom(cfg, k_n, mu0[i], nu0[i]) for i in range(len(mu0))])
    alpha0 = _least_squares_alpha(A, y)
    return mu0, nu0, alpha0


def estimate_pair(
    cfg: SystemConfig,
    k_n: int,
    R_nj: np.ndarray,
    n_paths: int,
    mu_bounds: tuple[float, float] = (0.02, 0.35),
    nu_bounds: tuple[float, float] = (-0.15, 0.15),
    max_nfev: int = 80,
) -> PairEstimate:
    """Practical baseline for Algorithm 1.

    The paper uses RL-ESPRIT followed by GDA. This reproducer exposes the same
    inputs/outputs but uses OMP on a 2-D grid followed by nonlinear least squares.
    """
    y = R_nj.reshape(-1, order="F")
    mu0, nu0, alpha0 = _omp_grid_init(cfg, k_n, y, n_paths, mu_bounds, nu_bounds)
    p0 = np.r_[mu0, nu0]

    def residual_real(p: np.ndarray) -> np.ndarray:
        mu = np.clip(p[:n_paths], mu_bounds[0], mu_bounds[1])
        nu = np.clip(p[n_paths:], nu_bounds[0], nu_bounds[1])
        A = np.column_stack([_atom(cfg, k_n, mu[i], nu[i]) for i in range(n_paths)])
        alpha = _least_squares_alpha(A, y)
        r = A @ alpha - y
        return np.r_[r.real, r.imag]

    if len(p0) != 2 * n_paths:
        # Pad rare duplicate-grid cases.
        pad = n_paths - len(mu0)
        mu0 = np.r_[mu0, np.linspace(mu_bounds[0], mu_bounds[1], pad)]
        nu0 = np.r_[nu0, np.linspace(nu_bounds[0], nu_bounds[1], pad)]
        p0 = np.r_[mu0, nu0]

    res = least_squares(residual_real, p0, max_nfev=max_nfev, xtol=1e-8, ftol=1e-8, gtol=1e-8)
    mu = np.clip(res.x[:n_paths], mu_bounds[0], mu_bounds[1])
    nu = np.clip(res.x[n_paths:], nu_bounds[0], nu_bounds[1])
    A = np.column_stack([_atom(cfg, k_n, mu[i], nu[i]) for i in range(n_paths)])
    alpha = _least_squares_alpha(A, y)
    return PairEstimate(mu=mu, nu=nu, alpha=alpha, residual_norm=float(np.linalg.norm(A @ alpha - y)))


def estimate_all_pairs(cfg: SystemConfig, dss_indices: np.ndarray, R: np.ndarray) -> dict[tuple[int, int], PairEstimate]:
    out: dict[tuple[int, int], PairEstimate] = {}
    for n, k_n in enumerate(dss_indices):
        for j in range(cfg.N_rx):
            out[(n, j)] = estimate_pair(cfg, int(k_n), R[n, j], cfg.L)
    return out
