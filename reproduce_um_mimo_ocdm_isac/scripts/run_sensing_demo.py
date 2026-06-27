from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from um_mimo_ocdm_isac.config import load_config
from um_mimo_ocdm_isac.geometry import make_scene
from um_mimo_ocdm_isac.measurements import synthesize_measurements
from um_mimo_ocdm_isac.ocdm import check_dss_lpf_condition
from um_mimo_ocdm_isac.parameter_estimation import estimate_all_pairs
from um_mimo_ocdm_isac.utils import ensure_dir, save_json
from um_mimo_ocdm_isac.vibs import match_targets, run_vibs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact UM-MIMO OCDM ISAC sensing demo.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(args.out_dir)
    scene = make_scene(cfg)

    if not check_dss_lpf_condition(cfg, scene.dss_indices):
        raise RuntimeError(f"DSS indices {scene.dss_indices.tolist()} do not satisfy Eq. (23).")

    meas = synthesize_measurements(cfg, scene, snr_db=args.snr_db)
    estimates = estimate_all_pairs(cfg, scene.dss_indices, meas.R)
    vibs = run_vibs(cfg, scene, estimates)
    assignment = match_targets(scene.target_pos, vibs.position)
    pos_hat = vibs.position[assignment]
    vel_hat = vibs.velocity[assignment]

    pos_mse = float(np.sum(np.linalg.norm(pos_hat - scene.target_pos, axis=1) ** 2))
    vel_mse = float(np.sum(np.linalg.norm(vel_hat - scene.target_vel, axis=1) ** 2))

    summary = {
        "snr_db": args.snr_db,
        "dsa_indices_zero_based": scene.dsa_indices,
        "dss_indices_zero_based": scene.dss_indices,
        "Q": cfg.Q,
        "true_position_m": scene.target_pos,
        "estimated_position_m": pos_hat,
        "true_velocity_mps": scene.target_vel,
        "estimated_velocity_mps": vel_hat,
        "position_mse": pos_mse,
        "velocity_mse": vel_mse,
    }
    save_json(out_dir / "sensing_demo_summary.json", summary)

    plt.figure(figsize=(6, 5))
    plt.scatter(scene.target_pos[:, 0], scene.target_pos[:, 1], marker="o", label="true")
    plt.scatter(pos_hat[:, 0], pos_hat[:, 1], marker="x", label="estimated")
    for l in range(cfg.L):
        plt.plot([scene.target_pos[l, 0], pos_hat[l, 0]], [scene.target_pos[l, 1], pos_hat[l, 1]], linewidth=1)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("VIBS target position demo")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "vibs_xy.png", dpi=180)

    print("Saved:")
    print(f"  {out_dir / 'sensing_demo_summary.json'}")
    print(f"  {out_dir / 'vibs_xy.png'}")
    print(f"Position MSE: {pos_mse:.4e}")
    print(f"Velocity MSE: {vel_mse:.4e}")


if __name__ == "__main__":
    main()
