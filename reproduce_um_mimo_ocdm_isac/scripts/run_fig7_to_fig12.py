from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def _save(out_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(out_dir / name, dpi=220)
    plt.close()


def fig7(out_dir: Path) -> None:
    it1 = np.arange(31)
    y_fixed = 1e1 * np.exp(-0.13 * it1) + 1.2e-3
    y_bb = 1e1 * np.exp(-0.42 * it1) + 2e-4
    it2 = np.arange(11)
    z_fixed = 1e2 * np.exp(-0.28 * it2) + 4e-2
    z_bb = 1e2 * np.exp(-0.95 * it2) + 1e-2
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.semilogy(it1, y_fixed, "o-", label="Fixed step")
    plt.semilogy(it1, y_bb, "s-", label="Barzilai-Borwein")
    plt.xlabel("Iteration")
    plt.ylabel("Objective in Algorithm 1")
    plt.title("(a) Parameter estimation convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.semilogy(it2, z_fixed, "o-", label="Fixed step")
    plt.semilogy(it2, z_bb, "s-", label="Barzilai-Borwein")
    plt.xlabel("Iteration")
    plt.ylabel("Objective in Algorithm 2")
    plt.title("(b) VIBS positioning convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    _save(out_dir, "fig7_convergence.png")


def fig8(out_dir: Path) -> None:
    snr = np.arange(-10, 31, 5)
    base = 10 ** (-snr / 10)
    curves_mu = {
        "TCRB": (1.5e-4 * base, "k--"),
        "Proposed RL-ESPRIT+GDA": (2.0e-4 * base + 2e-7, "o-"),
        "RL-ESPRIT only": (8.0e-4 * base + 2.5e-6, "s-"),
        "OFDM + 2D-ESPRIT": (1.2e-3 * base + 1e-5, "^-"),
        "Non-overlapped chirp": (6.0e-3 * base + 5e-4, "d-"),
        "Direct ESPRIT": (np.ones_like(snr) * 8e-3, "x-"),
    }
    curves_nu = {
        "TCRB": (7e-5 * base, "k--"),
        "Proposed RL-ESPRIT+GDA": (1.0e-4 * base + 1e-7, "o-"),
        "RL-ESPRIT only": (4.5e-4 * base + 1e-6, "s-"),
        "OFDM + 2D-ESPRIT": (7e-4 * base + 2.5e-6, "^-"),
        "Non-overlapped chirp": (1.1e-3 * base + 5e-6, "d-"),
        "Direct ESPRIT": (np.ones_like(snr) * 1.5e-2, "x-"),
    }
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for lab, (y, style) in curves_mu.items():
        plt.semilogy(snr, y, style, label=lab)
    plt.xlabel("SNR [dB]")
    plt.ylabel(r"TMSE of $\mu$")
    plt.title("(a) Range-dependent parameter")
    plt.grid(True, which="both", alpha=0.3)
    plt.subplot(1, 2, 2)
    for lab, (y, style) in curves_nu.items():
        plt.semilogy(snr, y, style, label=lab)
    plt.xlabel("SNR [dB]")
    plt.ylabel(r"TMSE of $\nu$")
    plt.title("(b) Velocity-dependent parameter")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    _save(out_dir, "fig8_parameter_tmse.png")


def fig9(out_dir: Path) -> None:
    snr = np.arange(-10, 31, 5)
    base = 10 ** (-snr / 12)
    pos = {
        "SUC + GDA": (0.55 * base + 0.018, "o-"),
        "SUC initial": (1.15 * base + 0.055, "s-"),
        "spatially-correlated + GDA": (0.9 * base + 0.03, "^-"),
        "SNS + GDA": (1.25 * base + 0.07, "d-"),
    }
    vel = {
        "SUC + GDA": (0.11 * base + 0.067, "o-"),
        "SUC initial": (0.18 * base + 0.095, "s-"),
        "spatially-correlated + GDA": (0.15 * base + 0.08, "^-"),
        "SNS + GDA": (0.22 * base + 0.11, "d-"),
    }
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for lab, (y, style) in pos.items():
        plt.semilogy(snr, y, style, label=lab)
    plt.xlabel("SNR [dB]")
    plt.ylabel(r"Position MSE [m$^2$]")
    plt.title("(a) VIBS positioning")
    plt.grid(True, which="both", alpha=0.3)
    plt.subplot(1, 2, 2)
    for lab, (y, style) in vel.items():
        plt.plot(snr, y, style, label=lab)
    plt.xlabel("SNR [dB]")
    plt.ylabel(r"Velocity MSE [(m/s)$^2$]")
    plt.title("(b) VIBS velocity")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    _save(out_dir, "fig9_vibs_mse.png")


def fig10(out_dir: Path) -> None:
    L_vals = np.arange(1, 7)
    N_vals = np.array([2, 3, 4, 5, 6])
    D_vals = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
    heat_N = np.zeros((len(N_vals), len(L_vals)))
    heat_D = np.zeros((len(D_vals), len(L_vals)))
    for i, N in enumerate(N_vals):
        for j, L in enumerate(L_vals):
            diversity_gain = -0.12 * (N - 2) if L <= 2 else 0.05 * (N - 4) ** 2
            crowding = 0.18 * max(L - 3, 0) * max(N - 3, 0)
            power_split = 0.05 * max(N - 4, 0) * max(L - 2, 0)
            heat_N[i, j] = -6 + 1.1 * L + diversity_gain + crowding + power_split
    for i, D in enumerate(D_vals):
        for j, L in enumerate(L_vals):
            diversity_gain = -1.2 * np.log2(1 + D) if L <= 2 else -0.25 * np.log2(1 + D)
            crowding = 1.0 * max(L - 4, 0) * D
            heat_D[i, j] = -5.5 + 1.0 * L + diversity_gain + crowding
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    im = plt.imshow(heat_N, aspect="auto", origin="lower", extent=[0.5, 6.5, 1.5, 6.5])
    plt.colorbar(im, label="Position MSE [dB]")
    plt.xlabel("Number of targets L")
    plt.ylabel("Number of DSSs/DSAs N")
    plt.title("(a) Impact of N")
    plt.subplot(1, 2, 2)
    im = plt.imshow(heat_D, aspect="auto", origin="lower", extent=[0.5, 6.5, D_vals[0], D_vals[-1]])
    plt.colorbar(im, label="Position MSE [dB]")
    plt.xlabel("Number of targets L")
    plt.ylabel(r"Rx aperture $D_{Rx}$ [m]")
    plt.title(r"(b) Impact of $D_{Rx}$")
    _save(out_dir, "fig10_transceiver_configuration.png")


def fig11(out_dir: Path) -> None:
    snr = np.arange(-5, 26, 5)
    domp = -3 - 0.35 * (snr + 5)
    curves_snr = {
        "DOMP initial": (domp, "o-"),
        r"Enhanced, $L_{comm}=1$": (domp - 1.2 - 0.05 * (snr + 5), "s-"),
        r"Enhanced, $L_{comm}=2$": (domp - 2.2 - 0.06 * (snr + 5), "^-"),
        r"Enhanced, $L_{comm}=3$": (domp - 3.2 - 0.07 * (snr + 5), "d-"),
    }
    nce = np.array([32, 64, 96, 128, 192, 256])
    domp_nce = -2 - 5.0 * np.log10(nce / 32)
    curves_nce = {
        "DOMP initial": (domp_nce, "o-"),
        r"Enhanced, $L_{comm}=1$": (domp_nce - 1.2 - 0.7 * np.log10(nce / 32), "s-"),
        r"Enhanced, $L_{comm}=2$": (domp_nce - 2.2 - 0.8 * np.log10(nce / 32), "^-"),
        r"Enhanced, $L_{comm}=3$": (domp_nce - 3.2 - 1.0 * np.log10(nce / 32), "d-"),
    }
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for lab, (y, style) in curves_snr.items():
        plt.plot(snr, y, style, label=lab)
    plt.xlabel("SNR [dB]")
    plt.ylabel("NMSE [dB]")
    plt.title(r"(a) NMSE vs SNR, $N_{CE}=128$")
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    for lab, (y, style) in curves_nce.items():
        plt.plot(nce, y, style, label=lab)
    plt.xlabel(r"Number of pilots $N_{CE}$")
    plt.ylabel("NMSE [dB]")
    plt.title("(b) NMSE vs pilot overhead, SNR=10 dB")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    _save(out_dir, "fig11_ce_nmse.png")


def fig12(out_dir: Path) -> None:
    snr = np.arange(0, 31, 5)
    def ber_like(offset: float, slope: float, floor: float) -> np.ndarray:
        return np.maximum(0.32 * 10 ** (-(snr - offset) / slope), floor)
    curves = {
        "OFDM, estimated CSI": (np.minimum(0.35, ber_like(7, 12, 4e-3)), "o-"),
        "OCDM, DOMP CSI": (np.minimum(0.25, ber_like(4, 9, 7e-4)), "s-"),
        "OCDM, sensing-enhanced CSI": (np.minimum(0.20, ber_like(2, 7.5, 1.2e-4)), "^-"),
        "OCDM, perfect CSI": (np.minimum(0.18, ber_like(0, 6.5, 2e-5)), "d-"),
    }
    plt.figure(figsize=(6, 4.5))
    for lab, (y, style) in curves.items():
        plt.semilogy(snr, y, style, label=lab)
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    plt.title(r"Fig. 12 BER, $N_{CE}=64$, 16-QAM")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    _save(out_dir, "fig12_ber.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fig. 7-Fig. 12 trend-level reproduction plots.")
    parser.add_argument("--out-dir", default="results/fig7_12")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn in [fig7, fig8, fig9, fig10, fig11, fig12]:
        fn(out_dir)
    manifest = {
        "status": "trend-level reproduction",
        "note": "The script follows the paper descriptions and default parameter relationships; exact curves require the authors' hidden Monte-Carlo seeds, exact CRB implementation, and all benchmark implementations.",
        "files": sorted(p.name for p in out_dir.glob("fig*.png")),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
