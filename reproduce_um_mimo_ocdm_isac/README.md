# Reproduction scaffold: UM-MIMO + OCDM for near-field ISAC

This directory contains a first-principles Python scaffold for reproducing the sensing part of arXiv:2512.23246v2, **Ultra-Massive MIMO with Orthogonal Chirp Division Multiplexing for Near-Field Sensing and Communication Integration**.

The current implementation focuses on the paper's central simulation chain: UM-MIMO near-field geometry, DSA/DSS selection, OCDM-FMCW low-rate measurements, pairwise bistatic range/velocity estimation, and virtual bistatic sensing (VIBS) fusion.

## Quick start

```bash
cd reproduce_um_mimo_ocdm_isac
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/run_sensing_demo.py --config configs/default.yaml --snr-db 20 --out-dir results
```

The demo writes `results/sensing_demo_summary.json` and `results/vibs_xy.png`.

## Generate Fig. 7-Fig. 12

```bash
python scripts/run_fig7_to_fig12.py --out-dir results/fig7_12
```

This creates:

```text
fig7_convergence.png
fig8_parameter_tmse.png
fig9_vibs_mse.png
fig10_transceiver_configuration.png
fig11_ce_nmse.png
fig12_ber.png
manifest.json
```

The Fig. 7-Fig. 12 script is a trend-level reproduction aligned with the paper descriptions and default parameter relationships. Exact numerical matching requires the authors' Monte-Carlo seeds, exact CRB implementation, exact OFDM/non-overlapped-chirp baselines, and full Algorithm 3 communication simulation details.

## Layout

```text
configs/default.yaml
src/um_mimo_ocdm_isac/config.py
src/um_mimo_ocdm_isac/geometry.py
src/um_mimo_ocdm_isac/ocdm.py
src/um_mimo_ocdm_isac/sensing_channel.py
src/um_mimo_ocdm_isac/measurements.py
src/um_mimo_ocdm_isac/parameter_estimation.py
src/um_mimo_ocdm_isac/vibs.py
scripts/run_sensing_demo.py
scripts/run_fig7_to_fig12.py
```

## Mapping to the paper

- Table I default system parameters: `configs/default.yaml`, `config.py`
- Equations (4)-(5), (13)-(14): bistatic distance and velocity: `geometry.py`
- Equations (23), (26)-(30): DSS feasibility and post-ADC signal model: `ocdm.py`, `measurements.py`
- Algorithm 1 interface: `parameter_estimation.py`
- Equations (36)-(38), Algorithm 2 VIBS: `vibs.py`
- Fig. 7-Fig. 12 plotting suite: `scripts/run_fig7_to_fig12.py`

## Current status

The scaffold directly generates the post-ADC model `R_nj = B_n(mu_nj) diag(alpha_nj) A^H(nu_nj) + N_nj`. Pairwise parameters are initialized by a 2-D FFT/OMP-style grid search and refined by nonlinear least squares. The module boundaries are prepared for replacing this baseline with exact RL-ESPRIT + GDA.

Next steps: exact RL-ESPRIT, Fig. 8 baselines/CRB, Algorithm 3 DOMP + sensing-enhanced CE, NMSE/BER scripts.
