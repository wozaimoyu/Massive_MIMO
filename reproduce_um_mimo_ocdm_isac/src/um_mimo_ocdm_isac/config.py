from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class SystemConfig:
    fc: float
    c: float
    K: int
    B: float
    M: int
    T_GI: float
    f_LPF: float
    f_ADC: float
    P_tx_dBm: float
    N_tx: int
    N_rx: int
    N_dss: int
    D_rx: float
    L: int
    range_min: float
    range_max: float
    elevation_min: float
    elevation_max: float
    azimuth_min: float
    azimuth_max: float
    velocity_min_kmh: float
    velocity_max_kmh: float
    H_max: int
    H_vibs: int
    T_com: float
    G_cp: int
    epsilon_ce: float
    seed: int
    channel_model: str = "suc"
    sns_null_count: int = 0

    @property
    def wavelength(self) -> float:
        return self.c / self.fc

    @property
    def T(self) -> float:
        # OCDM symbol duration: B = K / T.
        return self.K / self.B

    @property
    def Q(self) -> int:
        # Paper uses Q = floor(f_ADC * (T - T_GI)).
        return int(self.f_ADC * (self.T - self.T_GI))

    @property
    def frame_step(self) -> float:
        return self.T + self.T_GI + self.T_com

    @property
    def P_tx_watt(self) -> float:
        return 1e-3 * 10 ** (self.P_tx_dBm / 10.0)


def load_config(path: str | Path) -> SystemConfig:
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    return SystemConfig(**data)
