"""
Per-SNR DPS / cov λ knot tables for spatial pilot blend **γ = 0.5**
(√γ I_rect + √(1−γ) Gaussian).

**Placeholder:** values match ``spatial_pilot_schedule_g0.py`` until you tune γ=0.5.
"""

from __future__ import annotations

import math

SNR_KNOTS_DB: tuple[float, ...] = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
DPS_LAMBDA: tuple[float, ...] = (1, 0.7, 0.45, 0.3, 0.3, 0.25, 0.2, 0.15)
COV_LAMBDA: tuple[float, ...] = (5e-5, 6e-4, 4e-3, 2e-2, 5e-2, 0.06, 0.07, 0.08)

SNR_KNOTS_DB_DM: tuple[float, ...] = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
# with SNR matching
DPS_LAMBDA_DM: tuple[float, ...] = (1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.08, 0.07)
# without SNR matching
DPS_LAMBDA_DM: tuple[float, ...] = (1.0, 0.8, 0.5, 0.3, 0.2, 0.15, 0.08, 0.04)

_LOG_COV_LAMBDA: tuple[float, ...] = tuple(math.log(float(v)) for v in COV_LAMBDA)


def piecewise_linear_snr_db(snr_db: float, knots: tuple[float, ...], values: tuple[float, ...]) -> float:
    if len(knots) != len(values) or len(knots) < 2:
        raise ValueError("knots and values must have the same length >= 2")
    s = float(snr_db)
    if s <= knots[0]:
        return float(values[0])
    if s >= knots[-1]:
        return float(values[-1])
    for i in range(len(knots) - 1):
        lo, hi = knots[i], knots[i + 1]
        if lo <= s <= hi:
            if hi <= lo:
                return float(values[i])
            t = (s - lo) / (hi - lo)
            return float(values[i] + t * (values[i + 1] - values[i]))
    return float(values[-1])


def dps_lambda(snr_db: float, *, dm_no_cov: bool = False) -> float:
    if dm_no_cov:
        return piecewise_linear_snr_db(snr_db, SNR_KNOTS_DB_DM, DPS_LAMBDA_DM)
    return piecewise_linear_snr_db(snr_db, SNR_KNOTS_DB, DPS_LAMBDA)


def dps_lambda_dm(snr_db: float) -> float:
    return dps_lambda(snr_db, dm_no_cov=True)


def cov_lambda(snr_db: float) -> float:
    log_lam = piecewise_linear_snr_db(snr_db, SNR_KNOTS_DB, _LOG_COV_LAMBDA)
    return float(math.exp(log_lam))

