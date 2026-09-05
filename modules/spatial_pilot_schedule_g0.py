"""
Per-SNR DPS / cov λ **knot tables** for spatial pilot blend **γ = 0** (pure Gaussian pilots).

Unified API: ``dps_lambda``, ``dps_lambda_dm``, ``cov_lambda``. Selected at runtime via
``spatial_pilot_schedule_loader.load_spatial_pilot_schedule`` using ``--spatial_pilot_gamma 0``.
"""

from __future__ import annotations

import math

SNR_KNOTS_DB: tuple[float, ...] = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
# tr SNR MATCHING
# DPS_LAMBDA: tuple[float, ...] = (1.0, 0.8, 0.5, 0.4, 0.3, 0.32, 0.22, 0.14)
# COV_LAMBDA: tuple[float, ...] = (5e-5, 1e-4, 5e-3, 1e-2, 5e-2, 0.07, 0.06, 0.06)
# --gaussian_snr_match_mode worst # --n_pilot 16
DPS_LAMBDA: tuple[float, ...] = (
    1.15,   # -15
    0.82,   # -10
    0.3825, # -5
    0.26,   # 0
    0.315,  # 5
    0.304,  # 10
    0.22,   # 15
    0.145,  # 20
    0.076,  # 25 
    0.035   # 30 
)

COV_LAMBDA: tuple[float, ...] = (
    5.1e-5,   # -15
    1.9e-4,   # -10
    4.463e-3, # -5
    0.016,    # 0
    0.0575,   # 5
    0.091,    # 10
    0.07245,  # 15
    0.0759,   # 20
    0.070,    # 25  
    0.075     # 30  
)


# DPS_LAMBDA: tuple[float, ...] = (
#     1.07825,  # -15
#     0.861,    # -10
#     0.34425,  # -5
#     0.312,    # 0
#     0.252,    # 5
#     0.2432,   # 10
#     0.176,    # 15
#     0.116     # 20
# )

# COV_LAMBDA: tuple[float, ...] = (
#     4.6e-5,   # -15
#     2.85e-4,  # -10
#     6.693e-3, # -5
#     0.024,    # 0
#     0.0805,   # 5
#     0.0637,   # 10
#     0.07245,  # 15
#     0.06072   # 20
# )
SNR_KNOTS_DB_DM: tuple[float, ...] = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
# tr SNR MATCHING
# DPS_LAMBDA_DM: tuple[float, ...] = (1.0, 0.75, 0.5, 0.3, 0.18, 0.12, 0.11, 0.10)
# --n_pilot 16
DPS_LAMBDA: tuple[float, ...] = (
    1.15,   # -15 dB
    0.75,   # -10 dB 
    0.50,   # -5 dB 
    0.345,  # 0 dB
    0.2574, # 5 dB
    0.1872, # 10 dB
    0.1716, # 15 dB
    0.156,  # 20 dB
    0.168480, # 25
    0.137280 # 30
)
# # --n_pilot 32CD
# DPS_LAMBDA: tuple[float, ...] = (
#     1.15,     # -15 dB
#     0.75,     # -10 dB
#     0.50,     # -5 dB
#     0.345,    # 0 dB
#     0.2574,   # 5 dB
#     0.2028,   # 10 dB
#     0.16445,  # 15 dB
#     0.1495    # 20 dB
# )
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
