"""
Deprecated import path for **γ = 0** spatial-pilot knot tables.

Prefer ``modules.spatial_pilot_schedule_g0`` or
``spatial_pilot_schedule_loader.load_spatial_pilot_schedule(0.0)``.

Legacy names ``gamma0_dps_lambda`` / ``gamma0_cov_lambda`` map to the unified
``dps_lambda`` / ``cov_lambda`` API.
"""

from __future__ import annotations

from modules.spatial_pilot_schedule_g0 import (
    COV_LAMBDA,
    DPS_LAMBDA,
    DPS_LAMBDA_DM,
    SNR_KNOTS_DB,
    SNR_KNOTS_DB_DM,
    cov_lambda as gamma0_cov_lambda,
    dps_lambda as gamma0_dps_lambda,
    dps_lambda_dm as gamma0_dps_lambda_dm,
    piecewise_linear_snr_db,
)

__all__ = [
    "COV_LAMBDA",
    "DPS_LAMBDA",
    "DPS_LAMBDA_DM",
    "SNR_KNOTS_DB",
    "SNR_KNOTS_DB_DM",
    "gamma0_cov_lambda",
    "gamma0_dps_lambda",
    "gamma0_dps_lambda_dm",
    "piecewise_linear_snr_db",
]
