"""
Resolve per-SNR λ **knot tables** from ``--spatial_pilot_gamma`` (0, 0.5, or 1).

Schedules live in ``spatial_pilot_schedule_g0``, ``_g05``, ``_g1``. Each exposes
``dps_lambda(snr_db, *, dm_no_cov=False)``, ``dps_lambda_dm``, ``cov_lambda``.
"""

from __future__ import annotations

import importlib
from types import ModuleType

# Supported spatial pilot blend values (must match schedule modules).
SUPPORTED_SPATIAL_GAMMAS: tuple[float, ...] = (0.0, 0.5, 1.0)
_GAMMA_TOL = 1e-5

_GAMMA_TO_MODULE: dict[float, str] = {
    0.0: "modules.spatial_pilot_schedule_g0",
    0.5: "modules.spatial_pilot_schedule_g05",
    1.0: "modules.spatial_pilot_schedule_g1",
}


def normalize_spatial_pilot_gamma(spatial_pilot_gamma: float) -> float:
    """Return canonical γ ∈ {0, 0.5, 1} or raise ``ValueError``."""
    g = float(spatial_pilot_gamma)
    for cand in SUPPORTED_SPATIAL_GAMMAS:
        if abs(g - cand) <= _GAMMA_TOL:
            return cand
    raise ValueError(
        "pilot_table λ schedules require --spatial_pilot_gamma in {0, 0.5, 1} "
        f"(tol={_GAMMA_TOL:g}); got {g}."
    )


def load_spatial_pilot_schedule(spatial_pilot_gamma: float) -> ModuleType:
    """Import the schedule module for the given spatial pilot γ."""
    key = normalize_spatial_pilot_gamma(spatial_pilot_gamma)
    name = _GAMMA_TO_MODULE[key]
    return importlib.import_module(name)


def effective_spatial_gamma_for_pilot_table(pilot_mode: str, spatial_pilot_gamma: float) -> float:
    """
    γ used to pick ``spatial_pilot_schedule_g*``.

    ``pilot_mode=gaussian`` (pure G pilots) maps to γ=0 for table lookup; ``nonorthogonal`` uses
    ``spatial_pilot_gamma`` as in ``X_p = sqrt(γ) I + sqrt(1-γ) G``.
    """
    if pilot_mode == "gaussian":
        return 0.0
    return float(spatial_pilot_gamma)


def load_pilot_table_schedule(pilot_mode: str, spatial_pilot_gamma: float) -> ModuleType:
    """Resolve knot-table module from pilot mode and ``--spatial_pilot_gamma``."""
    g = effective_spatial_gamma_for_pilot_table(pilot_mode, spatial_pilot_gamma)
    return load_spatial_pilot_schedule(g)
