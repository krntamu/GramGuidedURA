"""
Spatial pilot matrix X_p (N_t × N_p) for multiuser / MIMO pilot simulations.

Blended orthonormal + Gaussian (user model):
    X_p = sqrt(gamma) * I_rect + sqrt(1 - gamma) * G

where:
  - I_rect has ones on (i,i) for i = 0 .. min(N_t, N_p)-1 and zeros elsewhere
    (reduces to I_{N_t} when N_p = N_t; gives [I_{N_t}, 0] when N_p > N_t).
  - G is i.i.d. complex Gaussian; variance matches the caller's convention
    (Torch eval uses complex(randn, randn) like load_and_eval_dm_dps;
     NumPy baselines use (randn + 1j*randn) / sqrt(2) for CN(0,1)).

gamma = 0  -> pure G (legacy Gaussian pilots)
gamma = 1  -> pure I_rect (orthogonal / identity-on-diagonal pilots)

power_norm:
  legacy   -> no extra scaling (torch: G = N(0,1)+jN(0,1); numpy: CN(0,1))
  align_i  -> scale G like above so E[G G^H] = I (Gaussian part only)
  row_norm -> build like align_i, then each row of final X_p has unit L2 norm
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
import torch

PilotPowerNorm = Literal["legacy", "align_i", "row_norm"]


def _row_normalize_pilot_torch(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Each row x_i <- x_i / ||x_i||_2 (complex Euclidean norm)."""
    nrm = torch.linalg.vector_norm(X, dim=1, keepdim=True)
    return X / torch.clamp(nrm, min=eps)


def row_normalize_pilot_numpy(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, eps)


def rect_identity_complex_torch(
    n_t: int,
    n_p: int,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Real rectangular identity embedded in C^{N_t × N_p}."""
    r = min(n_t, n_p)
    m = torch.zeros(n_t, n_p, dtype=torch.float32, device=device)
    if r > 0:
        idx = torch.arange(r, device=device)
        m[idx, idx] = 1.0
    return m.to(dtype)


def draw_xp_sqrt_gamma_identity_gaussian_torch(
    n_t: int,
    n_p: int,
    gamma: float,
    *,
    power_norm: PilotPowerNorm = "legacy",
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    X_p = sqrt(gamma) * I_rect + sqrt(1-gamma) * G with G_ij = N(0,1)+j N(0,1)
    (same complex noise convention as load_and_eval_dm_dps Gaussian pilots).

    power_norm:
      - "legacy": no additional scaling (backward compatible).
      - "align_i": scale the Gaussian part by 1/sqrt(2*n_p) so that E[G G^H] = I (power aligned with identity pilots).
      - "row_norm": same construction as align_i, then each row x_i <- x_i / ||x_i||_2 on the final X_p.
    """
    pnorm = str(power_norm).lower()
    if pnorm not in ("legacy", "align_i", "row_norm"):
        raise ValueError(f"power_norm must be legacy, align_i, or row_norm, got {power_norm!r}")

    g = float(max(0.0, min(1.0, gamma)))
    i_rect = rect_identity_complex_torch(n_t, n_p, device, dtype)
    align_like = pnorm in ("align_i", "row_norm")

    # Endpoints: no redundant Gaussian draw (γ=1 matches orthonormal I without touching RNG for G).
    if g >= 1.0 - 1e-15:
        out = i_rect
        return _row_normalize_pilot_torch(out) if pnorm == "row_norm" else out

    sg = math.sqrt(g)
    s1 = math.sqrt(1.0 - g)
    if generator is not None:
        xr = torch.randn(n_t, n_p, device=device, generator=generator, dtype=torch.float32)
        xi = torch.randn(n_t, n_p, device=device, generator=generator, dtype=torch.float32)
    else:
        xr = torch.randn(n_t, n_p, device=device, dtype=torch.float32)
        xi = torch.randn(n_t, n_p, device=device, dtype=torch.float32)
    big_g = torch.complex(xr, xi).to(dtype)
    if align_like:
        # For big_g = N(0,1) + jN(0,1): E[big_g big_g^H] = 2*n_p * I.
        # Scale by 1/sqrt(2*n_p) => E[G G^H] = I.
        big_g = big_g * (1.0 / math.sqrt(2.0 * float(n_p)))
    if g <= 1e-15:
        out = big_g
        return _row_normalize_pilot_torch(out) if pnorm == "row_norm" else out
    out = sg * i_rect + s1 * big_g
    return _row_normalize_pilot_torch(out) if pnorm == "row_norm" else out


def rect_identity_complex_numpy(n_t: int, n_p: int) -> np.ndarray:
    r = min(n_t, n_p)
    m = np.zeros((n_t, n_p), dtype=np.complex128)
    if r > 0:
        m[np.arange(r), np.arange(r)] = 1.0
    return m


def draw_xp_sqrt_gamma_identity_gaussian_numpy(
    n_t: int,
    n_p: int,
    gamma: float,
    rng: np.random.Generator,
    *,
    power_norm: PilotPowerNorm = "legacy",
) -> np.ndarray:
    """
    Same blend as torch path; G is CN(0,1) via (randn + 1j*randn)/sqrt(2)
    to match baselines.run_mimo_pilot_baselines_single_snr.

    power_norm:
      - "legacy": no additional scaling (backward compatible).
      - "align_i": additionally scale the Gaussian part by 1/sqrt(n_p) so that E[G G^H] = I.
      - "row_norm": same as align_i, then row-wise L2 normalization of the final X_p.
    """
    pnorm = str(power_norm).lower()
    if pnorm not in ("legacy", "align_i", "row_norm"):
        raise ValueError(f"power_norm must be legacy, align_i, or row_norm, got {power_norm!r}")

    g = float(np.clip(gamma, 0.0, 1.0))
    i_rect = rect_identity_complex_numpy(n_t, n_p)
    align_like = pnorm in ("align_i", "row_norm")

    if g >= 1.0 - 1e-15:
        out = i_rect
        return row_normalize_pilot_numpy(out) if pnorm == "row_norm" else out

    sg = math.sqrt(g)
    s1 = math.sqrt(1.0 - g)
    g_mat = (
        (rng.standard_normal((n_t, n_p)) + 1j * rng.standard_normal((n_t, n_p)))
        * (1.0 / np.sqrt(2.0))
    ).astype(np.complex128)
    if align_like:
        # For CN(0,1): E[g_mat g_mat^H] = n_p * I. Scale by 1/sqrt(n_p) => E[G G^H] = I.
        g_mat = g_mat * (1.0 / np.sqrt(float(n_p)))
    if g <= 1e-15:
        out = g_mat
        return row_normalize_pilot_numpy(out) if pnorm == "row_norm" else out
    out = sg * i_rect + s1 * g_mat
    return row_normalize_pilot_numpy(out) if pnorm == "row_norm" else out
