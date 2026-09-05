from __future__ import annotations

import math
from typing import Literal, Tuple

import numpy as np
import torch


DetMod = Literal["4qam", "qpsk", "bpsk"]


def _as_det_mod(modulation: str) -> DetMod:
    m = str(modulation).lower()
    if m == "4-qam":
        m = "4qam"
    if m in ("4qam", "qpsk", "bpsk"):
        return m  # type: ignore[return-value]
    raise ValueError(f"Unknown modulation: {modulation}. Use one of: 4qam, qpsk, bpsk.")


# =============================================================================
# Torch (batched) utilities
# =============================================================================


def generate_symbols_torch(
    batch: int,
    n_tx: int,
    n_sym: int,
    modulation: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    Generate i.i.d. symbols X of shape (B, N_t, N_sym) with unit average power per entry.
    """
    mod = _as_det_mod(modulation)
    if mod == "bpsk":
        xr = 2.0 * (torch.rand(batch, n_tx, n_sym, device=device) > 0.5).float() - 1.0
        X = xr.to(dtype=dtype)
        return X
    # QPSK / 4QAM: (±1 ± j)/sqrt(2)
    xr = 2.0 * (torch.rand(batch, n_tx, n_sym, device=device) > 0.5).float() - 1.0
    xi = 2.0 * (torch.rand(batch, n_tx, n_sym, device=device) > 0.5).float() - 1.0
    X = (xr + 1j * xi) / math.sqrt(2.0)
    return X.to(dtype=dtype)


def simulate_mimo_torch(
    H: torch.Tensor,  # (B, N_r, N_t) complex
    X: torch.Tensor,  # (B, N_t, N_sym) complex
    *,
    sigma2: float,
) -> torch.Tensor:
    """
    Y = H X + N, with CN(0, sigma2) noise per complex entry.
    Returns Y with shape (B, N_r, N_sym).
    """
    if H.dim() != 3 or X.dim() != 3:
        raise ValueError(f"Expected H,X to be 3D tensors, got H:{tuple(H.shape)} X:{tuple(X.shape)}")
    if H.size(0) != X.size(0) or H.size(2) != X.size(1):
        raise ValueError(f"Shape mismatch: H:{tuple(H.shape)} X:{tuple(X.shape)}")
    if not torch.is_complex(H) or not torch.is_complex(X):
        raise ValueError("H and X must be complex tensors.")

    B, N_r, _N_t = H.shape
    _B, _N_t2, N_sym = X.shape
    _ = _B, _N_t2
    Y_sig = H @ X
    s2 = float(sigma2)
    if s2 < 0 or not np.isfinite(s2):
        raise ValueError(f"sigma2 must be finite and >=0, got {sigma2}")
    if s2 == 0.0:
        return Y_sig
    std = math.sqrt(s2 / 2.0)
    n_re = torch.randn(B, N_r, N_sym, device=H.device, dtype=torch.float32)
    n_im = torch.randn(B, N_r, N_sym, device=H.device, dtype=torch.float32)
    N = (std * (n_re + 1j * n_im)).to(dtype=H.dtype)
    return Y_sig + N


def mmse_detect_torch(
    H_hat: torch.Tensor,  # (B, N_r, N_t) complex
    Y: torch.Tensor,      # (B, N_r, N_sym) complex
    *,
    sigma2: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Linear MMSE detector:
      X_hat = (H^H H + sigma2 I)^(-1) H^H Y
    Returns X_hat with shape (B, N_t, N_sym).
    """
    if H_hat.dim() != 3 or Y.dim() != 3:
        raise ValueError(f"Expected H_hat,Y to be 3D tensors, got H_hat:{tuple(H_hat.shape)} Y:{tuple(Y.shape)}")
    if H_hat.size(0) != Y.size(0) or H_hat.size(1) != Y.size(1):
        raise ValueError(f"Shape mismatch: H_hat:{tuple(H_hat.shape)} Y:{tuple(Y.shape)}")
    if not torch.is_complex(H_hat) or not torch.is_complex(Y):
        raise ValueError("H_hat and Y must be complex tensors.")

    B, N_r, N_t = H_hat.shape
    _B, _N_r, N_sym = Y.shape
    _ = _B, _N_r

    Hh = H_hat.conj().transpose(-1, -2)  # (B, N_t, N_r)
    A = Hh @ H_hat  # (B, N_t, N_t)
    s2 = float(sigma2)
    I = torch.eye(N_t, device=H_hat.device, dtype=H_hat.dtype).expand(B, N_t, N_t)
    A = A + (s2 + float(eps)) * I
    Bmat = Hh @ Y  # (B, N_t, N_sym)
    return torch.linalg.solve(A, Bmat)


def slice_constellation_torch(X_hat: torch.Tensor, modulation: str) -> torch.Tensor:
    """
    Hard slicing to nearest constellation point.
    """
    mod = _as_det_mod(modulation)
    if mod == "bpsk":
        xr = torch.where(X_hat.real >= 0, torch.tensor(1.0, device=X_hat.device), torch.tensor(-1.0, device=X_hat.device))
        return xr.to(dtype=X_hat.dtype)
    xr = torch.where(X_hat.real >= 0, torch.tensor(1.0, device=X_hat.device), torch.tensor(-1.0, device=X_hat.device))
    xi = torch.where(X_hat.imag >= 0, torch.tensor(1.0, device=X_hat.device), torch.tensor(-1.0, device=X_hat.device))
    Xs = (xr + 1j * xi) / math.sqrt(2.0)
    return Xs.to(dtype=X_hat.dtype)


def ser_torch(X_hat_sliced: torch.Tensor, X_true: torch.Tensor, modulation: str) -> float:
    """
    Symbol error rate over all entries.
    """
    mod = _as_det_mod(modulation)
    if mod == "bpsk":
        err = (torch.sign(X_hat_sliced.real) != torch.sign(X_true.real))
    else:
        err = (torch.sign(X_hat_sliced.real) != torch.sign(X_true.real)) | (torch.sign(X_hat_sliced.imag) != torch.sign(X_true.imag))
    return float(err.float().mean().item())


@torch.no_grad()
def ser_mmse_from_channel_estimates_torch(
    H_true_ri: torch.Tensor,  # (B, 2, N_r, N_t) real/imag
    H_hat_ri: torch.Tensor,   # (B, 2, N_r, N_t) real/imag
    *,
    snr_linear: float,
    n_sym: int,
    modulation: str = "4qam",
) -> Tuple[float, int]:
    """
    Convenience wrapper for SER evaluation from (H_true, H_hat) batches.
    Returns (ser, n_symbols_total).
    """
    if H_true_ri.dim() != 4 or H_true_ri.size(1) != 2:
        raise ValueError(f"Expected H_true_ri shape (B,2,Nr,Nt), got {tuple(H_true_ri.shape)}")
    if H_hat_ri.dim() != 4 or H_hat_ri.size(1) != 2:
        raise ValueError(f"Expected H_hat_ri shape (B,2,Nr,Nt), got {tuple(H_hat_ri.shape)}")
    if H_true_ri.shape != H_hat_ri.shape:
        raise ValueError(f"H_true_ri and H_hat_ri must have same shape, got {tuple(H_true_ri.shape)} vs {tuple(H_hat_ri.shape)}")
    if n_sym <= 0:
        raise ValueError(f"n_sym must be > 0, got {n_sym}")

    B, _two, N_r, N_t = H_true_ri.shape
    device = H_true_ri.device

    H_true = torch.complex(H_true_ri[:, 0], H_true_ri[:, 1]).to(torch.complex64)
    H_hat = torch.complex(H_hat_ri[:, 0], H_hat_ri[:, 1]).to(torch.complex64)

    snr_lin = float(snr_linear)
    if snr_lin <= 0 or not np.isfinite(snr_lin):
        raise ValueError(f"snr_linear must be finite and >0, got {snr_linear}")
    sigma2 = 1.0 / snr_lin

    X = generate_symbols_torch(B, N_t, n_sym, modulation, device=device, dtype=torch.complex64)
    Y = simulate_mimo_torch(H_true, X, sigma2=sigma2)
    X_hat = mmse_detect_torch(H_hat, Y, sigma2=sigma2)
    Xs = slice_constellation_torch(X_hat, modulation)
    ser = ser_torch(Xs, X, modulation)
    n_total = int(B * N_t * n_sym)
    return ser, n_total


# =============================================================================
# NumPy utilities (small-scale scripts)
# =============================================================================


def generate_symbols_np(n_tx: int, n_sym: int, modulation: str, rng: np.random.Generator) -> np.ndarray:
    mod = _as_det_mod(modulation)
    if mod == "bpsk":
        xr = 2.0 * (rng.random((n_tx, n_sym)) > 0.5).astype(np.float32) - 1.0
        return xr.astype(np.complex64)
    xr = 2.0 * (rng.random((n_tx, n_sym)) > 0.5).astype(np.float32) - 1.0
    xi = 2.0 * (rng.random((n_tx, n_sym)) > 0.5).astype(np.float32) - 1.0
    return (xr + 1j * xi) / np.sqrt(2.0)


def simulate_mimo_np(H: np.ndarray, X: np.ndarray, sigma2: float, rng: np.random.Generator) -> np.ndarray:
    Nr, Nt = H.shape
    if X.shape[0] != Nt:
        raise ValueError(f"X must have shape (Nt, Nsym), got {X.shape} with Nt={Nt}")
    Y_sig = H @ X
    if sigma2 == 0.0:
        return Y_sig
    std = np.sqrt(float(sigma2) / 2.0)
    N = std * (rng.standard_normal((Nr, X.shape[1])) + 1j * rng.standard_normal((Nr, X.shape[1])))
    return Y_sig + N


def mmse_detect_np(H_hat: np.ndarray, Y: np.ndarray, sigma2: float, eps: float = 1e-12) -> np.ndarray:
    Nr, Nt = H_hat.shape
    if Y.shape[0] != Nr:
        raise ValueError(f"Y must have shape (Nr, Nsym), got {Y.shape} with Nr={Nr}")
    Hh = H_hat.conj().T
    A = Hh @ H_hat + (float(sigma2) + float(eps)) * np.eye(Nt, dtype=H_hat.dtype)
    B = Hh @ Y
    return np.linalg.solve(A, B)


def slice_constellation_np(X_hat: np.ndarray, modulation: str) -> np.ndarray:
    mod = _as_det_mod(modulation)
    if mod == "bpsk":
        xr = np.where(X_hat.real >= 0, 1.0, -1.0).astype(np.float32)
        return xr.astype(np.complex64)
    xr = np.where(X_hat.real >= 0, 1.0, -1.0).astype(np.float32)
    xi = np.where(X_hat.imag >= 0, 1.0, -1.0).astype(np.float32)
    return (xr + 1j * xi) / np.sqrt(2.0)


def ser_np(X_hat_sliced: np.ndarray, X_true: np.ndarray, modulation: str) -> float:
    mod = _as_det_mod(modulation)
    if mod == "bpsk":
        err = np.sign(X_hat_sliced.real) != np.sign(X_true.real)
    else:
        err = (np.sign(X_hat_sliced.real) != np.sign(X_true.real)) | (np.sign(X_hat_sliced.imag) != np.sign(X_true.imag))
    return float(np.mean(err))

