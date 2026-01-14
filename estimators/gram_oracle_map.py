r"""
Gram-oracle MAP estimator for 3GPP (time-domain).

We replace the expensive conditional mean over V∈St(Nt,r) with a deterministic oracle baseline:
    H_hat_MAP(Y,R) = argmax_H p(H|Y)  s.t.  H H^H = R

Under the Gaussian prior vec(H)|delta ~ CN(0, C_tx ⊗ C_rx) and Y = H + Z, Z~CN(0, sigma2 I),
this becomes (up to constants):

    minimize_H  (1/sigma2)||Y - H||_F^2 + tr(C_rx^{-1} H C_tx^{-T} H^H)
    subject to  H H^H = R.

Using R = U diag(lam) U^H, lam>=0, rank r, write H = U Sigma V^H with Sigma = diag(sqrt(lam)).
Then the constrained problem reduces to optimizing V ∈ St(Nt,r).

We solve it with projected gradient descent on the complex Stiefel manifold using QR retraction.
This is stable and much cheaper than Monte Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import modules.utils as ut


def _hermitianize_np(C: np.ndarray) -> np.ndarray:
    return 0.5 * (C + C.conj().T)


def _eigh_rank_from_gram_np(R: np.ndarray, rank_tol: float) -> tuple[np.ndarray, np.ndarray]:
    R = _hermitianize_np(R)
    w, U = np.linalg.eigh(R)
    idx = np.argsort(w.real)[::-1]
    w = w.real[idx]
    U = U[:, idx]
    if w.size == 0 or w[0] <= 0:
        return U[:, :0], np.zeros((0,), dtype=float)
    keep = w > (rank_tol * w[0])
    r = int(np.sum(keep))
    return U[:, :r], np.sqrt(np.maximum(w[:r], 0.0))


def _procrustes_v_init_np(UHY: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Vh_init = argmin_{Vh Vh^H = I} ||U^H Y - Sigma Vh||_F^2  (Procrustes)
    Returns V (Nt,r) as initial point on Stiefel.
    """
    r, Nt = UHY.shape
    s_safe = np.maximum(s, 1e-12)
    A = UHY / s_safe.reshape(r, 1)  # (r,Nt)
    U_a, _, Vh_a = np.linalg.svd(A, full_matrices=False)  # Vh_a: (r,Nt)
    Vh = U_a @ Vh_a  # (r,Nt)
    return Vh.conj().T  # (Nt,r)


def _inv_psd_hermitian_np(C: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float]:
    C = _hermitianize_np(C)
    w, U = np.linalg.eigh(C)
    w = np.maximum(w.real, eps)
    logdet = float(np.sum(np.log(w)))
    inv_w = 1.0 / w
    C_inv = (U * inv_w.reshape(1, -1)) @ U.conj().T
    return C_inv, logdet


@dataclass
class MapOptions:
    rank_tol: float = 1e-6
    eps_cov: float = 1e-8
    n_iters: int = 80
    lr: float = 0.2
    # scale for prior energy (helps keep magnitudes comparable across dims)
    prior_q_scale: float = 1.0
    device: str = "cpu"
    dtype: torch.dtype = torch.complex64
    log_every: int = 0  # 0 disables; otherwise prints objective every N iters (debug)


def gram_oracle_map_given_delta(
    Y: np.ndarray,
    R: np.ndarray,
    *,
    t_rx: np.ndarray,
    t_tx: np.ndarray,
    sigma2: float,
    opts: MapOptions,
) -> tuple[np.ndarray, float]:
    """
    Solve MAP for a fixed delta, returning (H_hat, objective_value).
    Objective returned is the minimized value (smaller is better).
    """
    Nr, Nt = Y.shape
    U, s = _eigh_rank_from_gram_np(R, rank_tol=opts.rank_tol)
    r = s.size
    if r == 0:
        return np.zeros_like(Y), float("inf")

    # Build Toeplitz covariances and inverses (stable PSD projection)
    C_rx = ut.toeplitz(t_rx)
    C_tx = ut.toeplitz(t_tx)
    C_rx_inv, _ = _inv_psd_hermitian_np(C_rx, eps=opts.eps_cov)
    C_tx_inv, _ = _inv_psd_hermitian_np(C_tx, eps=opts.eps_cov)
    C_tx_inv_T = C_tx_inv.T  # needed for vec(H) with column-stacking

    # Precompute small matrices in numpy
    UHY = U.conj().T @ Y  # (r,Nt)
    s_row = s.reshape(r, 1)
    # Rx-side small matrix A = U^H C_rx^{-1} U
    A = U.conj().T @ C_rx_inv @ U
    G = (s.reshape(-1, 1) * A) * s.reshape(1, -1)  # Sigma A Sigma

    # Initialize V (Nt,r) using Procrustes
    V0 = _procrustes_v_init_np(UHY, s)

    # Torch optimization on complex Stiefel: V^H V = I (NO autograd; analytic gradient)
    dev = torch.device(opts.device)
    V = torch.tensor(V0, dtype=opts.dtype, device=dev)
    UHY_t = torch.tensor(UHY, dtype=opts.dtype, device=dev)  # (r,Nt)
    G_t = torch.tensor(G, dtype=opts.dtype, device=dev)      # (r,r) Hermitian
    C_t = torch.tensor(C_tx_inv_T, dtype=opts.dtype, device=dev)  # (Nt,Nt)
    s_t = torch.tensor(s, dtype=opts.dtype, device=dev).view(r, 1)  # (r,1)

    sigma2_f = float(sigma2)
    q_scale = float(opts.prior_q_scale)

    def retract_qr(Vm: torch.Tensor) -> torch.Tensor:
        # QR retraction to Stiefel
        Q, Rm = torch.linalg.qr(Vm, mode="reduced")
        d = torch.diagonal(Rm, 0)
        ph = d / torch.clamp(torch.abs(d), min=1e-12)
        Q = Q * ph.conj().unsqueeze(0)
        return Q

    # Reduced-form constants
    # B = (U^H Y)^H = Y^H U (Nt,r)
    B_t = UHY_t.conj().T  # (Nt,r)
    # s and s^2 as real vectors (length r)
    s_vec = torch.real(s_t.view(-1))
    s2_vec = s_vec * s_vec

    def stiefel_project(Vm: torch.Tensor, Gm: torch.Tensor) -> torch.Tensor:
        # Project Euclidean gradient onto tangent space at Vm: G - V sym(V^H G)
        VhG = Vm.conj().T @ Gm  # (r,r)
        sym = 0.5 * (VhG + VhG.conj().T)
        return Gm - Vm @ sym

    def objective(Vm: torch.Tensor) -> torch.Tensor:
        # like: ||B - V Sigma||^2 / sigma2
        VS = Vm * s_vec.view(1, -1)  # (Nt,r)
        diff = B_t - VS
        loss_like = (diff.abs() ** 2).sum() / sigma2_f
        # prior: q_scale * tr(G @ (V^H C V))
        Bmat = Vm.conj().T @ (C_t @ Vm)
        loss_prior = torch.real(torch.trace(G_t @ Bmat)) * q_scale
        return loss_like + loss_prior

    for it in range(int(opts.n_iters)):
        # Euclidean gradient
        # d/dV ||B - V Sigma||^2 = 2 (V Sigma^2 - B Sigma)
        grad_like = (V * s2_vec.view(1, -1) - B_t * s_vec.view(1, -1)) * (2.0 / sigma2_f)
        # d/dV tr(G V^H C V) = 2 C V G  (for Hermitian C,G)
        grad_prior = 2.0 * (C_t @ V @ G_t) * q_scale
        grad = grad_like + grad_prior

        grad = stiefel_project(V, grad)

        if not torch.all(torch.isfinite(grad)):
            break

        V = retract_qr(V - float(opts.lr) * grad)

        if opts.log_every and ((it + 1) % int(opts.log_every) == 0):
            with torch.no_grad():
                val = objective(V)
            print(f"[MAP] it={it+1} obj={float(val):.3e}")

    # Construct H_hat = U Sigma V^H (numpy)
    V_np = V.detach().cpu().numpy()
    H_hat = (U * s.reshape(1, -1)) @ V_np.conj().T

    # Return objective value in numpy for delta MAP selection
    # Recompute final objective in numpy (stable)
    like_obj = np.sum(np.abs(UHY - (s_row * V_np.conj().T)) ** 2) / max(sigma2, 1e-12)
    prior_obj = np.real(np.trace(G @ (V_np.conj().T @ (C_tx_inv_T @ V_np)))) * q_scale
    return H_hat, float(like_obj + prior_obj)


