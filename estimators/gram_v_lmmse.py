"""
Gram-aided LMMSE on the right singular factor V^H with oracle receive Gram / (U, Sigma).

Model (same pilots as baselines.py): Y_p = H P + W, W ~ CN(0, sigma^2 I) per entry
(variance sigma^2 = 10^{-SNR/10}, consistent with mimo_pilot_observation + LMMSE baselines).

Whitening: Y' = Sigma^{-1} U^H Y_p = V^H P + N' with N' = Sigma^{-1} U^H W.

Vectorized: y' = vec(Y') = (P^T ⊗ I_r) v + n', v = vec(V^H), A = P^T ⊗ I_r,
C_n = sigma^2 (I_{N_p} ⊗ Sigma^{-2}).

LMMSE: v_hat = C_v A^H (A C_v A^H + C_n)^{-1} y', then H_hat = U Sigma V_hat^H.

By default, raw V^H is projected to the Frobenius-nearest matrix with orthonormal columns in V = (V^H)^H
(Stiefel: V^H V = I): skinny SVD V_raw = U Σ W^H, V_proj = U W^H, V^H_proj = V_proj^H.

V^H recovery from (H, U, Sigma) for training statistics uses V^H = Sigma^{-1} U^H H (economy SVD; stable Sigma via floor).
"""

from __future__ import annotations

import numpy as np


def u_sigma_from_h_svd(
    H: np.ndarray, sing_tol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Economy SVD H = U diag(s) Vh with U (n_rx x r), s (r,), same U,s as HH^H eigendecomp.

    Returns U, s, s_eff (floored for stable inversion), and effective rank count
    (number of singular values with s_i / s_max > sing_tol).
    """
    H = np.asarray(H)
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    r = int(s.size)
    if r == 0:
        raise ValueError("Empty SVD (zero-sized channel).")
    s_max = float(s[0])
    s_eff = np.maximum(s, s_max * float(sing_tol))
    eff_rank = int(np.sum(s > s_max * float(sing_tol)))
    return U, s, s_eff, eff_rank


def vec_vh_training_from_h(H: np.ndarray, sing_tol: float) -> np.ndarray:
    """
    v = vec(V^H) with V^H = diag(1/s_eff) U^H H (matches user recipe with stable Sigma^{-1}).
    """
    H = np.asarray(H)
    U, s, s_eff, _ = u_sigma_from_h_svd(H, sing_tol)
    r = s.size
    n_tx = H.shape[1]
    Sigma_inv = np.diag(1.0 / s_eff)
    Vh = Sigma_inv @ (U.conj().T @ H)
    return Vh.reshape(-1, order="F")


def empirical_cv_from_training_H(
    H_train: np.ndarray,
    sing_tol: float,
    lambda_reg: float,
) -> tuple[np.ndarray, dict]:
    """
    H_train: (M, n_rx, n_tx). hat C_v = (1/M) sum_m v_m v_m^H + lambda_reg * I.
    """
    H_train = np.asarray(H_train)
    M = H_train.shape[0]
    if M < 1:
        raise ValueError("Need at least one training channel.")
    v0 = vec_vh_training_from_h(H_train[0], sing_tol)
    d = v0.size
    acc = np.zeros((d, d), dtype=np.result_type(H_train.dtype, np.complex128))
    for m in range(M):
        vm = vec_vh_training_from_h(H_train[m], sing_tol)
        acc += np.outer(vm, vm.conj())
    Cv_hat = acc / float(M)
    lam = float(lambda_reg)
    if lam > 0:
        Cv_hat = Cv_hat + lam * np.eye(d, dtype=Cv_hat.dtype)
    meta = _cv_diagnostics(Cv_hat, lam)
    return Cv_hat, meta


def vh_row_gram_orthogonality_metrics(Vh: np.ndarray) -> dict:
    """
    V^H from economy SVD is r×n_tx with orthonormal rows: Vh @ Vh^H = I_r.

    For estimated Vh_hat, deviations measure loss of row orthogonality / unit norms.
    """
    Vh = np.asarray(Vh)
    r, nt = Vh.shape
    G = Vh @ Vh.conj().T
    I = np.eye(r, dtype=G.dtype)
    err = G - I
    off = err - np.diag(np.diag(err))
    fro = float(np.linalg.norm(err, "fro"))
    ref = float(np.sqrt(r))
    try:
        g_herm = 0.5 * (G + G.conj().T)
        g_cond = float(np.linalg.cond(g_herm))
    except np.linalg.LinAlgError:
        g_cond = float("inf")
    return {
        "r": int(r),
        "n_tx": int(nt),
        "fro_gram_minus_I": fro,
        "rel_fro_gram_minus_I": fro / (ref + 1e-30),
        "max_abs_offdiag_G": float(np.max(np.abs(off)) if r > 1 else 0.0),
        "mean_abs_offdiag_G": float(np.mean(np.abs(off)) if r > 1 else 0.0),
        "max_diag_deviation_from_1": float(np.max(np.abs(np.diag(G) - 1.0))),
        "cond_VhVhH": g_cond,
    }


def true_vh_from_svd(H: np.ndarray) -> np.ndarray:
    """Economy SVD V^H (r×n_tx), exactly row-orthogonal."""
    _, _, Vh = np.linalg.svd(np.asarray(H), full_matrices=False)
    return Vh


def project_vh_to_stiefel_columns(Vh_raw: np.ndarray) -> np.ndarray:
    """
    V = Vh^H (n_tx × r). Target Stiefel: V^H V = I_r i.e. Vh @ Vh^H = I_r.

    Given raw Vh_hat (r×n_tx), Frobenius-nearest V with orthonormal columns is
    V_proj = U W^H from skinny SVD V_raw = U Σ W^H, then return Vh_proj = V_proj^H.
    """
    Vh_raw = np.asarray(Vh_raw)
    V_raw = Vh_raw.conj().T
    Uf, _, Wh = np.linalg.svd(V_raw, full_matrices=False)
    V_proj = Uf @ Wh
    return V_proj.conj().T


def _cv_diagnostics(Cv: np.ndarray, lambda_reg: float) -> dict:
    Ch = 0.5 * (Cv + Cv.conj().T)
    d = Ch.shape[0]
    diag_part = np.diag(np.diag(Ch))
    off = Ch - diag_part
    fro_all = np.linalg.norm(Ch, "fro") + 1e-30
    fro_off = np.linalg.norm(off, "fro")
    try:
        cond = float(np.linalg.cond(Ch))
    except np.linalg.LinAlgError:
        cond = float("inf")
    return {
        "cv_cond": cond,
        "cv_rel_fro_offdiag": float(fro_off / fro_all),
        "cv_lambda_reg": float(lambda_reg),
        "cv_dim": int(d),
    }


def gram_v_lmmse_estimate_h(
    H_oracle: np.ndarray,
    Y_p: np.ndarray,
    P: np.ndarray,
    sigma2: float,
    Cv: np.ndarray | None,
    sing_tol: float = 1e-8,
    ridge: float = 1e-10,
    return_vh: bool = False,
    project_v_to_stiefel: bool = True,
) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, np.ndarray]:
    """
    Oracle Gram via SVD of H_oracle. Returns (H_hat, diagnostics for this sample).

    Cv: (r*n_tx, r*n_tx) Hermitian PSD (regularized), or None for identity prior.

    If project_v_to_stiefel, map raw V^H estimate onto Frobenius-nearest V with V^H V = I
    (equivalently Vh Vh^H = I_r for Vh = V^H) via skinny SVD of V = Vh^H: V_proj = U W^H.
    """
    H_oracle = np.asarray(H_oracle)
    Y_p = np.asarray(Y_p)
    P = np.asarray(P)
    n_rx, n_tx = H_oracle.shape
    n_p = P.shape[1]
    if P.shape[0] != n_tx:
        raise ValueError(f"P shape {P.shape} incompatible with n_tx={n_tx}")
    if Y_p.shape != (n_rx, n_p):
        raise ValueError(f"Y_p shape {Y_p.shape}, expected ({n_rx}, {n_p})")

    U, s, s_eff, eff_rank = u_sigma_from_h_svd(H_oracle, sing_tol)
    r = int(s.size)
    Sigma = np.diag(s.astype(H_oracle.dtype))
    Sigma_inv = np.diag((1.0 / s_eff).astype(H_oracle.dtype))

    UpY = U.conj().T @ Y_p
    Y_prime = Sigma_inv @ UpY
    y_vec = Y_prime.reshape(-1, order="F")

    A = np.kron(P.T, np.eye(r, dtype=H_oracle.dtype))
    sigma_inv_sq = (1.0 / s_eff) ** 2
    C_sigma_inv2 = np.diag(sigma_inv_sq.astype(H_oracle.dtype))
    C_n = float(sigma2) * np.kron(np.eye(n_p, dtype=H_oracle.dtype), C_sigma_inv2)

    nh_v = r * n_tx
    if Cv is None:
        Cv_use = np.eye(nh_v, dtype=H_oracle.dtype)
    else:
        Cv_use = np.asarray(Cv)
        if Cv_use.shape != (nh_v, nh_v):
            raise ValueError(f"C_v shape {Cv_use.shape}, expected ({nh_v}, {nh_v})")

    S = A @ Cv_use @ A.conj().T + C_n
    if ridge > 0:
        S = S + float(ridge) * np.eye(S.shape[0], dtype=S.dtype)
    v_hat = Cv_use @ A.conj().T @ np.linalg.solve(S, y_vec)
    Vh_hat = v_hat.reshape(r, n_tx, order="F")
    if project_v_to_stiefel:
        Vh_hat = project_vh_to_stiefel_columns(Vh_hat)
    H_hat = U @ Sigma @ Vh_hat

    cond_sigma = float(s[0] / (s[-1] + 1e-30))
    cond_sigma_eff = float(s_eff[0] / (s_eff[-1] + 1e-30))

    diag = {
        "eff_rank": eff_rank,
        "r": r,
        "cond_sigma_raw": cond_sigma,
        "cond_sigma_eff": cond_sigma_eff,
        "v_stiefel_projected": bool(project_v_to_stiefel),
    }
    if return_vh:
        return H_hat, diag, Vh_hat
    return H_hat, diag


def gram_v_lmmse_identity(
    H_oracle: np.ndarray,
    Y_p: np.ndarray,
    P: np.ndarray,
    sigma2: float,
    sing_tol: float = 1e-8,
    ridge: float = 1e-10,
    return_vh: bool = False,
    project_v_to_stiefel: bool = True,
) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, np.ndarray]:
    return gram_v_lmmse_estimate_h(
        H_oracle,
        Y_p,
        P,
        sigma2,
        Cv=None,
        sing_tol=sing_tol,
        ridge=ridge,
        return_vh=return_vh,
        project_v_to_stiefel=project_v_to_stiefel,
    )


def gram_v_lmmse_empirical(
    H_oracle: np.ndarray,
    Y_p: np.ndarray,
    P: np.ndarray,
    sigma2: float,
    Cv_empirical: np.ndarray,
    sing_tol: float = 1e-8,
    ridge: float = 1e-10,
    return_vh: bool = False,
    project_v_to_stiefel: bool = True,
) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, np.ndarray]:
    return gram_v_lmmse_estimate_h(
        H_oracle,
        Y_p,
        P,
        sigma2,
        Cv=Cv_empirical,
        sing_tol=sing_tol,
        ridge=ridge,
        return_vh=return_vh,
        project_v_to_stiefel=project_v_to_stiefel,
    )
