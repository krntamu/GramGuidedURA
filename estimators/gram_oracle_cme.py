r"""
Gram-oracle conditional mean estimator for 3GPP (empirical delta prior).

Goal:
    H_hat_oracle(Y, R) = E[H | Y, R],  where R = H H^H (oracle Gram)

Model (matches existing repo conventions):
    vec(H) | delta ~ CN(0, C_delta), with C_delta = C_tx ⊗ C_rx (Kronecker, Toeplitz factors)
    Y = H + Z,  vec(Z) ~ CN(0, sigma2 I)

We approximate the delta-marginalized oracle:
    E[H | Y, R] = E_delta[ E[H | Y, R, delta] | Y, R ]

Conditioning on R is nonlinear. We use the SVD/eigendecomposition parameterization:
    R = U diag(lam) U^H, lam >= 0, rank r
    Sigma = diag(sqrt(lam_1..lam_r))
    Any H consistent with R can be written as:
        H = U Sigma V^H, with V ∈ St(Nt, r) (complex Stiefel manifold), V^H V = I_r.

Then for a fixed delta:
    p(V | Y, R, delta) ∝ exp( -||Y - U Sigma V^H||_F^2 / sigma2 ) * exp( - vec(H)^H C_delta^{-1} vec(H) )

We approximate E[V^H | ...] via importance sampling over V.

Important implementation detail:
The prior quadratic form can be computed efficiently using Kronecker structure:
    vec(H)^H (C_tx ⊗ C_rx)^{-1} vec(H) = tr( C_rx^{-1} H C_tx^{-1} H^H )
and for H = U Sigma V^H this reduces to small r×r matrices:
    tr( C_rx^{-1} H C_tx^{-1} H^H )
      = tr( (Sigma (U^H C_rx^{-1} U) Sigma) * (V^H C_tx^{-1} V) )

This avoids forming 1024×1024 matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

import modules.utils as ut


def _hermitianize(C: np.ndarray) -> np.ndarray:
    return 0.5 * (C + C.conj().T)


def _logdet_psd(C: np.ndarray, eps: float = 1e-10) -> float:
    """
    Log-det for Hermitian PSD-ish matrix using eigenvalues with floor eps.
    """
    C = _hermitianize(C)
    w = np.linalg.eigvalsh(C)
    w = np.maximum(w.real, eps)
    return float(np.sum(np.log(w)))


def _inv_psd_hermitian(C: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float]:
    """
    Stable inverse + logdet for (numerically) Hermitian PSD matrix via eigendecomposition.
    Returns (C_inv, logdet(C)).
    """
    C = _hermitianize(C)
    w, U = np.linalg.eigh(C)
    w = np.maximum(w.real, eps)
    logdet = float(np.sum(np.log(w)))
    inv_w = 1.0 / w
    C_inv = (U * inv_w.reshape(1, -1)) @ U.conj().T
    return C_inv, logdet


def sample_stiefel_haar(n: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """
    Haar-uniform sample on complex Stiefel St(n, r): V is n×r with V^H V = I.
    Uses QR on complex Gaussian.
    """
    G = (rng.standard_normal((n, r)) + 1j * rng.standard_normal((n, r))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(G, mode="reduced")
    d = np.diag(R)
    ph = d / np.maximum(np.abs(d), 1e-12)
    Q = Q * ph.conj()[None, :]
    return Q


def _softmax_logw(logw: np.ndarray) -> np.ndarray:
    lw = logw - np.max(logw)
    w = np.exp(lw)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w) / w.size
    return w / s


def _softmax_logw_tempered(logw: np.ndarray, alpha: float) -> np.ndarray:
    """
    Tempered softmax: w ∝ exp(alpha * logw). alpha in (0,1] reduces weight collapse.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    return _softmax_logw(alpha * logw)


@dataclass
class DeltaParticle:
    # Toeplitz first-columns for Rx and Tx
    t_rx: np.ndarray  # (Nr,)
    t_tx: np.ndarray  # (Nt,)
    log_prior: float = 0.0  # log p(delta) up to const; for empirical prior use 0


@dataclass
class OracleOptions:
    M_v: int = 128          # V samples per delta
    K_delta: int = 8        # delta particles
    rank_tol: float = 1e-6  # relative tolerance for rank selection from Gram eigenvalues
    eps_cov: float = 1e-8   # diagonal jitter for inverses
    proposal: str = "procrustes_mix"  # {"haar","procrustes_mix"}
    proposal_eps: float = 0.25        # mix strength for procrustes proposal (0 -> MAP, 1 -> Haar)
    prior_v_weight: float = 0.0       # weight on log p_prior(V|delta) energy term; 0 disables (sanity)
    prior_q_scale: float = 1.0        # scale factor for q (energy) before applying prior_v_weight
    include_logdet_prior: bool = False  # whether to include -log|C_delta| in delta evidence (often wrong when conditioning on R)
    v_temp: float = 1.0              # tempering for V-importance weights (alpha_V). <1 reduces overconfidence.


def _eigh_rank_from_gram(R: np.ndarray, rank_tol: float) -> tuple[np.ndarray, np.ndarray]:
    """
    R: (Nr,Nr) Hermitian PSD Gram. Return U (Nr,r) and sig (r,) where R ≈ U diag(sig^2) U^H.
    """
    R = _hermitianize(R)
    w, U = np.linalg.eigh(R)
    # sort descending
    idx = np.argsort(w.real)[::-1]
    w = w.real[idx]
    U = U[:, idx]
    w0 = w[0] if w.size else 0.0
    if w0 <= 0:
        return U[:, :0], np.zeros((0,), dtype=float)
    keep = w > (rank_tol * w0)
    r = int(np.sum(keep))
    return U[:, :r], np.sqrt(np.maximum(w[:r], 0.0))


def _procrustes_vh_map(UHY: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    MAP-like Procrustes solution for:
        argmin_{V^H V = I} || U^H Y - Sigma V^H ||_F^2
    where Sigma = diag(s) (r,), UHY is (r,Nt).
    Returns Vh_map with shape (r,Nt).
    """
    r, Nt = UHY.shape
    s_safe = np.maximum(s, 1e-12)
    A = (UHY / s_safe.reshape(r, 1))  # (r,Nt)
    U_a, _, Vh_a = np.linalg.svd(A, full_matrices=False)  # U_a: (r,r), Vh_a: (r,Nt)
    return U_a @ Vh_a  # (r,Nt), row-orthonormal


def _stiefel_from_vh(Vh: np.ndarray) -> np.ndarray:
    """
    Convert Vh (r,Nt) with row-orthonormal into V (Nt,r).
    """
    return Vh.conj().T


def sample_stiefel_procrustes_mix(
    Vh_map: np.ndarray,
    eps: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Proposal around V_map:
      V_raw = (1-eps)*V_map + eps*V_haar, then project back to Stiefel via QR.
    Returns V (Nt,r).
    """
    V_map = _stiefel_from_vh(Vh_map)  # (Nt,r)
    Nt, r = V_map.shape
    V_haar = sample_stiefel_haar(Nt, r, rng)
    V_raw = (1.0 - eps) * V_map + eps * V_haar
    Q, R = np.linalg.qr(V_raw, mode="reduced")
    d = np.diag(R)
    ph = d / np.maximum(np.abs(d), 1e-12)
    Q = Q * ph.conj()[None, :]
    return Q

def _delta_mats_from_toep(t_rx: np.ndarray, t_tx: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build C_rx, C_tx from Toeplitz first columns and return their inverses + logdet(C_delta).
    logdet(C_delta) = Nt*logdet(C_rx) + Nr*logdet(C_tx).
    """
    C_rx = ut.toeplitz(t_rx)
    C_tx = ut.toeplitz(t_tx)
    # Stable PSD projection + inverse/logdet
    C_rx_inv, logdet_rx = _inv_psd_hermitian(C_rx, eps=eps)
    C_tx_inv, logdet_tx = _inv_psd_hermitian(C_tx, eps=eps)
    # Kronecker logdet: log|C_tx ⊗ C_rx| = Nt log|C_rx| + Nr log|C_tx|
    logdet = logdet_rx * C_tx.shape[0] + logdet_tx * C_rx.shape[0]
    return C_rx_inv, C_tx_inv, float(logdet)


def oracle_e_h_given_y_r(
    Y: np.ndarray,
    R: np.ndarray,
    delta_particles: Iterable[DeltaParticle],
    sigma2: float,
    *,
    opts: OracleOptions,
    rng: np.random.Generator,
    return_debug: bool = False,
    true_delta_index: Optional[int] = None,
) -> np.ndarray:
    """
    Approximate E[H | Y, R] by delta-particle marginalization and V-importance sampling.

    Shapes:
      Y: (Nr, Nt) complex
      R: (Nr, Nr) complex Hermitian (oracle Gram)
    """
    Nr, Nt = Y.shape
    U, s = _eigh_rank_from_gram(R, rank_tol=opts.rank_tol)
    r = s.size
    if r == 0:
        return np.zeros_like(Y)

    # Precompute pieces for likelihood (work in r×Nt)
    UHY = U.conj().T @ Y  # (r,Nt)
    # constant part ||(I-UU^H)Y||^2 is ignored (independent of V)

    # Precompute V samples once. Same proposal for all deltas.
    if opts.proposal == "haar":
        V_samps = [sample_stiefel_haar(Nt, r, rng) for _ in range(opts.M_v)]
    elif opts.proposal == "procrustes_mix":
        Vh_map = _procrustes_vh_map(UHY, s)
        V_samps = [sample_stiefel_procrustes_mix(Vh_map, eps=opts.proposal_eps, rng=rng) for _ in range(opts.M_v)]
    else:
        raise ValueError(f"Unknown proposal: {opts.proposal}")

    # Precompute Sigma scaling
    # Sigma V^H => (r,Nt) where rows scaled by s
    s_row = s.reshape(r, 1)

    H_hats = []
    log_evidences = []

    for dp in delta_particles:
        C_rx_inv, C_tx_inv, logdet_Cdelta = _delta_mats_from_toep(dp.t_rx, dp.t_tx, eps=opts.eps_cov)
        # IMPORTANT: for column-stacked vec(H) with Kronecker covariance (C_tx ⊗ C_rx),
        # the quadratic form uses C_tx^{-T} (not C_tx^{-1}) in the trace identity:
        #   vec(H)^H (C_tx^{-1} ⊗ C_rx^{-1}) vec(H) = tr(C_rx^{-1} H C_tx^{-T} H^H)
        C_tx_inv_T = C_tx_inv.T

        # Small matrix for Rx side: A = U^H C_rx^{-1} U  (r×r)
        A = U.conj().T @ C_rx_inv @ U
        # G = Sigma A Sigma (r×r)
        G = (s.reshape(-1, 1) * A) * s.reshape(1, -1)

        logw = np.zeros(opts.M_v, dtype=float)
        Vh_list = []
        ll_list = []
        q_list = []
        for m, V in enumerate(V_samps):
            Vh = V.conj().T  # (r,Nt)
            Vh_list.append(Vh)

            # Likelihood term (up to const): -||U^H Y - Sigma V^H||^2 / sigma2
            diff = UHY - (s_row * Vh)
            ll = -float(np.sum(np.abs(diff) ** 2)) / float(sigma2)

            # Prior quadratic form using Kronecker structure:
            # q = tr( G * (V^H C_tx^{-1} V) ) = tr(G @ B)
            B = V.conj().T @ C_tx_inv_T @ V  # (r,r)
            q = float(np.real(np.trace(G @ B)))
            q_scaled = q * float(opts.prior_q_scale)
            lp = -opts.prior_v_weight * q_scaled

            logw[m] = ll + lp
            ll_list.append(ll)
            q_list.append(q_scaled)

        alpha = _softmax_logw_tempered(logw, alpha=opts.v_temp)
        EVh = np.zeros((r, Nt), dtype=complex)
        for m in range(opts.M_v):
            EVh += alpha[m] * Vh_list[m]
        H_hat_delta = (U * s.reshape(1, -1)) @ EVh  # U Sigma EVh

        # Evidence for delta: approx mean exp(logw) (up to constants).
        # NOTE: Including -log|C_delta| is correct for the *unconditioned* Gaussian prior density,
        # but after conditioning on R=H H^H the induced normalization on V generally changes.
        log_mean_exp = float(np.max(logw) + np.log(np.mean(np.exp(logw - np.max(logw)))))
        log_evidence = dp.log_prior + log_mean_exp - (logdet_Cdelta if opts.include_logdet_prior else 0.0)

        H_hats.append(H_hat_delta)
        log_evidences.append(log_evidence)

    w_delta = _softmax_logw(np.asarray(log_evidences, dtype=float))
    H_hat = np.zeros_like(Y)
    for k, Hk in enumerate(H_hats):
        H_hat += w_delta[k] * Hk
    if not return_debug:
        return H_hat

    dbg: dict = {
        "rank_r": int(r),
        "sigma2": float(sigma2),
        "w_delta": w_delta.copy(),
        "log_evidences": np.asarray(log_evidences, dtype=float),
    }
    if true_delta_index is not None and 0 <= true_delta_index < w_delta.size:
        w_true = float(w_delta[true_delta_index])
        rank_true = int(1 + np.sum(w_delta > w_true))
        dbg["w_true_delta"] = w_true
        dbg["rank_true_delta"] = rank_true
        dbg["true_is_argmax"] = bool(int(np.argmax(w_delta)) == int(true_delta_index))
    return H_hat, dbg


def make_delta_particles_from_empirical_toep(
    toep_train_rx: np.ndarray,
    toep_train_tx: np.ndarray,
    *,
    K: int,
    rng: np.random.Generator,
) -> list[DeltaParticle]:
    """
    Empirical prior over delta from training Toeplitz params.
    """
    idx = rng.choice(toep_train_rx.shape[0], size=K, replace=True)
    parts = []
    for i in idx:
        parts.append(DeltaParticle(t_rx=toep_train_rx[i].copy(), t_tx=toep_train_tx[i].copy(), log_prior=0.0))
    return parts


