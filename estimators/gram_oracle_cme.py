r"""
Gram-oracle conditional mean estimator for 3GPP (empirical delta prior).

Goal:
    H_hat_oracle(Y, R) = E[H | Y, R],  where R is an (N_r × N_r) Gram used to build U.
    By default R = H_true H_true^H (oracle).  Alternatively, build R from the **same sample
    covariance C as Scov-LMMSE**: C = (1/M) Σ vec(H_m) vec(H_m)^H (column-major vec), then
    R = Σ_t C^{(t)} where C^{(t)} is the t-th N_r×N_r block on the block diagonal of C
    (equivalently R_pq = Σ_t C_{p+N_r t, q+N_r t} = E[(H H^H)_pq] under that C).

Two modes depending on the dataset type:

── Pilot observation Y_p = H X_p + N (AWGN per entry, variance σ²) ───────────
    Reduced: Ŷ = Uᴴ Y_p,  vec(Ŷ) = A_r s + ñ,  A_r = X_pᵀ ⊗ I_r,  H = U S.
    Prior blocks Ψ_k = Uᴴ C_rx,k U  →  Σ_S = blkdiag(Ψ_1,…,Ψ_{Nt}) (independent cols)
    or Ψ repeated (shared C_rx).  MMSE: ŝ = Σ_S A_rᴴ (A_r Σ_S A_rᴴ + σ²I)⁻¹ ỹ.

    PilotConfig: identity | orthonormal_dft | gaussian | hybrid (√γ I + √(1−γ) G) | matrix.

── Independent-columns + X_p = I (legacy): per-column ANALYTICAL ───────────
    Same as blkdiag MMSE with A_r = I:  ẑ_k = S_k (S_k + σ² I)⁻¹ u_k,  ĥ_k = U ẑ_k.

── Kronecker + X_p = I only: STIEFEL MONTE CARLO ─────────────────────────────
    vec(H) | delta ~ CN(0, C_tx ⊗ C_rx); IS over V in H = U Σ Vᴴ.
    Non-identity pilots use reduced vector MMSE (approximation if single shared Ψ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional

import numpy as np

import modules.utils as ut
from modules.pilot_matrix import row_normalize_pilot_numpy


@dataclass
class PilotConfig:
    """
    Pilot matrix X_p ∈ C^{N_t × T} in Y_p = H X_p + N (elementwise AWGN variance σ²).

    vec(Ŷ) = A_r s + ñ with Ŷ = U^H Y_p, s = vec(S), H = U S, A_r = X_p^T ⊗ I_r.

    - identity: X_p = I_{N_t} (requires T = N_t); recovers Y = H + Z when T = N_t.
    - orthonormal_dft: unitary DFT matrix (T = N_t).
    - gaussian: i.i.d. CN(0,1) entries (not normalized orthogonal).
    - hybrid: X_p = √γ I + √(1-γ) G with standard complex Gaussian G (square N_t × N_t).
    - matrix: supply explicit X_p via make_pilot_matrix(..., kind=\"matrix\", X_p=...) or pass X_p into oracle.
    """

    kind: str = "identity"
    hybrid_gamma: float = 1.0
    power_norm: str = "legacy"  # legacy | align_i | row_norm (see make_pilot_matrix)
    T: Optional[int] = None  # pilot length; default N_t for built-in kinds
    X_p: Optional[np.ndarray] = None  # used when kind == "matrix"


def make_pilot_matrix(
    Nt: int,
    pilot: PilotConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build X_p (Nt × T)."""
    T = int(pilot.T) if pilot.T is not None else Nt
    kind = pilot.kind.lower()
    pnorm = str(getattr(pilot, "power_norm", "legacy")).lower()

    if kind == "matrix":
        if pilot.X_p is None:
            raise ValueError("PilotConfig.kind=='matrix' requires PilotConfig.X_p set")
        X = np.asarray(pilot.X_p, dtype=np.complex128)
        if X.shape != (Nt, T):
            raise ValueError(f"X_p shape {X.shape} expected ({Nt}, {T})")
        return X

    if kind == "identity":
        if T != Nt:
            raise ValueError("identity pilot requires T == N_t")
        return np.eye(Nt, dtype=np.complex128)

    if kind == "orthonormal_dft":
        if T != Nt:
            raise ValueError("orthonormal_dft requires T == N_t")
        F = np.fft.fft(np.eye(Nt, dtype=np.complex128), axis=0)
        return (F / np.sqrt(float(Nt))).astype(np.complex128)

    if kind == "gaussian":
        G = (rng.standard_normal((Nt, T)) + 1j * rng.standard_normal((Nt, T))) / np.sqrt(2.0)
        if pnorm in ("align_i", "row_norm"):
            # For CN(0,1): E[G G^H] = T * I. Scale by 1/sqrt(T) => E[X_p X_p^H] = I.
            G = G * (1.0 / np.sqrt(float(T)))
        X = G.astype(np.complex128)
        if pnorm == "row_norm":
            X = row_normalize_pilot_numpy(X)
        return X

    if kind == "hybrid":
        if T != Nt:
            raise ValueError("hybrid pilot uses square N_t × N_t (set T=N_t)")
        g = float(np.clip(pilot.hybrid_gamma, 0.0, 1.0))
        G = (rng.standard_normal((Nt, Nt)) + 1j * rng.standard_normal((Nt, Nt))) / np.sqrt(2.0)
        if pnorm in ("align_i", "row_norm"):
            # Scale only the Gaussian component so that when γ=0, E[X_p X_p^H]=I.
            G = G * (1.0 / np.sqrt(float(Nt)))
        X = (np.sqrt(g) * np.eye(Nt, dtype=np.complex128) + np.sqrt(max(0.0, 1.0 - g)) * G).astype(
            np.complex128
        )
        if pnorm == "row_norm":
            X = row_normalize_pilot_numpy(X)
        return X

    raise ValueError(f"Unknown PilotConfig.kind={pilot.kind!r}")


def simulate_pilot_observation(
    H: np.ndarray,
    X_p: np.ndarray,
    sigma2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Y_p = H X_p + N with N_ij i.i.d. CN(0, σ²) (complex variance σ²).
    """
    Nr, Nt = H.shape
    if X_p.shape[0] != Nt:
        raise ValueError(f"X_p rows {X_p.shape[0]} != N_t {Nt}")
    noise = np.sqrt(sigma2 / 2.0) * (
        rng.standard_normal((Nr, X_p.shape[1])) + 1j * rng.standard_normal((Nr, X_p.shape[1]))
    )
    return (H @ X_p + noise.astype(H.dtype)).astype(np.complex128, copy=False)


def _is_identity_pilot(X_p: np.ndarray, Nt: int, tol: float = 1e-8) -> bool:
    return X_p.shape == (Nt, Nt) and bool(np.allclose(X_p, np.eye(Nt, dtype=X_p.dtype), atol=tol))


def gram_cme_reduced_mmse(
    Y_p: np.ndarray,
    U: np.ndarray,
    X_p: np.ndarray,
    sigma2: float,
    psi_blocks: List[np.ndarray],
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """
    Gram-oracle conditional MMSE in reduced coordinates (user's Gram-CME).

    ŝ = Σ_S A_r^H (A_r Σ_S A_r^H + σ² I)^{-1} ỹ,  H_hat = U unvec(ŝ),
    with A_r = X_p^T ⊗ I_r, ỹ = vec(U^H Y_p), Σ_S = blkdiag(Ψ_1,…,Ψ_{N_t}).
    """
    Nr, T = Y_p.shape
    Nt, T2 = X_p.shape
    if T != T2:
        raise ValueError(f"Y_p columns {T} != X_p columns {T2}")
    if len(psi_blocks) != Nt:
        raise ValueError(f"psi_blocks length {len(psi_blocks)} != N_t {Nt}")
    r = U.shape[1]
    if r == 0:
        return np.zeros((Nr, Nt), dtype=np.complex128), 0.0

    Y_t = U.conj().T @ Y_p
    y_tilde = Y_t.reshape(-1, order="F")
    A_r = np.kron(X_p.T.astype(np.complex128), np.eye(r, dtype=np.complex128))

    rNt = Nt * r
    Sigma_S = np.zeros((rNt, rNt), dtype=np.complex128)
    for k, Pk in enumerate(psi_blocks):
        Sigma_S[k * r : (k + 1) * r, k * r : (k + 1) * r] = Pk

    Mmat = _hermitianize(A_r @ Sigma_S @ A_r.conj().T + sigma2 * np.eye(r * T, dtype=np.complex128))
    rhs = np.linalg.solve(Mmat, y_tilde)
    s_hat = Sigma_S @ A_r.conj().T @ rhs
    S_mat = s_hat.reshape((r, Nt), order="F")
    H_hat = (U @ S_mat).astype(np.complex128, copy=False)

    ew, _ = np.linalg.eigh(Mmat)
    ew = np.maximum(ew.real, eps)
    logdet = float(np.sum(np.log(ew)))
    quad = float(np.real(np.vdot(y_tilde, rhs)))
    log_ev = -quad - logdet
    return H_hat, log_ev


def _hermitianize(C: np.ndarray) -> np.ndarray:
    return 0.5 * (C + C.conj().T)


def sample_vec_covariance_scov(
    h_train_flat: np.ndarray,
) -> np.ndarray:
    """
    Sample covariance of vec(H), identical to baselines ``sample_channel_covariance_from_vec``:

        C = (1/M) Σ_m v_m v_m^H,   v_m = vec(H_m) column-major (same as ``H.reshape(-1, order='F')``).

    ``h_train_flat`` rows must be those vectors, shape (M, N_r * N_t).
    """
    V = np.asarray(h_train_flat, dtype=np.complex128)
    if V.ndim != 2:
        raise ValueError(f"h_train_flat must be 2D (M, N_r N_t), got shape {V.shape}")
    M = int(V.shape[0])
    if M == 0:
        raise ValueError("empty training set for Scov covariance")
    return _hermitianize(V.conj().T @ V / float(M))


def rx_gram_from_vec_channel_covariance(
    C: np.ndarray,
    *,
    n_rx: int,
    n_tx: int,
) -> np.ndarray:
    """
    Map Scov-LMMSE covariance C (N_h × N_h, N_h = N_r N_t) to an N_r × N_r matrix R for Gram-CME.

    With column-major vec, index i = p + N_r * t for receive row p and transmit column t. Then
    (H H^H)_pq = Σ_t H_{p,t} H_{q,t}^* and

        E[(H H^H)_pq] = Σ_t C_{p + N_r t, q + N_r t},

    i.e. R is the sum of the N_t diagonal N_r × N_r blocks of C in that partition.
    """
    nh = int(n_rx) * int(n_tx)
    C = np.asarray(C, dtype=np.complex128)
    C = _hermitianize(C)
    if C.shape != (nh, nh):
        raise ValueError(f"C shape {C.shape}, expected ({nh}, {nh}) for N_r={n_rx}, N_t={n_tx}")
    R = np.zeros((n_rx, n_rx), dtype=np.complex128)
    for t in range(int(n_tx)):
        sl = slice(n_rx * t, n_rx * (t + 1))
        R += C[sl, sl]
    return _hermitianize(R)


def rx_gram_from_scov_training_channels(
    h_train_flat: np.ndarray,
    *,
    n_rx: int,
    n_tx: int,
) -> np.ndarray:
    """
    C from training (Scov-LMMSE recipe), then R = rx_gram_from_vec_channel_covariance(C, ...).
    """
    C = sample_vec_covariance_scov(h_train_flat)
    return rx_gram_from_vec_channel_covariance(C, n_rx=n_rx, n_tx=n_tx)


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


def _psi_blocks_from_delta(
    U: np.ndarray,
    dp: DeltaParticle,
    Nt: int,
    eps: float,
) -> List[np.ndarray]:
    """Ψ_k = U^H C_rx,k U (each r×r). Kronecker 1D t_rx → same Ψ repeated N_t times."""
    r = U.shape[1]
    if dp.t_rx.ndim == 2:
        blocks: List[np.ndarray] = []
        for k in range(Nt):
            Ck = ut.toeplitz(dp.t_rx[k])
            Pk = _hermitianize(U.conj().T @ Ck @ U)
            w, V = np.linalg.eigh(Pk)
            w = np.maximum(w.real, eps)
            blocks.append((V * w) @ V.conj().T)
        return blocks
    C_rx = ut.toeplitz(dp.t_rx)
    Psi = _hermitianize(U.conj().T @ C_rx @ U)
    w, V = np.linalg.eigh(Psi)
    w = np.maximum(w.real, eps)
    Psi = (V * w) @ V.conj().T
    return [Psi.copy() for _ in range(Nt)]


@dataclass
class OracleOptions:
    M_v: int = 128          # V samples per delta
    K_delta: int = 8        # delta particles
    rank_tol: float = 1e-6  # relative tolerance for rank selection from Gram eigenvalues
    eps_cov: float = 1e-8   # diagonal jitter for inverses
    proposal: str = "procrustes_mix"  # {"haar","procrustes_mix"}
    proposal_eps: float = 0.25        # mix strength for procrustes proposal (0 -> MAP, 1 -> Haar)
    prior_v_weight: float = 1.0       # weight on log p_prior(V|delta) energy term (MUST be 1.0 for true CME)
    prior_q_scale: float = 1.0        # scale factor for q (energy) before applying prior_v_weight
    include_logdet_prior: bool = False  # whether to include -log|C_delta| in delta evidence (often wrong when conditioning on R)
    v_temp: float = 1.0              # tempering for V-importance weights (alpha_V). <1 reduces overconfidence.
    # After δ-mixture: Ŝ = Uᴴ Ĥ, then Gram projection so S_proj S_projᴴ ≈ Σ²
    #   wr: whitening–recoloring S_proj = Σ (Ŝ Ŝᴴ + ε I)^{-1/2} Ŝ
    #   fro: Frobenius / Procrustes S_proj = Σ · polar(Σ Ŝ) via SVD of M = Σ Ŝ
    project_s_to_sigma: bool = False
    project_s_eps: float = 1e-8      # ε in C_S = Ŝ Ŝᴴ + ε I (wr); ignored by fro unless used as SVD jitter
    gram_proj_method: Literal["wr", "fro"] = "wr"
    gram_proj_fro_svd_eps: float = 0.0  # if >0, add to every entry of M=ΣŜ before SVD (stability)


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


def project_s_hat_to_sigma_squared(
    S_hat: np.ndarray,
    sigma_vec: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Post-process reduced coefficients S_hat (r × N_t) so that S_proj S_proj^H ≈ Σ²,
    where Σ = diag(σ) and σ are singular values from the oracle Gram
    (R = U Σ² Uᴴ, same σ as in H = U Σ Vᴴ up to rank).

    Let C_S = S_hat S_hatᴴ + ε I (ε for numerical rank). Then
        S_proj = Σ C_S^{-1/2} S_hat
    satisfies S_proj S_projᴴ = Σ² when C_S = S_hat S_hatᴴ is full rank and ε = 0.
    """
    S_hat = np.asarray(S_hat, dtype=np.complex128)
    sigma_vec = np.asarray(sigma_vec, dtype=np.float64).real
    r = S_hat.shape[0]
    if r == 0:
        return S_hat
    C_S = _hermitianize(S_hat @ S_hat.conj().T) + float(eps) * np.eye(r, dtype=np.complex128)
    w, V = np.linalg.eigh(C_S)
    w = np.maximum(w.real, float(eps))
    inv_sqrt = (V * (1.0 / np.sqrt(w))) @ V.conj().T
    Sig = np.diag(np.maximum(sigma_vec, 0.0))
    return (Sig @ inv_sqrt @ S_hat).astype(np.complex128, copy=False)


def project_s_hat_to_sigma_frobenius(
    S_hat: np.ndarray,
    sigma_vec: np.ndarray,
    *,
    fro_svd_eps: float = 0.0,
) -> np.ndarray:
    """
    Frobenius-norm Procrustes projection onto {S : S Sᴴ = Σ²} with Σ = diag(σ).

    Let M = Σ Ŝ. With SVD M = U Λ Vᴴ, the unitary polar factor is Q = U Vᴴ and
        S_proj = Σ Q = Σ U Vᴴ
    minimizes ‖S_proj − Ŝ‖_F subject to S_proj S_projᴴ = Σ² when rank conditions
    match the usual orthogonal Procrustes setup (see e.g. tall Σ Ŝ with full row rank).

    If fro_svd_eps > 0, replace M by M + fro_svd_eps (scalar shift to all entries) before SVD.
    """
    S_hat = np.asarray(S_hat, dtype=np.complex128)
    sigma_vec = np.asarray(sigma_vec, dtype=np.float64).real
    r = S_hat.shape[0]
    if r == 0:
        return S_hat
    Sig = np.diag(np.maximum(sigma_vec, 0.0))
    M = Sig @ S_hat
    if fro_svd_eps > 0:
        M = M + float(fro_svd_eps) * (1.0 + 0.0j)
    Uv, _, Vh = np.linalg.svd(M, full_matrices=False)
    return (Sig @ Uv @ Vh).astype(np.complex128, copy=False)


GramProjMethod = Literal["wr", "fro"]


def project_h_to_sigma_gram(
    H_hat: np.ndarray,
    U: np.ndarray,
    sigma_vec: np.ndarray,
    *,
    eps: float = 1e-8,
    method: GramProjMethod = "wr",
    fro_svd_eps: float = 0.0,
) -> np.ndarray:
    """Ĥ_proj = U S_proj with S_proj from whitening–recoloring (wr) or Frobenius polar (fro)."""
    if sigma_vec.size == 0 or U.shape[1] == 0:
        return H_hat
    S_hat = U.conj().T @ H_hat
    if method == "wr":
        S_proj = project_s_hat_to_sigma_squared(S_hat, sigma_vec, eps=eps)
    elif method == "fro":
        S_proj = project_s_hat_to_sigma_frobenius(S_hat, sigma_vec, fro_svd_eps=fro_svd_eps)
    else:
        raise ValueError(f"Unknown gram projection method: {method!r}")
    return (U @ S_proj).astype(np.complex128, copy=False)


def project_h_to_sigma_gram_from_R(
    H_hat: np.ndarray,
    R: np.ndarray,
    *,
    rank_tol: float = 1e-6,
    eps: float = 1e-8,
    method: GramProjMethod = "wr",
    fro_svd_eps: float = 0.0,
) -> np.ndarray:
    """Recompute (U, σ) from R and apply project_h_to_sigma_gram."""
    U, s = _eigh_rank_from_gram(R, rank_tol=rank_tol)
    return project_h_to_sigma_gram(
        H_hat, U, s, eps=eps, method=method, fro_svd_eps=fro_svd_eps
    )


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
    # The true Procrustes solution for argmin ||A - V^H|| is from SVD of A
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


def _analytical_col_estimate(
    U: np.ndarray,
    S_k: np.ndarray,
    u_k: np.ndarray,
    sigma2: float,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """
    Column MMSE given oracle subspace U from R = H Hᴴ.

    Let z_k = Uᴴ h_k. Under h_k ~ CN(0, C_rx_k) and h_k ∈ span(U):
        Cov(z_k) = S_k = Uᴴ C_rx_k U   (r×r),  NOT (Uᴴ C_rx_k⁻¹ U)⁻¹.

    Observation u_k = Uᴴ y_k = z_k + ñ with ñ ~ CN(0, σ² I_r). Then
        ẑ_k = S_k (S_k + σ² I)⁻¹ u_k,   ĥ_k = U ẑ_k

    This matches full-vector LMMSE C_rx (C_rx + σ² I)⁻¹ y when projected; using
    A_k = Uᴴ C_rx⁻¹ U and (I + σ² A_k)⁻¹ is WRONG except when S_k = A_k⁻¹.

    Log-likelihood of u under CN(0, S_k + σ² I) (up to constants):
        −uᴴ(S_k+σ²I)⁻¹u − log|S_k+σ²I|
    """
    S_k = _hermitianize(S_k)
    d, Q = np.linalg.eigh(S_k)
    d = np.maximum(d.real, eps)

    Qu = Q.conj().T @ u_k
    # ẑ = Q diag(d/(d+σ²)) Qᴴ u
    z_hat = Q @ ((d / (d + sigma2)) * Qu)
    h_k_hat = U @ z_hat

    eff = d + sigma2
    quad = float(np.real(np.sum(np.abs(Qu) ** 2 / eff)))
    log_lik = -quad - float(np.sum(np.log(eff)))
    return h_k_hat, log_lik


def oracle_e_h_given_y_r(
    Y: np.ndarray,
    R: np.ndarray,
    delta_particles: Iterable[DeltaParticle],
    sigma2: float,
    *,
    opts: OracleOptions,
    rng: np.random.Generator,
    X_p: Optional[np.ndarray] = None,
    pilot: Optional[PilotConfig] = None,
    return_debug: bool = False,
    true_delta_index: Optional[int] = None,
) -> np.ndarray:
    """
    Compute E[H | Y_p, R] marginalised over delta particles.

    Shapes:
      Y (Y_p): (Nr, T)  pilot observation; T = N_t for identity / hybrid / DFT pilots.
      R: (Nr, Nr) Hermitian PSD Gram (oracle H H^H or sample-mean R̂ from training).
    If X_p and pilot are both None: X_p = I_{N_t}, Y must be (Nr, N_t) — legacy Y = H + Z.

    Stiefel IS is used only for Kronecker (1D t_rx) AND orthonormal identity pilot X_p = I.
    Otherwise the reduced Gram-MMSE with A_r = X_p^T ⊗ I_r is used.
    """
    delta_particles = list(delta_particles)
    if len(delta_particles) == 0:
        raise ValueError("delta_particles must be non-empty")

    dp0 = delta_particles[0]
    if dp0.t_rx.ndim == 2:
        Nt = int(dp0.t_rx.shape[0])
    else:
        Nt = int(np.asarray(dp0.t_tx).ravel().shape[0])

    if pilot is not None and X_p is None:
        X_p = make_pilot_matrix(Nt, pilot, rng)
    elif X_p is not None:
        X_p = np.asarray(X_p, dtype=np.complex128)
        Nt = int(X_p.shape[0])
    else:
        X_p = np.eye(Y.shape[1], dtype=np.complex128)
        Nt = Y.shape[1]

    Y_p = np.asarray(Y, dtype=np.complex128)
    Nr = Y_p.shape[0]
    if Y_p.shape[1] != X_p.shape[1]:
        raise ValueError(f"Y_p shape {Y_p.shape} incompatible with X_p {X_p.shape}")

    U, s = _eigh_rank_from_gram(R, rank_tol=opts.rank_tol)
    r = s.size
    if r == 0:
        H0 = np.zeros((Nr, Nt), dtype=np.complex128)
        if not return_debug:
            return H0
        return H0, {"rank_r": 0, "sigma2": float(sigma2), "w_delta": np.array([]), "log_evidences": np.array([])}

    UHY = U.conj().T @ Y_p  # (r, T) — only used for Stiefel / fast identity path

    is_independent_cols_mode = dp0.t_rx.ndim == 2
    use_stiefel = (not is_independent_cols_mode) and _is_identity_pilot(X_p, Nt)

    if use_stiefel:
        s_row = s.reshape(r, 1)
        if opts.proposal == "haar":
            V_samps = [sample_stiefel_haar(Nt, r, rng) for _ in range(opts.M_v)]
        elif opts.proposal == "procrustes_mix":
            Vh_map = _procrustes_vh_map(UHY, s)
            V_samps = [
                sample_stiefel_procrustes_mix(Vh_map, eps=opts.proposal_eps, rng=rng)
                for _ in range(opts.M_v)
            ]
        else:
            raise ValueError(f"Unknown proposal: {opts.proposal}")

    H_hats: list[np.ndarray] = []
    log_evidences: list[float] = []

    for dp in delta_particles:
        is_independent_cols = dp.t_rx.ndim == 2

        if use_stiefel:
            C_rx_inv, C_tx_inv, logdet_Cdelta = _delta_mats_from_toep(
                dp.t_rx, dp.t_tx, eps=opts.eps_cov
            )
            C_tx_inv_T = C_tx_inv.T
            A = U.conj().T @ C_rx_inv @ U
            G = (s.reshape(-1, 1) * A) * s.reshape(1, -1)

            logw = np.zeros(opts.M_v, dtype=float)
            Vh_list = []
            for m, V in enumerate(V_samps):
                Vh = V.conj().T
                Vh_list.append(Vh)
                diff = UHY - (s_row * Vh)
                ll = -float(np.sum(np.abs(diff) ** 2)) / float(sigma2)
                B = Vh @ C_tx_inv_T @ V
                q = float(np.real(np.trace(G @ B))) * float(opts.prior_q_scale)
                logw[m] = ll - opts.prior_v_weight * q

            alpha = _softmax_logw_tempered(logw, alpha=opts.v_temp)
            EVh = np.zeros((r, Nt), dtype=complex)
            for m in range(opts.M_v):
                EVh += alpha[m] * Vh_list[m]
            H_hat_delta = (U * s.reshape(1, -1)) @ EVh

            log_mean_exp = float(
                np.max(logw) + np.log(np.mean(np.exp(logw - np.max(logw))))
            )
            log_evidence = (
                dp.log_prior
                + log_mean_exp
                - (logdet_Cdelta if opts.include_logdet_prior else 0.0)
            )

        elif is_independent_cols and _is_identity_pilot(X_p, Nt):
            H_hat_delta = np.zeros((Nr, Nt), dtype=complex)
            log_evidence = float(dp.log_prior)
            for k in range(Nt):
                C_rx_k = ut.toeplitz(dp.t_rx[k])
                S_k = U.conj().T @ C_rx_k @ U
                h_k_hat, log_lik_k = _analytical_col_estimate(
                    U, S_k, UHY[:, k], sigma2, eps=opts.eps_cov
                )
                H_hat_delta[:, k] = h_k_hat
                log_evidence += log_lik_k

        else:
            psi_bl = _psi_blocks_from_delta(U, dp, Nt, opts.eps_cov)
            H_hat_delta, log_ev = gram_cme_reduced_mmse(
                Y_p, U, X_p, sigma2, psi_bl, eps=opts.eps_cov
            )
            log_evidence = float(dp.log_prior) + log_ev

        H_hats.append(H_hat_delta)
        log_evidences.append(log_evidence)

    w_delta = _softmax_logw(np.asarray(log_evidences, dtype=float))
    H_hat = np.zeros((Nr, Nt), dtype=np.complex128)
    for k, Hk in enumerate(H_hats):
        H_hat += w_delta[k] * Hk

    if opts.project_s_to_sigma and r > 0:
        H_hat = project_h_to_sigma_gram(
            H_hat,
            U,
            s,
            eps=float(opts.project_s_eps),
            method=opts.gram_proj_method,
            fro_svd_eps=float(opts.gram_proj_fro_svd_eps),
        )

    if not return_debug:
        return H_hat

    dbg: dict = {
        "rank_r": int(r),
        "sigma2": float(sigma2),
        "w_delta": w_delta.copy(),
        "log_evidences": np.asarray(log_evidences, dtype=float),
        "project_s_to_sigma": bool(opts.project_s_to_sigma),
        "gram_proj_method": str(opts.gram_proj_method),
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


