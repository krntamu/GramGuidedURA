r"""
Approximate CME baseline E[h | y] for 3GPP using importance sampling over per-sample
Toeplitz covariance parameters ("delta") drawn from the TRAIN set (empirical prior).

Model assumed (matches existing code):
  h | delta ~ CN(0, C_delta)
  y = h + n,  n ~ CN(0, sigma^2 I)

Then:
  mu(delta, y) = E[h | y, delta] = C_delta (C_delta + sigma^2 I)^{-1} y
  E[h | y] = \int mu(delta,y) p(delta | y) d delta

We approximate with delta_s ~ p_hat(delta) from training Toeplitz params:
  h_CME(y) ≈ sum_s w_s * mu(delta_s, y),
  w_s ∝ p(y | delta_s) = CN(y; 0, C_delta_s + sigma^2 I)

This is NOT the "true oracle" CME unless you have explicit p(delta).
It is a practical CME-style baseline under the empirical prior from training data.
"""

from __future__ import annotations

import argparse
import numpy as np

import modules.utils as ut
from estimators.lmmse import LMMSE


def _softmax_logweights(logw: np.ndarray) -> np.ndarray:
    lw = logw - np.max(logw)
    w = np.exp(lw)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        # Fallback: uniform weights
        return np.ones_like(w) / w.size
    return w / s


def _softmax_logweights_tempered(logw: np.ndarray, alpha: float) -> np.ndarray:
    """
    Tempered softmax: w ∝ exp(alpha * logw). For alpha in (0,1], this reduces weight collapse.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    return _softmax_logweights(alpha * logw)


def ess_from_weights(w: np.ndarray) -> float:
    """Effective sample size: 1/sum w^2 (for normalized weights)."""
    s2 = float(np.sum(np.square(w)))
    if s2 <= 0 or not np.isfinite(s2):
        return 0.0
    return float(1.0 / s2)


def _eig_psd_hermitian(C: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigen-decomposition of (numerically) Hermitian PSD matrix.
    Returns eigenvectors U and clipped eigenvalues lam (>= eps).
    """
    # Force Hermitian symmetry (numerical guard)
    C = 0.5 * (C + C.conj().T)
    lam, U = np.linalg.eigh(C)
    lam = np.maximum(lam.real, eps)
    return U, lam


def _toeplitz_first_col_from_mat(M: np.ndarray) -> np.ndarray:
    """
    Estimate the first column c of a (Hermitian) Toeplitz matrix by averaging diagonals.
    c[k] = mean(diag(M, -k)), k=0..n-1  (so c[0] is the main diagonal mean).
    """
    n = M.shape[0]
    c = np.zeros(n, dtype=M.dtype)
    for k in range(n):
        c[k] = np.mean(np.diag(M, -k))
    return c


def _moment_delta_hat_from_y(y_vec: np.ndarray, n_rx: int, n_tx: int, sigma2: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Rough moment-based estimate of delta = (t_rx, t_tx) from a single observation y = vec(Y):

      Y ≈ H + N,  vec(H) ~ CN(0, C_tx ⊗ C_rx),  vec(N) ~ CN(0, sigma2 I)

    For Kronecker covariance, E[H H^H] = Tr(C_tx) C_rx and E[H^H H] = Tr(C_rx) C_tx.
    We use sample analogs from a single Y to get a *shape* estimate (up to scaling):
      R_rx_hat ∝ (Y Y^H)/n_tx - sigma2 I
      R_tx_hat ∝ (Y^H Y)/n_rx - sigma2 I
    then project to Toeplitz by diagonal-averaging.
    """
    Y = np.reshape(y_vec, (n_rx, n_tx), order="F")

    Rrx = (Y @ Y.conj().T) / float(n_tx) - sigma2 * np.eye(n_rx, dtype=Y.dtype)
    Rtx = (Y.conj().T @ Y) / float(n_rx) - sigma2 * np.eye(n_tx, dtype=Y.dtype)

    t_rx = _toeplitz_first_col_from_mat(Rrx)
    t_tx = _toeplitz_first_col_from_mat(Rtx)
    return t_rx, t_tx


def _normalize_toep(t: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a Toeplitz first column by its zero-lag (power) to focus on correlation shape."""
    denom = np.abs(t[0])
    if not np.isfinite(denom) or denom < eps:
        return t
    return t / denom


def _kron_gaussian_terms(
    y_vec: np.ndarray,
    U_rx: np.ndarray,
    lam_rx: np.ndarray,
    U_tx: np.ndarray,
    lam_tx: np.ndarray,
    sigma2: float,
) -> tuple[float, np.ndarray]:
    """
    For Sigma = (C_tx ⊗ C_rx) + sigma2 I with eig(C_rx)=U_rx*diag(lam_rx)*U_rx^H and
    eig(C_tx)=U_tx*diag(lam_tx)*U_tx^H, compute:
      -logdet(Sigma) - y^H Sigma^{-1} y  (up to additive constants)
    and mu = C (C + sigma2 I)^{-1} y.

    Assumes y_vec corresponds to vec(Y) with Y shape (n_rx, n_tx) in Fortran order,
    consistent with np.kron(C_tx, C_rx).
    """
    n_rx = lam_rx.size
    n_tx = lam_tx.size
    Y = np.reshape(y_vec, (n_rx, n_tx), order="F")

    # z = U^H y in Kronecker basis:
    # (U_tx^H ⊗ U_rx^H) vec(Y) = vec(U_rx^H Y U_tx^*)
    Z = (U_rx.conj().T @ Y) @ (U_tx.conj())

    # Eigenvalues of kron(C_tx, C_rx) are outer products
    Lam = np.outer(lam_rx, lam_tx)  # (n_rx, n_tx)
    Den = Lam + sigma2

    # log p(y|delta) up to constant: -logdet(Sigma) - y^H Sigma^{-1} y
    logdet = np.sum(np.log(Den))
    quad = np.sum((np.abs(Z) ** 2) / Den)
    loglik_up_to_const = -logdet - quad

    # mu = U diag(Lam/(Lam+sigma2)) U^H y
    F = Lam / Den
    M = F * Z
    # vec(Ur M Ut^T) = (Ut ⊗ Ur) vec(M)
    Mu_mat = (U_rx @ M) @ (U_tx.T)
    mu_vec = np.reshape(Mu_mat, (-1,), order="F")

    return float(loglik_up_to_const), mu_vec


def cme_is_single(
    y_vec: np.ndarray,
    toep_rx_samples: np.ndarray,
    toep_tx_samples: np.ndarray,
    sigma2: float,
    eps_eig: float = 1e-12,
) -> np.ndarray:
    """
    Importance-sampling approximation of E[h|y] for a single observation y.
    delta samples are provided as Toeplitz first-columns (Rx and Tx).
    """
    S = toep_rx_samples.shape[0]
    logw = np.zeros(S, dtype=float)
    mu_list = np.zeros((S, y_vec.size), dtype=np.complex128)

    for s in range(S):
        C_rx = ut.toeplitz(toep_rx_samples[s, :])
        C_tx = ut.toeplitz(toep_tx_samples[s, :])

        U_rx, lam_rx = _eig_psd_hermitian(C_rx, eps=eps_eig)
        U_tx, lam_tx = _eig_psd_hermitian(C_tx, eps=eps_eig)

        ll, mu = _kron_gaussian_terms(y_vec, U_rx, lam_rx, U_tx, lam_tx, sigma2=sigma2)
        logw[s] = ll
        mu_list[s, :] = mu

    w = _softmax_logweights(logw)
    return (w[:, None] * mu_list).sum(axis=0)


def cme_is_single_precomp(
    y_vec: np.ndarray,
    precomp: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    sigma2: float,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Same as cme_is_single(), but takes precomputed eigenpairs for deltas:
      precomp[s] = (U_rx, lam_rx, U_tx, lam_tx)
    """
    S = len(precomp)
    logw = np.zeros(S, dtype=float)
    mu_list = np.zeros((S, y_vec.size), dtype=np.complex128)

    for s, (U_rx, lam_rx, U_tx, lam_tx) in enumerate(precomp):
        ll, mu = _kron_gaussian_terms(y_vec, U_rx, lam_rx, U_tx, lam_tx, sigma2=sigma2)
        logw[s] = ll
        mu_list[s, :] = mu

    w = _softmax_logweights_tempered(logw, alpha=alpha)
    return (w[:, None] * mu_list).sum(axis=0)


def cme_is_single_precomp_stats(
    y_vec: np.ndarray,
    precomp: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    sigma2: float,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (h_cme, weights, ess).
    """
    S = len(precomp)
    logw = np.zeros(S, dtype=float)
    mu_list = np.zeros((S, y_vec.size), dtype=np.complex128)

    for s, (U_rx, lam_rx, U_tx, lam_tx) in enumerate(precomp):
        ll, mu = _kron_gaussian_terms(y_vec, U_rx, lam_rx, U_tx, lam_tx, sigma2=sigma2)
        logw[s] = ll
        mu_list[s, :] = mu

    w = _softmax_logweights_tempered(logw, alpha=alpha)
    h_cme = (w[:, None] * mu_list).sum(axis=0)
    return h_cme, w, ess_from_weights(w)


def nmse(xhat: np.ndarray, x: np.ndarray) -> float:
    num = np.sum(np.abs(xhat - x) ** 2)
    den = np.sum(np.abs(x) ** 2)
    return float(num / den)


def main() -> None:
    p = argparse.ArgumentParser(description="CME baseline for 3GPP via IS over Toeplitz deltas (empirical prior)")
    p.add_argument("--snr_db", type=float, default=0.0, help="SNR in dB for y = h + n")
    p.add_argument("--n_path", type=int, default=3)
    p.add_argument("--n_rx", type=int, default=64)
    p.add_argument("--n_tx", type=int, default=16)
    p.add_argument("--n_train", type=int, default=100_000)
    p.add_argument("--n_val", type=int, default=10_000)
    p.add_argument("--n_test", type=int, default=10_000)
    p.add_argument("--n_eval", type=int, default=8, help="How many test channels to evaluate (small-scale)")
    p.add_argument("--S", type=int, default=64, help="Number of delta samples for IS (used if proposal='prior')")
    p.add_argument("--proposal", type=str, default="prior", choices=["prior", "knn"],
                   help="Delta proposal: 'prior' draws S deltas from train; 'knn' selects K nearest deltas to a moment estimate from a pool.")
    p.add_argument("--pool_size", type=int, default=20000,
                   help="For proposal='knn': number of train deltas to include in the candidate pool (subsample for speed).")
    p.add_argument("--K", type=int, default=256,
                   help="For proposal='knn': number of nearest deltas (from pool) to keep for IS.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Tempering exponent for IS weights: w ∝ exp(alpha * log p(y|delta)). alpha<1 reduces weight collapse.")
    p.add_argument("--refine_steps", type=int, default=1, choices=[0, 1, 2],
                   help="For proposal='knn': number of KNN refinement steps (0=single pass, 1-2=recompute center from weighted delta mean).")
    p.add_argument("--K_refine", type=int, default=None,
                   help="For proposal='knn': K used in refinement steps (defaults to K).")
    p.add_argument("--print_weight_stats", action="store_true",
                   help="Print per-sample ESS/max weight diagnostics (useful for debugging).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load data + Toeplitz params
    ch_type = "3gpp"
    channels_train, toep_train, _, _, channels_test, toep_test = ut.load_or_create_data(
        ch_type=ch_type,
        n_path=args.n_path,
        n_antennas_rx=args.n_rx,
        n_antennas_tx=args.n_tx,
        n_train_ch=args.n_train,
        n_val_ch=args.n_val,
        n_test_ch=args.n_test,
        return_toep=True,
    )

    if not isinstance(toep_train, tuple) or not isinstance(toep_test, tuple):
        raise RuntimeError("Expected (toep_rx, toep_tx) tuples for 3GPP MIMO case (n_tx > 1).")
    toep_train_rx, toep_train_tx = toep_train
    toep_test_rx, toep_test_tx = toep_test

    n_ant = args.n_rx * args.n_tx
    if channels_test.shape[1] != n_ant:
        raise RuntimeError(f"Unexpected channel shape: {channels_test.shape}, expected second dim {n_ant}.")

    # Generate observations for the chosen SNR
    y_all = ut.get_observation(channels_test, args.snr_db)
    sigma2 = 10 ** (-args.snr_db / 10.0)

    # Choose small subset of test indices
    test_idx = rng.choice(channels_test.shape[0], size=min(args.n_eval, channels_test.shape[0]), replace=False)

    # Prepare delta candidates from training set
    if args.proposal == "prior":
        # Sample delta from training set (empirical prior)
        train_idx = rng.choice(toep_train_rx.shape[0], size=args.S, replace=True)
        delta_rx_samp = toep_train_rx[train_idx, :]
        delta_tx_samp = toep_train_tx[train_idx, :]

        # Precompute eigendecompositions for sampled deltas (big speedup vs recomputing per test sample)
        precomp_global: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for s in range(args.S):
            C_rx = ut.toeplitz(delta_rx_samp[s, :])
            C_tx = ut.toeplitz(delta_tx_samp[s, :])
            U_rx, lam_rx = _eig_psd_hermitian(C_rx)
            U_tx, lam_tx = _eig_psd_hermitian(C_tx)
            precomp_global.append((U_rx, lam_rx, U_tx, lam_tx))
    else:
        # Build a candidate pool once (subsample of train deltas)
        pool_size = min(args.pool_size, toep_train_rx.shape[0])
        pool_idx = rng.choice(toep_train_rx.shape[0], size=pool_size, replace=False)
        pool_rx = toep_train_rx[pool_idx, :]
        pool_tx = toep_train_tx[pool_idx, :]

        # For KNN matching, compare normalized Toeplitz shapes (scale-invariant)
        pool_rx_n = pool_rx / (np.maximum(np.abs(pool_rx[:, [0]]), 1e-12))
        pool_tx_n = pool_tx / (np.maximum(np.abs(pool_tx[:, [0]]), 1e-12))

    # Evaluate
    nmse_ls = []
    nmse_genie = []
    nmse_cme = []
    ess_list = []
    wmax_list = []

    lmmse = LMMSE(args.snr_db)

    for k, i in enumerate(test_idx):
        h_true = channels_test[i, :]
        y = y_all[i, :]

        # LS (here simply y because A=I)
        h_ls = y

        # Genie LMMSE (upper bound): uses TRUE delta for this sample
        h_genie = lmmse.estimate_genie(y[None, :], (toep_test_rx[i : i + 1, :], toep_test_tx[i : i + 1, :]))[0]

        # CME via IS over training deltas (empirical prior)
        if args.proposal == "prior":
            h_cme, w, ess = cme_is_single_precomp_stats(y, precomp_global, sigma2=sigma2, alpha=args.alpha)
        else:
            # Moment-based delta estimate from y
            t_rx_center, t_tx_center = _moment_delta_hat_from_y(y, n_rx=args.n_rx, n_tx=args.n_tx, sigma2=sigma2)

            def knn_indices(t_rx_c: np.ndarray, t_tx_c: np.ndarray, K: int) -> np.ndarray:
                t_rx_c_n = _normalize_toep(t_rx_c)
                t_tx_c_n = _normalize_toep(t_tx_c)
                d_rx = np.sum(np.abs(pool_rx_n - t_rx_c_n[None, :]) ** 2, axis=1)
                d_tx = np.sum(np.abs(pool_tx_n - t_tx_c_n[None, :]) ** 2, axis=1)
                d = d_rx + d_tx
                K_eff = min(K, d.size)
                return np.argpartition(d, K_eff - 1)[:K_eff]

            K0 = int(args.K)
            nn_idx = knn_indices(t_rx_center, t_tx_center, K0)

            # Per-sample eig cache (keeps memory small, but allows reuse across refinement passes)
            eig_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

            def get_precomp(j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                jj = int(j)
                if jj in eig_cache:
                    return eig_cache[jj]
                C_rx = ut.toeplitz(pool_rx[jj, :])
                C_tx = ut.toeplitz(pool_tx[jj, :])
                U_rx, lam_rx = _eig_psd_hermitian(C_rx)
                U_tx, lam_tx = _eig_psd_hermitian(C_tx)
                eig_cache[jj] = (U_rx, lam_rx, U_tx, lam_tx)
                return eig_cache[jj]

            # Optional refinement steps: update center using posterior-weighted delta mean, then redo KNN
            K_ref = int(args.K_refine) if args.K_refine is not None else K0
            refine_steps = int(args.refine_steps)
            for _ in range(refine_steps):
                precomp_local = [get_precomp(int(j)) for j in nn_idx]
                _, w_tmp, _ = cme_is_single_precomp_stats(y, precomp_local, sigma2=sigma2, alpha=args.alpha)
                # Update center in Toeplitz parameter space (use unnormalized deltas to preserve scale)
                t_rx_center = (w_tmp[:, None] * pool_rx[nn_idx, :]).sum(axis=0)
                t_tx_center = (w_tmp[:, None] * pool_tx[nn_idx, :]).sum(axis=0)
                nn_idx = knn_indices(t_rx_center, t_tx_center, K_ref)

            precomp_local = [get_precomp(int(j)) for j in nn_idx]
            h_cme, w, ess = cme_is_single_precomp_stats(y, precomp_local, sigma2=sigma2, alpha=args.alpha)

        nmse_ls.append(nmse(h_ls, h_true))
        nmse_genie.append(nmse(h_genie, h_true))
        nmse_cme.append(nmse(h_cme, h_true))
        ess_list.append(float(ess))
        wmax_list.append(float(np.max(w)))

        if args.print_weight_stats:
            print(f"[idx={i}] ESS={ess:.1f}/{len(w)}  w_max={np.max(w):.3f}  NMSE(CME-IS)={nmse_cme[-1]:.4e}")

        # print(
        #     f"[{k+1:02d}/{len(test_idx)}] idx={i}  "
        #     f"NMSE(LS)={nmse_ls[-1]:.4e}  NMSE(CME-IS)={nmse_cme[-1]:.4e}  NMSE(genie)={nmse_genie[-1]:.4e}"
        # )

    print("\n=== Summary ===")
    if args.proposal == "prior":
        print(f"SNR(dB)={args.snr_db}  n_eval={len(test_idx)}  proposal=prior  S={args.S}  alpha={args.alpha}")
    else:
        print(
            f"SNR(dB)={args.snr_db}  n_eval={len(test_idx)}  proposal=knn  pool_size={min(args.pool_size, toep_train_rx.shape[0])}  K={args.K}  refine_steps={args.refine_steps}  K_refine={args.K_refine if args.K_refine is not None else args.K}  alpha={args.alpha}"
        )
    print(f"NMSE(LS):      mean={np.mean(nmse_ls):.4e}  std={np.std(nmse_ls):.4e}")
    print(f"NMSE(CME-IS):  mean={np.mean(nmse_cme):.4e}  std={np.std(nmse_cme):.4e}")
    print(f"NMSE(genie):   mean={np.mean(nmse_genie):.4e}  std={np.std(nmse_genie):.4e}")
    print(f"ESS:           mean={np.mean(ess_list):.2f}  std={np.std(ess_list):.2f}  (out of K={len(w)})")
    print(f"w_max:         mean={np.mean(wmax_list):.3f}  std={np.std(wmax_list):.3f}")


if __name__ == "__main__":
    main()


