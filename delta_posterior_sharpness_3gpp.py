r"""
Posterior sharpness of latent delta vs SNR for 3GPP (empirical prior).

We treat per-sample Toeplitz parameters (delta) as the latent variable defining the covariance:
  h|delta ~ CN(0, C_delta),  y = h + n,  n ~ CN(0, sigma^2 I)

Using delta_s ~ empirical prior from training Toeplitz params, compute:
  w_s ∝ p(y | delta_s)  (Gaussian, computed efficiently via Kronecker eig)
  entropy(y) = -Σ w_s log(w_s)
  ESS(y) = 1 / Σ w_s^2
  w_max(y) = max_s w_s

Aggregate over many test samples to get curves vs SNR.

Outputs:
  - results/useful_results/delta_posterior_sharpness_3gpp.npz
  - results/useful_results/delta_posterior_sharpness_3gpp.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
import os

import modules.utils as ut


def softmax_logweights(logw: np.ndarray) -> np.ndarray:
    lw = logw - np.max(logw)
    w = np.exp(lw)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w) / w.size
    return w / s


def ess_from_weights(w: np.ndarray) -> float:
    s2 = float(np.sum(np.square(w)))
    if s2 <= 0 or not np.isfinite(s2):
        return 0.0
    return float(1.0 / s2)


def entropy_from_weights(w: np.ndarray, eps: float = 1e-12) -> float:
    ww = np.clip(w, eps, 1.0)
    return float(-np.sum(ww * np.log(ww)))


def eig_psd_hermitian(C: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    C = 0.5 * (C + C.conj().T)
    lam, U = np.linalg.eigh(C)
    lam = np.maximum(lam.real, eps)
    return U, lam


def logpy_given_delta_kron_eig(
    y_vec: np.ndarray,
    U_rx: np.ndarray,
    lam_rx: np.ndarray,
    U_tx: np.ndarray,
    lam_tx: np.ndarray,
    sigma2: float,
) -> float:
    """
    y ~ CN(0, (C_tx ⊗ C_rx) + sigma2 I)
    Return log p(y|delta) up to an additive constant (dropping -N log pi).
    """
    n_rx = lam_rx.size
    n_tx = lam_tx.size
    Y = np.reshape(y_vec, (n_rx, n_tx), order="F")

    # Z = U_rx^H Y U_tx^*
    Z = (U_rx.conj().T @ Y) @ (U_tx.conj())

    Lam = np.outer(lam_rx, lam_tx)
    Den = Lam + sigma2

    logdet = np.sum(np.log(Den))
    quad = np.sum((np.abs(Z) ** 2) / Den)
    return float(-logdet - quad)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot posterior sharpness of delta vs SNR (3GPP, empirical prior)")
    p.add_argument("--n_path", type=int, default=3)
    p.add_argument("--n_rx", type=int, default=64)
    p.add_argument("--n_tx", type=int, default=16)
    p.add_argument("--n_train", type=int, default=100_000)
    p.add_argument("--n_val", type=int, default=10_000)
    p.add_argument("--n_test", type=int, default=10_000)
    p.add_argument("--snr_min", type=float, default=-15.0)
    p.add_argument("--snr_max", type=float, default=5.0)
    p.add_argument("--snr_step", type=float, default=1.0)
    p.add_argument("--n_eval", type=int, default=200, help="Number of test y per SNR")
    p.add_argument("--S", type=int, default=512, help="Number of delta samples per y (from training empirical prior)")
    p.add_argument("--reuse_deltas", action="store_true",
                   help="If set, reuse the same sampled deltas for all y within an SNR (faster, still valid MC).")
    p.add_argument("--include_true_delta", action="store_true",
                   help="Force-include each test sample's true delta in the candidate set and track its posterior weight w_true.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="results/useful_results")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    snr_grid = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step, dtype=float)

    # Load data + Toeplitz params
    channels_train, toep_train, _, _, channels_test, toep_test = ut.load_or_create_data(
        ch_type="3gpp",
        n_path=args.n_path,
        n_antennas_rx=args.n_rx,
        n_antennas_tx=args.n_tx,
        n_train_ch=args.n_train,
        n_val_ch=args.n_val,
        n_test_ch=args.n_test,
        return_toep=True,
    )
    if not isinstance(toep_train, tuple):
        raise RuntimeError("Expected (toep_rx, toep_tx) tuple for 3GPP MIMO case (n_tx > 1).")
    toep_train_rx, toep_train_tx = toep_train
    if not isinstance(toep_test, tuple):
        raise RuntimeError("Expected (toep_rx, toep_tx) tuple for 3GPP MIMO case (n_tx > 1).")
    toep_test_rx, toep_test_tx = toep_test

    n_ant = args.n_rx * args.n_tx
    if channels_test.shape[1] != n_ant:
        raise RuntimeError(f"Unexpected channel shape: {channels_test.shape}, expected second dim {n_ant}.")

    # Subsample test indices once (reused across SNRs for comparability)
    test_idx = rng.choice(channels_test.shape[0], size=min(args.n_eval, channels_test.shape[0]), replace=False)

    H_mean = np.zeros_like(snr_grid)
    H_median = np.zeros_like(snr_grid)
    ESS_mean = np.zeros_like(snr_grid)
    ESS_median = np.zeros_like(snr_grid)
    ESS_norm_mean = np.zeros_like(snr_grid)
    wmax_mean = np.zeros_like(snr_grid)
    wmax_median = np.zeros_like(snr_grid)
    wtrue_mean = np.full_like(snr_grid, np.nan)
    wtrue_median = np.full_like(snr_grid, np.nan)
    p_true_is_argmax = np.full_like(snr_grid, np.nan)
    true_rank_mean = np.full_like(snr_grid, np.nan)  # 1=best (highest weight)

    # Main loop
    for t, snr_db in enumerate(snr_grid):
        sigma2 = 10 ** (-snr_db / 10.0)

        # Generate y for this SNR on selected test subset
        h_sel = channels_test[test_idx, :]
        y_sel = ut.get_observation(h_sel, snr_db)

        ent_list = []
        ess_list = []
        wmax_list = []
        wtrue_list = []
        true_is_argmax_list = []
        true_rank_list = []

        # Optionally reuse sampled deltas across y (within this SNR)
        if args.reuse_deltas:
            train_idx = rng.choice(toep_train_rx.shape[0], size=args.S, replace=True)
            delta_rx_samp = toep_train_rx[train_idx, :]
            delta_tx_samp = toep_train_tx[train_idx, :]

            precomp = []
            for s in range(args.S):
                C_rx = ut.toeplitz(delta_rx_samp[s, :])
                C_tx = ut.toeplitz(delta_tx_samp[s, :])
                U_rx, lam_rx = eig_psd_hermitian(C_rx)
                U_tx, lam_tx = eig_psd_hermitian(C_tx)
                precomp.append((U_rx, lam_rx, U_tx, lam_tx))

        for i in range(y_sel.shape[0]):
            y = y_sel[i, :]

            if not args.reuse_deltas:
                train_idx = rng.choice(toep_train_rx.shape[0], size=args.S, replace=True)
                delta_rx_samp = toep_train_rx[train_idx, :]
                delta_tx_samp = toep_train_tx[train_idx, :]

                precomp = []
                for s in range(args.S):
                    C_rx = ut.toeplitz(delta_rx_samp[s, :])
                    C_tx = ut.toeplitz(delta_tx_samp[s, :])
                    U_rx, lam_rx = eig_psd_hermitian(C_rx)
                    U_tx, lam_tx = eig_psd_hermitian(C_tx)
                    precomp.append((U_rx, lam_rx, U_tx, lam_tx))

            # Compute log-likelihoods for sampled deltas
            S_eff = len(precomp)
            logp = np.zeros(S_eff + (1 if args.include_true_delta else 0), dtype=float)
            for s, (U_rx, lam_rx, U_tx, lam_tx) in enumerate(precomp):
                logp[s] = logpy_given_delta_kron_eig(y, U_rx, lam_rx, U_tx, lam_tx, sigma2=sigma2)

            # Sanity check: force-include the true delta for this test sample
            if args.include_true_delta:
                # Map back to original test index in full test set
                test_i = int(test_idx[i])
                C_rx_true = ut.toeplitz(toep_test_rx[test_i, :])
                C_tx_true = ut.toeplitz(toep_test_tx[test_i, :])
                U_rx_t, lam_rx_t = eig_psd_hermitian(C_rx_true)
                U_tx_t, lam_tx_t = eig_psd_hermitian(C_tx_true)
                logp_true = logpy_given_delta_kron_eig(y, U_rx_t, lam_rx_t, U_tx_t, lam_tx_t, sigma2=sigma2)
                logp[-1] = logp_true

            w = softmax_logweights(logp)
            ent_list.append(entropy_from_weights(w))
            ess = ess_from_weights(w)
            ess_list.append(ess)
            wmax_list.append(float(np.max(w)))
            if args.include_true_delta:
                w_true = float(w[-1])
                wtrue_list.append(w_true)
                true_is_argmax_list.append(float(1.0 if int(np.argmax(w)) == (w.size - 1) else 0.0))
                # rank of true in descending weights: 1..(S+1)
                rank = int(1 + np.sum(w[:-1] > w_true))
                true_rank_list.append(float(rank))

        ent = np.asarray(ent_list, dtype=float)
        ess = np.asarray(ess_list, dtype=float)
        wmx = np.asarray(wmax_list, dtype=float)

        H_mean[t] = float(np.mean(ent))
        H_median[t] = float(np.median(ent))
        ESS_mean[t] = float(np.mean(ess))
        ESS_median[t] = float(np.median(ess))
        ESS_norm_mean[t] = float(np.mean(ess / float(args.S)))
        wmax_mean[t] = float(np.mean(wmx))
        wmax_median[t] = float(np.median(wmx))
        if args.include_true_delta:
            wt = np.asarray(wtrue_list, dtype=float)
            wtrue_mean[t] = float(np.mean(wt))
            wtrue_median[t] = float(np.median(wt))
            p_true_is_argmax[t] = float(np.mean(true_is_argmax_list))
            true_rank_mean[t] = float(np.mean(true_rank_list))

        print(
            f"SNR={snr_db:>5.1f}dB | "
            f"H: mean={H_mean[t]:.3f} med={H_median[t]:.3f} | "
            f"ESS: mean={ESS_mean[t]:.2f} (ESS/S={ESS_norm_mean[t]:.3f}) med={ESS_median[t]:.2f} | "
            f"w_max: mean={wmax_mean[t]:.3f} med={wmax_median[t]:.3f}"
            + (
                f" | w_true: mean={wtrue_mean[t]:.3f} med={wtrue_median[t]:.3f} "
                f"P(true=argmax)={p_true_is_argmax[t]:.3f} rank_true(mean)={true_rank_mean[t]:.1f}"
                if args.include_true_delta
                else ""
            )
        )

    # Save results
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "delta_posterior_sharpness_3gpp.npz"
    np.savez(
        out_npz,
        snr=snr_grid,
        n_eval=args.n_eval,
        S=args.S,
        reuse_deltas=args.reuse_deltas,
        include_true_delta=args.include_true_delta,
        H_mean=H_mean,
        H_median=H_median,
        ESS_mean=ESS_mean,
        ESS_median=ESS_median,
        ESS_norm_mean=ESS_norm_mean,
        wmax_mean=wmax_mean,
        wmax_median=wmax_median,
        wtrue_mean=wtrue_mean,
        wtrue_median=wtrue_median,
        p_true_is_argmax=p_true_is_argmax,
        true_rank_mean=true_rank_mean,
        seed=args.seed,
    )

    # Plot
    if "DISPLAY" not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa

    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    ax2 = ax1.twinx()

    ax1.plot(snr_grid, H_mean, "-o", label=r"Entropy $H(\delta\mid y)$ (mean)", color="#1f77b4")
    ax2.plot(snr_grid, ESS_norm_mean, "-s", label=r"ESS/S (mean)", color="#d62728")
    if args.include_true_delta:
        ax2.plot(snr_grid, wtrue_mean, "-^", label=r"$w_{\mathrm{true}}$ (mean)", color="#2ca02c")

    ax1.set_xlabel("SNR [dB]")
    ax1.set_ylabel(r"Entropy  $-\sum w\log w$")
    ax2.set_ylabel("ESS / S   (and $w_{true}$)")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.95)

    fig.tight_layout()
    out_png = out_dir / "delta_posterior_sharpness_3gpp.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"[saved] {out_npz}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()


