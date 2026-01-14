"""
Sweep SNR and plot NMSE for the 3GPP Gram-oracle conditional mean estimator:
    H_hat_oracle(Y,R) = E[H | Y, R=H H^H]

This script is intentionally small-scale by default (the estimator is expensive).
It operates in the TIME-DOMAIN H representation:
  - No FFT is applied inside the oracle estimator.
  - Observation model matches other baselines in this repo: Y = H + Z (AWGN).

Run example:
    python run_gram_oracle_3gpp_nmse.py --snr_min -15 --snr_max 5 --snr_step 5 --n_mc 50 --M_v 128 --K_delta 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np
import matplotlib
import torch

import modules.utils as ut
from estimators.gram_oracle_cme import (
    OracleOptions,
    make_delta_particles_from_empirical_toep,
    DeltaParticle,
    oracle_e_h_given_y_r,
)
from estimators.gram_oracle_map import MapOptions, gram_oracle_map_given_delta


def nmse_fro(H_hat: np.ndarray, H: np.ndarray) -> float:
    num = np.sum(np.abs(H_hat - H) ** 2)
    den = np.sum(np.abs(H) ** 2)
    return float(num / max(den, 1e-12))


def _toeplitz_first_col_from_mat(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    c = np.zeros(n, dtype=M.dtype)
    for k in range(n):
        c[k] = np.mean(np.diag(M, -k))
    return c


def _normalize_toep(t: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.abs(t[0])
    if not np.isfinite(denom) or denom < eps:
        return t
    return t / denom


def select_delta_knn(
    Y: np.ndarray,
    R: np.ndarray,
    toep_train_rx: np.ndarray,
    toep_train_tx: np.ndarray,
    *,
    sigma2: float,
    pool_size: int,
    knn_k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a subset of delta candidates via a cheap KNN in Toeplitz space.
    - Rx center from oracle Gram R (noise-free): Toeplitz projection of R.
    - Tx center from Y^H Y (noisy): Toeplitz projection of (Y^H Y)/Nr - sigma2 I.
    Distances are computed on normalized Toeplitz first columns (scale-invariant).
    Returns (idx_pool, idx_knn_within_pool).
    """
    Nr, Nt = Y.shape
    # centers
    t_rx_hat = _toeplitz_first_col_from_mat(R)
    Ttx = (Y.conj().T @ Y) / float(Nr) - sigma2 * np.eye(Nt, dtype=Y.dtype)
    t_tx_hat = _toeplitz_first_col_from_mat(Ttx)
    t_rx_hat_n = _normalize_toep(t_rx_hat)
    t_tx_hat_n = _normalize_toep(t_tx_hat)

    pool_size = min(int(pool_size), toep_train_rx.shape[0])
    pool_idx = rng.choice(toep_train_rx.shape[0], size=pool_size, replace=False)
    pool_rx = toep_train_rx[pool_idx, :]
    pool_tx = toep_train_tx[pool_idx, :]
    pool_rx_n = pool_rx / np.maximum(np.abs(pool_rx[:, [0]]), 1e-12)
    pool_tx_n = pool_tx / np.maximum(np.abs(pool_tx[:, [0]]), 1e-12)

    d_rx = np.sum(np.abs(pool_rx_n - t_rx_hat_n[None, :]) ** 2, axis=1)
    d_tx = np.sum(np.abs(pool_tx_n - t_tx_hat_n[None, :]) ** 2, axis=1)
    d = d_rx + d_tx

    knn_k = min(int(knn_k), d.size)
    nn_in_pool = np.argpartition(d, knn_k - 1)[:knn_k]
    return pool_idx, nn_in_pool


def main() -> None:
    p = argparse.ArgumentParser(description="3GPP Gram-oracle CME: NMSE vs SNR sweep (time-domain)")
    p.add_argument("--n_rx", type=int, default=64)
    p.add_argument("--n_tx", type=int, default=16)
    p.add_argument("--n_path", type=int, default=3)
    p.add_argument("--n_train", type=int, default=100_000)
    p.add_argument("--n_val", type=int, default=10_000)
    p.add_argument("--n_test", type=int, default=10_000)

    p.add_argument("--snr_min", type=float, default=-15.0)
    p.add_argument("--snr_max", type=float, default=5.0)
    p.add_argument("--snr_step", type=float, default=5.0)

    p.add_argument("--n_mc", type=int, default=50, help="Monte Carlo test samples per SNR (subsample from test set)")
    p.add_argument("--M_v", type=int, default=128, help="V samples per delta")
    p.add_argument("--K_delta", type=int, default=8, help="delta particles per sample (empirical prior)")
    p.add_argument("--oracle_mode", type=str, default="mc", choices=["mc", "map"],
                   help="Oracle solver: 'mc' (Stiefel Monte Carlo for E[H|Y,R]) or 'map' (projected MAP on Stiefel).")
    p.add_argument("--proposal", type=str, default="procrustes_mix", choices=["haar", "procrustes_mix"])
    p.add_argument("--proposal_eps", type=float, default=0.25, help="For procrustes_mix: mix strength (0=MAP,1=Haar)")
    p.add_argument("--prior_v_weight", type=float, default=0.0, help="Weight for V-prior energy term (0 disables; start with 0 for sanity).")
    p.add_argument("--prior_q_scale", type=float, default=None, help="Scale factor for q before applying prior_v_weight. Default: 1/(Nr*Nt).")
    p.add_argument("--include_logdet_prior", action="store_true", help="Include -log|C_delta| in delta evidence (usually disable when conditioning on R).")
    p.add_argument("--v_temp", type=float, default=1.0, help="Tempering for V-weights softmax: alpha_V. Use <1 to reduce collapse.")
    p.add_argument("--delta_proposal", type=str, default="prior", choices=["prior", "knn"],
                   help="How to pick delta particles: 'prior' draws K_delta iid from train; 'knn' selects from a pool using (Y,R).")
    p.add_argument("--delta_pool_size", type=int, default=100000, help="For delta_proposal=knn: candidate pool size from train")
    p.add_argument("--delta_knn_k", type=int, default=512, help="For delta_proposal=knn: number of nearest deltas to keep before subsampling K_delta")
    p.add_argument("--include_true_delta", action="store_true", help="Sanity: include delta_true (from toep_test) in the delta particle set")
    p.add_argument("--debug_first", action="store_true", help="Print delta posterior diagnostics for the first sample at the first SNR.")
    p.add_argument("--map_iters", type=int, default=80, help="MAP mode: number of projected GD iterations on Stiefel")
    p.add_argument("--map_lr", type=float, default=0.2, help="MAP mode: learning rate for projected GD")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--out_dir", type=str, default="results/useful_results")
    p.add_argument("--no_save", action="store_true", help="Do not save npz results to disk (terminal output only).")
    p.add_argument("--no_plot", action="store_true", help="Do not generate/save plot images.")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    snr_grid = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step, dtype=float)

    # Load 3GPP dataset + Toeplitz latent (empirical p(delta))
    ch_type = "3gpp"
    H_train, toep_train, _, _, H_test, _toep_test = ut.load_or_create_data(
        ch_type=ch_type,
        n_path=args.n_path,
        n_antennas_rx=args.n_rx,
        n_antennas_tx=args.n_tx,
        n_train_ch=args.n_train,
        n_val_ch=args.n_val,
        n_test_ch=args.n_test,
        return_toep=True,
    )
    if not isinstance(toep_train, tuple):
        raise RuntimeError("Expected (toep_rx,toep_tx) tuple for 3GPP MIMO (n_tx>1).")
    toep_train_rx, toep_train_tx = toep_train
    toep_test = _toep_test
    if args.include_true_delta:
        if not isinstance(toep_test, tuple):
            raise RuntimeError("include_true_delta requires toep_test tuple (toep_rx,toep_tx).")
        toep_test_rx, toep_test_tx = toep_test

    n_ant = args.n_rx * args.n_tx
    if H_test.shape[1] != n_ant:
        raise RuntimeError(f"Unexpected H_test shape {H_test.shape}, expected second dim {n_ant}")

    # Subsample test indices once for speed
    idx_test = rng.choice(H_test.shape[0], size=min(args.n_mc, H_test.shape[0]), replace=False)

    opts = OracleOptions(
        M_v=args.M_v,
        K_delta=args.K_delta,
        proposal=args.proposal,
        proposal_eps=float(args.proposal_eps),
        prior_v_weight=float(args.prior_v_weight),
        prior_q_scale=float(args.prior_q_scale) if args.prior_q_scale is not None else 1.0 / float(args.n_rx * args.n_tx),
        include_logdet_prior=bool(args.include_logdet_prior),
        v_temp=float(args.v_temp),
    )

    map_opts = MapOptions(
        n_iters=int(args.map_iters),
        lr=float(args.map_lr),
        prior_q_scale=float(args.prior_q_scale) if args.prior_q_scale is not None else 1.0 / float(args.n_rx * args.n_tx),
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.complex64,
    )

    nmse_mean = []
    nmse_std = []
    timings = []

    for snr_db in snr_grid:
        t0 = time.perf_counter()
        sigma2 = 10 ** (-snr_db / 10.0)
        nmse_list = []

        for i in idx_test:
            h_vec = H_test[i, :]
            # observation Y = H + Z (vectorized)
            y_vec = ut.get_observation(h_vec[None, :], snr_db)[0]
            # reshape to matrices in Fortran order (matches kron(C_tx,C_rx))
            H = np.reshape(h_vec, (args.n_rx, args.n_tx), order="F")
            Y = np.reshape(y_vec, (args.n_rx, args.n_tx), order="F")
            R = H @ H.conj().T

            if args.delta_proposal == "prior":
                deltas = make_delta_particles_from_empirical_toep(
                    toep_train_rx, toep_train_tx, K=args.K_delta, rng=rng
                )
            else:
                pool_idx, nn_in_pool = select_delta_knn(
                    Y, R, toep_train_rx, toep_train_tx,
                    sigma2=sigma2,
                    pool_size=args.delta_pool_size,
                    knn_k=args.delta_knn_k,
                    rng=rng,
                )
                # pick K_delta uniformly from the KNN shortlist (to keep diversity)
                shortlist_global_idx = pool_idx[nn_in_pool]
                pick = rng.choice(shortlist_global_idx.size, size=min(args.K_delta, shortlist_global_idx.size), replace=False)
                chosen = shortlist_global_idx[pick]
                deltas = [DeltaParticle(t_rx=toep_train_rx[j].copy(), t_tx=toep_train_tx[j].copy(), log_prior=0.0) for j in chosen]

            if args.include_true_delta:
                # Append true delta for this sample
                dp_true = DeltaParticle(t_rx=toep_test_rx[i].copy(), t_tx=toep_test_tx[i].copy(), log_prior=0.0)
                deltas.append(dp_true)

            if args.oracle_mode == "mc":
                true_idx = (len(deltas) - 1) if args.include_true_delta else None
                if args.debug_first and (snr_db == snr_grid[0]) and (i == idx_test[0]):
                    H_hat, dbg = oracle_e_h_given_y_r(
                        Y, R, deltas, sigma2, opts=opts, rng=rng, return_debug=True, true_delta_index=true_idx
                    )
                    if "w_true_delta" in dbg:
                        print(f"  [debug] w_true_delta={dbg['w_true_delta']:.3g} rank_true={dbg['rank_true_delta']} true_is_argmax={dbg['true_is_argmax']}")
                        w = dbg["w_delta"]
                        top = np.argsort(w)[::-1][:3]
                        print("  [debug] top delta idx/weight: " + ", ".join([f"{int(j)}:{w[j]:.3g}" for j in top]))
                        le = dbg.get("log_evidences", None)
                        if le is not None:
                            print(f"  [debug] log_evidence range: [{float(np.min(le)):.3g}, {float(np.max(le)):.3g}]  span={float(np.max(le)-np.min(le)):.3g}")
                else:
                    H_hat = oracle_e_h_given_y_r(Y, R, deltas, sigma2, opts=opts, rng=rng)
            else:
                # MAP mode: choose the best delta by minimized objective (joint MAP over delta and V).
                best_obj = float("inf")
                best_hat = None
                for dp in deltas:
                    Hk, obj = gram_oracle_map_given_delta(
                        Y, R, t_rx=dp.t_rx, t_tx=dp.t_tx, sigma2=sigma2, opts=map_opts
                    )
                    if obj < best_obj:
                        best_obj = obj
                        best_hat = Hk
                H_hat = best_hat
            nmse_list.append(nmse_fro(H_hat, H))

        dt = time.perf_counter() - t0
        timings.append(dt)
        nmse_mean.append(float(np.mean(nmse_list)))
        nmse_std.append(float(np.std(nmse_list)))
        print(
            f"SNR={snr_db:+.1f} dB | NMSE mean={nmse_mean[-1]:.4e} std={nmse_std[-1]:.4e} | "
            f"n_mc={len(idx_test)} M_v={args.M_v} K_delta={args.K_delta} | time={dt:.1f}s"
        )

    # Save + plot (optional)
    if not args.no_save or not args.no_plot:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_save:
        out_npz = out_dir / f"gram_oracle_cme_3gpp_nmse_snr{args.snr_min:g}to{args.snr_max:g}_step{args.snr_step:g}.npz"
        np.savez(
            out_npz,
            snr_db=snr_grid,
            nmse_mean=np.asarray(nmse_mean),
            nmse_std=np.asarray(nmse_std),
            n_mc=int(len(idx_test)),
            M_v=int(args.M_v),
            K_delta=int(args.K_delta),
            seed=int(args.seed),
            timings_sec=np.asarray(timings),
        )
        print(f"[saved] {out_npz}")

    if not args.no_plot:
        if "DISPLAY" not in os.environ:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa

        plt.figure(figsize=(8.0, 5.5))
        plt.semilogy(snr_grid, nmse_mean, marker="P", linewidth=2.0, label="Gram-oracle CME (empirical prior)")
        plt.grid(True, which="both", linestyle=":", alpha=0.6)
        plt.xlabel("SNR [dB]")
        plt.ylabel("NMSE")
        plt.title("3GPP: Gram-oracle conditional mean (time-domain)")
        plt.legend(loc="upper right", framealpha=0.95)
        plt.tight_layout()
        out_png = out_dir / "gram_oracle_cme_3gpp_nmse.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()
        print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()


