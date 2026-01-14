"""
A1/A2 Gram-matrix diagnostics for 3GPP (NLOS) vs QuaDRiGa (LOS).

This script implements:
  - A1: per-sample spectral structure (energy concentration, effective rank, condition number)
  - A2: cross-sample Gram diversity (pairwise Frobenius distances of trace-normalized Grams)

Outputs:
  - Publication-quality plots saved to results/useful_results/
  - Printed numeric summaries (mean/std/percentiles)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
import os

import sys

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so `import modules.*` works when running from subdirectories.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import modules.utils as ut


EPS = 1e-12


def compute_gram(H: np.ndarray) -> np.ndarray:
    """
    H: complex ndarray, shape (..., N_r, N_t)
    Returns: G = H H^H, shape (..., N_r, N_r)
    """
    return H @ np.conjugate(np.swapaxes(H, -1, -2))


def trace_normalize_gram(G: np.ndarray) -> np.ndarray:
    tr = np.real(np.trace(G, axis1=-2, axis2=-1))
    tr = np.maximum(tr, EPS)
    return G / tr[..., None, None]


def gram_eigvals(G: np.ndarray) -> np.ndarray:
    """
    Hermitian PSD eigenvalues, descending.
    Input G: shape (..., N_r, N_r) complex, Hermitian.
    Returns: shape (..., N_r) real, sorted descending.
    """
    # eigvalsh returns ascending
    w = np.linalg.eigvalsh(G)
    w = np.real(w)
    w = np.flip(w, axis=-1)
    return w


def effective_rank(p: np.ndarray) -> np.ndarray:
    """
    p: normalized eigenvalues (sum to 1), shape (..., N_r)
    r_eff = exp( - sum p log p )
    """
    p = np.clip(p, EPS, 1.0)
    H = -np.sum(p * np.log(p), axis=-1)
    return np.exp(H)


@dataclass
class SpectrumMetrics:
    top1: np.ndarray
    top2: np.ndarray
    r_eff: np.ndarray
    cond: np.ndarray
    eigs_norm: np.ndarray  # (M, N_r)


def gram_spectrum_metrics(G: np.ndarray) -> SpectrumMetrics:
    eigs = gram_eigvals(G)  # (M, N_r)
    total = np.sum(eigs, axis=-1, keepdims=True)
    total = np.maximum(total, EPS)
    p = eigs / total
    top1 = p[:, 0]
    top2 = np.sum(p[:, :2], axis=-1)
    r_eff = effective_rank(p)
    # condition number (optional; robust with eps)
    lam_max = np.maximum(eigs[:, 0], EPS)
    lam_min = np.maximum(eigs[:, -1], EPS)
    cond = lam_max / lam_min
    return SpectrumMetrics(top1=top1, top2=top2, r_eff=r_eff, cond=cond, eigs_norm=p)


def gram_pairwise_distances(G_tilde: np.ndarray, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """
    G_tilde: trace-normalized Gram matrices, shape (M, N_r, N_r)
    Returns: distances shape (n_pairs,)
    """
    M = G_tilde.shape[0]
    i = rng.integers(0, M, size=n_pairs)
    j = rng.integers(0, M, size=n_pairs)
    # avoid i==j (optional)
    same = i == j
    if np.any(same):
        j[same] = (j[same] + 1) % M
    diff = G_tilde[i] - G_tilde[j]
    # Frobenius norm for complex matrices
    d = np.sqrt(np.sum(np.abs(diff) ** 2, axis=(-2, -1)))
    return d


def _ensure_shape(H: np.ndarray, n_rx: int, n_tx: int, ch_type: str) -> np.ndarray:
    """
    Normalize channel array to shape (N, n_rx, n_tx) complex.
    Handles common repo formats (flattened 3GPP, Quadriga already 3D).
    """
    H = np.asarray(H)
    if H.ndim == 3:
        # Try to interpret as (N, n_rx, n_tx); if swapped, fix if obvious
        if H.shape[1:] == (n_rx, n_tx):
            return H
        if H.shape[1:] == (n_tx, n_rx):
            return np.transpose(H, (0, 2, 1))
        raise ValueError(f"Unexpected 3D H shape {H.shape}, expected (*,{n_rx},{n_tx})")
    if H.ndim == 2:
        if H.shape[1] != n_rx * n_tx:
            raise ValueError(f"Unexpected 2D H shape {H.shape}, expected second dim {n_rx*n_tx}")
        # Match existing evaluation scripts: reshape with Fortran order for 3GPP MIMO
        order = "F" if ch_type.startswith("3gpp") and n_tx > 1 else "C"
        return np.reshape(H, (-1, n_rx, n_tx), order=order)
    raise ValueError(f"Unexpected H ndim={H.ndim}, shape={H.shape}")


def load_channels(
    *,
    ch_type: str,
    n_path: int,
    n_rx: int,
    n_tx: int,
    split: str,
    n_samples: int,
) -> np.ndarray:
    """
    Returns complex channels H with shape (n_samples, n_rx, n_tx).

    Uses repo data loader `modules.utils.load_or_create_data`.
    """
    # These sizes must match files in bin/ for 3GPP, and slicing for Quadriga.
    n_train, n_val, n_test = 100_000, 10_000, 10_000
    data_train, data_val, data_test = ut.load_or_create_data(
        ch_type=ch_type,
        n_path=n_path,
        n_antennas_rx=n_rx,
        n_antennas_tx=n_tx,
        n_train_ch=n_train,
        n_val_ch=n_val,
        n_test_ch=n_test,
        return_toep=False,
    )
    if split == "train":
        H = data_train
    elif split == "val":
        H = data_val
    elif split == "test":
        H = data_test
    else:
        raise ValueError("split must be one of: train, val, test")
    H = _ensure_shape(H, n_rx=n_rx, n_tx=n_tx, ch_type=ch_type)
    if n_samples > H.shape[0]:
        raise ValueError(f"Requested n_samples={n_samples} but only have {H.shape[0]} samples for {ch_type}/{split}")
    return H[:n_samples]


def plot_spectrum_bands(
    out_path: Path,
    eigs_norm_a: np.ndarray,
    eigs_norm_b: np.ndarray,
    label_a: str,
    label_b: str,
    *,
    percentile_lo: float = 10.0,
    percentile_hi: float = 90.0,
    max_k: int = 32,
    logy: bool = True,
) -> None:
    """
    Plot mean normalized eigenvalue spectrum with percentile bands.
    """
    k = np.arange(1, eigs_norm_a.shape[1] + 1)
    k = k[:max_k]

    def _summ(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(x, axis=0)[:max_k]
        lo = np.percentile(x, percentile_lo, axis=0)[:max_k]
        hi = np.percentile(x, percentile_hi, axis=0)[:max_k]
        return mean, lo, hi

    mean_a, lo_a, hi_a = _summ(eigs_norm_a)
    mean_b, lo_b, hi_b = _summ(eigs_norm_b)

    plt.figure(figsize=(8.0, 5.5))
    ax = plt.gca()
    ax.plot(k, mean_a, linewidth=2.0, label=f"{label_a} (mean)")
    ax.fill_between(k, lo_a, hi_a, alpha=0.18, label=f"{label_a} ({percentile_lo:.0f}-{percentile_hi:.0f}%)")
    ax.plot(k, mean_b, linewidth=2.0, label=f"{label_b} (mean)")
    ax.fill_between(k, lo_b, hi_b, alpha=0.18, label=f"{label_b} ({percentile_lo:.0f}-{percentile_hi:.0f}%)")
    ax.set_xlabel("Eigenvalue index (sorted desc)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized eigenvalue", fontsize=12, fontweight="bold")
    ax.set_xticks(k[::4])
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.4)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim([1e-4, 1.0])
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, fancybox=True, edgecolor="gray", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"[saved] {out_path}")


def plot_boxplot(
    out_path: Path,
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(6.5, 5.0))
    ax = plt.gca()
    ax.boxplot([a, b], labels=[label_a, label_b], showfliers=False)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"[saved] {out_path}")


def plot_hist(
    out_path: Path,
    d_a: np.ndarray,
    d_b: np.ndarray,
    label_a: str,
    label_b: str,
    xlabel: str,
) -> None:
    plt.figure(figsize=(8.0, 5.5))
    ax = plt.gca()
    bins = 60
    ax.hist(d_a, bins=bins, alpha=0.55, density=True, label=label_a)
    ax.hist(d_b, bins=bins, alpha=0.55, density=True, label=label_b)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.4)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95, fancybox=True, edgecolor="gray", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"[saved] {out_path}")


def _print_summary(name: str, m: SpectrumMetrics) -> None:
    def _fmt(x: np.ndarray) -> str:
        return f"mean={np.mean(x):.4f}, std={np.std(x):.4f}, p10={np.percentile(x,10):.4f}, p50={np.percentile(x,50):.4f}, p90={np.percentile(x,90):.4f}"

    print(f"\n[{name}] A1 summary")
    print(f"  top1 energy: {_fmt(m.top1)}")
    print(f"  top2 energy: {_fmt(m.top2)}")
    print(f"  effective rank: {_fmt(m.r_eff)}")
    print(f"  cond(G): {_fmt(m.cond)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="A1/A2 Gram diagnostics: 3GPP (NLOS) vs Quadriga (LOS)")
    ap.add_argument("--n_rx", type=int, default=64)
    ap.add_argument("--n_tx", type=int, default=16)
    ap.add_argument("--n_path", type=int, default=3, help="3GPP paths (ignored for quadriga)")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--n_samples", type=int, default=2000, help="Samples per case (same for both)")
    ap.add_argument("--n_pairs", type=int, default=20000, help="Random Gram pairs for A2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_k", type=int, default=32, help="How many leading eigenvalues to plot")
    ap.add_argument("--no_log_spectrum", action="store_true", help="Disable log-y for eigen spectrum plot")
    ap.add_argument("--out_dir", type=str, default=str(Path("results") / "useful_results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Load channels
    H_3gpp = load_channels(
        ch_type="3gpp",
        n_path=args.n_path,
        n_rx=args.n_rx,
        n_tx=args.n_tx,
        split=args.split,
        n_samples=args.n_samples,
    )
    H_quad = load_channels(
        ch_type="quadriga_LOS",
        n_path=args.n_path,
        n_rx=args.n_rx,
        n_tx=args.n_tx,
        split=args.split,
        n_samples=args.n_samples,
    )

    # A1: per-sample spectral metrics
    G_3gpp = compute_gram(H_3gpp)
    G_quad = compute_gram(H_quad)

    m_3gpp = gram_spectrum_metrics(G_3gpp)
    m_quad = gram_spectrum_metrics(G_quad)

    _print_summary("3GPP (NLOS)", m_3gpp)
    _print_summary("QuaDRiGa (LOS)", m_quad)

    # Plot eigenvalue spectrum bands
    plot_spectrum_bands(
        out_dir / "A1_eigspectrum_bands_3gpp_vs_quadriga.png",
        m_3gpp.eigs_norm,
        m_quad.eigs_norm,
        "3GPP (NLOS)",
        "QuaDRiGa (LOS)",
        percentile_lo=10.0,
        percentile_hi=90.0,
        max_k=args.max_k,
        logy=not args.no_log_spectrum,
    )

    # Effective rank boxplot
    plot_boxplot(
        out_dir / "A1_effective_rank_boxplot.png",
        m_3gpp.r_eff,
        m_quad.r_eff,
        "3GPP",
        "QuaDRiGa",
        ylabel="Effective rank (exp entropy)",
    )

    # A2: Gram diversity (trace-normalized pairwise distances)
    Gt_3gpp = trace_normalize_gram(G_3gpp)
    Gt_quad = trace_normalize_gram(G_quad)

    d_3gpp = gram_pairwise_distances(Gt_3gpp, n_pairs=args.n_pairs, rng=rng)
    d_quad = gram_pairwise_distances(Gt_quad, n_pairs=args.n_pairs, rng=rng)

    print(f"\n[A2] Pairwise Frobenius distances (trace-normalized Grams), n_pairs={args.n_pairs}")
    print(f"  3GPP: mean={np.mean(d_3gpp):.4f}, std={np.std(d_3gpp):.4f}")
    print(f"  Quad: mean={np.mean(d_quad):.4f}, std={np.std(d_quad):.4f}")

    plot_hist(
        out_dir / "A2_gram_distance_hist.png",
        d_3gpp,
        d_quad,
        "3GPP (NLOS)",
        "QuaDRiGa (LOS)",
        xlabel=r"$\|\tilde G_i - \tilde G_j\|_F$",
    )

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()


