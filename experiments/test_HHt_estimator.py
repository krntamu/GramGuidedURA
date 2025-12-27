"""
Diagnostic: Estimate channel covariance R_h = H H^H from noisy data observations using TIME AVERAGING.

Model:
  - Pick a SINGLE channel matrix H (N_R × N_T)
  - Generate N_d i.i.d. data symbols X_d: (N_T × N_d)
  - Received signals: Y_d = H @ X_d + N, where Y_d: (N_R × N_d), each column is y_d(t)
  - Time-averaged covariance: R_y_hat = (1/N_d) * Y_d @ Y_d.conj().T
  - Estimated channel covariance: R_h_hat = R_y_hat - sigma_n^2 * I_N_R
  - True reference: R_h_true = H @ H.conj().T

This implements PURE TIME AVERAGING: averaging over time samples t=1..N_d for a fixed channel H.
No ensemble averaging is used in the covariance computation itself.

Example ablation commands:
  python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode none --snrs -15 --n_time_samples 300
  python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode hermitian --snrs -15 --n_time_samples 300
  python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd --snrs -15 --n_time_samples 300
  python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd_trace --snrs -15 --n_time_samples 300
  python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd_rankk --rank_k 16 --snrs -15 --n_time_samples 300
  python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd_rankk_trace --rank_k 16 --snrs -15 --n_time_samples 300
"""

import argparse
import numpy as np
import os
import sys

# Ensure repo root is on sys.path so `modules` can be imported when running directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import modules.utils as ut


def parse_snr_range(snr_str: str) -> list[float]:
    """
    Parse SNR specification, supporting both list and range syntax.
    
    Examples:
        "-15 -10 -5" -> [-15.0, -10.0, -5.0] (space-separated list)
        "-15:5:1" -> [-15.0, -14.0, ..., 4.0, 5.0]  (start:end:step)
        "-15:5" -> [-15.0, -14.0, ..., 4.0, 5.0]   (start:end, step=1)
        "-15" -> [-15.0]                            (single value)
    
    Args:
        snr_str: SNR specification string
    
    Returns:
        List of SNR values
    """
    # Check if it's a range specification (contains ':')
    if ':' in snr_str:
        parts = snr_str.split(':')
        if len(parts) == 2:
            # start:end (default step=1)
            start, end = float(parts[0]), float(parts[1])
            step = 1.0
        elif len(parts) == 3:
            # start:end:step
            start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
        else:
            raise ValueError(f"Invalid SNR range format: '{snr_str}'. Use 'start:end' or 'start:end:step'")
        
        # Generate range
        if step > 0:
            snr_list = np.arange(start, end + step/2, step).tolist()  # +step/2 to include end
        else:
            snr_list = np.arange(start, end - step/2, step).tolist()  # -step/2 to include end for negative step
        return [float(x) for x in snr_list]
    else:
        # Parse as space-separated list
        return [float(x) for x in snr_str.split()]


def compute_matrix_stats(R: np.ndarray, R_true: np.ndarray = None) -> dict:
    """
    Compute comprehensive statistics for a covariance matrix.
    
    Args:
        R: Matrix to analyze (N_R, N_R)
        R_true: True reference matrix (optional, for error computation)
    
    Returns:
        Dictionary with: trace, fro_norm, min_eig, neg_eig_energy_ratio, rank, rel_fro_err
    """
    # Ensure Hermitian symmetry for analysis
    R_herm = 0.5 * (R + R.conj().T)
    
    # Eigendecomposition
    evals, _ = np.linalg.eigh(R_herm)
    
    # Basic stats
    trace = np.real(np.trace(R_herm))
    fro_norm = np.linalg.norm(R_herm, "fro")
    min_eig = np.min(evals)
    
    # Negative eigenvalue energy ratio
    evals_abs = np.abs(evals)
    neg_energy = np.sum(np.maximum(-evals, 0.0))
    total_energy = np.sum(evals_abs)
    eps = 1e-12
    neg_eig_energy_ratio = neg_energy / (total_energy + eps) if total_energy > eps else 0.0
    
    # Rank (eigenvalues > threshold)
    max_eig = np.max(np.abs(evals))
    rank_threshold = 1e-8 * max_eig
    rank = np.sum(evals > rank_threshold)
    
    # Relative Frobenius error (if R_true provided)
    rel_fro_err = None
    if R_true is not None:
        error_matrix = R_herm - R_true
        abs_error = np.linalg.norm(error_matrix, "fro")
        norm_true = np.linalg.norm(R_true, "fro")
        rel_fro_err = abs_error / (norm_true + eps) if norm_true > eps else np.inf
    
    return {
        'trace': trace,
        'fro_norm': fro_norm,
        'min_eig': min_eig,
        'neg_eig_energy_ratio': neg_eig_energy_ratio,
        'rank': rank,
        'rel_fro_err': rel_fro_err,
    }


def apply_postprocessing(R_raw: np.ndarray, mode: str, rank_k: int = None, gamma: float = 0.0) -> np.ndarray:
    """
    Apply post-processing to estimated covariance matrix.
    
    Args:
        R_raw: Raw estimated covariance (N_R, N_R)
        mode: Post-processing mode: 'none', 'hermitian', 'psd', 'psd_trace', 'psd_rankk', 'psd_rankk_trace', 
              'psd_shrink', 'psd_rankk_shrink'
        rank_k: Rank for rank-k truncation (required for psd_rankk modes)
        gamma: Shrinkage parameter for shrink modes (default: 0.0)
    
    Returns:
        R_post: Post-processed matrix
    """
    if mode == 'none':
        return R_raw
    
    # Always apply Hermitian symmetrization when any postproc is enabled
    R = 0.5 * (R_raw + R_raw.conj().T)
    
    if mode == 'hermitian':
        return R
    
    # Eigendecomposition
    evals, evecs = np.linalg.eigh(R)
    
    if mode == 'psd':
        # Clamp negative eigenvalues to zero
        evals_pos = np.maximum(evals, 0.0)
        R_psd = evecs @ np.diag(evals_pos) @ evecs.conj().T
        return R_psd
    
    elif mode == 'psd_trace':
        # PSD + trace scaling
        evals_pos = np.maximum(evals, 0.0)
        R_psd = evecs @ np.diag(evals_pos) @ evecs.conj().T
        trace_before = np.real(np.trace(R))
        trace_after = np.real(np.trace(R_psd))
        if trace_after > 0:
            scale_factor = trace_before / trace_after
            R_psd = R_psd * scale_factor
        return R_psd
    
    elif mode == 'psd_rankk':
        if rank_k is None:
            raise ValueError("rank_k must be specified for psd_rankk mode")
        # PSD + rank-k truncation
        evals_pos = np.maximum(evals, 0.0)
        # Keep top-k eigenvalues
        k = min(rank_k, len(evals_pos))
        idx_sorted = np.argsort(evals_pos)[::-1]  # Descending order
        idx_topk = idx_sorted[:k]
        evals_topk = np.zeros_like(evals_pos)
        evals_topk[idx_topk] = evals_pos[idx_topk]
        R_rankk = evecs @ np.diag(evals_topk) @ evecs.conj().T
        return R_rankk
    
    elif mode == 'psd_rankk_trace':
        if rank_k is None:
            raise ValueError("rank_k must be specified for psd_rankk_trace mode")
        # PSD + rank-k + trace scaling
        evals_pos = np.maximum(evals, 0.0)
        k = min(rank_k, len(evals_pos))
        idx_sorted = np.argsort(evals_pos)[::-1]
        idx_topk = idx_sorted[:k]
        evals_topk = np.zeros_like(evals_pos)
        evals_topk[idx_topk] = evals_pos[idx_topk]
        R_rankk = evecs @ np.diag(evals_topk) @ evecs.conj().T
        trace_before = np.real(np.trace(R))
        trace_after = np.real(np.trace(R_rankk))
        if trace_after > 0:
            scale_factor = trace_before / trace_after
            R_rankk = R_rankk * scale_factor
        return R_rankk
    
    elif mode == 'psd_shrink':
        # PSD + shrinkage
        evals_pos = np.maximum(evals, 0.0)
        R_base = evecs @ np.diag(evals_pos) @ evecs.conj().T
        # Apply shrinkage: R_post = (1-γ) * R_base + γ * (tr(R_base)/N_R) * I
        N_R = R_base.shape[0]
        trace_base = np.real(np.trace(R_base))
        # Only apply shrinkage if gamma > 0
        if gamma > 0:
            R_shrink = (1.0 - gamma) * R_base + gamma * (trace_base / N_R) * np.eye(N_R, dtype=R_base.dtype)
        else:
            R_shrink = R_base.copy()
        # Ensure Hermitian symmetry for numerical stability
        R_shrink = 0.5 * (R_shrink + R_shrink.conj().T)
        return R_shrink
    
    elif mode == 'psd_rankk_shrink':
        if rank_k is None:
            raise ValueError("rank_k must be specified for psd_rankk_shrink mode")
        # PSD + rank-k + shrinkage
        evals_pos = np.maximum(evals, 0.0)
        k = min(rank_k, len(evals_pos))
        idx_sorted = np.argsort(evals_pos)[::-1]
        idx_topk = idx_sorted[:k]
        evals_topk = np.zeros_like(evals_pos)
        evals_topk[idx_topk] = evals_pos[idx_topk]
        R_base = evecs @ np.diag(evals_topk) @ evecs.conj().T
        # Apply shrinkage
        N_R = R_base.shape[0]
        trace_base = np.real(np.trace(R_base))
        # Only apply shrinkage if gamma > 0
        if gamma > 0:
            R_shrink = (1.0 - gamma) * R_base + gamma * (trace_base / N_R) * np.eye(N_R, dtype=R_base.dtype)
        else:
            R_shrink = R_base.copy()
        # Ensure Hermitian symmetry for numerical stability
        R_shrink = 0.5 * (R_shrink + R_shrink.conj().T)
        return R_shrink
    
    else:
        raise ValueError(f"Unknown postproc_mode: {mode}")


def project_to_psd(R: np.ndarray, preserve_trace: bool = False) -> tuple[np.ndarray, dict]:
    """
    Project a Hermitian matrix onto the positive semidefinite (PSD) cone.
    
    Args:
        R: Hermitian matrix (N_R, N_R) to project
        preserve_trace: If True, scale result to preserve trace of original matrix
    
    Returns:
        R_psd: Projected PSD matrix
        info: Dictionary with diagnostic info:
            - trace_before: trace of input matrix
            - trace_after: trace after projection (before scaling)
            - trace_after_scaled: trace after scaling (if preserve_trace)
            - neg_eig_energy_ratio: ratio of negative eigenvalue energy to total
    """
    # Ensure Hermitian symmetry
    R_herm = 0.5 * (R + R.conj().T)
    
    # Eigendecomposition (for Hermitian matrices, eigh is appropriate)
    evals, evecs = np.linalg.eigh(R_herm)
    
    # Compute diagnostic info before clamping
    evals_abs = np.abs(evals)
    neg_mask = evals < 0
    neg_energy = np.sum(np.abs(evals[neg_mask]))
    total_energy = np.sum(evals_abs)
    eps = 1e-12
    neg_eig_energy_ratio = neg_energy / (total_energy + eps) if total_energy > eps else 0.0
    
    trace_before = np.real(np.trace(R_herm))
    
    # Clamp negative eigenvalues to zero
    evals_pos = np.maximum(evals, 0.0)
    
    # Reconstruct: R_psd = evecs @ diag(evals_pos) @ evecs.conj().T
    R_psd = evecs @ np.diag(evals_pos) @ evecs.conj().T
    
    trace_after = np.real(np.trace(R_psd))
    
    # Trace-preserving scaling
    trace_after_scaled = trace_after
    if preserve_trace and trace_after > 0:
        scale_factor = trace_before / trace_after
        R_psd = R_psd * scale_factor
        trace_after_scaled = np.real(np.trace(R_psd))
    
    info = {
        'trace_before': trace_before,
        'trace_after': trace_after,
        'trace_after_scaled': trace_after_scaled,
        'neg_eig_energy_ratio': neg_eig_energy_ratio,
    }
    
    return R_psd, info


def generate_data_symbols(n_tx: int, n_time: int, modulation: str = "bpsk") -> np.ndarray:
    """
    Generate i.i.d. data symbols X_d over time.
    
    Args:
        n_tx: Number of transmit antennas (or TX dimensions)
        n_time: Number of time samples N_d
        modulation: "bpsk" or "qpsk"
    
    Returns:
        X_d: (n_tx, n_time) complex array - each column is a data symbol vector x_d(t)
    """
    if modulation.lower() == "bpsk":
        # BPSK: ±1 per entry (real-valued, but stored as complex)
        x_real = 2 * (np.random.rand(n_tx, n_time) > 0.5).astype(float) - 1
        X_d = x_real.astype(complex)
    elif modulation.lower() == "qpsk":
        # QPSK: (±1 ± j)/√2 per entry (normalized to unit power)
        x_real = 2 * (np.random.rand(n_tx, n_time) > 0.5).astype(float) - 1
        x_imag = 2 * (np.random.rand(n_tx, n_time) > 0.5).astype(float) - 1
        X_d = (x_real + 1j * x_imag) / np.sqrt(2.0)
    else:
        raise ValueError(f"Unknown modulation: {modulation}")
    
    return X_d


def main():
    parser = argparse.ArgumentParser(
        description="Covariance estimation from noisy data observations Y_d = H @ X_d + N using TIME AVERAGING"
    )
    parser.add_argument(
        "--snrs", type=str, default="-15.0", metavar='SNR_SPEC',
        # Examples (use = syntax to avoid shell parsing issues):
        #   "--snrs=-15:5:1" (range from -15 to 5 with step 1, RECOMMENDED)
        #   "--snrs=-15:5" (range from -15 to 5 with default step 1)
        #   "--snrs=-15 -10 -5" (space-separated list, use quotes: '--snrs="-15 -10 -5"')
        #   "--snrs '-15:5:1'" (alternative: use quotes in bash/zsh)
        help="SNRs in dB to test. Can be space-separated list or range. RECOMMENDED: Use '=' syntax to avoid shell issues (e.g., '--snrs=-15:5:1' for range from -15 to 5 with step 1, or '--snrs=-15:5' for default step 1). For lists, use quotes: '--snrs=\"-15 -10 -5\"'."
    )
    parser.add_argument(
        "--ch_type", type=str, default="3gpp",
        help="Channel type passed to load_or_create_data"
    )
    parser.add_argument("--n_path", type=int, default=3, help="n_path for 3gpp loader")
    parser.add_argument("--n_antennas_rx", type=int, default=64, help="Number of RX antennas N_R")
    parser.add_argument("--n_antennas_tx", type=int, default=16, help="Number of TX antennas N_T")
    parser.add_argument("--n_train_ch", type=int, default=100_000, help="Training channels to load (unused)")
    parser.add_argument("--n_val_ch", type=int, default=10_000, help="Validation channels to load (unused)")
    parser.add_argument("--n_test_ch", type=int, default=10_000, help="Test channels to load (we use one)")
    parser.add_argument(
        "--n_time_samples", type=int, default=300,
        help="Number of time samples N_d for time-averaged covariance estimation"
    )
    parser.add_argument(
        "--channel_idx", type=int, default=0,
        help="Index of channel realization to use (default: 0)"
    )
    parser.add_argument(
        "--modulation", type=str, default="bpsk", choices=["bpsk", "qpsk"],
        help="Modulation for data symbols (default: bpsk)"
    )
    parser.add_argument(
        "--n_trials", type=int, default=1,
        help="Number of Monte Carlo trials per SNR (default: 1). "
             "In each trial, X_d and noise N are regenerated, but H remains fixed."
    )
    parser.add_argument(
        "--psd_project", action="store_true",
        help="[Backward compatibility] If set and --postproc_mode not specified, sets postproc_mode='psd'."
    )
    parser.add_argument(
        "--preserve_trace", action="store_true",
        help="[Backward compatibility] If set with --psd_project, sets postproc_mode='psd_trace'."
    )
    parser.add_argument(
        "--postproc_mode", type=str, default=None,
        choices=['none', 'hermitian', 'psd', 'psd_trace', 'psd_rankk', 'psd_rankk_trace', 'psd_shrink', 'psd_rankk_shrink'],
        help="Post-processing mode for estimated covariance. Default: 'none' (or 'psd' if --psd_project is set)."
    )
    parser.add_argument(
        "--rank_k", type=int, default=16,
        help="Rank for rank-k truncation in psd_rankk modes (default: 16)."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility (default: 0)."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.0,
        help="Shrinkage parameter for shrink modes (default: 0.0)."
    )
    parser.add_argument(
        "--gamma_list", type=str, default="",
        help="Comma-separated list of gamma values for sweep (e.g., '0,0.01,0.05,0.1,0.2'). Empty string disables sweep."
    )
    parser.add_argument(
        "--run_gamma_sweep", action="store_true",
        help="If set, run estimator for each gamma in --gamma_list and print summary table."
    )
    args = parser.parse_args()
    
    # Parse SNR specification (supports both list and range syntax)
    args.snrs = parse_snr_range(args.snrs)
    
    # Handle backward compatibility: --psd_project sets postproc_mode if not explicitly set
    if args.postproc_mode is None:
        if args.psd_project:
            if args.preserve_trace:
                args.postproc_mode = 'psd_trace'
            else:
                args.postproc_mode = 'psd'
        else:
            args.postproc_mode = 'none'
    
    # Parse gamma list if sweep is enabled
    gamma_values = []
    if args.run_gamma_sweep:
        if not args.gamma_list:
            raise ValueError("--gamma_list must be provided when --run_gamma_sweep is set")
        try:
            gamma_values = [float(g.strip()) for g in args.gamma_list.split(',')]
        except ValueError:
            raise ValueError(f"Invalid --gamma_list format: '{args.gamma_list}'. Use comma-separated floats like '0,0.01,0.05'")
    else:
        gamma_values = [args.gamma]
    
    # Set random seed for reproducibility (will be re-seeded per gamma in sweep mode)
    np.random.seed(args.seed)
    
    # Print configuration
    print(f"Configuration:")
    print(f"  postproc_mode: {args.postproc_mode}")
    if args.postproc_mode in ('psd_rankk', 'psd_rankk_trace', 'psd_rankk_shrink'):
        print(f"  rank_k: {args.rank_k}")
    print(f"  seed: {args.seed}")
    if args.run_gamma_sweep:
        print(f"  gamma_list: {gamma_values}")
    else:
        print(f"  gamma: {args.gamma}")
    
    # Warn if gamma is set but mode doesn't use it
    if not args.run_gamma_sweep and args.gamma > 0:
        if args.postproc_mode not in ('psd_shrink', 'psd_rankk_shrink'):
            print(f"  WARNING: gamma={args.gamma} is set but postproc_mode='{args.postproc_mode}' does not use shrinkage.")
            print(f"           Shrinkage is only applied in 'psd_shrink' or 'psd_rankk_shrink' modes.")
            print(f"           gamma will be ignored in this mode.")
    elif args.postproc_mode in ('psd_shrink', 'psd_rankk_shrink'):
        if not args.run_gamma_sweep and args.gamma == 0:
            print(f"  NOTE: Using shrink mode '{args.postproc_mode}' with gamma=0 (no shrinkage applied).")
    print()

    N_R = args.n_antennas_rx
    N_T = args.n_antennas_tx
    N_d = args.n_time_samples

    # ========================================================================
    # STEP 1: Load channels and select a SINGLE channel matrix H
    # ========================================================================
    try:
        _, _, channels = ut.load_or_create_data(
            ch_type=args.ch_type,
            n_path=args.n_path,
            n_antennas_rx=N_R,
            n_antennas_tx=N_T,
            n_train_ch=args.n_train_ch,
            n_val_ch=args.n_val_ch,
            n_test_ch=args.n_test_ch,
            return_toep=False,
        )
        # channels shape for 3gpp, tx>1: stored vectorized (Fortran order).
        h_all = channels
        if h_all.ndim > 2:
            # Flatten spatial dimensions so h_all has shape (N, D) where D = N_R * N_T
            h_all = h_all.reshape(h_all.shape[0], -1, order="F")
    except FileNotFoundError:
        # Fallback: synthetic i.i.d. Rayleigh channels when dataset is unavailable.
        print("Dataset not found; generating synthetic Rayleigh channels for the diagnostic.")
        D = N_R * N_T
        h_all = ut.crandn(args.n_test_ch, D)

    N_channels, D_expected = h_all.shape
    if args.channel_idx >= N_channels:
        raise ValueError(f"channel_idx={args.channel_idx} exceeds available channels (N={N_channels})")
    if D_expected != N_R * N_T:
        raise ValueError(f"Channel dimension mismatch: expected {N_R * N_T}, got {D_expected}")

    # Select a SINGLE channel and reshape to matrix form H: (N_R, N_T)
    h0_vec = h_all[args.channel_idx]  # shape (N_R * N_T,)
    H = h0_vec.reshape(N_R, N_T, order="F")  # Reshape to (N_R, N_T) - Fortran order to match storage
    print(f"Using channel realization {args.channel_idx} from {N_channels} available channels")
    print(f"Channel matrix H: shape ({N_R}, {N_T})")
    print(f"Number of time samples N_d = {N_d}")
    print()

    # ========================================================================
    # STEP 2: Define the TRUE reference covariance R_h_true = H @ H.conj().T
    # ========================================================================
    # True second-order statistic for this single channel matrix:
    R_h_true = H @ H.conj().T  # shape (N_R, N_R)
    norm_R_true = np.linalg.norm(R_h_true, "fro")
    
    print(f"True covariance R_h_true = H @ H.conj().T: shape ({N_R}, {N_R})")
    print(f"||R_h_true||_F = {norm_R_true:.6e}")
    print(f"rank(R_h_true) = {np.linalg.matrix_rank(R_h_true)} (max possible: {min(N_R, N_T)})")
    print()

    # ========================================================================
    # STEP 3: For each gamma (if sweep) or single gamma, run estimator
    # ========================================================================
    # Store results for gamma sweep summary
    gamma_sweep_results = []
    
    for gamma_idx, gamma in enumerate(gamma_values):
        # Re-seed before each gamma to ensure identical Monte Carlo noise
        np.random.seed(args.seed)
        
        if args.run_gamma_sweep:
            print(f"\n{'='*60}")
            print(f"Gamma = {gamma} ({gamma_idx + 1}/{len(gamma_values)})")
            print(f"{'='*60}")
            # Print headers for this gamma
            if args.n_trials == 1:
                print("SNR_dB | rel_err (||R_h_hat - R_h_true||_F / ||R_h_true||_F) | ||R_h_hat - R_h_true||_F")
                print("-" * 80)
            else:
                print(f"Monte Carlo averaging with {args.n_trials} trials per SNR")
                print("SNR_dB | mean_rel_err | std_rel_err")
                print("-" * 50)
        else:
            # Normal mode: print headers once (only for first gamma)
            if gamma_idx == 0:
                if args.n_trials == 1:
                    # Original single-run format (backward compatible)
                    print("SNR_dB | rel_err (||R_h_hat - R_h_true||_F / ||R_h_true||_F) | ||R_h_hat - R_h_true||_F")
                    print("-" * 80)
                else:
                    # Monte Carlo format
                    print(f"Monte Carlo averaging with {args.n_trials} trials per SNR")
                    print("SNR_dB | mean_rel_err | std_rel_err")
                    print("-" * 50)
        
        # Store results for this gamma (for summary table)
        gamma_snr_results = []
        
        for snr_db in args.snrs:
            # Noise variance per complex entry: sigma_n^2 = 10^(-snr/10)
            # This matches the noise variance in get_observation:
            #   get_observation uses: y = h + 10^(-snr/20) * crandn(...)
            #   where crandn has variance 1 per complex entry
            #   so noise variance = [10^(-snr/20)]^2 = 10^(-snr/10)
            sigma2 = 10 ** (-snr_db / 10)

            # Store relative errors from all trials
            rel_err_list = []
            # Store matrices from last trial for diagnostics
            R_raw_last = None
            R_post_last = None
            
            # Monte Carlo loop: repeat experiment n_trials times
            for trial in range(args.n_trials):
                # --------------------------------------------------------------------
                # STEP 3a: Generate N_d i.i.d. data symbols X_d: (N_T, N_d)
                #          Each column x_d(t) is a data symbol vector at time t
                #          NOTE: Regenerated for each trial
                # --------------------------------------------------------------------
                X_d = generate_data_symbols(N_T, N_d, modulation=args.modulation)  # shape (N_T, N_d)

                # --------------------------------------------------------------------
                # STEP 3b: Generate received signals Y_d = H @ X_d + N: (N_R, N_d)
                #          Each column y_d(t) = H @ x_d(t) + n(t) is the received signal at time t
                #          NOTE: Noise N is regenerated for each trial, but H remains fixed
                # --------------------------------------------------------------------
                # Compute signal part: H @ X_d
                Y_signal = H @ X_d  # shape (N_R, N_d)
                
                # Generate noise N: (N_R, N_d) - independent across antennas and time
                # Noise power per complex entry = sigma2 = 10^(-snr/10)
                noise_power_per_entry = np.sqrt(sigma2)  # std dev for crandn (which has unit variance)
                N = noise_power_per_entry * ut.crandn(N_R, N_d)  # shape (N_R, N_d)
                
                # Received signals
                Y_d = Y_signal + N  # shape (N_R, N_d)

                # --------------------------------------------------------------------
                # STEP 3c: Compute TIME-AVERAGED covariance
                #          R_y_hat = (1/N_d) * Y_d @ Y_d.conj().T
                #          This implements: R_y_hat = (1/N_d) * sum_{t=1}^{N_d} y_d(t) y_d(t)^H
                # --------------------------------------------------------------------
                # IMPORTANT: The sum is over time index t (columns of Y_d), denominator is N_d
                R_y_hat = (Y_d @ Y_d.conj().T) / N_d  # shape (N_R, N_R)
                
                # --------------------------------------------------------------------
                # STEP 3d: Estimate channel covariance by subtracting noise covariance
                #          R_h_hat = R_y_hat - sigma_n^2 * I_N_R
                # --------------------------------------------------------------------
                R_h_hat_raw = R_y_hat - sigma2 * np.eye(N_R, dtype=R_y_hat.dtype)  # shape (N_R, N_R)

                # --------------------------------------------------------------------
                # STEP 3d_postproc: Apply post-processing
                # --------------------------------------------------------------------
                R_h_hat = apply_postprocessing(R_h_hat_raw, mode=args.postproc_mode, rank_k=args.rank_k, gamma=gamma)

                # --------------------------------------------------------------------
                # STEP 3e: Compute estimation error
                # --------------------------------------------------------------------
                error_matrix = R_h_hat - R_h_true  # shape (N_R, N_R)
                abs_error = np.linalg.norm(error_matrix, "fro")
                rel_err = abs_error / norm_R_true if norm_R_true > 0 else np.inf
                
                rel_err_list.append(rel_err)
                
                # Store raw and post-processed matrices from last trial for diagnostics
                if trial == args.n_trials - 1:
                    R_raw_last = R_h_hat_raw.copy()
                    R_post_last = R_h_hat.copy()

            # Report results
            if args.n_trials == 1:
                # Single trial: print original format
                # Recompute abs_error for display (from the single trial)
                rel_err = rel_err_list[0]
                abs_error = rel_err * norm_R_true if norm_R_true > 0 else np.inf
                print(f"{snr_db:6.1f} | {rel_err: .6e} | {abs_error: .6e}")
                mean_rel_err = rel_err
                std_rel_err = 0.0
            else:
                # Multiple trials: print mean and std
                mean_rel_err = np.mean(rel_err_list)
                std_rel_err = np.std(rel_err_list, ddof=1)  # Use sample std (ddof=1)
                print(f"{snr_db:6.1f} | {mean_rel_err: .6e} | {std_rel_err: .6e}")
            
            # Print comprehensive diagnostics (using last trial's matrices)
            if R_raw_last is not None and R_post_last is not None:
                stats_raw = compute_matrix_stats(R_raw_last, R_true=R_h_true)
                stats_post = compute_matrix_stats(R_post_last, R_true=R_h_true)
                
                # if not args.run_gamma_sweep:
                #     # Print detailed diagnostics in normal mode
                #     print(f"  Diagnostics: trace_raw={stats_raw['trace']:.6e}, trace_post={stats_post['trace']:.6e}, "
                #           f"fro_raw={stats_raw['fro_norm']:.6e}, fro_post={stats_post['fro_norm']:.6e}, "
                #           f"min_eig_raw={stats_raw['min_eig']:.6e}, min_eig_post={stats_post['min_eig']:.6e}, "
                #           f"neg_eig_energy_ratio_raw={stats_raw['neg_eig_energy_ratio']:.6e}, "
                #           f"neg_eig_energy_ratio_post={stats_post['neg_eig_energy_ratio']:.6e}, "
                #           f"rank_raw={stats_raw['rank']}, rank_post={stats_post['rank']}, "
                #           f"rel_fro_err_raw={stats_raw['rel_fro_err']:.6e}, "
                #           f"rel_fro_err_post={stats_post['rel_fro_err']:.6e}")
                
                # # Store results for gamma sweep summary
                # if args.run_gamma_sweep:
                #     gamma_snr_results.append({
                #         'snr_db': snr_db,
                #         'mean_rel_err': mean_rel_err,
                #         'std_rel_err': std_rel_err,
                #         'rel_fro_err_post': stats_post['rel_fro_err'] if stats_post['rel_fro_err'] is not None else np.nan,
                #         'trace_post': stats_post['trace'],
                #         'rank_post': stats_post['rank'],
                #         'min_eig_post': stats_post['min_eig'],
                #     })
        
        # Store results for this gamma
        if args.run_gamma_sweep:
            gamma_sweep_results.append({
                'gamma': gamma,
                'snr_results': gamma_snr_results,
            })
    
    # Print gamma sweep summary table
    if args.run_gamma_sweep:
        print(f"\n{'='*60}")
        print("Gamma Sweep Summary")
        print(f"{'='*60}")
        # For simplicity, aggregate across all SNRs (or use first SNR if multiple)
        # User can modify to show per-SNR if needed
        print("gamma | mean_rel_err | std_rel_err | rel_fro_err_post | trace_post | rank_post | min_eig_post")
        print("-" * 100)
        for gamma_result in gamma_sweep_results:
            gamma = gamma_result['gamma']
            # Aggregate across SNRs (take mean of means, or use first SNR)
            if gamma_result['snr_results']:
                # Use first SNR result (or could aggregate)
                first_snr = gamma_result['snr_results'][0]
                print(f"{gamma:5.3f} | {first_snr['mean_rel_err']: .6e} | {first_snr['std_rel_err']: .6e} | "
                      f"{first_snr['rel_fro_err_post']: .6e} | {first_snr['trace_post']: .6e} | "
                      f"{first_snr['rank_post']:3d} | {first_snr['min_eig_post']: .6e}")

    # ========================================================================
    # NOTES ON THE IMPLEMENTATION
    # ========================================================================
    # print()
    # print("=" * 80)
    # print("Implementation Notes:")
    # print("=" * 80)
    # print("1. TIME AVERAGING (not ensemble averaging):")
    # print("   - We pick ONE channel matrix H and generate N_d time samples")
    # print("   - Each time sample t corresponds to a data symbol x_d(t) and received signal y_d(t)")
    # print("   - The covariance R_y_hat = (1/N_d) * Y_d @ Y_d.conj().T averages OVER TIME")
    # print()
    # print("2. Data symbol generation:")
    # print(f"   - Modulation: {args.modulation.upper()}")
    # print("   - X_d: (N_T, N_d) - each column is an i.i.d. data symbol vector")
    # print()
    # print("3. Received signal model:")
    # print("   - Y_d = H @ X_d + N, where Y_d: (N_R, N_d)")
    # print("   - Each column y_d(t) = H @ x_d(t) + n(t) is the received signal at time t")
    # print("   - Noise N has variance sigma_n^2 = 10^(-SNR/10) per complex entry")
    # print()
    # print("4. Covariance estimation:")
    # print("   - R_y_hat = (1/N_d) * Y_d @ Y_d.conj().T  [TIME AVERAGE over columns]")
    # print("   - R_h_hat = R_y_hat - sigma_n^2 * I_N_R   [subtract noise covariance]")
    # print("   - R_h_true = H @ H.conj().T               [true reference]")
    # print()
    # print("5. Expected behavior:")
    # print("   - rel_err should DECREASE as SNR increases (sigma_n^2 -> 0)")
    # print("   - rel_err should DECREASE as N_d increases (more time samples)")
    # print("   - In limit: sigma_n^2 -> 0 and N_d -> inf, rel_err -> 0")
    # print()
    # print("6. Key difference from ensemble averaging:")
    # print("   - Ensemble: R = (1/N) * sum_{i=1}^N h_i h_i^H  [averaging over channels]")
    # print("   - Time:     R = (1/N_d) * sum_{t=1}^{N_d} y_d(t) y_d(t)^H  [averaging over time]")
    # print("   - This diagnostic uses PURE TIME AVERAGING for a fixed channel H")


if __name__ == "__main__":
    main()
