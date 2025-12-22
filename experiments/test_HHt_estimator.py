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
"""

import argparse
import os
import sys
import numpy as np

# Ensure repo root is on sys.path so `modules` can be imported when running directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import modules.utils as ut


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
        "--snrs", type=float, nargs="+", default=[-10.0, 0.0, 10.0, 20.0, 30.0],
        help="List of SNRs in dB to test"
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
        "--n_time_samples", type=int, default=500,
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
    args = parser.parse_args()

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
    # STEP 3: For each SNR, generate data symbols and estimate covariance
    # ========================================================================
    print("SNR_dB | rel_err (||R_h_hat - R_h_true||_F / ||R_h_true||_F) | ||R_h_hat - R_h_true||_F")
    print("-" * 80)
    
    for snr_db in args.snrs:
        # Noise variance per complex entry: sigma_n^2 = 10^(-snr/10)
        # This matches the noise variance in get_observation:
        #   get_observation uses: y = h + 10^(-snr/20) * crandn(...)
        #   where crandn has variance 1 per complex entry
        #   so noise variance = [10^(-snr/20)]^2 = 10^(-snr/10)
        sigma2 = 10 ** (-snr_db / 10)

        # --------------------------------------------------------------------
        # STEP 3a: Generate N_d i.i.d. data symbols X_d: (N_T, N_d)
        #          Each column x_d(t) is a data symbol vector at time t
        # --------------------------------------------------------------------
        X_d = generate_data_symbols(N_T, N_d, modulation=args.modulation)  # shape (N_T, N_d)

        # --------------------------------------------------------------------
        # STEP 3b: Generate received signals Y_d = H @ X_d + N: (N_R, N_d)
        #          Each column y_d(t) = H @ x_d(t) + n(t) is the received signal at time t
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
        R_h_hat = R_y_hat - sigma2 * np.eye(N_R, dtype=R_y_hat.dtype)  # shape (N_R, N_R)

        # --------------------------------------------------------------------
        # STEP 3e: Compute estimation error
        # --------------------------------------------------------------------
        error_matrix = R_h_hat - R_h_true  # shape (N_R, N_R)
        abs_error = np.linalg.norm(error_matrix, "fro")
        rel_err = abs_error / norm_R_true if norm_R_true > 0 else np.inf

        print(f"{snr_db:6.1f} | {rel_err: .6e} | {abs_error: .6e}")

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
