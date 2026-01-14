"""
Evaluate the pretrained DMCE diffusion model using Diffusion Posterior Sampling (DPS).
This script mirrors `load_and_eval_dm.py` but toggles the Tester into DPS mode so that
we can obtain posterior samples without retraining the unconditional DiffusionModel.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import DMCE
from DMCE.utils import cmplx2real
import modules.utils as ut


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate DMCE with DPS sampling')
    # Auto-detect device: use CUDA if available, otherwise CPU
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', '-d', default=default_device, type=str)
    parser.add_argument(
        '--dps_lambda',
        type=str,
        default='0.1',
        help='DPS correction strength(s). Can be single value or comma-separated list (e.g., "0.1,0.3,0.5")',
    )
    parser.add_argument('--sigma_y2', type=float, default=1.0, help='AWGN variance used in likelihood')
    parser.add_argument(
        '--use_fixed_sigma_y2',
        action='store_true',
        help='Use a fixed --sigma_y2 for likelihood (and closed-form variants) instead of computing sigma_y2 from SNR. '
             'Default: disabled (sigma_y2 is derived from SNR and the model noise_multiplier to match functional.awgn).',
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=None,
        help='Number of reverse sampling steps. '
             'Default: None (use all steps from SNR-matched timestep down to 0, same as DM).',
    )
    parser.add_argument('--return_all_timesteps', action='store_true',
                        help='Store DPS estimates for each reverse step')
    parser.add_argument('--reverse_add_random', action='store_true',
                        help='Use stochastic prior reverse steps (same meaning as load_and_eval_dm)')

    # Covariance-guided DPS options
    parser.add_argument(
        '--method',
        type=str,
        default='dps',
        choices=['dps', 'dps_cov_oracle', 'dps_cov_est'],
        help='DPS evaluation mode: '
             '"dps" (no covariance guidance), '
             '"dps_cov_oracle" (oracle R_h = H H^H), '
             '"dps_cov_est" (time-averaged covariance estimate from Y_d).',
    )
    parser.add_argument(
        '--cov_lambda',
        type=float,
        default=0.01,
        help='Strength of covariance guidance term (multiplies cov-grad correction).',
    )
    parser.add_argument(
        '--cov_scale_mode',
        type=str,
        default='beta_t',
        choices=['beta_t', 'sqrt_beta_t', 'constant', 'snr_aware'],
        help='Scaling mode for covariance guidance: "beta_t" (default, backward compatible), '
             '"sqrt_beta_t", "constant", "snr_aware".',
    )
    parser.add_argument(
        '--cov_beta_power',
        type=float,
        default=None,
        help='Optional override for cov scaling: use zeta_t = beta_t**p instead of --cov_scale_mode. '
             'Examples: p=1.0 -> beta_t, p=0.5 -> sqrt_beta_t, p=0.0 -> constant, p=-0.5 -> snr_aware-like.',
    )
    parser.add_argument(
        '--cov_grad_norm',
        type=str,
        default='none',
        choices=['none', 'by_x', 'by_r', 'global'],
        help='Normalization mode for covariance gradient: "none" (default, backward compatible), '
             '"by_x", "by_r", "global".',
    )
    parser.add_argument(
        '--cov_step_clip',
        type=float,
        default=None,
        help='Separate clipping threshold for covariance correction. '
             'If None, uses default step_clip (backward compatible).',
    )
    parser.add_argument(
        '--cov_clip_mode',
        type=str,
        default='auto',
        choices=['auto', 'elementwise', 'norm'],
        help="Clipping mode for covariance correction. "
             "'auto' (default) preserves backward compatibility: elementwise clamp for legacy beta_t path, norm clip otherwise. "
             "'elementwise' clamps each element to [-C, C]. "
             "'norm' rescales the whole update to have L2 norm <= C (preserves direction).",
    )
    parser.add_argument(
        '--use_t_start_scaling',
        action='store_true',
        help='Enable t_start-based scaling: cov_lambda_eff = cov_lambda_base * sqrt(beta[t_start]). '
             'This makes cov guidance strength adaptive to the starting timestep.',
    )
    parser.add_argument(
        '--debug_cov_scaling',
        action='store_true',
        help='Enable debug logging for cov scaling. Records per-timestep statistics to CSV.',
    )
    parser.add_argument(
        '--n_time_samples',
        type=int,
        default=2000,
        help='Number of time samples N_d for time-averaged covariance estimation (only used for dps_cov_est).',
    )
    parser.add_argument(
        '--modulation',
        type=str,
        default='bpsk',
        choices=['bpsk', 'qpsk'],
        help='Modulation used when simulating Y_d for covariance estimation (dps_cov_est); '
             'must mirror experiments/test_HHt_estimator.py.',
    )
    parser.add_argument(
        '--ch_type',
        type=str,
        default='3gpp',
        choices=['3gpp', 'quadriga_LOS'],
        help='Channel type: "3gpp" (default) or "quadriga_LOS". '
             'Must match the available best_models_dm_paper subfolder.',
    )
    parser.add_argument(
        '--n_path',
        type=int,
        default=3,
        help='Number of paths for 3gpp channel (ignored for quadriga_LOS).',
    )
    parser.add_argument(
        '--sanity_snrs',
        action='store_true',
        help='Use sanity check SNR range: [-15, -10, -5, 0, 5] instead of full range.',
    )
    parser.add_argument(
        '--single_snr_db',
        type=float,
        default=None,
        help='If set, evaluate only this single SNR point (in dB), e.g. --single_snr_db -15. '
             'This overrides --sanity_snrs and the default full sweep.',
    )
    parser.add_argument(
        '--record_diagnostics',
        action='store_true',
        help='Record diagnostic metrics (c_t/b_t, clip_rate) for each SNR point.',
    )
    parser.add_argument(
        '--run_fft_diagnostics',
        action='store_true',
        help='Run FFT invariance diagnostics to verify NMSE is invariant to IFFT transformation.',
    )
    parser.add_argument(
        '--fft_diagnostics_debug',
        action='store_true',
        help='Enable debug prints in FFT diagnostics (shows detailed FFT operation info).',
    )
    parser.add_argument(
        '--exp_key',
        type=str,
        default='A',
        choices=['A', 'B', 'C', 'D', 'E', 'Eprime', 'F', 'G', 'H'],
        help='Ablation experiment key: A=baseline (current), B=scalar Jacobian (1/sqrt(alpha_bar_t)), C=autograd gold gradient, D=soft-gated (beta_t * alpha_bar_t^(gamma-1/2)), E=closed-form likelihood (score injection), Eprime=diagnostic closed-form post-add (sigma_t^2 scaling), F=paper-style reverse optimization update, G=paper Algorithm-3 style (DDIM step + posterior correction), H=classic DPS likelihood: x0_hat(H_t) from prior score then grad_like≈(y-x0_hat)/sigma_y2',
    )
    parser.add_argument(
        '--like_snr_gate',
        action='store_true',
        help='EXP H only: enable sigmoid SNR(dB) gate on likelihood strength: '
             'w_like = dps_lambda * beta_t * gate(SNR_dB), gate = 1/(1+exp(-(SNR_dB-SNR0)/Delta)).',
    )
    parser.add_argument(
        '--like_snr0_db',
        type=float,
        default=-10.5,
        help='EXP H only: SNR0 (dB) midpoint for the sigmoid gate. Default: -10.5.',
    )
    parser.add_argument(
        '--like_snr_delta_db',
        type=float,
        default=2.0,
        help='EXP H only: Delta (dB) smoothness for the sigmoid gate. Default: 2.0.',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=1.0,
        help='Gamma parameter for experiment D (soft-gated likelihood guidance). Default: 1.0. Formula: lambda_t = beta_t * alpha_bar_t^(gamma - 1/2)',
    )
    parser.add_argument(
        '--debug_likelihood',
        action='store_true',
        help='Enable debug mode for likelihood guidance: print detailed info for first 3 reverse steps',
    )
    parser.add_argument(
        '--debug_cf',
        action='store_true',
        help="Enable closed-form likelihood debug prints (exp E/F/G): print coefficient/norm diagnostics at a few timesteps for the first batch.",
    )
    parser.add_argument(
        '--debug_cf_microtest',
        action='store_true',
        help="Run a fast closed-form micro-test: only 3 reverse steps (t=T-1, T//2, 0). Useful for quick sanity checks.",
    )
    parser.add_argument(
        '--g_tau1',
        type=int,
        default=0,
        help="Exp G only: stop the Algorithm-3 posterior-correction loop at timestep tau1 (>=1 recommended). "
             "If 0, runs all the way to t=0 (often unstable).",
    )
    parser.add_argument(
        '--enable_snr_matching',
        action='store_true',
        help="Enable SNR-matched start timestep selection (t_start) even for ablation experiments. "
             "By default, ablations disable SNR matching to use the full reverse chain.",
    )
    parser.add_argument(
        '--like_weight',
        type=float,
        default=1.0,
        help='Scale factor for exp E closed-form likelihood score (score_total = score_prior + like_weight * score_like_cf).',
    )
    parser.add_argument(
        '--lw_schedule',
        type=str,
        default='const',
        choices=['const', 'ramp', 'lastk'],
        help="Likelihood weight schedule for exp E. 'const' uses --like_weight. 'ramp' uses gated linear ramp based on alpha_bar_t. 'lastk' boosts only the last K steps.",
    )
    parser.add_argument(
        '--lw_tau',
        type=float,
        default=0.95,
        help='Ramp gate threshold tau for lw_schedule=ramp. Default: 0.95.',
    )
    parser.add_argument(
        '--lw_max',
        type=float,
        default=8.0,
        help='Ramp maximum w_max for lw_schedule=ramp. Default: 8.0.',
    )
    parser.add_argument(
        '--lw_end',
        type=float,
        default=1.0,
        help='Last-K schedule end weight for lw_schedule=lastk. Used when t <= lw_k. Default: 1.0.',
    )
    parser.add_argument(
        '--lw_k',
        type=int,
        default=0,
        help='Last-K schedule K for lw_schedule=lastk. Boost applies when t <= lw_k. Default: 0 (only t=0).',
    )
    parser.add_argument(
        '--record_like_balance',
        action='store_true',
        help='Record per-step likelihood-vs-prior strength diagnostics for exp E (saves CSV for first batch of each SNR).',
    )
    parser.add_argument(
        '--like_beta_power',
        type=float,
        default=1.0,
        help='Ablation knob for post-add DPS (exp A/B/C/D only): replace beta_t with beta_t**p in the likelihood correction operator. '
             'Default: 1.0 (no change). Example: p=2.0 tests beta_t^2.',
    )
    args = parser.parse_args()
    
    # Parse lambda values
    try:
        lambda_strs = [s.strip() for s in args.dps_lambda.split(',')]
        args.dps_lambdas = [float(lam) for lam in lambda_strs]
    except ValueError:
        raise ValueError(f'Invalid dps_lambda format: {args.dps_lambda}. Use comma-separated floats like "0.1,0.3,0.5"')
    
    # Debug: print parsed dps_lambda values
    print(f"[DEBUG] Parsed --dps_lambda: '{args.dps_lambda}' -> {args.dps_lambdas}")
    
    return args


def prepare_test_data(ch_type: str,
                      n_dim: int,
                      n_dim2: int,
                      n_path: int,
                      n_train: int,
                      n_val: int,
                      n_test: int) -> Tuple[torch.Tensor, str]:
    _, _, data_test = ut.load_or_create_data(
        ch_type=ch_type,
        n_path=n_path,
        n_antennas_rx=n_dim,
        n_antennas_tx=n_dim2,
        n_train_ch=n_train,
        n_val_ch=n_val,
        n_test_ch=n_test,
        return_toep=False,
    )
    del _

    if ch_type.startswith('3gpp') and n_dim2 > 1:
        data_test = np.reshape(data_test, (-1, n_dim, n_dim2), order='F')

    data_test = torch.from_numpy(np.asarray(data_test[:, None, :]))
    data_test = cmplx2real(data_test, dim=1, new_dim=False).float()

    if ch_type.startswith('3gpp'):
        ch_type = f'{ch_type}_path={n_path}'

    return data_test, ch_type


def load_diffusion_model(model_dir: Path, device: str) -> Tuple[DMCE.DiffusionModel, dict]:
    sim_params = DMCE.utils.load_params(str(model_dir / 'sim_params'))
    cnn_dict = sim_params['unet_dict']
    diff_model_dict = sim_params['diff_model_dict']

    cnn_dict['device'] = device
    cnn = DMCE.CNN(**cnn_dict)
    diffusion_model = DMCE.DiffusionModel(cnn, **diff_model_dict)

    ckpt_dir = model_dir / 'train_models'
    checkpoint = ckpt_dir / sorted(os.listdir(ckpt_dir))[-1]
    model_state = torch.load(checkpoint, map_location=device)
    diffusion_model.load_state_dict(model_state['model'])

    print(f'T = {diffusion_model.num_timesteps}')
    print(f'len(betas) = {diffusion_model.betas.numel()}')
    print('diff_model_dict =', diff_model_dict)

    return diffusion_model, sim_params


def build_tester(model: DMCE.DiffusionModel,
                 data: torch.Tensor,
                 mode: str,
                 return_all_timesteps: bool,
                 dps_lambda: float,
                 sigma_y2: float) -> DMCE.Tester:
    tester_cfg = dict(
        batch_size=512,
        criteria=['nmse'],
        complex_data=False,
        return_all_timesteps=return_all_timesteps,
        fft_pre=True,
        mode=mode,
        use_dps=True,
        dps_lambda=dps_lambda,
        sigma_y2=sigma_y2,
    )
    return DMCE.Tester(model, data=data, **tester_cfg)


def export_table(base_dir: Path,
                 prefix: str,
                 headers: Sequence[str],
                 rows: Sequence[Sequence],
                 suffix: str) -> None:
    path = base_dir / f'{prefix}{suffix}.csv'
    with open(path, 'w', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows([headers, *rows])

def generate_cov_batch(data_batch: torch.Tensor) -> torch.Tensor:
    """
    Generate complex covariance matrices via X X^H.
 
    Args:
        data_batch: real/imag tensor shaped (B, 2, d_out, d_in)
                    channel 0 real, channel 1 imag.
 
    Returns:
        Covariance as real/imag channels shaped (B, 2, d_out, d_out).
    """
    if data_batch.dim() != 4 or data_batch.size(1) != 2:
        raise ValueError("data_batch must have shape (B, 2, d_out, d_in)")
 
    x_comp = torch.complex(data_batch[:, 0], data_batch[:, 1])  # (B, d_out, d_in)
    cov = x_comp @ x_comp.conj().transpose(-1, -2)             # (B, d_out, d_out)
    return torch.stack((cov.real, cov.imag), dim=1)  # (B, 2, d_out, d_out)
 

def _generate_data_symbols_torch(
    n_tx: int,
    n_time: int,
    modulation: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Torch implementation of generate_data_symbols from experiments/test_HHt_estimator.py.

    Returns
    -------
    X_d : complex tensor of shape (n_tx, n_time)
    """
    modulation = modulation.lower()
    if modulation == "bpsk":
        # BPSK: ±1 per entry (real-valued, but stored as complex)
        x_real = 2.0 * (torch.rand(n_tx, n_time, device=device) > 0.5).float() - 1.0
        X_d = x_real.to(dtype=torch.complex64)
    elif modulation == "qpsk":
        # QPSK: (±1 ± j)/√2 per entry (normalized to unit power)
        x_real = 2.0 * (torch.rand(n_tx, n_time, device=device) > 0.5).float() - 1.0
        x_imag = 2.0 * (torch.rand(n_tx, n_time, device=device) > 0.5).float() - 1.0
        X_d = (x_real + 1j * x_imag) / np.sqrt(2.0)
        X_d = X_d.to(dtype=torch.complex64)
    else:
        raise ValueError(f"Unknown modulation: {modulation}")

    return X_d


def test_fft_ifft_unitarity(mode: str = '2D', verbose: bool = True) -> dict:
    """
    Test 1: FFT/IFFT Unitarity Check (Normalization)
    
    Goal: Verify whether the FFT/IFFT pair preserves Frobenius norm.
    
    Steps:
    1. Generate a random complex matrix A ∈ C^{64×16}
    2. Apply FFT to get A_fft
    3. Apply IFFT to get A_rec
    4. Check if ||A||_F^2 ≈ ||A_fft||_F^2 ≈ ||A_rec||_F^2
    
    Parameters
    ----------
    mode : str
        FFT mode: '1D' or '2D' (default: '2D')
    verbose : bool
        If True, print diagnostic information
    
    Returns
    -------
    dict
        Dictionary containing test results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate random complex matrix A ∈ C^{64×16}
    N_R, N_T = 64, 16
    A_real = torch.randn(N_R, N_T, device=device)
    A_imag = torch.randn(N_R, N_T, device=device)
    A = torch.complex(A_real, A_imag)  # (64, 16)
    
    # Convert to real/imag format for FFT function
    # When mode='2D' and _4d_array=False (default), complex_1d_fft expects input shape (batches, 2, N_R, N_T)
    # where dimension 1 is real/imag (0=real, 1=imag) based on utils.py line 273
    # This is the standard format used in the codebase (see load_and_eval_dm_dps.py line 939)
    A_input = torch.stack([A.real, A.imag], dim=0)  # (2, 64, 16)
    A_input = A_input.unsqueeze(0)  # (1, 2, 64, 16) for batch dimension
    # Verify the input format matches what complex_1d_fft expects
    assert A_input.shape == (1, 2, N_R, N_T), f"Expected input shape (1, 2, {N_R}, {N_T}), got {A_input.shape}"
    
    # Apply FFT (same as preprocessing)
    # Use _4d_array=False for mode='2D' (this is the default behavior in the codebase)
    _4d_array = False
    A_fft = ut.complex_1d_fft(A_input, ifft=False, mode=mode, _4d_array=_4d_array)
    
    # Apply IFFT (same as evaluation)
    A_rec = ut.complex_1d_fft(A_fft, ifft=True, mode=mode, _4d_array=_4d_array)
    
    # Convert back to complex
    # Output shape when _4d_array=True: (batches, 2, N_R, N_T)
    # Dimension 1: 0=real, 1=imag
    # Verify shapes before conversion
    if _4d_array:
        # A_fft and A_rec should be (1, 2, N_R, N_T) based on utils.py line 289
        # Check actual shapes and print debug info if mismatch
        if A_fft.shape != (1, 2, N_R, N_T):
            raise RuntimeError(f"Unexpected A_fft shape: expected (1, 2, {N_R}, {N_T}), got {A_fft.shape}. "
                             f"This indicates a mismatch in complex_1d_fft input/output format.")
        if A_rec.shape != (1, 2, N_R, N_T):
            raise RuntimeError(f"Unexpected A_rec shape: expected (1, 2, {N_R}, {N_T}), got {A_rec.shape}. "
                             f"This indicates a mismatch in complex_1d_fft input/output format.")
        # Extract real and imag: A_fft[0, 0] is real part (N_R, N_T), A_fft[0, 1] is imag part (N_R, N_T)
        A_fft_real = A_fft[0, 0]  # Should be (N_R, N_T)
        A_fft_imag = A_fft[0, 1]  # Should be (N_R, N_T)
        A_rec_real = A_rec[0, 0]  # Should be (N_R, N_T)
        A_rec_imag = A_rec[0, 1]  # Should be (N_R, N_T)
        # Verify shapes before creating complex
        if A_fft_real.shape != (N_R, N_T) or A_fft_imag.shape != (N_R, N_T):
            raise RuntimeError(f"Shape mismatch when extracting real/imag: "
                             f"A_fft_real.shape={A_fft_real.shape}, A_fft_imag.shape={A_fft_imag.shape}, "
                             f"expected ({N_R}, {N_T})")
        A_fft_comp = torch.complex(A_fft_real, A_fft_imag)  # (N_R, N_T)
        A_rec_comp = torch.complex(A_rec_real, A_rec_imag)  # (N_R, N_T)
        # Final verification
        if A_fft_comp.shape != A.shape or A_rec_comp.shape != A.shape:
            raise RuntimeError(f"Final shape mismatch: A_fft_comp.shape={A_fft_comp.shape}, "
                             f"A_rec_comp.shape={A_rec_comp.shape}, A.shape={A.shape}")
    else:
        # For _4d_array=False case (standard for mode='2D'), output is (batches, 2, N_R, N_T)
        # Same extraction as _4d_array=True case
        A_fft_comp = torch.complex(A_fft[0, 0], A_fft[0, 1])  # (N_R, N_T) for mode='2D', or (dim,) for mode='1D'
        A_rec_comp = torch.complex(A_rec[0, 0], A_rec[0, 1])  # (N_R, N_T) for mode='2D', or (dim,) for mode='1D'
    
    # Compute Frobenius norms squared
    norm_A_sq = torch.linalg.matrix_norm(A, ord='fro').pow(2).item()
    norm_A_fft_sq = torch.linalg.matrix_norm(A_fft_comp, ord='fro').pow(2).item()
    norm_A_rec_sq = torch.linalg.matrix_norm(A_rec_comp, ord='fro').pow(2).item()
    
    # Reconstruction error
    recon_error = torch.linalg.matrix_norm(A_rec_comp - A, ord='fro').item()
    recon_error_rel = recon_error / (torch.linalg.matrix_norm(A, ord='fro').item() + 1e-12)
    
    # Energy preservation errors
    energy_error_fft = abs(norm_A_sq - norm_A_fft_sq) / (norm_A_sq + 1e-12)
    energy_error_rec = abs(norm_A_sq - norm_A_rec_sq) / (norm_A_sq + 1e-12)
    
    results = {
        'norm_A_sq': norm_A_sq,
        'norm_A_fft_sq': norm_A_fft_sq,
        'norm_A_rec_sq': norm_A_rec_sq,
        'recon_error': recon_error,
        'recon_error_rel': recon_error_rel,
        'energy_error_fft': energy_error_fft,
        'energy_error_rec': energy_error_rec,
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Test 1: FFT/IFFT Unitarity Check (Normalization)")
        print(f"{'='*70}")
        print(f"  Mode: {mode}, Shape: ({N_R}, {N_T})")
        print(f"\n  ||A||_F^2 = {norm_A_sq:.10e}")
        print(f"  ||A_fft||_F^2 = {norm_A_fft_sq:.10e}")
        print(f"  ||A_rec||_F^2 = {norm_A_rec_sq:.10e}")
        print(f"\n  Energy preservation (FFT): |||A||^2 - ||A_fft||^2| / ||A||^2 = {energy_error_fft:.10e}")
        print(f"  Energy preservation (IFFT): |||A||^2 - ||A_rec||^2| / ||A||^2 = {energy_error_rec:.10e}")
        print(f"\n  Reconstruction error: ||A_rec - A||_F = {recon_error:.10e}")
        print(f"  Relative reconstruction error: ||A_rec - A||_F / ||A||_F = {recon_error_rel:.10e}")
        print(f"\n  Expected: Energy errors < 1e-6, Reconstruction error < 1e-6")
        if energy_error_fft < 1e-6 and energy_error_rec < 1e-6 and recon_error_rel < 1e-6:
            print(f"  ✓ PASS: FFT/IFFT preserves energy and reconstructs correctly")
        else:
            print(f"  ✗ FAIL: FFT/IFFT normalization issue detected!")
        print(f"{'='*70}\n")
    
    return results


def test_complex_fft_dimensions(H_sample: torch.Tensor, mode: str = '2D', verbose: bool = True) -> dict:
    """
    Test 2: Complex Conversion & Dimension Check
    
    Goal: Verify that FFT is applied to a true complex matrix on correct dimensions.
    
    Steps:
    1. Take one sample from H_gt with shape [2, 64, 16]
    2. Convert explicitly to complex: Hc = torch.complex(H[0], H[1])
    3. Apply FFT and IFFT only on spatial dimensions
    4. Check energy preservation and reconstruction
    
    Parameters
    ----------
    H_sample : torch.Tensor
        Sample channel matrix, shape (2, N_R, N_T) or (2, ...)
    mode : str
        FFT mode: '1D' or '2D' (default: '2D')
    verbose : bool
        If True, print diagnostic information
    
    Returns
    -------
    dict
        Dictionary containing test results
    """
    device = H_sample.device
    
    # Extract one sample: shape (2, N_R, N_T)
    if H_sample.dim() == 4:
        # Take first batch element: (B, 2, N_R, N_T) -> (2, N_R, N_T)
        H = H_sample[0]  # (2, N_R, N_T)
    elif H_sample.dim() == 3:
        H = H_sample  # Already (2, N_R, N_T)
    else:
        raise ValueError(f"Unexpected shape: {H_sample.shape}")
    
    N_R, N_T = H.shape[1], H.shape[2]
    
    # Convert explicitly to complex
    Hc = torch.complex(H[0], H[1])  # (N_R, N_T)
    
    # Prepare for FFT function
    # When mode='2D', use _4d_array=False (default behavior in codebase)
    # Input format: (batches, 2, N_R, N_T) where dimension 1 is real/imag
    _4d_array = False  # Standard format for mode='2D' (see load_and_eval_dm_dps.py line 939)
    H_input = H.unsqueeze(0)  # (1, 2, N_R, N_T)
    
    # Apply FFT (same as preprocessing)
    Hc_fft = ut.complex_1d_fft(H_input, ifft=False, mode=mode, _4d_array=_4d_array)
    
    # Apply IFFT (same as evaluation)
    Hc_rec = ut.complex_1d_fft(Hc_fft, ifft=True, mode=mode, _4d_array=_4d_array)
    
    # Convert back to complex
    # Output shape when _4d_array=False: (batches, 2, N_R, N_T) based on utils.py line 291
    # Dimension 1: 0=real, 1=imag
    Hc_fft_comp = torch.complex(Hc_fft[0, 0], Hc_fft[0, 1])  # (N_R, N_T) for mode='2D'
    Hc_rec_comp = torch.complex(Hc_rec[0, 0], Hc_rec[0, 1])  # (N_R, N_T) for mode='2D'
    
    # Compute Frobenius norms squared
    norm_Hc_sq = torch.linalg.matrix_norm(Hc, ord='fro').pow(2).item()
    norm_Hc_fft_sq = torch.linalg.matrix_norm(Hc_fft_comp, ord='fro').pow(2).item()
    norm_Hc_rec_sq = torch.linalg.matrix_norm(Hc_rec_comp, ord='fro').pow(2).item()
    
    # Reconstruction error
    recon_error = torch.linalg.matrix_norm(Hc_rec_comp - Hc, ord='fro').item()
    recon_error_rel = recon_error / (torch.linalg.matrix_norm(Hc, ord='fro').item() + 1e-12)
    
    # Energy preservation errors
    energy_error_fft = abs(norm_Hc_sq - norm_Hc_fft_sq) / (norm_Hc_sq + 1e-12)
    energy_error_rec = abs(norm_Hc_sq - norm_Hc_rec_sq) / (norm_Hc_sq + 1e-12)
    
    results = {
        'norm_Hc_sq': norm_Hc_sq,
        'norm_Hc_fft_sq': norm_Hc_fft_sq,
        'norm_Hc_rec_sq': norm_Hc_rec_sq,
        'recon_error': recon_error,
        'recon_error_rel': recon_error_rel,
        'energy_error_fft': energy_error_fft,
        'energy_error_rec': energy_error_rec,
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Test 2: Complex Conversion & Dimension Check")
        print(f"{'='*70}")
        print(f"  Input shape: {H.shape}, Mode: {mode}")
        print(f"  Hc shape (complex): ({N_R}, {N_T})")
        print(f"\n  ||Hc||_F^2 = {norm_Hc_sq:.10e}")
        print(f"  ||Hc_fft||_F^2 = {norm_Hc_fft_sq:.10e}")
        print(f"  ||Hc_rec||_F^2 = {norm_Hc_rec_sq:.10e}")
        print(f"\n  Energy preservation (FFT): |||Hc||^2 - ||Hc_fft||^2| / ||Hc||^2 = {energy_error_fft:.10e}")
        print(f"  Energy preservation (IFFT): |||Hc||^2 - ||Hc_rec||^2| / ||Hc||^2 = {energy_error_rec:.10e}")
        print(f"\n  Reconstruction error: ||Hc_rec - Hc||_F = {recon_error:.10e}")
        print(f"  Relative reconstruction error: ||Hc_rec - Hc||_F / ||Hc||_F = {recon_error_rel:.10e}")
        print(f"\n  Expected: Energy errors < 1e-6, Reconstruction error < 1e-6")
        if energy_error_fft < 1e-6 and energy_error_rec < 1e-6 and recon_error_rel < 1e-6:
            print(f"  ✓ PASS: Complex FFT handling is correct")
        else:
            print(f"  ✗ FAIL: Complex FFT handling issue detected!")
        print(f"{'='*70}\n")
    
    return results


# ============================================================================
# FFT Diagnostics (optional, imported from separate module)
# ============================================================================

# Import FFT diagnostics module (optional feature)
try:
    import fft_diagnostics
    FFT_DIAGNOSTICS_AVAILABLE = True
except ImportError:
    FFT_DIAGNOSTICS_AVAILABLE = False
    fft_diagnostics = None

# ============================================================================
# Helper functions for end-to-end invariance audit (kept for backward compatibility)
# ============================================================================

def to_complex(H_ri: torch.Tensor) -> torch.Tensor:
    """
    Convert real/imag tensor to complex tensor.
    
    Parameters
    ----------
    H_ri : torch.Tensor
        Shape [B, 2, R, T] where dim 1 is real/imag (0=real, 1=imag)
    
    Returns
    -------
    torch.Tensor
        Complex tensor of shape [B, R, T]
    """
    return torch.complex(H_ri[:, 0], H_ri[:, 1])


def fro2(X: torch.Tensor) -> float:
    """
    Compute squared Frobenius norm: ||X||_F^2
    
    Parameters
    ----------
    X : torch.Tensor
        Any tensor (will be flattened and squared)
    
    Returns
    -------
    float
        Sum of squares of all entries
    """
    return torch.sum(torch.abs(X) ** 2).item()


def relerr(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Compute relative Frobenius error: ||A - B||_F / ||A||_F
    
    Parameters
    ----------
    A : torch.Tensor
        Reference tensor
    B : torch.Tensor
        Comparison tensor
    
    Returns
    -------
    float
        Relative error
    """
    num = fro2(A - B)
    den = fro2(A)
    return (num / (den + 1e-12)) ** 0.5


def run_end_to_end_invariance_audit(
    H_gt_ang: torch.Tensor,
    H_hat_ang: torch.Tensor,
    mode: str = '2D',
    verbose: bool = True,
    debug: bool = False,
) -> dict:
    """
    Wrapper function that calls the FFT diagnostics module.
    This function is kept for backward compatibility.
    """
    if FFT_DIAGNOSTICS_AVAILABLE:
        return fft_diagnostics.run_end_to_end_invariance_audit(H_gt_ang, H_hat_ang, mode, verbose, debug)
    else:
        raise ImportError("fft_diagnostics module not available. Please ensure fft_diagnostics.py exists.")


def run_nmse_fft_diagnostics(data_batch: torch.Tensor, mode: str = '2D', verbose: bool = True) -> dict:
    """
    Run both diagnostic tests to determine why NMSE differs before and after IFFT.
    
    Parameters
    ----------
    data_batch : torch.Tensor
        Sample batch of channel data, shape (B, 2, N_R, N_T) or (B, 2, ...)
    mode : str
        FFT mode: '1D' or '2D' (default: '2D')
    verbose : bool
        If True, print diagnostic information
    
    Returns
    -------
    dict
        Dictionary containing results from both tests
    """
    if verbose:
        print(f"\n{'#'*70}")
        print(f"NMSE FFT Invariance Diagnostic Tests")
        print(f"{'#'*70}")
    
    # Test 1: FFT/IFFT Unitarity Check
    test1_results = test_fft_ifft_unitarity(mode=mode, verbose=verbose)
    
    # Test 2: Complex Conversion & Dimension Check
    # Use first sample from data_batch
    H_sample = data_batch[0] if data_batch.dim() == 4 else data_batch
    test2_results = test_complex_fft_dimensions(H_sample, mode=mode, verbose=verbose)
    
    # Summary
    if verbose:
        print(f"\n{'#'*70}")
        print(f"Diagnostic Summary")
        print(f"{'#'*70}")
        print(f"Test 1 (Unitarity):")
        print(f"  Energy error (FFT): {test1_results['energy_error_fft']:.10e}")
        print(f"  Energy error (IFFT): {test1_results['energy_error_rec']:.10e}")
        print(f"  Reconstruction error: {test1_results['recon_error_rel']:.10e}")
        test1_pass = (test1_results['energy_error_fft'] < 1e-6 and 
                     test1_results['energy_error_rec'] < 1e-6 and 
                     test1_results['recon_error_rel'] < 1e-6)
        print(f"  Status: {'✓ PASS' if test1_pass else '✗ FAIL'}")
        
        print(f"\nTest 2 (Complex & Dimensions):")
        print(f"  Energy error (FFT): {test2_results['energy_error_fft']:.10e}")
        print(f"  Energy error (IFFT): {test2_results['energy_error_rec']:.10e}")
        print(f"  Reconstruction error: {test2_results['recon_error_rel']:.10e}")
        test2_pass = (test2_results['energy_error_fft'] < 1e-6 and 
                     test2_results['energy_error_rec'] < 1e-6 and 
                     test2_results['recon_error_rel'] < 1e-6)
        print(f"  Status: {'✓ PASS' if test2_pass else '✗ FAIL'}")
        
        print(f"\nInterpretation:")
        if not test1_pass:
            print(f"  → Root cause #1: FFT normalization mismatch detected")
        if not test2_pass:
            print(f"  → Root cause #2: Incorrect complex handling or FFT dimension usage")
        if test1_pass and test2_pass:
            print(f"  → Both tests pass: NMSE difference must come from metric aggregation, not FFT")
        print(f"{'#'*70}\n")
    
    return {
        'test1': test1_results,
        'test2': test2_results,
    }


def estimate_cov_time_averaged_batch(
    data_batch: torch.Tensor,
    snr_db: float,
    n_time_samples: int,
    modulation: str,
) -> torch.Tensor:
    """
    Time-averaged covariance estimate per channel (batch) and per SNR.

    This mirrors the logic in experiments/test_HHt_estimator.py:
      - Y_d = H @ X_d + N
      - R_y_hat = (1/N_d) * Y_d Y_d^H
      - R_h_hat = R_y_hat - sigma_n^2 I
      - Enforce Hermitian symmetry.

    Parameters
    ----------
    data_batch : Tensor
        Real/imag tensor shaped (B, 2, N_R, N_T) representing channels H.
    snr_db : float
        SNR in dB for this sweep point.
    n_time_samples : int
        Number of time samples N_d.
    modulation : {"bpsk", "qpsk"}
        Modulation used to generate X_d.

    Returns
    -------
    cov_est : Tensor
        Estimated covariance as real/imag channels shaped (B, 2, N_R, N_R).
    """
    if data_batch.dim() != 4 or data_batch.size(1) != 2:
        raise ValueError("data_batch must have shape (B, 2, N_R, N_T)")

    device = data_batch.device
    B, _, N_R, N_T = data_batch.shape

    # Complex channel matrices H: (B, N_R, N_T)
    H_batch = torch.complex(data_batch[:, 0], data_batch[:, 1]).to(device=device)

    # Noise variance per complex entry: sigma_n^2 = 10^(-snr/10)
    sigma2 = float(10 ** (-snr_db / 10.0))
    sigma2_t = torch.tensor(sigma2, dtype=torch.float32, device=device)

    cov_list = []
    for b in range(B):
        H = H_batch[b]  # (N_R, N_T)

        # STEP 3a: Generate N_d i.i.d. data symbols X_d: (N_T, N_d)
        X_d = _generate_data_symbols_torch(N_T, n_time_samples, modulation, device=device)  # (N_T, N_d)

        # STEP 3b: Generate received signals Y_d = H @ X_d + N: (N_R, N_d)
        Y_signal = H @ X_d  # (N_R, N_d)

        # Noise with variance sigma2 per complex entry
        noise_std = torch.sqrt(sigma2_t)
        noise_real = torch.randn(N_R, n_time_samples, device=device)
        noise_imag = torch.randn(N_R, n_time_samples, device=device)
        noise = np.sqrt(0.5) * noise_std * (noise_real + 1j * noise_imag)  # (N_R, N_d)

        Y_d = Y_signal + noise  # (N_R, N_d)

        # STEP 3c: Time-averaged covariance
        R_y_hat = (Y_d @ Y_d.conj().transpose(-1, -2)) / float(n_time_samples)  # (N_R, N_R)

        # STEP 3d: Subtract noise covariance
        eye = torch.eye(N_R, dtype=R_y_hat.dtype, device=device)
        R_h_hat = R_y_hat - sigma2_t * eye

        # STEP 3e: Enforce Hermitian symmetry
        R_h_hat = 0.5 * (R_h_hat + R_h_hat.conj().transpose(-1, -2))

        cov_list.append(torch.stack((R_h_hat.real, R_h_hat.imag), dim=0))  # (2, N_R, N_R)

    cov_est = torch.stack(cov_list, dim=0)  # (B, 2, N_R, N_R)
    return cov_est


def maybe_plot_timesteps(base_dir: Path,
                         prefix: str,
                         num_timesteps: int,
                         snrs: Sequence[float],
                         nmse_curves: Sequence[Sequence[float]]) -> None:
    plt.figure()
    lines = []
    for snr, curve in zip(snrs, nmse_curves):
        start_step = num_timesteps - len(curve) + 1
        xs = range(start_step, num_timesteps + 1)
        (line,) = plt.semilogy(xs, curve, label=f'SNR = {int(snr)}')
        lines.append(line)
    plt.xlabel('Timesteps')
    plt.ylabel('nMSE')
    plt.legend(lines, [l.get_label() for l in lines])
    plt.savefig(base_dir / f'{prefix}_dps_perstep.png')


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    device = args.device
    
    # Print device information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    print()

    # These mirror load_and_eval_dm.py
    n_dim, n_dim2 = 64, 16
    num_train, num_val, num_test = 100_000, 10_000, 10_000
    ch_type, n_path = args.ch_type, args.n_path
    mode = '2D' if n_dim2 > 1 else '1D'

    data_test, ch_type = prepare_test_data(
        ch_type, n_dim, n_dim2, n_path, num_train, num_val, num_test
    )

    cwd = Path(os.getcwd())
    model_dir = cwd / 'results/best_models_dm_paper' / ch_type
    diffusion_model, sim_params = load_diffusion_model(model_dir, device)

    # Initialize tester with first lambda value (will be updated in loop)
    tester = build_tester(
        diffusion_model,
        data=data_test,
        mode=mode,
        return_all_timesteps=args.return_all_timesteps,
        dps_lambda=args.dps_lambdas[0],  # Use first lambda for initialization
        sigma_y2=args.sigma_y2,
    )

    diffusion_model.reverse_add_random = args.reverse_add_random
    
    # SNR range: -15 to 5 dB, step 1 (or custom range for sanity check / single point)
    if args.single_snr_db is not None:
        target_snrs = [float(args.single_snr_db)]
    elif args.sanity_snrs:
        target_snrs = [-15, -10, -5, 0, 5]
    else:
        target_snrs = list(range(-15, 6, 1))
    
    # Test multiple lambda values
    print(f"\n{'='*60}")
    print(f"Testing {len(args.dps_lambdas)} lambda value(s): {args.dps_lambdas}")
    print(f"SNR range: {target_snrs} dB")
    print(f"Method: {args.method}")
    if args.method in ('dps_cov_oracle', 'dps_cov_est'):
        print(f"Cov scale mode: {args.cov_scale_mode}")
        if args.cov_beta_power is not None:
            print(f"Cov beta power override: {args.cov_beta_power}   (zeta_t = beta_t**p)")
        print(f"Cov grad norm: {args.cov_grad_norm}")
        print(f"Cov lambda: {args.cov_lambda}")
        print(f"Cov step clip: {args.cov_step_clip}")
        print(f"Cov clip mode: {args.cov_clip_mode}")
    if args.use_fixed_sigma_y2:
        print(f"Likelihood sigma_y2: FIXED ({args.sigma_y2})")
    else:
        print("Likelihood sigma_y2: SNR-derived (matches functional.awgn)")
        if args.use_t_start_scaling:
            print(f"Per-step scaling: ENABLED (cov_lambda_eff(t) = cov_lambda_base * sqrt(beta[t]) for each step)")
    if args.num_steps is None:
        print("Number of reverse steps: None (use all steps from SNR-matched timestep to 0)")
    else:
        print(f"Number of reverse steps: {args.num_steps}")
    if args.record_diagnostics:
        print("Diagnostic recording: ENABLED")
    if args.exp_key in ['A', 'B', 'C', 'D', 'E', 'Eprime', 'F', 'G']:
        print(f"Ablation experiment: {args.exp_key}")
        if args.exp_key == 'A':
            print("  Baseline: no Jacobian correction")
        elif args.exp_key == 'B':
            print("  Using scalar Jacobian approximation: 1/sqrt(alpha_bar_t)")
        elif args.exp_key == 'C':
            print("  Using autograd gold gradient (full Jacobian)")
        elif args.exp_key == 'D':
            print(f"  Using soft-gated likelihood guidance: beta_t * alpha_bar_t^(gamma - 1/2)")
            print(f"    gamma = {args.gamma}")
            if args.gamma == 1.0:
                print("    (gamma=1.0: lambda_t = beta_t * sqrt(alpha_bar_t), strong early suppression)")
            elif args.gamma == 0.75:
                print("    (gamma=0.75: lambda_t = beta_t * alpha_bar_t^(1/4), milder suppression)")
            elif args.gamma == 0.5:
                print("    (gamma=0.5: lambda_t = beta_t, reduces to baseline A)")
        elif args.exp_key == 'E':
            print("  Using closed-form likelihood (paper-style): score_total = score_prior + like_weight * score_like_cf")
            print(f"    like_weight = {args.like_weight}")
            if args.debug_cf:
                print("  Debug CF enabled: will run only 3 reverse steps (t=T-1, T//2, 0) for a fast sanity check.")
        elif args.exp_key == 'Eprime':
            print("  Diagnostic: closed-form post-add using sigma_t^2 * score_like_cf (no beta_t)")
        elif args.exp_key == 'F':
            print("  Paper-style reverse optimization update: H_t <- H_t + post_var_t * score_post, then noise injection")
            print(f"    like_weight = {args.like_weight}")
        elif args.exp_key == 'G':
            print("  Paper Algorithm-3 style: deterministic DDIM step + explicit posterior correction with closed-form likelihood gradient")
            print("    (In exp G, dps_lambda acts as the paper's lambda in Step 5 correction.)")
        if args.enable_snr_matching:
            print("  IMPORTANT: SNR matching is ENABLED (t_start uses user SNR).")
        else:
            print("  IMPORTANT: For ablation, SNR matching is DISABLED (using full T)")
    print(f"{'='*60}\n")
    
    # Custom test function that uses specified SNR range
    def custom_test_nmse():
        """Custom _test_nmse that tests SNR -15 to 5 dB (step 1)"""
        # Copy the original method but modify SNR range
        import time
        from tqdm import tqdm
        from DMCE import functional
        import modules.utils as ut
        
        # DEBUG: Print to confirm this function is being called
        print(f"\n[DEBUG] Using custom_test_nmse with SNR range: -15 to 5 dB (step 1)\n")
        
        # Use target_snrs from outer scope (can be custom for sanity check)
        # For debug mode, only test the first SNR to speed up
        if args.debug_cov_scaling and len(target_snrs) > 1:
            # Only test first SNR for debug
            debug_snrs = [target_snrs[0]]
            snr_db_range = torch.tensor(debug_snrs, dtype=torch.float32, device=tester.device)
            print(f"[Debug mode] Testing only SNR={debug_snrs[0]} dB")
        else:
            snr_db_range = torch.tensor(target_snrs, dtype=torch.float32, device=tester.device)
        # corresponding linear SNR ρ = 10^(SNR/10)
        snr_range = 10 ** (snr_db_range / 10)
        
        nmse_total_power_list = []
        timings_sec = []
        steps_list = []
        tps_ms_list = []
        diagnostic_summaries = []  # Store diagnostic summaries for each SNR
        
        # if DPS is enabled, import the sampler + grad builder
        if tester.use_dps:
            from dps_sampler import DpsSampler, make_awgn_likelihood_grad
            try:
                from dps_diagnostic_recorder import DpsDiagnosticRecorder
            except ImportError:
                DpsDiagnosticRecorder = None
            if tester.return_all_timesteps:
                raise NotImplementedError('return_all_timesteps is not supported when use_dps=True')
        else:
            DpsSampler = None
            make_awgn_likelihood_grad = None
            DpsDiagnosticRecorder = None
        
        with torch.no_grad():
            snr_db_list = snr_db_range.tolist()
            for snr_idx, snr in enumerate(tqdm(iterable=snr_range, desc="SNR sweep")):
                snr_db = snr_db_list[snr_idx]
                t_hat = int(torch.abs(tester.model.snrs - snr).argmin())
                steps_list.append(int(t_hat))
                
                if tester.use_dps:
                    # Note: for method='dps' we set cov_lambda=0.0 below (no covariance guidance),
                    # but args.cov_lambda is still printed elsewhere for reproducibility of settings.
                    print(f"\n[SNR {snr_db:.1f} dB] Processing with DPS (lambda={tester.dps_lambda:.3f}, "
                          f"method={args.method}, cov_lambda={args.cov_lambda})...")
                    if args.exp_key == 'H' and args.like_snr_gate:
                        import math
                        gate = 1.0 / (1.0 + math.exp(-((float(snr_db) - float(args.like_snr0_db)) / float(args.like_snr_delta_db))))
                        print(f"  [EXP H] like_snr_gate: SNR_dB={snr_db:.1f}, SNR0={args.like_snr0_db:.2f}, Delta={args.like_snr_delta_db:.2f} -> gate={gate:.6f}")
                else:
                    print(f"\n[SNR {snr_db:.1f} dB] Processing baseline...")
                
                # build DPS sampler for this SNR (if enabled)
                if tester.use_dps:
                    rho = float(snr)
                    noise_mult = float(tester.model.noise_multiplier)
                    sigma_y2_snr = (noise_mult ** 2) / rho
                    sigma_y2_like = float(args.sigma_y2) if args.use_fixed_sigma_y2 else float(sigma_y2_snr)
                    likelihood_grad_fn = make_awgn_likelihood_grad(sigma_y2_like)

                    # Select covariance guidance strength based on method
                    if args.method == 'dps':
                        cov_lambda = 0.0  # no covariance guidance
                    else:
                        cov_lambda = float(args.cov_lambda)

                    dps_sampler = DpsSampler(
                        dm=tester.model,
                        likelihood_grad_fn=likelihood_grad_fn,
                        lambda_dps=tester.dps_lambda,
                        cov_lambda=cov_lambda,
                        cov_scale_mode=args.cov_scale_mode,
                        cov_beta_power=args.cov_beta_power,
                        cov_grad_norm=args.cov_grad_norm,
                        cov_step_clip=args.cov_step_clip,
                        cov_clip_mode=args.cov_clip_mode,
                        sigma_y2=sigma_y2_like,
                        add_random=False,
                        exp_key=args.exp_key,
                        gamma=args.gamma,
                        like_weight=args.like_weight,
                        lw_schedule=args.lw_schedule,
                        lw_tau=args.lw_tau,
                        lw_max=args.lw_max,
                        lw_end=args.lw_end,
                        lw_k=args.lw_k,
                        g_tau1=args.g_tau1,
                        like_beta_power=args.like_beta_power,
                        like_snr_gate=bool(args.like_snr_gate),
                        like_snr0_db=float(args.like_snr0_db),
                        like_snr_delta_db=float(args.like_snr_delta_db),
                    )
                    # Enable t_start-based scaling if requested
                    if args.use_t_start_scaling:
                        dps_sampler.use_t_start_scaling = True
                        dps_sampler.cov_lambda_base = float(cov_lambda)
                    
                    # Enable debug logging if requested
                    if args.debug_cov_scaling:
                        dps_sampler.debug_cov_scaling = True
                else:
                    dps_sampler = None
                
                # timing start
                if tester.device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                x_hat = []
                debug_csv_saved = False  # Flag to ensure CSV is only saved once per SNR
                for batch_idx, data_batch in enumerate(tester.dataloader):
                    try:
                        data_batch = data_batch.to(device=tester.device)
                        y = functional.awgn(data_batch, snr, multiplier=tester.model.noise_multiplier)

                        # --------------------------------------------------------------
                        # Per-batch debug toggles (only for the first batch)
                        # --------------------------------------------------------------
                        if tester.use_dps and dps_sampler is not None:
                            if args.debug_likelihood and batch_idx == 0:
                                dps_sampler.debug_likelihood = True
                                dps_sampler._debug_step_count = 0
                            else:
                                dps_sampler.debug_likelihood = False

                            if args.debug_cf and args.exp_key in ('E', 'Eprime', 'F', 'G') and batch_idx == 0:
                                dps_sampler.debug_cf = True
                            else:
                                dps_sampler.debug_cf = False

                            # Optional micro-test mode (3 reverse steps). Independent from debug prints.
                            dps_sampler.debug_cf_microtest = bool(args.debug_cf_microtest and batch_idx == 0)
                            
                            if args.record_like_balance and args.exp_key == 'E' and batch_idx == 0:
                                dps_sampler.record_like_balance = True
                                dps_sampler.like_balance_history = []
                            else:
                                dps_sampler.record_like_balance = False
                        
                        # Only debug first sample
                        if args.debug_cov_scaling and batch_idx == 0 and not debug_csv_saved:
                            # Reset debug history for this sample
                            if tester.use_dps and dps_sampler is not None:
                                dps_sampler.debug_history = []

                        # ------------------------------------------------------------------
                        # Build covariance term depending on selected method.
                        # ------------------------------------------------------------------
                        if args.method in ('dps', 'dps_cov_oracle'):
                            # Oracle covariance R_h = H H^H via X X^H
                            cov = generate_cov_batch(data_batch)
                        elif args.method == 'dps_cov_est':
                            # Time-averaged covariance estimate from Y_d = H X_d + N
                            cov = estimate_cov_time_averaged_batch(
                                data_batch,
                                snr_db=float(snr_db),
                                n_time_samples=args.n_time_samples,
                                modulation=args.modulation,
                            )
                        else:
                            raise ValueError(f"Unknown DPS method: {args.method}")

                        if tester.use_dps:
                            # Create diagnostic recorder if enabled
                            diagnostic_recorder = None
                            if args.record_diagnostics and args.method in ('dps_cov_oracle', 'dps_cov_est') and DpsDiagnosticRecorder is not None:
                                diagnostic_recorder = DpsDiagnosticRecorder(record_enabled=True)
                            
                            # For ablation experiments, default is to disable SNR matching (use full T),
                            # but the user can override with --enable_snr_matching.
                            if args.exp_key in ['A', 'B', 'C', 'D', 'E', 'Eprime', 'F', 'G'] and (not args.enable_snr_matching):
                                snr_for_loop = None
                            else:
                                snr_for_loop = snr
                            
                            x_est = dps_sampler.generate_posterior_sample(
                                y.to(device=tester.device),
                                cov=cov.to(device=tester.device),
                                return_all_timesteps=tester.return_all_timesteps,
                                num_steps=args.num_steps,
                                snr=snr_for_loop,  # Pass None for ablation to use full T
                                obs_snr_db=float(snr_db),
                                diagnostic_recorder=diagnostic_recorder,
                            )
                            
                            # NMSE FFT invariance diagnostic tests (optional, only if --run_fft_diagnostics is set)
                            if args.run_fft_diagnostics and batch_idx == 0 and snr_idx == 0 and tester.fft_pre and not tester.return_all_timesteps:
                                if not FFT_DIAGNOSTICS_AVAILABLE:
                                    print("\n[WARNING] FFT diagnostics requested but fft_diagnostics module not available.")
                                    print("  Skipping FFT diagnostics. Please ensure fft_diagnostics.py exists.")
                                else:
                                    # Run basic diagnostics
                                    run_nmse_fft_diagnostics(
                                        data_batch=data_batch,
                                        mode=tester.mode,
                                        verbose=True,
                                    )
                                    
                                    # Run corrected NMSE FFT invariance diagnostic
                                    # Uses the same ground truth and estimate tensors in both domains
                                    run_end_to_end_invariance_audit(
                                        H_gt_ang=data_batch,  # Angular domain ground truth
                                        H_hat_ang=x_est,      # Angular domain estimate
                                        mode=tester.mode,
                                        verbose=True,
                                        debug=args.fft_diagnostics_debug,  # Enable debug prints if requested
                                    )
                            
                            # Log per-step scaling info if enabled (only for first batch to avoid spam)
                            if args.use_t_start_scaling and hasattr(dps_sampler, '_last_t_start') and batch_idx == 0:
                                t_start = dps_sampler._last_t_start
                                beta_t_start = dps_sampler._last_beta_t_start
                                # Compute effective lambda at t_start for logging
                                cov_lambda_eff_at_start = dps_sampler.cov_lambda_base * (beta_t_start + 1e-12) ** 0.5
                                print(f"  [per-step scaling] t_start={t_start}, beta[{t_start}]={beta_t_start:.6e}, "
                                      f"cov_lambda_eff(t) = cov_lambda_base * sqrt(beta[t]) for each step")
                            
                            # Store diagnostic summary if recorded
                            if diagnostic_recorder is not None and diagnostic_recorder.record_enabled:
                                summary = diagnostic_recorder.get_late_stage_summary(t_start=0.6, t_end=0.9)
                                summary['snr_db'] = float(snr_db)
                                diagnostic_summaries.append(summary)
                        else:
                            x_est = tester.model.generate_estimate(
                                y.to(device=tester.device), snr,
                                return_all_timesteps=tester.return_all_timesteps,
                            )
                        
                        if tester.fft_pre:
                            if tester.return_all_timesteps:
                                x_est = ut.complex_1d_fft(x_est, ifft=True, mode=tester.mode, _4d_array=True)
                            else:
                                x_est = ut.complex_1d_fft(x_est, ifft=True, mode=tester.mode)
                        
                        x_hat.append(x_est)
                        
                        # Save debug CSV after first sample
                        if args.debug_cov_scaling and batch_idx == 0 and not debug_csv_saved:
                            if tester.use_dps and dps_sampler is not None and len(dps_sampler.debug_history) > 0:
                                import csv
                                from pathlib import Path
                                
                                debug_dir = Path('results') / 'dm_dps' / 'debug_cov_scaling'
                                debug_dir.mkdir(parents=True, exist_ok=True)
                                
                                cov_scale_tag = args.cov_scale_mode
                                if args.cov_beta_power is not None:
                                    cov_scale_tag = f'beta_pow{args.cov_beta_power:g}'
                                csv_file = debug_dir / f'debug_snr{snr_db:.1f}_scale_{cov_scale_tag}_norm_{args.cov_grad_norm}_clip_{args.cov_clip_mode}.csv'
                                
                                with open(csv_file, 'w', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([
                                        't', 'beta_t', 'zeta_t', 'grad_cov_raw_norm',
                                        'grad_cov_normed_norm', 'dx_cov_preclip_norm',
                                        'dx_cov_postclip_norm', 'cov_clip_applied'
                                    ])
                                    for record in dps_sampler.debug_history:
                                        writer.writerow([
                                            record['t'],
                                            record['beta_t'],
                                            record['zeta_t'],
                                            record['grad_cov_raw_norm'],
                                            record['grad_cov_normed_norm'],
                                            record['dx_cov_preclip_norm'],
                                            record['dx_cov_postclip_norm'],
                                            record['cov_clip_applied'],
                                        ])
                                
                                # Print summary statistics
                                zeta_vals = [r['zeta_t'] for r in dps_sampler.debug_history]
                                dx_preclip_vals = [r['dx_cov_preclip_norm'] for r in dps_sampler.debug_history]
                                dx_postclip_vals = [r['dx_cov_postclip_norm'] for r in dps_sampler.debug_history]
                                
                                if args.cov_beta_power is not None:
                                    print(f"\n  [Debug Summary] cov_beta_power={args.cov_beta_power} (override, zeta_t=beta_t**p)")
                                else:
                                    print(f"\n  [Debug Summary] cov_scale_mode={args.cov_scale_mode}")
                                print(f"    zeta_t: min={min(zeta_vals):.6e}, mean={sum(zeta_vals)/len(zeta_vals):.6e}, max={max(zeta_vals):.6e}")
                                print(f"    dx_cov_preclip_norm: min={min(dx_preclip_vals):.6e}, mean={sum(dx_preclip_vals)/len(dx_preclip_vals):.6e}, max={max(dx_preclip_vals):.6e}")
                                print(f"    dx_cov_postclip_norm: min={min(dx_postclip_vals):.6e}, mean={sum(dx_postclip_vals)/len(dx_postclip_vals):.6e}, max={max(dx_postclip_vals):.6e}")
                                print(f"    Debug CSV saved to: {csv_file}")
                                
                                debug_csv_saved = True
                                
                                # Disable debug for remaining batches
                                dps_sampler.debug_cov_scaling = False

                        # Save like-balance diagnostics after first sample (exp E only)
                        if args.record_like_balance and args.exp_key == 'E' and batch_idx == 0:
                            if tester.use_dps and dps_sampler is not None and len(getattr(dps_sampler, 'like_balance_history', [])) > 0:
                                import csv
                                from pathlib import Path

                                debug_dir = Path('results') / 'dm_dps' / 'debug_like_balance'
                                debug_dir.mkdir(parents=True, exist_ok=True)
                                if args.lw_schedule == 'ramp':
                                    csv_file = debug_dir / f'like_balance_snr{snr_db:.1f}_ramp_tau{args.lw_tau:.2f}_wmax{args.lw_max:.0f}.csv'
                                elif args.lw_schedule == 'lastk':
                                    csv_file = debug_dir / f'like_balance_snr{snr_db:.1f}_lastk_k{args.lw_k}_wend{args.lw_end:.0f}.csv'
                                else:
                                    csv_file = debug_dir / f'like_balance_snr{snr_db:.1f}_lw{args.like_weight:.3f}.csv'

                                rows = sorted(dps_sampler.like_balance_history, key=lambda r: r['t'], reverse=True)
                                with open(csv_file, 'w', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(['t', 'alpha_bar_t', 'beta_t', 'like_weight', 'lw_schedule', 'lw_tau', 'lw_max', 'w_t', 'P_t', 'L_t', 'r_t', 'P0', 'L0', 'r0'])
                                    for r in rows:
                                        writer.writerow([r['t'], r['alpha_bar_t'], r['beta_t'], r['like_weight'], r.get('lw_schedule', ''), r.get('lw_tau', ''), r.get('lw_max', ''), r.get('w_t', ''), r['P_t'], r['L_t'], r['r_t'], r['P0'], r['L0'], r['r0']])

                                r_vals = [r['r_t'] for r in rows]
                                print(f"\n  [Like-balance] Saved CSV: {csv_file}")
                                print(f"    r_t = L_t/(P_t+eps): min={min(r_vals):.3e}, mean={sum(r_vals)/len(r_vals):.3e}, max={max(r_vals):.3e}")
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {e}")
                        raise
                
                x_hat = torch.cat(x_hat, dim=0).cpu()
                
                # timing end
                if tester.device.type == 'cuda':
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                timings_sec.append(dt)
                tps_ms_list.append(dt * 1000.0 / tester.num_samples)
                
                # compute NMSE
                if len(tester.data.shape) == 4:
                    dim = int(tester.data.shape[-1] * tester.data.shape[-2])
                    x_hat_rs = ut.reshape_fortran(x_hat, (-1, dim))
                    nmse_val = functional.nmse_torch(
                        ut.reshape_fortran(torch.squeeze(tester.data), (-1, dim)),
                        x_hat_rs,
                        norm_per_sample=False
                    )
                    nmse_total_power_list.append(nmse_val)
                else:
                    nmse_val = functional.nmse_torch(
                        torch.squeeze(tester.data),
                        torch.squeeze(x_hat),
                        norm_per_sample=False
                    )
                    nmse_total_power_list.append(nmse_val)
                
                # Log t_start scaling info with NMSE if enabled
                if tester.use_dps and args.use_t_start_scaling and hasattr(dps_sampler, '_last_t_start'):
                    t_start = dps_sampler._last_t_start
                    beta_t_start = dps_sampler._last_beta_t_start
                    # Compute effective lambda at t_start for logging
                    cov_lambda_eff_at_start = dps_sampler.cov_lambda_base * (beta_t_start + 1e-12) ** 0.5
                    # Handle both Tensor and float types
                    if isinstance(nmse_val, torch.Tensor):
                        nmse_float = float(nmse_val.item())
                    else:
                        nmse_float = float(nmse_val)
                    print(f"  [SNR {snr_db:.1f} dB] t_start={t_start}, beta[{t_start}]={beta_t_start:.6e}, "
                          f"cov_lambda_eff(t_start)={cov_lambda_eff_at_start:.6e}, NMSE={nmse_float:.6e}")
        
        result_dict = {
            'SNRs': snr_db_range.tolist(),
            'NMSEs_total_power': nmse_total_power_list,
            'Steps': steps_list,
            'Timings_sec': timings_sec,
            'Time_per_sample_ms': tps_ms_list,
        }
        
        # Add diagnostic summaries if recorded
        if diagnostic_summaries:
            result_dict['diagnostic_summaries'] = diagnostic_summaries
        
        return result_dict
    
    # Test each lambda value
    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    num_timesteps = sim_params['diff_model_dict']['num_timesteps']
    # Save DPS results in a separate folder from pure DM results
    # (DM uses results/dm_est, DPS uses results/dm_dps)
    base_dir = cwd / 'results' / 'dm_dps'
    base_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'{timestamp}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val}_T={num_timesteps}'
    
    # Store results for all lambdas (for comparison)
    all_results = []
    
    for lambda_idx, dps_lambda in enumerate(args.dps_lambdas):
        print(f"\n{'='*60}")
        print(f"Testing lambda = {dps_lambda} ({lambda_idx + 1}/{len(args.dps_lambdas)})")
        print(f"{'='*60}\n")
        
        # Update tester with current lambda
        tester.dps_lambda = dps_lambda
        
        # Replace the method temporarily
        # IMPORTANT: Need to update both _test_nmse and test_funcs list
        tester._test_nmse = custom_test_nmse
        # Update the test_funcs list to use the new function
        if 'nmse' in tester.criteria:
            nmse_idx = tester.criteria.index('nmse')
            tester.test_funcs[nmse_idx] = custom_test_nmse
        
        test_dict = tester.test()
        
        # Process results
        snrs: List[float] = test_dict['nmse']['SNRs']
        nmse_curves: List = test_dict['nmse']['NMSEs_total_power']
        steps = test_dict['nmse'].get('Steps', [None] * len(snrs))
        times = test_dict['nmse'].get('Timings_sec', [None] * len(snrs))
        tps = test_dict['nmse'].get('Time_per_sample_ms', [None] * len(snrs))
        diagnostic_summaries = test_dict['nmse'].get('diagnostic_summaries', [])
        
        # Note: diagnostic_summaries are collected per SNR point in custom_test_nmse
        
        if args.return_all_timesteps:
            maybe_plot_timesteps(base_dir, prefix, num_timesteps, snrs, nmse_curves)
            nmse_final = [curve[-1] for curve in nmse_curves]
        else:
            nmse_final = nmse_curves
        
        # Save individual results for this lambda and method
        if args.method == 'dps':
            method_name = 'DPS'
        elif args.method == 'dps_cov_oracle':
            method_name = 'DPS_COV_ORACLE'
        elif args.method == 'dps_cov_est':
            method_name = 'DPS_COV_EST'
        else:
            method_name = args.method.upper()

        suffix = f'_method={method_name}_dps_lambda={dps_lambda}_sigma={args.sigma_y2}'
        if args.method in ('dps_cov_oracle', 'dps_cov_est'):
            suffix += f'_cov_lambda={args.cov_lambda}'
            if args.cov_beta_power is not None:
                suffix += f'_scale=beta_pow{args.cov_beta_power:g}'
            else:
                suffix += f'_scale={args.cov_scale_mode}'
            suffix += f'_clip={args.cov_clip_mode}'
            if args.use_t_start_scaling:
                suffix += '_tstart_scaled'
        if args.method == 'dps_cov_est':
            suffix += f'_ntime={args.n_time_samples}_mod={args.modulation}'
        # Add experiment key to suffix for ablation experiments
        if args.exp_key != 'A':
            suffix += f'_exp={args.exp_key}'
            # For experiment D, also include gamma value
            if args.exp_key == 'D':
                suffix += f'_gamma={args.gamma:.2f}'
            # For experiment E/Eprime, add explicit tag
            if args.exp_key == 'E':
                suffix += '_closedform'
            if args.exp_key == 'Eprime':
                suffix += '_closedform_postadd'
            if args.exp_key == 'F':
                suffix += '_paper'
        
        rows_full = list(zip(snrs, nmse_final, steps, times, tps))
        export_table(
            base_dir,
            prefix,
            headers=['SNR', 'nmse_dm_dps', 'steps', 'time_s', 'time_per_sample_ms'],
            rows=rows_full,
            suffix=suffix,
        )
        
        rows_best = list(zip(snrs, nmse_final))
        export_table(
            base_dir,
            prefix,
            headers=['SNR', 'nmse_dm_dps'],
            rows=rows_best,
            suffix=suffix + '_best',
        )
        
        # Store for comparison
        all_results.append({
            'lambda': dps_lambda,
            'snrs': snrs,
            'nmse': nmse_final,
            'exp_key': args.exp_key,
        })
        
        # Save diagnostic summaries if recorded
        if diagnostic_summaries and args.record_diagnostics:
            diag_dir = base_dir / 'diagnostics'
            diag_dir.mkdir(parents=True, exist_ok=True)
            for summary in diagnostic_summaries:
                snr_val = summary.get('snr_db', 'unknown')
                diag_file = diag_dir / f'{timestamp}_snr{snr_val:.1f}_summary.txt'
                with open(diag_file, 'w') as f:
                    f.write(f"DPS-COV Diagnostic Summary (SNR={snr_val} dB)\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Configuration:\n")
                    f.write(f"  Method: {method_name}\n")
                    f.write(f"  DPS lambda: {dps_lambda}\n")
                    if args.method in ('dps_cov_oracle', 'dps_cov_est'):
                        f.write(f"  Cov lambda: {args.cov_lambda}\n")
                        f.write(f"  Cov scale mode: {args.cov_scale_mode}\n")
                        if args.cov_beta_power is not None:
                            f.write(f"  Cov beta power override: {args.cov_beta_power} (zeta_t = beta_t**p)\n")
                        f.write(f"  Cov grad norm: {args.cov_grad_norm}\n")
                        f.write(f"  Cov step clip: {args.cov_step_clip}\n")
                        f.write(f"  Cov clip mode: {args.cov_clip_mode}\n")
                    f.write(f"\nMid-to-Late Stage Statistics (t in [0.6T, 0.9T]):\n")
                    f.write(f"  Mean(c_t / b_t): {summary.get('mean_c_over_b', 'N/A'):.6f}\n")
                    f.write(f"  Mean(clip_rate_cov): {summary.get('mean_clip_rate_cov', 'N/A'):.6f}\n")
                    f.write(f"  Mean(||Δx_cov||): {summary.get('mean_c_t', 'N/A'):.6e}\n")
                    f.write(f"  Mean(||Δx_like||): {summary.get('mean_b_t', 'N/A'):.6e}\n")
            print(f"  Diagnostic summaries saved to: {diag_dir}/")
        
        print(f"\n✓ Lambda {dps_lambda} completed. Results saved.\n")
    
    # Print summary table for ablation experiments (if multiple exp_keys or non-A)
    unique_exp_keys = set(r.get('exp_key', 'A') for r in all_results)
    if len(unique_exp_keys) > 1 or (len(unique_exp_keys) == 1 and 'A' not in unique_exp_keys):
        print("\n" + "="*70)
        print("Ablation Summary: Mean NMSE at each SNR")
        print("="*70)
        print(f"{'SNR (dB)':>10} ", end="")
        for result in all_results:
            exp_key = result.get('exp_key', 'A')
            print(f"{'Exp ' + exp_key:>15} ", end="")
        print()
        print("-"*70)
        
        # Find common SNR values
        all_snrs_set = set()
        for result in all_results:
            all_snrs_set.update(result['snrs'])
        common_snrs = sorted(all_snrs_set)
        
        for snr in common_snrs:
            print(f"{snr:>10.1f} ", end="")
            for result in all_results:
                snrs = result['snrs']
                nmse = result['nmse']
                # Find closest SNR
                idx = min(range(len(snrs)), key=lambda i: abs(snrs[i] - snr))
                nmse_val = nmse[idx]
                print(f"{nmse_val:>15.6e} ", end="")
            print()
        print("="*70)
    
    # Save comparison table (all lambdas in one file)
    if len(args.dps_lambdas) > 1:
        print(f"\n{'='*60}")
        print("Saving comparison table for all lambda values...")
        print(f"{'='*60}\n")
        
        # Create comparison table
        comparison_rows = []
        for result in all_results:
            for snr, nmse in zip(result['snrs'], result['nmse']):
                comparison_rows.append([result['lambda'], snr, nmse])
        
        comparison_suffix = f'_method={method_name}_comparison_all_lambdas_sigma={args.sigma_y2}'
        if args.method in ('dps_cov_oracle', 'dps_cov_est'):
            comparison_suffix += f'_cov_lambda={args.cov_lambda}'
            comparison_suffix += f'_scale={args.cov_scale_mode}'
        if args.method == 'dps_cov_est':
            comparison_suffix += f'_ntime={args.n_time_samples}_mod={args.modulation}'
        export_table(
            base_dir,
            prefix,
            headers=['lambda', 'SNR', 'nmse_dm_dps'],
            rows=comparison_rows,
            suffix=comparison_suffix,
        )
        
        print(f"✓ Comparison table saved.\n")
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"Results saved to: {base_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

