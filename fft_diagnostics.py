"""
FFT Invariance Diagnostics Module

This module provides diagnostic functions to verify that NMSE is invariant to IFFT transformation.
It includes helper functions and comprehensive tests for FFT/IFFT unitarity, energy preservation,
and NMSE invariance in both angular and spatial domains.
"""

import torch
import modules.utils as ut


# ============================================================================
# Helper functions
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


# ============================================================================
# Test 1: FFT/IFFT Unitarity Check
# ============================================================================

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
    # Output shape when _4d_array=False: (batches, 2, N_R, N_T) based on utils.py line 291
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


# ============================================================================
# Test 2: Complex Conversion & Dimension Check
# ============================================================================

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
# Main diagnostic function
# ============================================================================

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


# ============================================================================
# End-to-end invariance audit
# ============================================================================

def run_end_to_end_invariance_audit(
    H_gt_ang: torch.Tensor,
    H_hat_ang: torch.Tensor,
    mode: str = '2D',
    verbose: bool = True,
    debug: bool = False,
) -> dict:
    """
    Corrected NMSE FFT Invariance Diagnostic.
    
    Uses the same ground truth and estimate tensors in both angular and spatial domains.
    Computes NMSE consistently in complex form.
    
    Parameters
    ----------
    H_gt_ang : torch.Tensor
        Ground truth channels in angular domain, shape [B, 2, R, T]
    H_hat_ang : torch.Tensor
        Estimated channels in angular domain, shape [B, 2, R, T]
    mode : str
        FFT mode: '1D' or '2D' (default: '2D')
    verbose : bool
        If True, print diagnostic information
    debug : bool
        If True, enable debug prints in complex_1d_fft
    
    Returns
    -------
    dict
        Dictionary containing audit results
    """
    device = H_gt_ang.device
    B, _, N_R, N_T = H_gt_ang.shape
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Corrected NMSE FFT Invariance Diagnostic")
        print(f"{'='*70}")
        print(f"  Input shapes: H_gt_ang={H_gt_ang.shape}, H_hat_ang={H_hat_ang.shape}")
        print(f"  Mode: {mode}")
        print(f"  Using _4d_array=False (same as normal NMSE_sp computation path)")
    
    # Use _4d_array=False to match the actual code path used in NMSE_sp computation
    # (see load_and_eval_dm_dps.py line 1113: default is False when return_all_timesteps=False)
    _4d_array = False
    
    # ========================================================================
    # Step 1: Convert to spatial domain using IFFT
    # ========================================================================
    # Define spatial GT strictly as IFFT(H_gt_ang)
    # Input format: [B, 2, R, T] where dim 1 is real/imag (standard format)
    H_gt_sp = ut.complex_1d_fft(H_gt_ang, ifft=True, mode=mode, _4d_array=_4d_array, debug=debug)
    
    # Define spatial estimate as IFFT(H_hat_ang) using the same function
    H_hat_sp = ut.complex_1d_fft(H_hat_ang, ifft=True, mode=mode, _4d_array=_4d_array, debug=debug)
    
    # Verify shapes
    if H_gt_sp.shape != H_gt_ang.shape:
        raise RuntimeError(f"Shape mismatch: H_gt_sp.shape={H_gt_sp.shape}, H_gt_ang.shape={H_gt_ang.shape}")
    if H_hat_sp.shape != H_hat_ang.shape:
        raise RuntimeError(f"Shape mismatch: H_hat_sp.shape={H_hat_sp.shape}, H_hat_ang.shape={H_hat_ang.shape}")
    
    # ========================================================================
    # Step 2: Convert to complex form for consistent NMSE computation
    # ========================================================================
    # Convert [B, 2, R, T] tensors to complex form [B, R, T]
    Hc_gt_ang = to_complex(H_gt_ang)
    Hc_hat_ang = to_complex(H_hat_ang)
    Hc_gt_sp = to_complex(H_gt_sp)
    Hc_hat_sp = to_complex(H_hat_sp)
    
    # ========================================================================
    # Step 3: Compute NMSE in both domains
    # ========================================================================
    # NMSE = ||H_hat - H_gt||_F^2 / ||H_gt||_F^2
    # Compute in angular domain
    diff_ang = Hc_hat_ang - Hc_gt_ang
    nmse_ang = fro2(diff_ang) / (fro2(Hc_gt_ang) + 1e-12)
    
    # Compute in spatial domain
    diff_sp = Hc_hat_sp - Hc_gt_sp
    nmse_sp = fro2(diff_sp) / (fro2(Hc_gt_sp) + 1e-12)
    
    # Compute differences
    abs_diff = abs(nmse_ang - nmse_sp)
    rel_diff = abs_diff / (abs(nmse_ang) + 1e-12)
    
    # ========================================================================
    # Step 4: Print results
    # ========================================================================
    if verbose:
        print(f"\n  Results:")
        print(f"  {'-'*66}")
        print(f"    NMSE_ang = {nmse_ang:.10e}")
        print(f"    NMSE_sp  = {nmse_sp:.10e}")
        print(f"    Absolute difference |NMSE_ang - NMSE_sp| = {abs_diff:.10e}")
        print(f"    Relative difference = {rel_diff:.10e}")
        print(f"\n  Expected: NMSE_ang ≈ NMSE_sp")
        print(f"  Expected: Absolute difference ≲ 1e-6 (numerical precision)")
        if abs_diff < 1e-6:
            print(f"  ✓ PASS: NMSE is invariant to IFFT (within numerical precision)")
        else:
            print(f"  ✗ FAIL: NMSE differs before and after IFFT!")
            print(f"    → This indicates a problem with FFT/IFFT implementation or NMSE computation")
        print(f"{'='*70}\n")
    
    return {
        'nmse_ang': nmse_ang,
        'nmse_sp': nmse_sp,
        'abs_diff': abs_diff,
        'rel_diff': rel_diff,
    }

