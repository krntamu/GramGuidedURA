## DPS Complete User Guide — Appendix (Diagnostics & FFT)

This appendix contains the long-form diagnostic recipes and FFT invariance reference material that used to live inside `DPS_COMPLETE_USER_GUIDE.md`.

---

## Diagnostic Tools

### DPS Sampler Diagnostic Tool

`dps_diagnostic_recorder.py` is a diagnostic tool for detecting magnitude issues in DPS sampler guidance, helping to quickly identify hyperparameter configuration problems.

#### Core Features

1. **Magnitude Check**: Records gradient/correction magnitudes at each timestep
2. **Pathology Diagnosis**: Automatically identifies three common issues (A: Too weak, B: Late-stage failure, C: Late-stage explosion)
3. **Scaling Mode Testing**: Supports four scaling modes: `beta_t`, `sqrt_beta_t`, `constant`, `snr_aware`
4. **Lambda Suggestions**: Automatically suggests reasonable initial `lambda_cov` values based on statistics
5. **Proxy Testing**: Optional use of proxy gradient to verify if it's a `cov_grad` implementation issue

#### Usage

Enable diagnostic recording:

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_step_clip 2.0 \
    --record_diagnostics \
    --dps_lambda 0.1
```

#### Output Description

**Console Output:**
- **Pathology Diagnosis Results**: Whether issues A/B/C are detected
- **Suggestions**: Specific repair suggestions for detected issues
- **Lambda Suggestions**: Suggested `lambda_cov` values based on statistics

**Diagnostic Summary Files:**
Saved in `results/dm_dps/diagnostics/{timestamp}_snr{X}_summary.txt`:
- `mean(c_t / b_t)`: Ratio of covariance correction to likelihood correction (target: 0.1-0.3)
- `mean(clip_rate_cov)`: Proportion of covariance correction that is clipped (target: < 0.2)
- `mean(||Δx_cov||)`: Magnitude of covariance correction
- `mean(||Δx_like||)`: Magnitude of likelihood correction

**Debug CSV Files** (if `--debug_cov_scaling` is enabled):
Saved in `results/dm_dps/debug_cov_scaling/debug_snr{X}_scale_{mode}_norm_{norm}.csv`:

Contains detailed statistics for each timestep:
- `t`: Timestep
- `beta_t`: Noise variance
- `zeta_t`: Scaling factor
- `grad_cov_raw_norm`: Raw gradient norm
- `grad_cov_normed_norm`: Normalized gradient norm
- `dx_cov_preclip_norm`: Correction norm before clipping
- `dx_cov_postclip_norm`: Correction norm after clipping
- `cov_clip_applied`: Whether clipping was applied

#### Diagnosis Guide

**Pathology A: Side Info Too Weak (`c_t << b_t`)**

**Symptoms**:
- `c_t / b_t` consistently < 0.1 (side correction < 10% likelihood correction)
- `||grad_cov||` may be normal, but `c_t` is very small

**Possible Causes**:
1. `lambda_cov` too small
2. `cov_grad` incorrectly normalized/scaled
3. `sigma_y2` placed incorrectly

**Fix Suggestions**:
- Increase `lambda_cov`
- Check `compute_cov_grad` implementation
- Try `constant` or `sqrt_beta_t` scaling mode

**Pathology B: Late-Stage Failure (`c_t` → 0 in later stages)**

**Symptoms**:
- `c_t` is normal in early stages, suddenly drops to near 0 in later stages
- `zeta_t` becomes very small in later stages (if using `beta_t` scaling)

**Possible Causes**:
- `beta_t` scaling causes late-stage failure (`beta_t` becomes very small in later stages)

**Fix Suggestions**:
- Use `constant` or `sqrt_beta_t` scaling mode
- Or use `snr_aware` mode (stronger in later stages)

**Pathology C: Late-Stage Explosion + High Clip Rate**

**Symptoms**:
- `c_t` becomes very large in later stages (> 2x early stages)
- `clip_rate_cov` > 50%
- Curves show plateaus or rebounds

**Possible Causes**:
- Side guidance too strong in low-noise stages
- `lambda_cov` too large
- Scaling mode inappropriate

**Fix Suggestions**:
- Decrease `lambda_cov`
- Use `beta_t` scaling (natural decay in later stages)
- Increase `step_clip` threshold

#### Lambda Suggestion Calculation

The diagnostic tool automatically calculates suggested `lambda_cov` values based on:
- Target: side correction ≈ 20% likelihood correction
- Statistics from mid-to-late timesteps (t ≈ 0.6T - 0.9T)
- Formula: `lambda_cov ≈ 0.2 * mean_like / (scale_t * mean_grad_cov)`

**Note**: This is only an initial suggestion and may need fine-tuning based on actual results.

---

## FFT Invariance Diagnostics

### Overview

The FFT invariance diagnostic module (`fft_diagnostics.py`) verifies that NMSE (Normalized Mean Squared Error) is invariant to IFFT transformation. This is critical because channel estimation is performed in the frequency domain (angular domain), but evaluation is done in the spatial domain after IFFT.

### Why This Matters

In the DPS-COV pipeline (spatial-first, paper-faithful execution):
1. **Spatial-domain observation & side-info**: Generate \(\mathbf{Y}\) and estimate Gram/covariance in the **spatial** domain.
2. **Angular-domain diffusion**: Apply a unitary FFT to map to the **angular** domain and run DM/DPS sampling there.
3. **Spatial-domain evaluation**: Apply IFFT to map the estimate back to the **spatial** domain and compute final NMSE.

Note:
- This avoids “domain-mixed” covariance estimation (i.e., estimating Gram in the already-FFT’ed domain by construction).
- FFT/IFFT is unitary, so NMSE is invariant *if computed consistently* (up to numerical precision), but intermediate estimators (e.g., sample Gram estimates) are now explicitly computed in the spatial domain to match Algorithm~1.

If NMSE differs significantly between angular and spatial domains, it indicates:
- FFT/IFFT normalization issues
- Incorrect complex tensor handling
- Domain mismatch in metric computation

### Features

The FFT diagnostics module provides:

1. **Test 1: FFT/IFFT Unitarity Check**
   - Verifies that FFT/IFFT preserves Frobenius norm
   - Checks energy preservation: `||A||_F^2 ≈ ||A_fft||_F^2 ≈ ||A_rec||_F^2`
   - Validates reconstruction: `||A_rec - A||_F / ||A||_F < 1e-6`

2. **Test 2: Complex Conversion & Dimension Check**
   - Verifies correct complex tensor handling
   - Tests FFT on actual channel data format `[B, 2, R, T]`
   - Ensures correct dimension usage for 2D FFT

3. **End-to-End Invariance Audit**
   - Compares NMSE computed in angular vs spatial domains
   - Uses the same ground truth and estimate tensors
   - Expected: `|NMSE_ang - NMSE_sp| < 1e-6`

### Usage

Enable FFT diagnostics during evaluation:

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --run_fft_diagnostics \
    --dps_lambda 0.1 \
    --ch_type 3gpp \
    --n_path 3
```

With debug output:

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --run_fft_diagnostics \
    --fft_diagnostics_debug \
    --dps_lambda 0.1 \
    --ch_type 3gpp \
    --n_path 3
```

Parameters:
- `--run_fft_diagnostics`: enable FFT invariance diagnostics
- `--fft_diagnostics_debug`: more detailed debug prints

---

