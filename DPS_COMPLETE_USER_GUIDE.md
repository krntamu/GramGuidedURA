# DPS-COV Complete User Guide

## Overview

This comprehensive guide covers the DPS (Diffusion Posterior Sampling) implementation with covariance guidance, including hyperparameter tuning, diagnostic tools, and FFT invariance testing.

---

## Table of Contents

1. [Implementation Details](#implementation-details)
2. [Hyperparameters](#hyperparameters)
3. [DPS-COV Parameters](#dps-cov-parameters)
4. [Diagnostic Tools](#diagnostic-tools)
5. [FFT Invariance Diagnostics](#fft-invariance-diagnostics)
6. [Usage Examples](#usage-examples)
7. [Troubleshooting](#troubleshooting)

---

## Implementation Details

### 1. **Gradient Correction Order (CRITICAL)**

**Correct implementation:**
```python
# First do prior reverse step, then apply gradient correction
x_prior = reverse_step(x_t, t)
grad = likelihood_grad_fn(x_prior, y, t)  # Evaluate at x_prior, not x_t
x_prev = x_prior + eta_t * grad
```

**Why this matters:**
- The likelihood gradient should be evaluated at the **denoised estimate** (x_prior), not the noisy state (x_t)
- This matches DPS theory and ensures the gradient points toward the observation y from the current best estimate

### 2. **Deterministic Sampling (add_random=False)**

**Default:** `add_random=False` (deterministic DDIM-style)

**Why:**
- DPS relies on consistent gradient directions
- Stochastic noise can interfere with the likelihood gradient
- Deterministic steps are more stable and perform better in practice

### 3. **sigma_y^2 Calculation**

The observation noise variance is correctly computed as:
```python
sigma_y2_snr = (noise_multiplier^2) / rho
```
where `rho = 10^(SNR_dB / 10)` is the linear SNR.

This matches the AWGN function: `y = x + (1/sqrt(rho)) * multiplier * n`

### 4. **SNR-Based Timestep Selection**

The implementation automatically selects the starting timestep based on SNR:
```python
t_start = int(torch.abs(self.dm.snrs - snr).argmin())
```

This ensures that the reverse process starts from an appropriate noise level matching the observation SNR.

---

## Hyperparameters

### lambda_dps (DPS Guidance Strength)

**Default:** 0.1

**Tuning Guidelines:**
- **Too small (< 0.05):** DPS has minimal effect, results similar to plain DDIM
- **Too large (> 0.5):** Can overshoot and hurt performance, especially at low SNR
- **Recommended:** Start with 0.1, then tune based on results

**Expected Behavior:**
- **Low SNR (< 0 dB):** Smaller lambda (0.05-0.1) often better (y is mostly noise)
- **Mid SNR (0-15 dB):** Medium lambda (0.1-0.2) should show clear improvement
- **High SNR (> 15 dB):** Larger lambda (0.2-0.3) can help, but be careful not to overfit

### add_random

**Default:** `False` (deterministic)

**Recommendation:** Always use `False` for DPS. Stochastic steps can interfere with the likelihood gradient.

### step_clip

**Default:** `2.0`

**When to use:**
- Prevents overly large corrections that can cause instability
- Recommended value: `2.0` for most cases
- Can be adjusted based on observed behavior

---

## DPS-COV Parameters

### Version Comparison

#### Original Version (Default)
- **Formula**: `cov_correction = cov_lambda * beta_t * grad_cov`
- **Characteristics**: Simple and direct, but `beta_t` becomes very small in later timesteps, potentially causing guidance to become ineffective
- **Activation**: Default parameters (`cov_scale_mode=beta_t`, `cov_grad_norm=none`)

#### New Version
- **Formula**: `cov_correction = cov_lambda * zeta_t * normalize(grad_cov)`
- **Characteristics**: 
  - Optional gradient normalization (stabilizes gradient scale)
  - Flexible scaling modes (avoids late-stage failure)
  - Independent clipping control

### 1. `--cov_scale_mode` (Scaling Mode)

Controls how `zeta_t` is computed:

| Mode | Formula | Characteristics | Use Cases |
|------|---------|----------------|-----------|
| `beta_t` (default) | `zeta_t = beta_t` | Original version, becomes smaller in later stages | Backward compatibility |
| `sqrt_beta_t` | `zeta_t = sqrt(beta_t)` | Decays slower than `beta_t` | Medium-strength guidance |
| `constant` | `zeta_t = 1.0` | Constant strength, no late-stage failure | Stable guidance |
| `snr_aware` | `zeta_t = 1/sqrt(beta_t)` | Enhanced in later stages, may explode | Requires clipping |

**Recommendations**: 
- Low SNR: `constant` or `sqrt_beta_t`
- Medium to high SNR: `sqrt_beta_t` or `beta_t`

#### Optional: `--cov_beta_power` (Continuous Beta Power Override)

If you want a *continuous* knob instead of switching modes, you can override `zeta_t` with:

\[
zeta_t = \beta_t^p
\]

- `p=1.0` ≈ `beta_t`
- `p=0.5` ≈ `sqrt_beta_t`
- `p=0.0` ≈ `constant`
- `p=-0.5` ≈ `snr_aware`-like

If `--cov_beta_power` is set, it overrides `--cov_scale_mode`.

#### Optional: `--cov_clip_mode` (Clipping Mode for Cov Guidance)

The cov guidance can be clipped in two different ways:

- `auto` (default): backward-compatible behavior
  - element-wise clip for the legacy `beta_t` path
  - norm-based clip otherwise
- `elementwise`: clamp each element of the cov correction to \([-C, C]\)
- `norm`: scale the whole correction so its L2 norm is \(\le C\) (preserves direction)

When you are sweeping `--cov_beta_power`, it can be useful to **fix `--cov_clip_mode`** so the sweep reflects only the scaling operator, not a change in clipping behavior.

### 2. `--cov_grad_norm` (Gradient Normalization)

Controls gradient normalization method:

| Mode | Normalization | Characteristics | Use Cases |
|------|--------------|----------------|-----------|
| `none` (default) | No normalization | Original gradient scale | Backward compatibility |
| `by_x` | `grad / (||x|| + eps)` | Relative to x scale | When x scale varies significantly |
| `by_r` | `grad / (||R|| + eps)` | Relative to covariance matrix scale | When covariance scale varies significantly |
| `global` | `grad / mean(||grad||)` | Normalized to unit scale | **Recommended**: Stable and interpretable |

**Recommendation**: `global` - Normalizes gradient to `||grad|| ≈ 1`, making `cov_lambda` easier to tune

### 3. `--cov_lambda` (Guidance Strength)

Strength coefficient for covariance guidance. **Note**: If using `global` normalization, the meaning of `cov_lambda` changes.

| Normalization Mode | Recommended Range | Notes |
|-------------------|-------------------|-------|
| `none` | `0.001 - 0.01` | Original gradient scale, requires smaller values |
| `global` | `0.01 - 0.1` | After normalization, larger values can be used |
| `by_x` / `by_r` | `0.01 - 0.1` | Depends on x/R scale |

**Tuning Tips**: 
- Start with `0.01`
- If guidance is too weak (NMSE improvement is not obvious), increase to `0.03-0.05`
- If guidance is too strong (NMSE degrades or becomes unstable), decrease to `0.005-0.01`

### 4. `--cov_step_clip` (Clipping Threshold)

Independent control over covariance correction clipping. Uses **norm-based clipping** (not element-wise).

| Value | Effect |
|-------|--------|
| `None` | Uses default `step_clip` (typically `2.0`) |
| `1.0` | More conservative, prevents explosion |
| `2.0` | **Recommended**: Balances stability and effectiveness |
| `5.0` | More lenient, allows larger corrections |

**Recommendation**: `2.0` - Balances stability and effectiveness

### 5. `--use_t_start_scaling` (Optional)

Enables timestep-based scaling:
- `cov_lambda_eff(t) = cov_lambda_base * sqrt(beta[t])`
- Makes guidance strength adaptive to starting noise level

**Use Case**: When different SNRs use different `t_start`, this can automatically adjust guidance strength.

### Default Configuration

#### For 3GPP Channels

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_step_clip 2.0 \
    --dps_lambda 0.1 \
    --ch_type 3gpp \
    --n_path 3
```

#### For Quadriga LOS Channels

```bash
python load_and_eval_dm_dps.py \
    --ch_type quadriga_LOS \
    --method dps_cov_oracle \
    --cov_lambda 0.0005 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_step_clip 2.0 \
    --dps_lambda 0.3
```

**Parameter Explanation:**
- `--cov_lambda 0.01`: Moderate covariance guidance strength
- `--cov_scale_mode sqrt_beta_t`: Balanced scaling that decays slower than `beta_t`
- `--cov_grad_norm none`: No normalization (backward compatible)
- `--cov_step_clip 2.0`: Prevents explosion while maintaining effectiveness
- `--dps_lambda 0.1`: Standard DPS likelihood guidance strength

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

In the DPS-COV pipeline:
1. **Training/Inference**: Channels are transformed to frequency domain using FFT (`fft_pre=True`)
2. **Evaluation**: Estimates are transformed back to spatial domain using IFFT for NMSE computation
3. **Requirement**: Since FFT/IFFT is a unitary transformation, NMSE should be identical in both domains (up to numerical precision)

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

#### Basic Usage

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

#### With Debug Output

Enable detailed FFT operation logging:

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

#### Parameters

- `--run_fft_diagnostics`: Enable FFT invariance diagnostics (default: disabled)
- `--fft_diagnostics_debug`: Enable detailed debug prints showing FFT operation details

**Note**: Diagnostics run only once (on first batch, first SNR) to avoid cluttering output.

### Output Description

#### Test 1: FFT/IFFT Unitarity Check

```
======================================================================
Test 1: FFT/IFFT Unitarity Check (Normalization)
======================================================================
  Mode: 2D, Shape: (64, 16)

  ||A||_F^2 = 1.234567e+02
  ||A_fft||_F^2 = 1.234567e+02
  ||A_rec||_F^2 = 1.234567e+02

  Energy preservation (FFT): |||A||^2 - ||A_fft||^2| / ||A||^2 = 1.234e-10
  Energy preservation (IFFT): |||A||^2 - ||A_rec||^2| / ||A||^2 = 1.234e-10

  Reconstruction error: ||A_rec - A||_F = 1.234e-10
  Relative reconstruction error: ||A_rec - A||_F / ||A||_F = 1.234e-10

  Expected: Energy errors < 1e-6, Reconstruction error < 1e-6
  ✓ PASS: FFT/IFFT preserves energy and reconstructs correctly
======================================================================
```

#### Test 2: Complex Conversion & Dimension Check

```
======================================================================
Test 2: Complex Conversion & Dimension Check
======================================================================
  Input shape: torch.Size([2, 64, 16]), Mode: 2D
  Hc shape (complex): (64, 16)

  ||Hc||_F^2 = 1.234567e+02
  ||Hc_fft||_F^2 = 1.234567e+02
  ||Hc_rec||_F^2 = 1.234567e+02

  Energy preservation (FFT): |||Hc||^2 - ||Hc_fft||^2| / ||Hc||^2 = 1.234e-10
  Energy preservation (IFFT): |||Hc||^2 - ||Hc_rec||^2| / ||Hc||^2 = 1.234e-10

  Reconstruction error: ||Hc_rec - Hc||_F = 1.234e-10
  Relative reconstruction error: ||Hc_rec - Hc||_F / ||Hc||_F = 1.234e-10

  Expected: Energy errors < 1e-6, Reconstruction error < 1e-6
  ✓ PASS: Complex FFT handling is correct
======================================================================
```

#### End-to-End Invariance Audit

```
======================================================================
Corrected NMSE FFT Invariance Diagnostic
======================================================================
  Input shapes: H_gt_ang=torch.Size([512, 2, 64, 16]), H_hat_ang=torch.Size([512, 2, 64, 16])
  Mode: 2D
  Using _4d_array=False (same as normal NMSE_sp computation path)

  Results:
  ------------------------------------------------------------------
    NMSE_ang = 1.234567e-02
    NMSE_sp  = 1.234567e-02
    Absolute difference |NMSE_ang - NMSE_sp| = 1.234e-10
    Relative difference = 1.234e-10

  Expected: NMSE_ang ≈ NMSE_sp
  Expected: Absolute difference ≲ 1e-6 (numerical precision)
  ✓ PASS: NMSE is invariant to IFFT (within numerical precision)
======================================================================
```

### Interpretation Guide

**If Test 1 fails:**
- **Root cause #1**: FFT normalization mismatch detected
- **Action**: Check `modules/utils.py` `complex_1d_fft` function, verify `norm="ortho"` is used

**If Test 2 fails:**
- **Root cause #2**: Incorrect complex handling or FFT dimension usage
- **Action**: Verify tensor format `[B, 2, R, T]` and dimension indexing

**If both tests pass but end-to-end audit fails:**
- **Root cause #3**: Metric aggregation or domain mixing issue
- **Action**: Check NMSE computation code, ensure consistent tensor representations

**If all tests pass:**
- ✓ FFT/IFFT implementation is correct
- ✓ NMSE computation is consistent across domains
- ✓ No action needed

### Module Structure

The FFT diagnostics are implemented in a separate module `fft_diagnostics.py`:

- **Helper functions**: `to_complex()`, `fro2()`, `relerr()`
- **Test functions**: `test_fft_ifft_unitarity()`, `test_complex_fft_dimensions()`
- **Main functions**: `run_nmse_fft_diagnostics()`, `run_end_to_end_invariance_audit()`

The module is optional - if `fft_diagnostics.py` is missing, the main script will warn but continue normally.

---

## Usage Examples

### Basic DPS Evaluation

```bash
python load_and_eval_dm_dps.py \
    --method dps \
    --dps_lambda 0.1 \
    --ch_type 3gpp \
    --n_path 3
```

### DPS with Covariance Guidance (Oracle)

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_step_clip 2.0 \
    --dps_lambda 0.1 \
    --ch_type 3gpp \
    --n_path 3
```

### DPS with Estimated Covariance

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_est \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_step_clip 2.0 \
    --n_time_samples 2000 \
    --modulation bpsk \
    --dps_lambda 0.1 \
    --ch_type 3gpp \
    --n_path 3
```

### Quick Sanity Check with Diagnostics

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_step_clip 2.0 \
    --dps_lambda 0.1 \
    --record_diagnostics \
    --run_fft_diagnostics \
    --sanity_snrs
```

### Comprehensive Testing Workflow

```bash
# 1. Test with default parameters and enable all diagnostics
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_step_clip 2.0 \
    --record_diagnostics \
    --run_fft_diagnostics \
    --dps_lambda 0.1 \
    --sanity_snrs

# 2. If Pathology B (late-stage failure) is detected, try constant mode
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode constant \
    --cov_grad_norm global \
    --cov_step_clip 2.0 \
    --record_diagnostics \
    --dps_lambda 0.1 \
    --sanity_snrs

# 3. If Pathology A (too weak) is detected, use suggested lambda_cov
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda <suggested_value> \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_step_clip 2.0 \
    --record_diagnostics \
    --dps_lambda 0.1 \
    --sanity_snrs

# 4. Compare different scaling modes
for mode in beta_t sqrt_beta_t constant snr_aware; do
    python load_and_eval_dm_dps.py \
        --method dps_cov_oracle \
        --cov_lambda 0.01 \
        --cov_scale_mode $mode \
        --cov_grad_norm none \
        --cov_step_clip 2.0 \
        --record_diagnostics \
        --dps_lambda 0.1 \
        --sanity_snrs
done
```

---

## Expected Performance

### Low SNR (< 0 dB)
- **DPS may be similar or slightly worse than DDIM**
- This is acceptable: y is mostly noise, forcing toward y can hurt
- Use smaller lambda_dps (0.05-0.1)

### Mid SNR (0-15 dB)
- **DPS should show clear improvement over DDIM**
- This is where DPS shines: both prior and data are informative
- Use medium lambda_dps (0.1-0.2)

### High SNR (> 15 dB)
- **DPS should not be significantly worse than DDIM**
- Ideally, DPS should be better or close to "use y directly"
- Use larger lambda_dps (0.2-0.3), but watch for overfitting

---

## Troubleshooting

### DPS performs worse than DDIM at all SNRs

**Possible causes:**
1. **lambda_dps too large:** Try smaller values (0.05, 0.1)
2. **add_random=True:** Should be False
3. **sigma_y2 mismatch:** Verify sigma_y2 calculation matches AWGN noise variance
4. **Gradient clipping too aggressive:** Adjust or disable clip values

**Solution:** Use diagnostic tools (`--record_diagnostics`) to check gradient magnitudes

### DPS works at mid SNR but fails at high/low SNR

**This is expected behavior:**
- Low SNR: DPS may be worse (acceptable)
- High SNR: May need different lambda_dps, or DPS may not help much

**Solution:** Use SNR-dependent lambda_dps or adjust parameters per SNR range

### Numerical instability (NaN/Inf)

**Solution:**
1. Enable step clipping: `--cov_step_clip 2.0`
2. Reduce lambda_dps or cov_lambda
3. Check that sigma_y2 is not too small (avoid division by very small numbers)
4. Use gradient normalization: `--cov_grad_norm global`

### Covariance guidance not effective

**Possible causes:**
1. **cov_lambda too small:** Increase to 0.02-0.05
2. **Scaling mode inappropriate:** Try `constant` or `sqrt_beta_t`
3. **Late-stage failure:** Use `constant` scaling mode
4. **Gradient normalization needed:** Try `--cov_grad_norm global`

**Solution:** Use diagnostic tools (`--record_diagnostics`) to check `c_t / b_t` ratio and clip rates

### FFT diagnostics fail

**If Test 1 (Unitarity) fails:**
- Check `modules/utils.py` `complex_1d_fft` function
- Verify `norm="ortho"` is used for FFT/IFFT
- Check that input/output shapes match expected format

**If Test 2 (Complex/Dimensions) fails:**
- Verify tensor format is `[B, 2, R, T]` where dim 1 is real/imag
- Check that FFT is applied to correct dimensions (spatial dims, not batch/channel)
- Ensure complex conversion is correct: `torch.complex(H[:, 0], H[:, 1])`

**If End-to-End Audit fails:**
- Check that same tensors are used for both angular and spatial NMSE
- Verify IFFT is applied consistently to both ground truth and estimate
- Ensure no extra normalization or post-processing between domains

**Solution:** Enable `--fft_diagnostics_debug` to see detailed FFT operation logs

### Module import errors

**If `fft_diagnostics` module not found:**
- Ensure `fft_diagnostics.py` exists in the project root directory
- Check Python path includes the project directory
- The main script will warn but continue normally if module is missing

---

## Results Format

Results are saved in `results/dm_dps/` directory:
- `{timestamp}_nmse_results.csv`: NMSE vs SNR for each method
- `{timestamp}_nmse_plot.png`: Visualization of results
- `results/dm_dps/diagnostics/`: Diagnostic summaries (if `--record_diagnostics` is enabled)
- `results/dm_dps/debug_cov_scaling/`: Per-timestep debug data (if `--debug_cov_scaling` is enabled)

---

## Code Structure

- `dps_sampler.py`: Core DPS implementation
- `load_and_eval_dm_dps.py`: Main evaluation script
- `dps_diagnostic_recorder.py`: Diagnostic recording tool
- `fft_diagnostics.py`: FFT invariance diagnostic module (optional)
- `DMCE/diffusion_model.py`: Base diffusion model and tester integration
- `modules/utils.py`: Utility functions including FFT/IFFT operations

---

## Additional Notes

1. **Test Samples**: Default sample counts are sufficient for diagnosis
2. **SNR Selection**: Recommend testing multiple SNRs (-15, -10, -5, 0, 5 dB) for comprehensive diagnosis
3. **Timestep Selection**: Default uses all timesteps, can be limited via `--num_steps`
4. **Device**: Automatically detects CUDA, can force CPU with `--device cpu`
5. **FFT Diagnostics**: Run only once per evaluation (first batch, first SNR) to avoid output clutter

---

## Version History

- **v2.0**: Added flexible scaling modes and gradient normalization
- **v2.1**: Added FFT invariance diagnostics module
- **v2.2**: Enhanced diagnostic tools with automatic pathology detection

