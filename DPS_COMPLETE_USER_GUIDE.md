# DPS-COV Complete User Guide

## Overview

This comprehensive guide covers the DPS (Diffusion Posterior Sampling) implementation with covariance guidance, including hyperparameter tuning, diagnostic tools, and FFT invariance testing.

---

## Table of Contents

1. [Implementation Details](#implementation-details)
2. [Hyperparameters](#hyperparameters)
3. [Likelihood EXP Keys (`--exp_key` A–H)](#likelihood-exp-keys---exp_key-a-h)
4. [DPS-COV Parameters](#dps-cov-parameters)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)
7. [Appendix (Diagnostics & FFT)](#appendix-diagnostics--fft)

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

**Current CLI semantics (important):**
- By default, `load_and_eval_dm_dps.py` derives `sigma_y2` from the current SNR (matches `functional.awgn`).
- Use **fixed** `sigma_y2` only for ablations via:
  - `--use_fixed_sigma_y2 --sigma_y2 <value>`

### 4. **SNR-Based Timestep Selection**

The implementation automatically selects the starting timestep based on SNR:
```python
t_start = int(torch.abs(self.dm.snrs - snr).argmin())
```

This ensures that the reverse process starts from an appropriate noise level matching the observation SNR.

**Important (current repo behavior):**
- In `load_and_eval_dm_dps.py`, likelihood ablations **A–G** default to **full reverse chain** (SNR matching disabled) unless you pass `--enable_snr_matching`.
- `exp_key=H` is **not** in that ablation list, so it uses SNR matching by default.

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

## Likelihood EXP Keys (`--exp_key` A–H)

This section merges and replaces the older standalone EXP cheat sheet. The implementation lives in `dps_sampler.py` (`DpsSampler.reverse_step_dps()`).

### Status (what we actually use)

For this repo’s current results and curves, treat the likelihood side as having **two supported modes**:

- **EXP H**: Tweedie-form / “x0-hat” likelihood guidance (post-add)
- **EXP E**: closed-form likelihood score injection (paper-style)

All other EXP keys are considered **legacy / ablation-only** and are good candidates for removal after code review (see cleanup checklist).

### What `exp_key` controls (and what it does NOT)

- **`--exp_key` controls likelihood guidance only** (how the likelihood term is formed/applied).
- **Covariance / Gram guidance is orthogonal** and controlled by:
  - `--method dps_cov_oracle|dps_cov_est`
  - `--cov_lambda > 0`

### Quick reference (recommended)

| exp_key | Likelihood mode | When to use | Main knobs |
|---|---|---|---|
| **H** | **Tweedie / \(\hat x_0(H_t)\)** post-add | Default likelihood guidance for current experiments | `--dps_lambda`, optional `--like_snr_gate --like_snr0_db --like_snr_delta_db` |
| **E** | **Closed-form likelihood** score injection | Paper-style likelihood variant / sanity checks | `--like_weight`, `--lw_schedule` (+ `--lw_tau/--lw_max/--lw_end/--lw_k`) |

### EXP H (sigmoid-gated likelihood weight)

For `exp_key=H`, you can optionally apply an observation-SNR gate:
\[
w_{\text{like}}(t,\mathrm{SNR}_{dB})=\lambda_{\text{dps}}\cdot \beta_t \cdot \text{gate}(\mathrm{SNR}_{dB})
\]
\[
\text{gate}(\mathrm{SNR}_{dB})=\frac{1}{1+\exp\left(-\frac{\mathrm{SNR}_{dB}-\mathrm{SNR0}}{\Delta}\right)}
\]

CLI:
- `--like_snr_gate`
- `--like_snr0_db` (default `-10.5`)
- `--like_snr_delta_db` (default `2.0`)

### Legacy EXP keys (kept for ablation only)

The following are **not** used for current headline curves and are candidates for removal after code checking:
`A/B/C/D/Eprime/F/G`.
If you still need them temporarily, keep them as “legacy ablations” but do not add new features on top.

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

Independent control over covariance correction clipping. The **clipping operator** is controlled by `--cov_clip_mode`:
- `auto`: elementwise for legacy `beta_t` path, norm otherwise
- `elementwise`: element-wise clamp
- `norm`: norm-based scaling clip (preserves direction)

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
    --cov_clip_mode norm \
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
    --cov_clip_mode norm \
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

## Appendix (Diagnostics & FFT)

The long-form diagnostics sections were moved to:

- `DPS_COMPLETE_USER_GUIDE_APPENDIX.md`

This keeps the main guide concise for code review while preserving the full diagnostic recipes and reference outputs.

---

## Usage Examples

### Recommended baseline: EXP H (Tweedie-form likelihood)

```bash
python load_and_eval_dm_dps.py \
    --method dps \
    --exp_key H \
    --dps_lambda 0.1 \
    --ch_type 3gpp \
    --n_path 3
```

Quadriga LOS variant (same setup; only `--dps_lambda` changes):

```bash
python load_and_eval_dm_dps.py \
    --method dps \
    --exp_key H \
    --dps_lambda 0.3 \
    --ch_type quadriga_LOS
```

### EXP H + covariance guidance (oracle)

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --exp_key H \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_clip_mode norm \
    --cov_step_clip 2.0 \
    --dps_lambda 0.1 \
    --ch_type 3gpp \
    --n_path 3
```

Quadriga LOS variant (only `--cov_lambda` and `--dps_lambda` change):

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --exp_key H \
    --cov_lambda 0.0005 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_clip_mode norm \
    --cov_step_clip 2.0 \
    --dps_lambda 0.3 \
    --ch_type quadriga_LOS
```

### EXP H + covariance guidance (estimated)

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_est \
    --exp_key H \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_clip_mode norm \
    --cov_step_clip 2.0 \
    --n_time_samples 2000 \
    --modulation bpsk \
    --dps_lambda 0.1 \
    --ch_type 3gpp \
    --n_path 3
```

Quadriga LOS variant (only `--cov_lambda` and `--dps_lambda` change):

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_est \
    --exp_key H \
    --cov_lambda 0.0005 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_clip_mode norm \
    --cov_step_clip 2.0 \
    --n_time_samples 2000 \
    --modulation bpsk \
    --dps_lambda 0.3 \
    --ch_type quadriga_LOS
```
### Disable likelihood / disable Gram (how to toggle)

- **Disable likelihood**: set `--dps_lambda 0` (works for EXP H).
- **Disable Gram/cov**: use `--method dps` (or set `--cov_lambda 0` with a cov method).

Example (Gram-only, no likelihood):

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --exp_key H \
    --dps_lambda 0 \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_clip_mode norm \
    --cov_step_clip 2.0 \
    --ch_type 3gpp \
    --n_path 3
```

### EXP E (closed-form likelihood score injection)

```bash
python load_and_eval_dm_dps.py \
    --method dps \
    --exp_key E \
    --like_weight 1.0 \
    --lw_schedule const \
    --ch_type 3gpp \
    --n_path 3
```

Notes:
- EXP E’s likelihood strength is controlled by `--like_weight` / `--lw_schedule` (not `--dps_lambda`).
- SNR matching for EXP E is **disabled by default** (full reverse chain). Enable it explicitly with `--enable_snr_matching`.

### Single-SNR quick runs

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_est \
    --exp_key H \
    --single_snr_db 5 \
    --enable_snr_matching \
    --dps_lambda 0.1 \
    --cov_lambda 0.01
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

