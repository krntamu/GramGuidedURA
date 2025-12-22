# DPS Implementation Notes and Hyperparameter Tuning Guide

## Overview

This document explains the DPS (Diffusion Posterior Sampling) implementation, key design decisions, and how to tune hyperparameters for optimal performance.

## Key Implementation Details

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

## DPS-COV Parameters (3GPP Default Configuration)

For 3GPP channel estimation with covariance guidance, the recommended default parameters are:

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_step_clip 2.0 \
    --dps_lambda 0.1 \
    --sanity_snrs
```

**Parameter Explanation:**
- `--cov_lambda 0.01`: Moderate covariance guidance strength
- `--cov_scale_mode sqrt_beta_t`: Balanced scaling that decays slower than `beta_t`
- `--cov_grad_norm none`: No normalization (backward compatible)
- `--cov_step_clip 2.0`: Prevents explosion while maintaining effectiveness
- `--dps_lambda 0.1`: Standard DPS likelihood guidance strength

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

## Running Experiments

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

### Quick Sanity Check

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_step_clip 2.0 \
    --dps_lambda 0.1 \
    --sanity_snrs
```

## Results Format

Results are saved in `results/dm_dps/` directory:
- `{timestamp}_nmse_results.csv`: NMSE vs SNR for each method
- `{timestamp}_nmse_plot.png`: Visualization of results
- `results/dm_dps/diagnostics/`: Diagnostic summaries (if `--record_diagnostics` is enabled)

## Troubleshooting

### DPS performs worse than DDIM at all SNRs

**Possible causes:**
1. **lambda_dps too large:** Try smaller values (0.05, 0.1)
2. **add_random=True:** Should be False
3. **sigma_y2 mismatch:** Verify sigma_y2 calculation matches AWGN noise variance
4. **Gradient clipping too aggressive:** Adjust or disable clip values

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

## Code Structure

- `dps_sampler.py`: Core DPS implementation
- `load_and_eval_dm_dps.py`: Main evaluation script
- `dps_diagnostic_recorder.py`: Diagnostic recording tool
- `DMCE/diffusion_model.py`: Base diffusion model and tester integration


