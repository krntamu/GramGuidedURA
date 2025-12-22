# DPS-COV New Version User Guide

## Overview

The new version of DPS-COV provides more flexible gradient normalization and scaling modes, allowing better control over the strength and stability of covariance guidance.

## Version Comparison

### Original Version (Default)
- **Formula**: `cov_correction = cov_lambda * beta_t * grad_cov`
- **Characteristics**: Simple and direct, but `beta_t` becomes very small in later timesteps, potentially causing guidance to become ineffective
- **Activation**: Default parameters (`cov_scale_mode=beta_t`, `cov_grad_norm=none`)

### New Version
- **Formula**: `cov_correction = cov_lambda * zeta_t * normalize(grad_cov)`
- **Characteristics**: 
  - Optional gradient normalization (stabilizes gradient scale)
  - Flexible scaling modes (avoids late-stage failure)
  - Independent clipping control

## Parameter Description

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

## Default Configuration for 3GPP Channels

For 3GPP channel estimation, the recommended default parameters are:

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

**Parameter Explanation:**
- `--cov_lambda 0.01`: Moderate covariance guidance strength
- `--cov_scale_mode sqrt_beta_t`: Balanced scaling that decays slower than `beta_t`
- `--cov_grad_norm none`: No normalization (backward compatible)
- `--cov_step_clip 2.0`: Prevents explosion while maintaining effectiveness
- `--dps_lambda 0.1`: Standard DPS likelihood guidance strength

This configuration provides a good balance between stability and effectiveness for 3GPP channels.

## Usage Examples

### Example 1: Using Recommended New Version Configuration

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.03 \
    --cov_scale_mode constant \
    --cov_grad_norm global \
    --cov_step_clip 2.0 \
    --dps_lambda 0.1 \
    --sanity_snrs
```

**Characteristics**:
- `constant` scaling: No late-stage failure
- `global` normalization: Stable gradient scale
- `cov_lambda=0.03`: Moderate guidance strength
- `cov_step_clip=2.0`: Prevents explosion

### Example 2: Using sqrt_beta_t Scaling (Medium Strength)

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.02 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm global \
    --cov_step_clip 2.0 \
    --dps_lambda 0.1
```

**Characteristics**:
- `sqrt_beta_t`: Decays slower than `beta_t`, more conservative than `constant`
- Suitable for medium to high SNR scenarios

### Example 3: Original Version (Backward Compatible)

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode beta_t \
    --cov_grad_norm none \
    --dps_lambda 0.1
```

**Characteristics**:
- Default parameters, equivalent to original version
- `cov_correction = cov_lambda * beta_t * grad_cov`

### Example 4: Low SNR Optimized Configuration

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.04 \
    --cov_scale_mode constant \
    --cov_grad_norm global \
    --cov_step_clip 2.0 \
    --dps_lambda 0.1 \
    --sanity_snrs
```

**Characteristics**:
- `constant`: Ensures guidance doesn't fail at low SNR
- `cov_lambda=0.04`: Slightly stronger guidance to compensate for weak likelihood guidance at low SNR

### Example 5: Using Estimated Covariance (dps_cov_est)

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_est \
    --cov_lambda 0.03 \
    --cov_scale_mode constant \
    --cov_grad_norm global \
    --cov_step_clip 2.0 \
    --n_time_samples 2000 \
    --modulation bpsk \
    --dps_lambda 0.1
```

**Characteristics**:
- Uses time-averaged estimated covariance (not oracle)
- `n_time_samples=2000`: Number of time samples (more is more accurate but slower)

## Tuning Workflow

### Step 1: Choose Base Configuration

Start with recommended configuration:
```bash
--cov_scale_mode constant \
--cov_grad_norm global \
--cov_step_clip 2.0
```

### Step 2: Tune `cov_lambda`

1. Start with `0.01`
2. Observe NMSE curve:
   - If improvement is not obvious → Increase to `0.02, 0.03, 0.04`
   - If NMSE degrades or becomes unstable → Decrease to `0.005`
3. Goal: Find balance between stability and effectiveness

### Step 3: Adjust Scaling Mode (If Needed)

If `constant` performs poorly at high SNR:
- Try `sqrt_beta_t` (more conservative)
- Or keep `constant` but decrease `cov_lambda`

### Step 4: Adjust Clipping (If Needed)

If instability occurs:
- Decrease `cov_step_clip` to `1.0` or `1.5`
- Or decrease `cov_lambda`

## Diagnostic Tools

### Enable Diagnostic Recording

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.03 \
    --cov_scale_mode constant \
    --cov_grad_norm global \
    --cov_step_clip 2.0 \
    --record_diagnostics \
    --dps_lambda 0.1
```

Generates diagnostic files: `results/dm_dps/diagnostics/{timestamp}_snr{X}_summary.txt`

Contains:
- `mean(c_t / b_t)`: Ratio of covariance correction to likelihood correction (target: 0.1-0.3)
- `mean(clip_rate_cov)`: Proportion of covariance correction that is clipped (target: < 0.2)
- `mean(||Δx_cov||)`: Magnitude of covariance correction
- `mean(||Δx_like||)`: Magnitude of likelihood correction

### Enable Debug Mode (View Each Timestep)

```bash
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.03 \
    --cov_scale_mode constant \
    --cov_grad_norm global \
    --cov_step_clip 2.0 \
    --debug_cov_scaling \
    --dps_lambda 0.1 \
    --sanity_snrs
```

Generates CSV file: `results/dm_dps/debug_cov_scaling/debug_snr{X}_scale_{mode}_norm_{norm}.csv`

Contains for each timestep:
- `t`: timestep
- `beta_t`: beta value
- `zeta_t`: scaling factor
- `grad_cov_raw_norm`: Raw gradient norm
- `grad_cov_normed_norm`: Normalized gradient norm
- `dx_cov_preclip_norm`: Correction norm before clipping
- `dx_cov_postclip_norm`: Correction norm after clipping
- `cov_clip_applied`: Whether clipping was applied

## FAQ

### Q1: What's the difference between the new version and the original version?

**A**: 
- **Original version**: `cov_correction = cov_lambda * beta_t * grad_cov` (no normalization, fixed scaling)
- **New version**: `cov_correction = cov_lambda * zeta_t * normalize(grad_cov)` (optional normalization, flexible scaling)

Advantages of the new version:
1. Gradient normalization makes `cov_lambda` easier to tune
2. `constant` scaling avoids late-stage failure
3. Independent clipping control

### Q2: When should I use the new version?

**A**: 
- When the original version performs poorly at low SNR or in later timesteps
- When you want more stable guidance
- When you want finer control

### Q3: How large should `cov_lambda` be?

**A**: 
- **No normalization** (`cov_grad_norm=none`): `0.001 - 0.01`
- **Global normalization** (`cov_grad_norm=global`): `0.01 - 0.1` (recommended to start with `0.03`)

### Q4: How to judge if guidance is effective?

**A**: 
1. Check NMSE curve: New version should be better than or at least as good as the original version
2. Check diagnostic files: `mean(c_t / b_t)` should be between `0.1 - 0.3`
3. Check clip rate: `mean(clip_rate_cov)` should be `< 0.2` (if too high, guidance is too strong)

### Q5: Why does the new version sometimes perform worse than the original version?

**A**: 
- `cov_lambda` may be set incorrectly (too large or too small)
- Scaling mode may not be suitable for current SNR
- Normalization may have changed the effective scale of gradients

**Solutions**: 
1. Use diagnostic tools to check `c_t / b_t` and `clip_rate_cov`
2. Adjust `cov_lambda` to appropriate range
3. Try different scaling modes

## Summary

**Recommended Configuration** (suitable for most scenarios):
```bash
--cov_scale_mode constant \
--cov_grad_norm global \
--cov_lambda 0.03 \
--cov_step_clip 2.0
```

This configuration:
- ✅ Uses `constant` scaling to avoid late-stage failure
- ✅ Uses `global` normalization to stabilize gradient scale
- ✅ `cov_lambda=0.03` provides moderate guidance strength
- ✅ `cov_step_clip=2.0` prevents explosion

Start tuning from here and fine-tune `cov_lambda` based on results!
