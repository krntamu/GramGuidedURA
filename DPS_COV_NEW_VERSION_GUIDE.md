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

