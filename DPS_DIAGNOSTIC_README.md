# DPS Sampler Diagnostic Tool User Guide

## Overview

`dps_diagnostic_recorder.py` is a diagnostic tool for detecting magnitude issues in DPS sampler guidance, helping to quickly identify hyperparameter configuration problems.

## Core Features

1. **Magnitude Check**: Records gradient/correction magnitudes at each timestep
2. **Pathology Diagnosis**: Automatically identifies three common issues (A: Too weak, B: Late-stage failure, C: Late-stage explosion)
3. **Scaling Mode Testing**: Supports four scaling modes: `beta_t`, `sqrt_beta_t`, `constant`, `snr_aware`
4. **Lambda Suggestions**: Automatically suggests reasonable initial `lambda_cov` values based on statistics
5. **Proxy Testing**: Optional use of proxy gradient to verify if it's a `cov_grad` implementation issue

## Usage

### Basic Usage

The diagnostic recorder is integrated into `load_and_eval_dm_dps.py`. To enable diagnostics:

```bash
# Enable diagnostic recording
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_step_clip 2.0 \
    --record_diagnostics \
    --dps_lambda 0.1
```

### Parameters

- `--record_diagnostics`: Enable diagnostic recording
- `--cov_lambda`: Covariance guidance strength, default 0.01
- `--cov_scale_mode`: Scaling mode
  - `beta_t`: Default, consistent with DPS likelihood
  - `sqrt_beta_t`: Uses sqrt(β_t), more balanced
  - `constant`: Fixed strength, no decay with t
  - `snr_aware`: Stronger in later stages (when cleaner)
- `--cov_grad_norm`: Gradient normalization mode
  - `none`: No normalization (default)
  - `global`: Normalize to unit scale (recommended)
  - `by_x`: Normalize relative to x scale
  - `by_r`: Normalize relative to covariance matrix scale
- `--cov_step_clip`: Clipping threshold for covariance correction
- `--sanity_snrs`: Test with limited SNR range for quick checks

## Output Description

### 1. Console Output

The script outputs:
- **Pathology Diagnosis Results**: Whether issues A/B/C are detected
- **Suggestions**: Specific repair suggestions for detected issues
- **Lambda Suggestions**: Suggested `lambda_cov` values based on statistics

### 2. Diagnostic Summary Files

Saved in `results/dm_dps/diagnostics/{timestamp}_snr{X}_summary.txt`:
- `mean(c_t / b_t)`: Ratio of covariance correction to likelihood correction (target: 0.1-0.3)
- `mean(clip_rate_cov)`: Proportion of covariance correction that is clipped (target: < 0.2)
- `mean(||Δx_cov||)`: Magnitude of covariance correction
- `mean(||Δx_like||)`: Magnitude of likelihood correction

### 3. Debug CSV Files (if `--debug_cov_scaling` is enabled)

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

## Diagnosis Guide

### Pathology A: Side Info Too Weak (`c_t << b_t`)

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

### Pathology B: Late-Stage Failure (`c_t` → 0 in later stages)

**Symptoms**:
- `c_t` is normal in early stages, suddenly drops to near 0 in later stages
- `zeta_t` becomes very small in later stages (if using `beta_t` scaling)

**Possible Causes**:
- `beta_t` scaling causes late-stage failure (`beta_t` becomes very small in later stages)

**Fix Suggestions**:
- Use `constant` or `sqrt_beta_t` scaling mode
- Or use `snr_aware` mode (stronger in later stages)

### Pathology C: Late-Stage Explosion + High Clip Rate

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

## Lambda Suggestion Calculation

The diagnostic tool automatically calculates suggested `lambda_cov` values based on:
- Target: side correction ≈ 20% likelihood correction
- Statistics from mid-to-late timesteps (t ≈ 0.6T - 0.9T)
- Formula: `lambda_cov ≈ 0.2 * mean_like / (scale_t * mean_grad_cov)`

**Note**: This is only an initial suggestion and may need fine-tuning based on actual results.

## Example Workflow

```bash
# 1. Test with default parameters and enable diagnostics
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda 0.01 \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm none \
    --cov_step_clip 2.0 \
    --record_diagnostics \
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

## Output File Locations

All outputs are saved in:
- `results/dm_dps/diagnostics/{timestamp}_snr{X}_summary.txt`: Diagnostic summaries
- `results/dm_dps/debug_cov_scaling/debug_snr{X}_scale_{mode}_norm_{norm}.csv`: Per-timestep debug data (if `--debug_cov_scaling` is enabled)

## Notes

1. **Test Samples**: Default sample counts are sufficient for diagnosis
2. **SNR Selection**: Recommend testing multiple SNRs (-15, -10, -5, 0, 5 dB) for comprehensive diagnosis
3. **Timestep Selection**: Default uses all timesteps, can be limited via `--num_steps`
4. **Device**: Automatically detects CUDA, can force CPU with `--device cpu`

## Relationship to Main DPS Implementation

This diagnostic tool:
- Based on `dps_sampler.py` implementation
- Adds detailed statistical recording functionality
- Supports different scaling modes (original only supported `beta_t`)
- Does not modify original code, runs independently

After diagnosis, suggested parameters can be applied to `load_and_eval_dm_dps.py` for formal evaluation.
