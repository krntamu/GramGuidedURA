# Jacobian Ablation Test: Implementation Guide

## Overview

This ablation test verifies whether DPS/likelihood guidance failed previously because the chain-rule/Jacobian term was missed when converting ∇_{H0} log p(Y|H0) into ∇_{Ht} log p(Y|Ht).

## Theory

**Observation model**: Y = H0 + N (after pilot decorrelation)

**Current implementation** (baseline):
- Uses plug-in estimate H0_hat(Ht) (via Tweedie/denoiser output)
- Computes: `g_like ≈ (1/eta^2)(Y - H0_hat)`
- **Missing**: Jacobian J_t = ∂H0_hat/∂Ht

**Correct chain rule**:
```
∇_{Ht} log p(Y|Ht) ≈ J_t^H * (1/eta^2)(Y - H0_hat(Ht))
```

## Experiment Definitions

### Experiment A: Baseline (Current Behavior)
- **Formula**: `g_like_A = (1/eta2) * (Y - H0_hat)`
- **Description**: Ignores Jacobian, uses gradient w.r.t. H0 directly
- **Implementation**: Direct use of `likelihood_grad_fn(x_prior, y, t)`

### Experiment B: Scalar Jacobian Approximation
- **Formula**: `g_like_B = (1/sqrt(alpha_bar_t)) * (1/eta2) * (Y - H0_hat)`
- **Description**: Multiplies baseline by scalar approximation of Jacobian
- **Implementation**: Multiplies gradient by `1/sqrt(alpha_bar_t)`
- **Note**: Uses `alpha_bar_t` from `dm.alphas_cumprod[t]` (consistent with q(x_t|x_0) = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps)

### Experiment C: Autograd "Gold" Gradient
- **Formula**: Computes full gradient via autograd
- **Description**: Defines `loss_like(Ht) = (1/(2*eta2)) * ||Y - H0_hat(Ht)||_F^2`, then computes `gradHt = -d(loss_like)/d(Ht)`
- **Implementation**: Uses `torch.autograd.grad()` to compute full Jacobian automatically
- **Safety**: Uses `torch.enable_grad()` locally, detaches result to avoid graph growth

## Usage

### Running Individual Experiments

**Experiment A (Baseline)**:
```bash
python load_and_eval_dm_dps.py \
    --method dps \
    --dps_lambda 0.1 \
    --exp_key A \
    --ch_type 3gpp \
    --n_path 3
```

**Experiment B (Scalar Jacobian)**:
```bash
python load_and_eval_dm_dps.py \
    --method dps \
    --dps_lambda 0.1 \
    --exp_key B \
    --ch_type 3gpp \
    --n_path 3
```

**Experiment C (Autograd Gold)**:
```bash
python load_and_eval_dm_dps.py \
    --method dps \
    --dps_lambda 0.1 \
    --exp_key C \
    --ch_type 3gpp \
    --n_path 3
```

### Running All Experiments (Recommended)

Run each experiment separately with the **same random seed** and **same test set**:

```bash
# Set a fixed seed for reproducibility
export PYTHONHASHSEED=0

# Experiment A
python load_and_eval_dm_dps.py --method dps --dps_lambda 0.1 --exp_key A --ch_type 3gpp --n_path 3

# Experiment B
python load_and_eval_dm_dps.py --method dps --dps_lambda 0.1 --exp_key B --ch_type 3gpp --n_path 3

# Experiment C
python load_and_eval_dm_dps.py --method dps --dps_lambda 0.1 --exp_key C --ch_type 3gpp --n_path 3
```

### Plotting Results

After running all three experiments, use the plotting script:

```bash
python plot_ablation_jacobian.py \
    --base_dir results/dm_dps \
    --method dps \
    --dps_lambda 0.1
```

This will:
1. Load results from experiments A, B, and C
2. Create an overlayed plot showing NMSE vs SNR for all three
3. Print a summary table to stdout

## Important Implementation Details

### 1. SNR Matching Disabled for Ablation

**Critical**: For ablation experiments (exp_key != 'A'), SNR matching is **automatically disabled**. The code uses the **full T** (original failed setting) instead of SNR-matched timestep selection.

This ensures:
- All experiments start from the same timestep (T-1)
- Identical evaluation protocol across A/B/C
- Fair comparison without SNR-dependent starting points

### 2. Identical Evaluation Protocol

All experiments use:
- **Same random seed** (set via environment or code)
- **Same test set** (same data loading)
- **Same SNR grid** (default: -15 to 5 dB, step 1)
- **Same number of Monte Carlo trials** (same batch processing)
- **Same diffusion steps T** (full T, no SNR matching for ablation)
- **Same starting strategy** (from T-1, not SNR-matched)

### 3. Alpha_bar_t Definition

The code uses `dm.alphas_cumprod[t]` which corresponds to:
```
q(x_t|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
```

This is consistent with the forward diffusion process used in the codebase.

### 4. Autograd Safety (Experiment C)

Experiment C uses careful gradient management:
- `torch.enable_grad()` only when computing the gradient
- `torch.autograd.grad()` with `retain_graph=False, create_graph=False`
- Result is immediately detached to avoid graph growth
- Rest of computation uses `torch.no_grad()` context

### 5. Result File Naming

Results are saved with experiment key in the filename:
- Experiment A: `*_method=DPS_*_dps_lambda=0.1_*_best.csv` (no exp_key suffix)
- Experiment B: `*_method=DPS_*_dps_lambda=0.1_*_exp=B_best.csv`
- Experiment C: `*_method=DPS_*_dps_lambda=0.1_*_exp=C_best.csv`

## Expected Outputs

### Console Output

Each experiment prints:
- Configuration summary (including exp_key)
- Per-SNR NMSE results
- Summary table (if multiple exp_keys are run)

### CSV Files

Saved in `results/dm_dps/`:
- `{timestamp}_{ch_type}_*_method=DPS_*_dps_lambda={lambda}_*_exp={key}_best.csv`
- Contains columns: `SNR`, `nmse_dm_dps`

### Plot File

Generated by `plot_ablation_jacobian.py`:
- `ablation_jacobian_method=dps_lambda=0.1.png`
- Shows overlayed NMSE curves for A/B/C
- Includes legend and summary table

## Interpretation Guide

### If Experiment B improves over A:
- **Conclusion**: Scalar Jacobian approximation helps
- **Implication**: Missing Jacobian was indeed a problem
- **Next step**: Consider using B or investigating full Jacobian (C)

### If Experiment C improves over A and B:
- **Conclusion**: Full Jacobian (autograd) provides best results
- **Implication**: Scalar approximation is insufficient
- **Next step**: Consider using C or finding better approximation

### If All experiments perform similarly:
- **Conclusion**: Jacobian term may not be critical for this problem
- **Implication**: Other factors (hyperparameters, model, etc.) may be more important
- **Next step**: Investigate other potential issues

### If Experiment C is worse:
- **Possible causes**: 
  - Numerical instability in autograd
  - Memory issues (graph too large)
  - Implementation bug
- **Next step**: Check for NaN/Inf, verify autograd computation

## Code Changes Summary

### Modified Files

1. **`dps_sampler.py`**:
   - Added `exp_key` parameter to `DpsSampler.__init__`
   - Modified `reverse_step_dps()` to implement A/B/C logic
   - Removed `@torch.no_grad()` decorator (managed manually for experiment C)
   - Added sigma_y2 extraction for experiment C

2. **`load_and_eval_dm_dps.py`**:
   - Added `--exp_key` CLI argument
   - Modified `generate_posterior_sample()` call to pass `exp_key`
   - Disabled SNR matching for ablation experiments (pass `snr=None`)
   - Added experiment key to result file suffix
   - Added summary table output for ablation experiments

3. **`plot_ablation_jacobian.py`** (new):
   - Script to load and plot results from A/B/C
   - Creates overlayed NMSE curves
   - Prints summary table

## Safety Checks

✅ **Shape consistency**: `g_like` has same shape/dtype/device as `Ht`  
✅ **Complex handling**: Uses existing real/imag format `[B, 2, R, T]`  
✅ **Gradient safety**: Experiment C properly manages autograd context  
✅ **No hyperparameter changes**: All other parameters remain identical  
✅ **Identical evaluation**: Same seed, test set, SNR grid, T  

## Troubleshooting

### Experiment C fails with memory error:
- **Cause**: Autograd graph too large
- **Solution**: Reduce batch size or use smaller model

### Experiment C produces NaN:
- **Cause**: Numerical instability in autograd
- **Solution**: Check sigma_y2 value, verify loss computation

### Results are identical across A/B/C:
- **Possible causes**: 
  - Jacobian term is negligible
  - Implementation bug (check exp_key is being used)
  - Hyperparameters dominate
- **Solution**: Verify exp_key is passed correctly, check gradient magnitudes

### Plot script can't find files:
- **Cause**: File naming mismatch
- **Solution**: Check actual filenames in `results/dm_dps/`, adjust pattern in script

## Reproducibility

To ensure reproducible results:

1. **Set random seeds**:
   ```python
   import torch
   import numpy as np
   torch.manual_seed(0)
   np.random.seed(0)
   ```

2. **Use same test set**: Don't change data loading between experiments

3. **Use same device**: Run all experiments on same device (CPU/GPU)

4. **Document versions**: Note PyTorch version, CUDA version, etc.

## Example Workflow

```bash
# 1. Run all three experiments
for exp in A B C; do
    python load_and_eval_dm_dps.py \
        --method dps \
        --dps_lambda 0.1 \
        --exp_key $exp \
        --ch_type 3gpp \
        --n_path 3 \
        --sanity_snrs  # Quick test with limited SNRs
done

# 2. Plot results
python plot_ablation_jacobian.py \
    --base_dir results/dm_dps \
    --method dps \
    --dps_lambda 0.1

# 3. Compare results
# Check the printed summary table and plot file
```

## Notes

- **Default behavior**: If `--exp_key` is not specified, defaults to 'A' (baseline)
- **Backward compatibility**: Existing code without `exp_key` continues to work
- **Performance**: Experiment C is slower due to autograd overhead
- **Memory**: Experiment C uses more memory due to gradient computation

