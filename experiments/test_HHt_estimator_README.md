# test_HHt_estimator.py User Guide

## Overview

`test_HHt_estimator.py` is a diagnostic script for evaluating the performance of channel covariance matrix estimation `R_h = H H^H` using **time averaging**.

### Mathematical Model

The script implements the following model:

1. **Select a single channel matrix** `H` (N_R × N_T)
2. **Generate data symbols** `X_d`: (N_T × N_d), where each column vector `x_d(t)` is a data symbol at time t
3. **Received signals** `Y_d = H @ X_d + N`, where `Y_d`: (N_R × N_d), each column `y_d(t)` is the received signal at time t
4. **Time-averaged covariance** `R_y_hat = (1/N_d) * Y_d @ Y_d.conj().T`
5. **Estimated channel covariance** `R_h_hat = R_y_hat - sigma_n^2 * I_N_R`
6. **True reference** `R_h_true = H @ H.conj().T`

This implements **pure time averaging**: averaging over time samples t=1..N_d for a fixed channel H. No ensemble averaging is used in the covariance computation itself.

---

## Main Parameters

### Required Parameters

None (all parameters have default values)

### Core Parameters

#### SNR Settings

- `--snrs <SNR_SPEC>` (default: `-15.0`)
  - SNR value specification, supports two formats:
    - **Range syntax** (recommended): `--snrs=-15:5:1` means from -15 to 5 with step 1
    - **List syntax**: `--snrs="-15 -10 -5"` means testing three specific SNR values: -15, -10, -5
  - **Note**: When the value starts with `-`, you must use `=` syntax or quotes, e.g., `--snrs=-15:5:1` or `--snrs="-15:5:1"`
  - Examples:
    ```bash
    --snrs=-15:5:1      # Range: -15 to 5, step 1
    --snrs=-15:5        # Range: -15 to 5, default step 1
    --snrs="-15 -10 -5" # List: three specific values
    ```

#### Channel Settings

- `--ch_type <TYPE>` (default: `3gpp`)
  - Channel type, passed to `load_or_create_data`
  
- `--n_antennas_rx <N_R>` (default: `64`)
  - Number of receive antennas N_R
  
- `--n_antennas_tx <N_T>` (default: `16`)
  - Number of transmit antennas N_T
  
- `--n_time_samples <N_d>` (default: `300`)
  - Number of time samples N_d for time-averaged covariance estimation
  
- `--channel_idx <IDX>` (default: `0`)
  - Index of channel realization to use

#### Data Generation

- `--modulation <MOD>` (default: `bpsk`)
  - Data symbol modulation, options: `bpsk`, `qpsk`
  
- `--n_trials <N>` (default: `1`)
  - Number of Monte Carlo trials per SNR
  - In each trial, `X_d` and noise `N` are regenerated, but `H` remains fixed

#### Post-processing Modes

- `--postproc_mode <MODE>` (default: `none`)
  - Post-processing mode, options:
    - `none`: No post-processing
    - `hermitian`: Hermitian symmetrization only
    - `psd`: PSD projection (clamp negative eigenvalues to zero)
    - `psd_trace`: PSD projection + trace-preserving scaling
    - `psd_rankk`: PSD projection + rank-k truncation
    - `psd_rankk_trace`: PSD projection + rank-k truncation + trace-preserving scaling
    - `psd_shrink`: PSD projection + shrinkage (diagonal loading)
    - `psd_rankk_shrink`: PSD projection + rank-k truncation + shrinkage
  
- `--rank_k <K>` (default: `16`)
  - Rank for rank-k truncation (only used in `psd_rankk*` modes)

#### Shrinkage Parameters

- `--gamma <VAL>` (default: `0.0`)
  - Shrinkage parameter, only used in `psd_shrink` and `psd_rankk_shrink` modes
  - Shrinkage formula: `R_post = (1-γ) * R_base + γ * (tr(R_base)/N_R) * I`
  - **Note**: If `gamma > 0` but `postproc_mode` is not a shrink mode, gamma will be ignored and a warning will be displayed

- `--gamma_list <LIST>` (default: `""`)
  - Comma-separated list of gamma values for sweep, e.g., `"0,0.01,0.05,0.1,0.2"`
  
- `--run_gamma_sweep` (default: `False`)
  - If set, run the estimator for each gamma value in `--gamma_list` and print a summary table

#### Other Parameters

- `--seed <SEED>` (default: `0`)
  - Random seed for reproducibility
  
- `--psd_project` (backward compatibility flag)
  - If set and `--postproc_mode` is not specified, sets `postproc_mode='psd'`
  
- `--preserve_trace` (backward compatibility flag)
  - If set with `--psd_project`, sets `postproc_mode='psd_trace'`

---

## Usage Examples

### Basic Usage

```bash
# Single SNR, no post-processing
python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode none --snrs=-15 --n_time_samples 300

# Multiple SNR values (range syntax)
python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd --snrs=-15:5:1 --n_time_samples 300

# Multiple SNR values (list syntax)
python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd --snrs="-15 -10 -5 0 5" --n_time_samples 300
```

### Post-processing Mode Comparison

```bash
# No post-processing
python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode none --snrs=-15 --n_time_samples 300

# Hermitian symmetrization
python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode hermitian --snrs=-15 --n_time_samples 300

# PSD projection
python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd --snrs=-15 --n_time_samples 300

# PSD projection + trace-preserving scaling
python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd_trace --snrs=-15 --n_time_samples 300

# PSD projection + rank-k truncation
python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd_rankk --rank_k 16 --snrs=-15 --n_time_samples 300

# PSD projection + rank-k truncation + trace-preserving scaling
python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd_rankk_trace --rank_k 16 --snrs=-15 --n_time_samples 300
```

### Shrinkage Modes

```bash
# PSD projection + shrinkage (single gamma)
python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd_shrink --gamma 0.1 --snrs=-15 --n_time_samples 300

# PSD projection + rank-k truncation + shrinkage (single gamma)
python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd_rankk_shrink --rank_k 16 --gamma 0.05 --snrs=-15 --n_time_samples 300

# Gamma sweep (multiple gamma values)
python experiments/test_HHt_estimator.py --modulation bpsk --n_trials 100 --seed 0 --postproc_mode psd_rankk_shrink --rank_k 16 --run_gamma_sweep --gamma_list "0,0.01,0.05,0.1,0.2" --snrs=-15 --n_time_samples 300
```

### Complete Example

```bash
# Full configuration: multiple SNRs, Monte Carlo averaging, rank-k truncation + shrinkage
python experiments/test_HHt_estimator.py \
    --modulation bpsk \
    --n_trials 100 \
    --seed 0 \
    --postproc_mode psd_rankk_shrink \
    --rank_k 16 \
    --gamma 0.05 \
    --snrs=-15:5:1 \
    --n_time_samples 1500 \
    --n_antennas_rx 64 \
    --n_antennas_tx 16
```

---

## Output Description

### Configuration Information

At startup, the script prints configuration information:

```
Configuration:
  postproc_mode: psd_rankk_shrink
  rank_k: 16
  seed: 0
  gamma: 0.05
```

If gamma > 0 but the mode is not a shrink mode, a warning will be displayed.

### Single Trial Output (n_trials=1)

```
SNR_dB | rel_err (||R_h_hat - R_h_true||_F / ||R_h_true||_F) | ||R_h_hat - R_h_true||_F
--------------------------------------------------------------------------------
 -15.0 |  1.234567e-02 |  5.678901e-03
```

### Monte Carlo Output (n_trials>1)

```
Monte Carlo averaging with 100 trials per SNR
SNR_dB | mean_rel_err | std_rel_err
--------------------------------------------------
 -15.0 |  1.234567e-02 |  2.345678e-03
```

### Detailed Diagnostic Information

For each SNR, detailed diagnostic information is printed:

```
  Diagnostics: trace_raw=..., trace_post=..., fro_raw=..., fro_post=..., 
               min_eig_raw=..., min_eig_post=..., neg_eig_energy_ratio_raw=..., 
               neg_eig_energy_ratio_post=..., rank_raw=..., rank_post=..., 
               rel_fro_err_raw=..., rel_fro_err_post=...
```

### Gamma Sweep Summary Table

When using `--run_gamma_sweep`, a summary table is printed at the end:

```
============================================================
Gamma Sweep Summary
============================================================
gamma | mean_rel_err | std_rel_err | rel_fro_err_post | trace_post | rank_post | min_eig_post
----------------------------------------------------------------------------------------------------
0.000 |  1.234567e-02 |  2.345678e-03 |  1.234567e-02 |  5.678901e+00 |  16 |  1.234567e-05
0.010 |  1.123456e-02 |  2.234567e-03 |  1.123456e-02 |  5.567890e+00 |  16 |  1.345678e-05
...
```

---

## Important Notes

1. **SNR Parameter Format**:
   - Use `=` syntax to avoid shell parsing issues: `--snrs=-15:5:1`
   - Or use quotes: `--snrs="-15:5:1"`

2. **Gamma Parameter**:
   - Only effective in `psd_shrink` or `psd_rankk_shrink` modes
   - If gamma is set but a different mode is used, a warning will be displayed

3. **Reproducibility**:
   - Use `--seed` parameter to ensure reproducible results
   - In gamma sweep mode, the random seed is reset before each gamma value to ensure identical Monte Carlo noise

4. **Performance Considerations**:
   - Increasing `--n_trials` improves statistical accuracy but increases runtime
   - Increasing `--n_time_samples` improves estimation accuracy but increases computation

---

## Frequently Asked Questions

**Q: Why doesn't the gamma parameter have any effect?**

A: Make sure you're using the correct `postproc_mode`:
- Use `psd_shrink` or `psd_rankk_shrink` mode
- Check the warning message in the configuration output

**Q: How to test multiple SNR values?**

A: Use range syntax or list syntax:
- Range: `--snrs=-15:5:1` (from -15 to 5, step 1)
- List: `--snrs="-15 -10 -5"` (specific values)

**Q: How to compare different post-processing modes?**

A: Run the same command, only changing the `--postproc_mode` parameter, keeping other parameters (especially `--seed`) the same.

---

## Technical Details

### Shrinkage Formula

For `psd_shrink` and `psd_rankk_shrink` modes, the shrinkage formula is:

```
R_post = (1-γ) * R_base + γ * (tr(R_base)/N_R) * I
```

where:
- `R_base` is the matrix after PSD projection (and optional rank-k truncation)
- `γ` is the shrinkage parameter (gamma)
- `N_R` is the number of receive antennas
- `I` is the identity matrix

### Post-processing Pipeline

1. **Hermitian Symmetrization**: `R = 0.5 * (R + R.conj().T)`
2. **PSD Projection**: Eigendecomposition, clamp negative eigenvalues to zero
3. **Rank-k Truncation** (optional): Keep top k largest eigenvalues
4. **Trace Preservation** (optional): Scale to preserve original trace
5. **Shrinkage** (optional): Apply diagonal loading

