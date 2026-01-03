## DPS Tests: EXP (`--exp_key`) Cheat Sheet

This document explains the **DPS test / ablation experiment keys** in `load_and_eval_dm_dps.py` via `--exp_key`:
`A/B/C/D/E/Eprime/F/G`.

The implementation lives in `dps_sampler.py` under `DpsSampler.reverse_step_dps()`.

---

## 0. What are we testing?

In this repo, “DPS testing” typically means evaluating a pretrained DMCE diffusion model via `load_and_eval_dm_dps.py`, and guiding sampling with:

- **Likelihood guidance** (data consistency): default AWGN observation model \(y=x+n\), \(n\sim\mathcal N(0,\sigma_y^2 I)\).
- **Optional covariance side information** (cov guidance): using \(R=HH^H\) (oracle or estimated).

> `--exp_key` controls **likelihood guidance** behavior only.  
> `--method dps_cov_oracle|dps_cov_est` controls whether covariance guidance is enabled.

---

## 1. Entry points & the most relevant flags

### 1.1 Entry points

- `load_and_eval_dm_dps.py`: main evaluation script (DPS / cov guidance / EXP switches).
- `dps_sampler.py`: DPS sampler implementation (EXP logic lives here).

### 1.2 Flag quick reference (EXP-related)

- **`--exp_key`**: choose experiment key (A/B/C/D/E/Eprime/F/G).
- **`--dps_lambda`**: likelihood guidance strength (applies to A/B/C/D/Eprime/G; for E/F/G interpretation see below).
- **`--sigma_y2`**: AWGN variance \(\sigma_y^2\) (critical for closed-form likelihood variants, especially E/Eprime/F/G).
- **`--reverse_add_random`**: whether the **prior reverse step** is stochastic (DDPM-style).  
  - For DPS, deterministic is usually recommended.
  - **EXP G requires deterministic** steps (`add_random=False`), otherwise it errors.
- **`--enable_snr_matching`**: enable SNR-matched start timestep `t_start`.  
  Note: for ablations, the script defaults to **disabling** SNR matching (full reverse chain) unless you explicitly enable it.
- **`--like_beta_power`**: for EXP **A/B/C/D** post-add likelihood update, replace \(\beta_t\) with \(\beta_t^p\) (default \(p=1\)).  
  Example: test \(\beta_t^2\) via `--like_beta_power 2`.

---

## 2. Notation & baseline DPS structure

Let diffusion timesteps be \(t\in\{T-1,\dots,0\}\):

- \(\bar\alpha_t\): `dm.alphas_cumprod[t]`
- \(\beta_t\): `dm.betas[t]`
- \(\sigma_t^2 = \frac{1-\bar\alpha_t}{\bar\alpha_t}\) (used by Eprime “sigma_t^2 scaling”)

Default AWGN likelihood gradient:

\[
\nabla_x \log p(y|x) = \frac{y-x}{\sigma_y^2}
\]

In `dps_sampler.py`, the “standard DPS” structure (A/B/D) is:

1) prior reverse: `x_prior = dm.reverse_step(x_t, t, add_random=...)`  
2) evaluate likelihood gradient at `x_prior`: `grad_like_H0 = g(x_prior, y, t)`  
3) post-add update:
\[
x_{t-1} = x_{\text{prior}} + \lambda_{\text{dps}} \cdot s_t \cdot \text{grad}
\]
where \(s_t\) depends on `exp_key` (A/B/D/Eprime/C).

---

## 3. EXP overview

| exp_key | Purpose | Update form | Extra knobs |
|---|---|---|---|
| **A** | baseline DPS | post-add with \(\beta_t\) | none |
| **B** | scalar Jacobian approx | post-add with \(\beta_t\), but grad scaled by \(1/\sqrt{\bar\alpha_t}\) | none |
| **C** | autograd “gold” gradient | post-add with \(\beta_t\), grad via autograd | `sigma_y2` (used in loss) |
| **D** | soft-gated scaling | post-add with \(\beta_t\bar\alpha_t^{\gamma-1/2}\) | `--gamma` (\(\gamma=0.5\Rightarrow\) equals A) |
| **E** | closed-form likelihood (score injection into reverse mean) | **score injection** (not post-add) | `--like_weight`, `--lw_schedule`, … |
| **Eprime** | diagnostic closed-form, but keep post-add | post-add with \(\sigma_t^2\) + closed-form score | none (still needs `sigma_y2`) |
| **F** | “reverse optimization update” (paper-style) | **MAP + noise** (not post-add) | `--like_weight`, `sigma_y2` |
| **G** | Algorithm-3 style (paper) | **DDIM + explicit posterior correction** | `--g_tau1` (recommend ≥1), must be deterministic |

---

## 4. EXP details (as implemented)

### 4.1 EXP A: baseline

- Gradient at `x_prior`: `grad_like = (y - x_prior)/sigma_y2`
- Update:
\[
x_{t-1} = x_{\text{prior}} + \lambda_{\text{dps}} \cdot \beta_t \cdot \frac{y-x_{\text{prior}}}{\sigma_y^2}
\]

Example:

```bash
python load_and_eval_dm_dps.py --method dps --exp_key A --dps_lambda 0.1 --sigma_y2 1.0
```

---

### 4.2 EXP B: scalar Jacobian \((1/\sqrt{\bar\alpha_t})\)

- Compute `grad_like_H0 = g(x_prior, y, t)`
- Multiply a scalar Jacobian approx:
\[
\text{grad} = \frac{1}{\sqrt{\bar\alpha_t}} \cdot \text{grad\_like\_H0}
\]
- Post-add with \(\beta_t\):
\[
x_{t-1} = x_{\text{prior}} + \lambda_{\text{dps}}\cdot \beta_t \cdot \text{grad}
\]

```bash
python load_and_eval_dm_dps.py --method dps --exp_key B --dps_lambda 0.1 --sigma_y2 1.0
```

---

### 4.3 EXP C: autograd gold gradient (full chain rule)

- Builds a differentiable copy of `reverse_mean` (since `dm.reverse_step()` is `@torch.no_grad()`).
- Defines likelihood loss:
\[
\mathcal L = \frac{1}{2\sigma_y^2}\|y-\hat x_0(x_t)\|_F^2
\]
- Backprop to get `grad_like = - dL/dx_t`
- Still uses post-add with \(\beta_t\):
\[
x_{t-1} = x_{\text{prior}} + \lambda_{\text{dps}}\cdot \beta_t \cdot \text{grad\_autograd}
\]

Notes:

- Very expensive (autograd per timestep); mainly for verification.
- `sigma_y2` is extracted from the likelihood closure when possible; otherwise pass `--sigma_y2`.

```bash
python load_and_eval_dm_dps.py --method dps --exp_key C --dps_lambda 0.1 --sigma_y2 1.0
```

---

### 4.4 EXP D: soft-gated scaling \(\beta_t \bar\alpha_t^{\gamma-1/2}\)

- Uses baseline gradient `grad_like_H0`
- Per-step scalar:
\[
s_t = \beta_t\cdot \bar\alpha_t^{\gamma-\frac12}
\]
- Update:
\[
x_{t-1} = x_{\text{prior}} + \lambda_{\text{dps}}\cdot s_t \cdot \text{grad}
\]

Equivalence check:

- When \(\gamma=0.5\), \(\bar\alpha_t^{\gamma-1/2}=1\), so **D equals A**.

```bash
python load_and_eval_dm_dps.py --method dps --exp_key D --gamma 1.0 --dps_lambda 0.1 --sigma_y2 1.0
```

---

### 4.5 EXP E: closed-form likelihood (score injection into reverse mean)

This EXP changes structure: it does **not** use post-add correction. Instead it injects likelihood score into the reverse-mean score.

Supported only for `objective='pred_noise'` and `reverse_method='reverse_mean'`.

- Prior score (DDPM-style approximation):
\[
\text{score}_{\text{prior}} = -\frac{\hat\varepsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}
\]
- Closed-form likelihood score (defined at **\(x_t\)**, not `x_prior`):
\[
\text{score}_{\text{like}} =
\frac{y - x_t/\sqrt{\bar\alpha_t}}
{\sqrt{\bar\alpha_t}\left(\sigma_y^2 + \frac{1-\bar\alpha_t}{\bar\alpha_t}\right)}
\]
(`score_like_cf` in code)

- Combine:
\[
\text{score}_{\text{total}}=\text{score}_{\text{prior}} + w_t\cdot \text{score}_{\text{like}}
\]
where \(w_t\) is controlled by `--lw_schedule`:
  - `const`: \(w_t=\text{like\_weight}\)
  - `ramp`: gated linear ramp based on \(\bar\alpha_t\) (`--lw_tau`, `--lw_max`)
  - `lastk`: boost only last K steps (`--lw_k`, `--lw_end`)

- Inject into reverse mean (equivalent form used in code):
\[
\mu_t = \frac{x_t + \beta_t\cdot \text{score}_{\text{total}}}{\sqrt{\alpha_t}}
\]
and optionally add \(\sqrt{\text{posterior\_var}}\cdot z\) if `add_random=True`.

Because likelihood is already injected, the code sets `correction=0`.

Example:

```bash
python load_and_eval_dm_dps.py --method dps --exp_key E --sigma_y2 1.0 --like_weight 1.0 --lw_schedule const --debug_cf
```

Fast sanity check (3 timesteps only: T-1 / T//2 / 0):

```bash
python load_and_eval_dm_dps.py --method dps --exp_key E --sigma_y2 1.0 --like_weight 1.0 --debug_cf_microtest
```

Record per-step “likelihood vs prior” balance (first batch per SNR):

```bash
python load_and_eval_dm_dps.py --method dps --exp_key E --sigma_y2 1.0 --like_weight 1.0 --record_like_balance
```

---

### 4.6 EXP Eprime: diagnostic closed-form, but keep post-add

- Uses the same closed-form likelihood score as EXP E (still defined at \(x_t\)):
\[
\text{score}_{\text{like}} = \frac{y - x_t/\sqrt{\bar\alpha_t}}
{\sqrt{\bar\alpha_t}\left(\sigma_y^2 + \frac{1-\bar\alpha_t}{\bar\alpha_t}\right)}
\]
- Treat it as the post-add “gradient term”, and use:
\[
s_t = \sigma_t^2 = \frac{1-\bar\alpha_t}{\bar\alpha_t}
\]
- Update:
\[
x_{t-1} = x_{\text{prior}} + \lambda_{\text{dps}}\cdot \sigma_t^2 \cdot \text{score}_{\text{like}}
\]

```bash
python load_and_eval_dm_dps.py --method dps --exp_key Eprime --dps_lambda 0.1 --sigma_y2 1.0 --debug_cf
```

---

### 4.7 EXP F: paper-style reverse optimization update (MAP + noise)

Supported only for `objective='pred_noise'`.

- Compute `score_prior` and `score_like_cf` (same closed-form as EXP E)
- Posterior score:
\[
\text{score}_{\text{post}}=\text{score}_{\text{prior}} + w\cdot \text{score}_{\text{like}}
\]
with \(w=\text{like\_weight}\) (constant)

- MAP-style update with posterior variance:
\[
x_t^{\text{map}} = x_t + \text{post\_var}_t\cdot \text{score}_{\text{post}}
\]
- Then:
\[
x_{t-1}=x_t^{\text{map}} + \sqrt{\text{post\_var}_t}\cdot z
\]

The code sets `correction=0` since `x_prev` is produced directly.

```bash
python load_and_eval_dm_dps.py --method dps --exp_key F --sigma_y2 1.0 --like_weight 1.0 --debug_cf
```

---

### 4.8 EXP G: paper Algorithm-3 style (DDIM + posterior correction)

Key points:

- **Requires deterministic** steps (`add_random=False`), otherwise it errors.
- Step 3 (DDIM, eta=0):
  - \(\hat x_0 = (x_t-\sqrt{1-\bar\alpha_t}\hat\varepsilon)/\sqrt{\bar\alpha_t}\)
  - \(x_{t-1}^{\text{ddim}}=\sqrt{\bar\alpha_{t-1}}\hat x_0 + \sqrt{1-\bar\alpha_{t-1}}\hat\varepsilon\)
- Step 4: closed-form likelihood score (same as EXP E)
- Step 5: explicit correction using the code’s `coef_step5`:
\[
x_{t-1} \leftarrow x_{t-1}^{\text{ddim}} + \lambda_{\text{dps}}\cdot \text{coef\_step5}\cdot \text{score}_{\text{like}}
\]

Important knob: `--g_tau1`

- If `--g_tau1 > 0`, the loop stops at \(t=\tau_1\) and outputs \(\hat x_0\) computed from \(x_{\tau_1}\).
- Recommended `tau1 >= 1` (running to 0 is often unstable).

```bash
python load_and_eval_dm_dps.py --method dps --exp_key G --sigma_y2 1.0 --dps_lambda 0.1 --g_tau1 5 --debug_cf
```

---

## 5. Common run templates

### 5.1 Likelihood-only (no covariance side info)

```bash
python load_and_eval_dm_dps.py \
  --method dps \
  --exp_key A \
  --dps_lambda 0.1 \
  --sigma_y2 1.0
```

### 5.2 DPS + oracle covariance (orthogonal to EXP)

```bash
python load_and_eval_dm_dps.py \
  --method dps_cov_oracle \
  --exp_key A \
  --dps_lambda 0.1 \
  --sigma_y2 1.0 \
  --cov_lambda 0.01 \
  --cov_scale_mode sqrt_beta_t \
  --cov_grad_norm none
```

---

## 6. Debug / diagnostics (fastest way to sanity-check scales)

- **Likelihood update magnitude**: `--debug_likelihood` (prints only the first 3 reverse steps)
- **Closed-form coefficients & norms (E/Eprime/F/G)**: `--debug_cf`
- **Fast 3-step run**: `--debug_cf_microtest` (runs fewer steps; may not print)
- **EXP E: record likelihood-vs-prior balance**: `--record_like_balance`

---

## 7. Common pitfall: SNR matching is disabled by default for ablations

In `load_and_eval_dm_dps.py`, ablation experiments A~G default to `snr_for_loop=None` (i.e., start from \(t=T-1\) and run the full reverse chain).  
If you want SNR-matched start for ablations, add:

```bash
--enable_snr_matching
```

---

## 8. Suggested practical workflow

- Start with **A** to validate `sigma_y2`, `dps_lambda`, deterministic steps, and reasonable NMSE.
- Try **D** (and verify the \(\gamma=0.5\) equivalence to A) if you want per-stage gating.
- Use **E / Eprime** to sanity-check closed-form likelihood scaling (use `--debug_cf_microtest` to iterate fast).
- Touch **F / G** last (closest to paper variants, but more sensitive/unstable).


