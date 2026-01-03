## DPS-COV: Study of `cov_beta_power` (p) and `cov_clip_mode` (clipping) — experiments & analysis

This note summarizes an ablation study for **DPS-COV** (covariance guidance using \(HH^H\) side information). The core questions are:

1) How much does the cov-guidance scaling operator \(\zeta_t\) matter when using \(\beta_t\) vs \(\sqrt{\beta_t}\) (generalized to \(\beta_t^p\))?  
2) Does the performance improvement come mainly from **scaling (p)** or from **changing the clipping rule**?

Takeaways (for this configuration):

- **Clipping mode is decisive**: `elementwise` can completely break performance, while `norm` works and yields strong gains.
- With `norm` clipping fixed, **\(\sqrt{\beta_t}\) (\(p=0.5\)) still outperforms \(\beta_t\) (\(p=1.0\))**, but that gain is “fine-tuning level” (roughly ~10%–20% relative improvement at mid/high SNR).

---

## 1. Background & notation

In `dps_sampler.py`, the cov guidance update can be abstracted (ignoring normalization and other details) as:

\[
\Delta x_{\text{cov}}(t) \propto \lambda_{\text{cov}}\;\zeta_t\;\nabla_{\!x} \mathcal{L}_{\text{cov}}
\]

Historically, \(\zeta_t\) was selected via discrete modes:

- `beta_t`: \(\zeta_t=\beta_t\)
- `sqrt_beta_t`: \(\zeta_t=\sqrt{\beta_t}\)
- `constant`: \(\zeta_t=1\)
- `snr_aware`: \(\zeta_t=1/\sqrt{\beta_t}\)

To make sweeps easier, we added:

- **`--cov_beta_power p`**: overrides mode selection and uses \(\zeta_t=\beta_t^p\)  
  - \(p=1.0 \Rightarrow \beta_t\)
  - \(p=0.5 \Rightarrow \sqrt{\beta_t}\)
  - \(p=0 \Rightarrow 1\)
  - \(p=-0.5 \Rightarrow 1/\sqrt{\beta_t}\)-like

---

## 2. Two clipping rules (this is critical)

Cov guidance updates \(\Delta x_{\text{cov}}\) are often huge, so every step needs clipping (to bound step size/energy). Without clipping, numerical blow-ups or divergence are common.

This repo contains two fundamentally different clipping rules:

### 2.1 `elementwise`: per-element clamp (legacy behavior)

Form:

\[
\Delta x \leftarrow \operatorname{clip}(\Delta x,\,-C,\,+C)
\]

Meaning:

- Each tensor element is independently clamped to \([-C,C]\).
- Large components saturate; small components remain unchanged.
- **The update direction changes** because different coordinates are clipped by different amounts.
- In high-dimensional tensors, if many entries exceed the threshold, the update tends to look “saturated / sign-like”, potentially deviating strongly from the true gradient direction.

### 2.2 `norm`: global L2-norm scaling (new behavior)

Form:

\[
\Delta x \leftarrow \Delta x \cdot \min\left(1,\frac{C}{\|\Delta x\|_2+\varepsilon}\right)
\]

Meaning:

- Treat each per-sample update tensor as one vector and compute its L2 norm.
- If the norm exceeds \(C\), multiply the whole update by a scalar factor.
- **Direction is preserved** (only the length changes), which matches the optimization meaning of “step size control / energy bound”.

> Practical intuition: when cov gradients are extremely large and coordinate scales differ, `elementwise` can severely distort directions and break performance; `norm` is typically more stable.

### 2.3 Why this can hide the effect of \(p\)

If \(\Delta x_{\text{cov}}\) is clipped (nearly) every timestep, changing \(\zeta_t=\beta_t^p\) often only changes the **pre-clip magnitude**, while the **post-clip effective step** is pinned by \(C\). As a result, curves for different \(p\) can “collapse” together.

Therefore, when sweeping \(p\), it’s best to fix the clip rule (e.g., always `norm`) and check whether clipping is overly saturated.

---

## 3. Attribution experiment design

To isolate contributions, we ran two orthogonal comparisons:

### 3.1 Fix scaling (\(p=0.5\)), vary clipping

Same settings:

- `cov_beta_power=0.5` (\(\sqrt{\beta_t}\))
- `cov_lambda=0.01`
- `cov_step_clip=2.0`
- `dps_lambda=0.1`
- `--enable_snr_matching`
- `--sanity_snrs`

Only difference:

- `cov_clip_mode=elementwise` vs `cov_clip_mode=norm`

Expected output files in `results/dm_dps`:

- `..._scale=beta_pow0.5_clip=elementwise.csv`
- `..._scale=beta_pow0.5_clip=norm.csv`

### 3.2 Fix clipping (`norm`), vary scaling (\(p\))

Same settings:

- `cov_clip_mode=norm`
- `cov_lambda=0.01`
- `cov_step_clip=2.0`
- `dps_lambda=0.1`
- `--enable_snr_matching`
- `--sanity_snrs`

Only difference:

- `cov_beta_power=0.5` vs `cov_beta_power=1.0`

Expected files:

- `..._scale=beta_pow0.5_clip=norm.csv`
- `..._scale=beta_pow1_clip=norm.csv`

---

## 4. Commands (reproducible)

### 4.1 Fix p=0.5, compare clip

```bash
# sqrt(beta) + elementwise clip
python load_and_eval_dm_dps.py \
  --method dps_cov_oracle \
  --cov_beta_power 0.5 \
  --cov_clip_mode elementwise \
  --cov_lambda 0.01 \
  --cov_grad_norm none \
  --cov_step_clip 2.0 \
  --dps_lambda 0.1 \
  --enable_snr_matching \
  --sanity_snrs

# sqrt(beta) + norm clip
python load_and_eval_dm_dps.py \
  --method dps_cov_oracle \
  --cov_beta_power 0.5 \
  --cov_clip_mode norm \
  --cov_lambda 0.01 \
  --cov_grad_norm none \
  --cov_step_clip 2.0 \
  --dps_lambda 0.1 \
  --enable_snr_matching \
  --sanity_snrs
```

### 4.2 Fix norm clip, compare p

```bash
# p=0.5
python load_and_eval_dm_dps.py \
  --method dps_cov_oracle \
  --cov_beta_power 0.5 \
  --cov_clip_mode norm \
  --cov_lambda 0.01 \
  --cov_grad_norm none \
  --cov_step_clip 2.0 \
  --dps_lambda 0.1 \
  --enable_snr_matching \
  --sanity_snrs

# p=1.0
python load_and_eval_dm_dps.py \
  --method dps_cov_oracle \
  --cov_beta_power 1.0 \
  --cov_clip_mode norm \
  --cov_lambda 0.01 \
  --cov_grad_norm none \
  --cov_step_clip 2.0 \
  --dps_lambda 0.1 \
  --enable_snr_matching \
  --sanity_snrs
```

---

## 5. Key results & interpretation

> The tables below use the “latest attribution runs” on sanity SNRs \(-15/-10/-5/0/5\) dB.

### 5.1 Fix p=0.5: elementwise vs norm (huge difference)

| SNR(dB) | p=0.5 + elementwise | p=0.5 + norm |
|---:|---:|---:|
| -15 | 2.8547 | 0.7984 |
| -10 | 2.4309 | 0.4507 |
| -5  | 1.8386 | 0.2177 |
| 0   | 1.3452 | 0.0939 |
| 5   | 0.8406 | 0.0388 |

Interpretation:

- `elementwise` is effectively “broken” here (NMSE drastically worsens).
- `norm` performs well and matches the expected magnitude of repo plots / historical results.
- **Therefore: whether the method works at all is largely determined by the clipping rule.**

### 5.2 Fix norm clip: p=0.5 vs p=1.0 (moderate difference, clear at mid/high SNR)

| SNR(dB) | p=0.5 + norm | p=1.0 + norm |
|---:|---:|---:|
| -15 | 0.7997 | 0.8024 |
| -10 | 0.4502 | 0.4541 |
| -5  | 0.2175 | 0.2269 |
| 0   | 0.0940 | 0.1059 |
| 5   | 0.0388 | 0.0480 |

Interpretation:

- At very low SNR (-15 dB), the gap is small (side guidance is more easily “washed out” by observation uncertainty).
- At mid/high SNR (0/5 dB), the gap is clear and p=0.5 is better:
  - ~11% relative improvement at 0 dB
  - ~19% relative improvement at 5 dB
- **So, with stable clipping fixed (`norm`), \(\sqrt{\beta_t}\) still helps, but it is a second-order tuning knob.**

---

## 6. Why curves can “collapse” for p<1

Common reasons:

1) **Clipping saturation**: if \(\|\Delta x_{\text{cov}}\|\gg C\) most of the time, the post-clip step is nearly fixed to \(C\), flattening the effect of \(p\).  
2) **Limited numeric range of \(\beta_t\)**: typically \(\beta_t\in(10^{-4},10^{-1})\); within a narrow range \(p\in[0.35,0.95]\), changes in \(\beta_t^p\) may be only a finite factor and not reshape NMSE dramatically.  
3) **No strength re-calibration**: sweeping \(p\) without re-tuning `cov_lambda` changes both the time-weighting profile and the overall effective strength, making many settings look similar after clipping.

Recommendations:

- Fix `--cov_clip_mode norm` during p sweeps.
- Re-tune `--cov_lambda` per p (aim: avoid persistent clipping saturation / keep clip-rate reasonable).
- Use `--debug_cov_scaling` to check whether `dx_cov_postclip_norm` is pinned to the threshold for most steps.

---

## 7. Practical recommendation: a “fair” p-sweep protocol

If you want to study the *shape* effect of \(\zeta_t=\beta_t^p\) without confounds:

- Fix `--cov_clip_mode norm`
- Fix `--cov_step_clip` (e.g., 2.0)
- Choose `cov_lambda(p)` per p so a target statistic is matched across p, e.g.:
  - mid/late-stage `mean(dx_cov_postclip_norm)` in the same range
  - or `mean(c_t/b_t)` in the same range (if using a diagnostic recorder)

This produces a comparison closer to the intrinsic contribution of the power schedule itself.


