## DPS Code Cleanup Checklist (for code review)

Goal: reduce maintenance burden and reviewer load by focusing on what is used in current experiments:
- **EXP H** (Tweedie-form likelihood) and **EXP E** (closed-form likelihood).

This document lists **candidates** to delete/archieve and how to do it safely. No code changes are performed by this file.

---

## 0) Principles

- **One canonical entrypoint**: `load_and_eval_dm_dps.py`
- **Two supported likelihood modes**: `exp_key ∈ {E, H}`
- Everything else should be either:
  - moved to `legacy/` or
  - deleted behind a short deprecation window

---

## 1) Likelihood EXP keys to deprecate (code)

File: `dps_sampler.py`

- **Candidates to remove**: `A/B/C/D/Eprime/F/G`
  - Reason: not used for current final curves; increases branching complexity in `reverse_step_dps()`.
  - Risk: medium if you still need old ablations; low if team agrees to standardize on E/H.
  - Safer alternative: move to `dps_sampler_legacy.py` and keep import path for a short time.

**Suggested deletion order**
1. `C` (autograd “gold”): expensive + hard to maintain.
2. `Eprime`, `F`, `G`: paper-style variants with extra coefficients/debug.
3. `A/B/D`: old post-add variants once H fully replaces them.

---

## 2) CLI flags to deprecate (entrypoint)

File: `load_and_eval_dm_dps.py`

If you commit to E/H only, candidates include:
- `--gamma` (only used by EXP D)
- `--g_tau1` (only used by EXP G)
- `--like_beta_power` (only used by A/B/C/D)
- `--debug_cf_microtest` / `--record_like_balance` (if EXP E debug is no longer needed)

Recommendation: keep flags for 1–2 weeks behind a warning, then remove.

---

## 3) Diagnostic / FFT code duplication

Files:
- `load_and_eval_dm_dps.py` (has embedded FFT diagnostic helpers)
- `fft_diagnostics.py` (optional module, if present in your repo)

Candidates:
- Keep **one** FFT diagnostic implementation; remove duplicates.
- Keep only “one-shot” diagnostic entrypoints (first batch, first SNR).

---

## 4) Plot scripts cleanup

Files in repo root:
- `plot_3GPP_NLOS.py`
- `plot_Quadriga_LOS.py`
- `plot_Shrinkage_Rankk.py`
- `plot_likelihood.py`

Candidates:
- Remove commented-out legacy curves once final curves are locked.
- Standardize:
  - output directory `results/useful_results/`
  - consistent color map / labels across plots

---

## 5) Experimental scripts / one-offs

Directory: `experiments/`

Candidates to move to `experiments/legacy/` or delete:
- old estimator tests that are no longer referenced by docs or CI

Keep:
- `conditional_gram_consistency.py` (useful for mechanism checks)
- `gram_spectral_diagnostics.py` (dataset structure diagnostics)

---

## 6) Documentation cleanup

- Keep `DPS_COMPLETE_USER_GUIDE.md` as the canonical doc.
- Move long-form diagnostics into `DPS_COMPLETE_USER_GUIDE_APPENDIX.md`.
- Avoid separate “cheat sheet” docs unless auto-generated.


