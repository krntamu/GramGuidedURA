"""
Plot: 3GPP NLOS Channel Estimation Results

This script visualizes DPS and DPS-COV performance for 3GPP NLOS channel.

Methods compared:
- DM: DDIM (unconditional diffusion model, baseline)
- DM + First-Order Guidance: DPS (lambda=0.1)
- DM + Closed-Form Likelihood (Exp E): DPS with closed-form likelihood score injection (paper Theorem-1 style)
- DM + Multi-Order Guidance (oracle, sqrt(beta_t)): DPS-COV with oracle covariance
- DM + Multi-Order Guidance (est, sqrt(beta_t)): DPS-COV with estimated covariance
- LMMSE: Linear Minimum Mean Square Error (genie-aided, upper bound)

Channel Type: 3GPP NLOS
SNR Range: -15 to 5 dB (step 1 dB)
"""

import numpy as np
import matplotlib
import os
from pathlib import Path

# Set backend based on environment
# In WSL, if DISPLAY is not set, use non-interactive backend
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
    print("Note: Using non-interactive backend. Plot will be saved to file.")

import matplotlib.pyplot as plt

# Create output directory if it doesn't exist
output_dir = Path('results') / 'useful_results'
output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# SNR grid
# =========================
snr = np.arange(-15, 6, 1, dtype=float)

# =========================
# Data Definitions
# =========================
# DM: DDIM (unconditional diffusion model, baseline)
nmse_dm = np.array([
    0.956674993, 0.932828784, 0.898368776, 0.851259053, 0.79441011,
    0.727668762, 0.654289067, 0.574952424, 0.499485046, 0.427704483,
    0.364863485, 0.309671074, 0.262683958, 0.220814109, 0.1872347,
    0.157922253, 0.133347675, 0.112339757, 0.095159866, 0.079534099,
    0.067043602
])

# CME-IS baseline (empirical-prior CME via IS over deltas; knn proposal; n_eval=200; alpha=0.7)
# NOTE: Only computed at a subset of SNRs; missing SNR points are NaN and will be skipped in plotting.
# Source: terminal runs of cme_baseline_3gpp_is.py (proposal=knn, pool_size=min(120000, n_train)=100000, K=512, refine_steps=1).
nmse_cme_is = np.full_like(snr, np.nan, dtype=float)
_cme_is_points = {
    -15.0: 8.5324e-01,
    -13.0: 7.1445e-01,
    -11.0: 5.7753e-01,
    -10.0: 5.2179e-01,
    -9.0:  4.7881e-01,
    -7.0:  3.7813e-01,
    -5.0:  3.0617e-01,
    -3.0:  2.3403e-01,
     0.0:  1.5444e-01,
     5.0:  6.6497e-02,
}
for _snr_db, _nmse in _cme_is_points.items():
    _idx = int(np.where(snr == _snr_db)[0][0])
    nmse_cme_is[_idx] = _nmse

# FOG: DPS (First-Order Guidance) with lambda=0.1
nmse_fog = np.array([
    0.953405201, 0.923455656, 0.883167565, 0.831169307, 0.767774642,
    0.697487533, 0.620056272, 0.542053878, 0.466282755, 0.400504202,
    0.343076289, 0.292655855, 0.250314415, 0.211691454, 0.180478558,
    0.153672323, 0.13030912,  0.11008589,  0.094051592, 0.078737386,
    0.066486768
])

# FOG_CF: DPS (Exp E) closed-form likelihood score injection (exp_key=E)
# Source: terminal output table SNR -15..5 (step 1) from load_and_eval_dm_dps.py
# - nmse_fog_cf_e_snr_match: with SNR matching (t_start) enabled
# - nmse_fog_cf_e_fullT: without SNR matching (full reverse chain)
nmse_fog_cf_e = np.array([
    9.093432e-01, 8.700707e-01, 8.176520e-01, 7.546990e-01, 6.863842e-01,
    6.193473e-01, 5.470487e-01, 4.820425e-01, 4.200711e-01, 3.654773e-01,
    3.175669e-01, 2.756021e-01, 2.390080e-01, 2.061705e-01, 1.774397e-01,
    1.521594e-01, 1.301131e-01, 1.107215e-01, 9.371015e-02, 7.937575e-02,
    6.680968e-02
], dtype=float)

nmse_fog_cf_e_fullT = np.array([
    9.602736e-01, 9.445305e-01, 9.233588e-01, 8.954188e-01, 8.574855e-01,
    8.141812e-01, 7.576047e-01, 6.917177e-01, 6.189027e-01, 5.443640e-01,
    4.699221e-01, 4.004216e-01, 3.382464e-01, 2.837459e-01, 2.375206e-01,
    1.984276e-01, 1.654709e-01, 1.374218e-01, 1.140154e-01, 9.470135e-02,
    7.823673e-02
], dtype=float)

# MOG_ORACLE_BETA: DPS-COV with oracle covariance, using beta_t scaling
nmse_mog_oracle_beta = np.array([
    0.942350209, 0.854864717, 0.764093459, 0.67789036,  0.594796062,
    0.524424136, 0.454492867, 0.390313804, 0.3288095,   0.279698402,
    0.237899065, 0.202439368, 0.172186002, 0.146007895, 0.124765158,
    0.10631869,  0.090707541, 0.07726799,  0.066082083, 0.056151293,
    0.047963966
])

# MOG_EST_BETA: DPS-COV with estimated covariance, using beta_t scaling
nmse_mog_est_beta = np.array([
    0.948509872, 0.860948324, 0.767477989, 0.679588377, 0.594359457,
    0.525992751, 0.456567824, 0.390627414, 0.32931754,  0.281405777,
    0.238511965, 0.202675998, 0.172925904, 0.146775916, 0.125355572,
    0.106843434, 0.091108084, 0.077716768, 0.066501848, 0.056575477,
    0.048371933
])

# MOG_ORACLE_SQRTBETA: DPS-COV with oracle covariance, using sqrt(beta_t) scaling
nmse_mog_oracle_sqrtbeta = np.array([
    0.801430464, 0.720343709, 0.648588121, 0.57714802, 0.512298167,
    0.451562792, 0.393757194, 0.342622906, 0.295749933, 0.254043192,
    0.217772707, 0.185272902, 0.157341585, 0.13236919, 0.111656912,
    0.094020076, 0.078858458, 0.065990075, 0.055385161, 0.046284296,
    0.038781099
])

# MOG_EST_SQRTBETA: DPS-COV with estimated covariance, using sqrt(beta_t) scaling
nmse_mog_est_sqrtbeta = np.array([
    0.805450201,
    0.727815747,
    0.651870728,
    0.580896378,
    0.514808476,
    0.453535736,
    0.395390272,
    0.34342137,
    0.297424614,
    0.255440831,
    0.218208566,
    0.186128631,
    0.157960027,
    0.132972807,
    0.112462915,
    0.094539598,
    0.079208583,
    0.066684335,
    0.055959918,
    0.046914622,
    0.039299436
])

# -----------------------------------------------------------------------------
# Exp H (GRAM + Likelihood) results (from terminal output, SNR -15..5 step 1)
# -----------------------------------------------------------------------------
# GRAM only (Exp H, likelihood off / baseline for this ablation)
nmse_gram_only_exp_h = np.array([
    7.998869e-01, 7.224673e-01, 6.470816e-01, 5.771204e-01, 5.121256e-01,
    4.515603e-01, 3.962944e-01, 3.437672e-01, 2.971370e-01, 2.554634e-01,
    2.186876e-01, 1.864926e-01, 1.583882e-01, 1.333226e-01, 1.125727e-01,
    9.487968e-02, 7.938926e-02, 6.656365e-02, 5.602067e-02, 4.675360e-02,
    3.917487e-02
], dtype=float)

# GRAM + Likelihood (oracle)
nmse_gram_like_oracle_exp_h = np.array([
    7.991148e-01, 7.217503e-01, 6.479988e-01, 5.772156e-01, 5.121858e-01,
    4.507651e-01, 3.942465e-01, 3.425176e-01, 2.951238e-01, 2.532761e-01,
    2.170824e-01, 1.842092e-01, 1.562552e-01, 1.317736e-01, 1.111409e-01,
    9.336518e-02, 7.842468e-02, 6.561569e-02, 5.512220e-02, 4.600573e-02,
    3.846021e-02
], dtype=float)

# GRAM + Likelihood (est)
nmse_gram_like_est_exp_h = np.array([
    8.046212e-01, 7.268641e-01, 6.512193e-01, 5.809162e-01, 5.137955e-01,
    4.520712e-01, 3.944145e-01, 3.428931e-01, 2.966368e-01, 2.543182e-01,
    2.176813e-01, 1.852136e-01, 1.575337e-01, 1.325100e-01, 1.118462e-01,
    9.405626e-02, 7.904492e-02, 6.623647e-02, 5.558591e-02, 4.664312e-02,
    3.911159e-02
], dtype=float)

# Likelihood only (Exp H) — from terminal output, SNR -15..5 step 1
nmse_like_only_exp_h = np.array([
    9.525881e-01, 9.248568e-01, 8.880149e-01, 8.341098e-01, 7.732138e-01,
    7.018799e-01, 6.236374e-01, 5.442631e-01, 4.674852e-01, 3.998596e-01,
    3.414079e-01, 2.907076e-01, 2.473322e-01, 2.094728e-01, 1.783003e-01,
    1.516210e-01, 1.287495e-01, 1.088099e-01, 9.236980e-02, 7.785374e-02,
    6.560829e-02
], dtype=float)



# LMMSE: Linear Minimum Mean Square Error (genie-aided, upper bound)
nmse_lmmse_genie = np.array([
    0.654871892, 0.614210594, 0.571697074, 0.528890459, 0.485627649,
    0.44325059,  0.402287903, 0.361847727, 0.323922238, 0.28793767,
    0.25426745,  0.223241458, 0.194940835, 0.169181755, 0.145924788,
    0.125577844, 0.10756921,  0.091593267, 0.077720248, 0.06574859,
    0.055502986
])

# =========================
# Figure: Main Comparison Plot
# =========================

plt.figure(figsize=(8.0, 5.5))

# Define colors for consistent styling (using more distinct colors)
color_dm = '#1f77b4'          # Blue
color_fog = '#ff7f0e'         # Orange
color_fog_cf = '#17becf'      # Cyan
color_mog_oracle = '#2ca02c'  # Green
color_mog_est = '#d62728'     # Red
color_lmmse = '#9467bd'       # Purple
color_cme = '#8c564b'         # Brown

# Line width and marker settings
linewidth = 2.0
markersize = 7
markevery = 2  # Show marker every 2 points to reduce clutter

# Plot baseline: DDIM (unconditional diffusion model)
plt.semilogy(
    snr, nmse_dm,
    marker='o', markersize=markersize, markevery=markevery,
    label='DM', color=color_dm, linewidth=linewidth
)

# Plot CME-IS (only where computed)
_mask_cme = np.isfinite(nmse_cme_is)
plt.semilogy(
    snr[_mask_cme], nmse_cme_is[_mask_cme],
    marker='P', markersize=markersize, linestyle='-',
    label='CME (empirical prior)', color=color_cme, linewidth=linewidth
)

# # Plot DPS (First-Order Guidance) with lambda=0.1
# plt.semilogy(
#     snr, nmse_fog,
#     marker='s', markersize=markersize, markevery=markevery,
#     label='FOG', color=color_fog, linewidth=linewidth
#     # label='FOG (Tweedie)', color=color_fog, linewidth=linewidth
# )

# Plot DPS (Exp E) closed-form likelihood score injection
plt.semilogy(
    snr, nmse_fog_cf_e,
    marker='D', markersize=markersize, markevery=markevery,
    label='Likelihood (closed-form)', color=color_fog_cf, linewidth=linewidth
)

# # Plot DPS (Exp E) closed-form likelihood score injection (no SNR matching, full T)
# plt.semilogy(
#     snr, nmse_fog_cf_e_fullT,
#     marker='D', markersize=markersize, markevery=markevery,
#     linestyle='--',
#     label='FOG (closed-form, full T)', color=color_fog_cf, linewidth=linewidth, alpha=0.9
# )

# # Plot DPS-COV with oracle covariance, using beta_t scaling
# plt.semilogy(
#     snr, nmse_mog_oracle_beta,
#     marker='s', linestyle='--', markersize=markersize, markevery=markevery,
#     label='MOG (oracle, $\\beta_t$)', color=color_mog_oracle, linewidth=linewidth
# )

# # Plot DPS-COV with estimated covariance, using beta_t scaling
# plt.semilogy(
#     snr, nmse_mog_est_beta,
#     marker='^', linestyle='--', markersize=markersize, markevery=markevery,
#     label='MOG (est, $\\beta_t$)', color=color_mog_est, linewidth=linewidth
# )

# Plot Exp H (GRAM + Likelihood) curves
plt.semilogy(
    snr, nmse_gram_only_exp_h,
    marker='s', linestyle='-', markersize=markersize, markevery=markevery,
    label='GRAM only', color=color_fog, linewidth=linewidth
)

plt.semilogy(
    snr, nmse_gram_like_oracle_exp_h,
    marker='^', linestyle='-', markersize=markersize, markevery=markevery,
    label='GRAM + Likelihood (oracle)', color=color_mog_oracle, linewidth=linewidth
)

plt.semilogy(
    snr, nmse_gram_like_est_exp_h,
    marker='d', linestyle='-', markersize=markersize, markevery=markevery,
    label='GRAM + Likelihood (est)', color=color_mog_est, linewidth=linewidth
)

# Plot Likelihood only (Exp H)
plt.semilogy(
    snr, nmse_like_only_exp_h,
    marker='o', linestyle='--', markersize=markersize, markevery=markevery,
    label='Likelihood (Tweedie)', color=color_fog_cf, linewidth=linewidth, alpha=0.95
)

# Plot LMMSE (genie-aided, upper bound)
plt.semilogy(
    snr, nmse_lmmse_genie,
    marker='x', markersize=markersize+1, markevery=markevery,
    label='LMMSE', color=color_lmmse, linewidth=linewidth
)

# # Add annotation for line style meaning (with better styling)
# plt.text(
#     0.02, 0.02,
#     r'Solid: $\sqrt{\beta_t}$   Dashed: $\beta_t$',
#     transform=plt.gca().transAxes,
#     fontsize=11,
#     verticalalignment='bottom',
#     bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8, edgecolor='gray', linewidth=1)
# )

# Formatting
plt.xlim([-15, 5])
plt.ylim([4e-2, 1])
plt.xticks(np.arange(-15, 6, 1))
plt.xlabel('SNR [dB]', fontsize=12, fontweight='bold')
plt.ylabel('NMSE', fontsize=12, fontweight='bold')
# plt.title('3GPP NLOS Channel: Performance Comparison', fontsize=13, fontweight='bold', pad=15)
plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.6)
plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.4)

# Improve legend styling
legend = plt.legend(loc='upper right', fontsize=10, framealpha=0.95, 
                    fancybox=True, shadow=True, edgecolor='gray', frameon=True)
legend.get_frame().set_linewidth(1.2)

# Improve tick labels
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tick_params(axis='both', which='minor', labelsize=8)

plt.tight_layout()

# Save the figure instead of showing (for WSL/headless environments)
output_path = output_dir / 'plot_3GPP_NLOS.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Plot saved to '{output_path}'")

# Try to show if display is available
try:
    plt.show()
except Exception as e:
    print(f"Could not display plot: {e}")
    print(f"Plot has been saved to '{output_path}' instead")

