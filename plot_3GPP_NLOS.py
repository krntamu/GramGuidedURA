"""
Plot: 3GPP NLOS Channel Estimation Results

This script visualizes DPS and DPS-COV performance for 3GPP NLOS channel.

Methods compared:
- DM: DDIM (unconditional diffusion model, baseline)
- DM + First-Order Guidance: DPS (lambda=0.1)
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

# FOG: DPS (First-Order Guidance) with lambda=0.1
nmse_fog = np.array([
    0.953405201, 0.923455656, 0.883167565, 0.831169307, 0.767774642,
    0.697487533, 0.620056272, 0.542053878, 0.466282755, 0.400504202,
    0.343076289, 0.292655855, 0.250314415, 0.211691454, 0.180478558,
    0.153672323, 0.13030912,  0.11008589,  0.094051592, 0.078737386,
    0.066486768
])

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
    0.801430464,
    0.720343709,
    0.648588121,
    0.57714802,
    0.512298167,
    0.451562792,
    0.393757194,
    0.342622906,
    0.295749933,
    0.254043192,
    0.217772707,
    0.185272902,
    0.157341585,
    0.13236919,
    0.111656912,
    0.094020076,
    0.078858458,
    0.065990075,
    0.055385161,
    0.046284296,
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
color_mog_oracle = '#2ca02c'  # Green
color_mog_est = '#d62728'     # Red
color_lmmse = '#9467bd'       # Purple

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

# Plot DPS (First-Order Guidance) with lambda=0.1
plt.semilogy(
    snr, nmse_fog,
    marker='s', markersize=markersize, markevery=markevery,
    label='FOG', color=color_fog, linewidth=linewidth
)

# Plot DPS-COV with oracle covariance, using beta_t scaling
plt.semilogy(
    snr, nmse_mog_oracle_beta,
    marker='s', linestyle='--', markersize=markersize, markevery=markevery,
    label='MOG (oracle, $\\beta_t$)', color=color_mog_oracle, linewidth=linewidth
)

# Plot DPS-COV with estimated covariance, using beta_t scaling
plt.semilogy(
    snr, nmse_mog_est_beta,
    marker='^', linestyle='--', markersize=markersize, markevery=markevery,
    label='MOG (est, $\\beta_t$)', color=color_mog_est, linewidth=linewidth
)

# Plot DPS-COV with oracle covariance, using sqrt(beta_t) scaling
plt.semilogy(
    snr, nmse_mog_oracle_sqrtbeta,
    marker='^', linestyle='-', markersize=markersize, markevery=markevery,
    label='MOG (oracle, $\\sqrt{\\beta_t}$)', color=color_mog_oracle, linewidth=linewidth
)

# Plot DPS-COV with estimated covariance, using sqrt(beta_t) scaling
plt.semilogy(
    snr, nmse_mog_est_sqrtbeta,
    marker='v', linestyle='-', markersize=markersize, markevery=markevery,
    label='MOG (est, $\\sqrt{\\beta_t}$)', color=color_mog_est, linewidth=linewidth
)

# Plot LMMSE (genie-aided, upper bound)
plt.semilogy(
    snr, nmse_lmmse_genie,
    marker='x', markersize=markersize+1, markevery=markevery,
    label='LMMSE', color=color_lmmse, linewidth=linewidth
)

# Add annotation for line style meaning (with better styling)
plt.text(
    0.02, 0.02,
    r'Solid: $\sqrt{\beta_t}$   Dashed: $\beta_t$',
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment='bottom',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8, edgecolor='gray', linewidth=1)
)

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

