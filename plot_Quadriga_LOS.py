"""
Plot Case 2: Quadriga LOS Channel Estimation Results

This script visualizes DPS and DPS-COV performance for Quadriga LOS channel.

Methods compared:
- DM: DDIM (unconditional diffusion model, baseline)
- FOG: DPS (First-Order Guidance, lambda=0.3)
- MOG (oracle, beta_t): DPS-COV with oracle covariance, beta_t scaling
- MOG (est, beta_t): DPS-COV with estimated covariance, beta_t scaling
- MOG (oracle, sqrt(beta_t)): DPS-COV with oracle covariance, sqrt(beta_t) scaling
- MOG (est, sqrt(beta_t)): DPS-COV with estimated covariance, sqrt(beta_t) scaling

Channel Type: Quadriga LOS
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
snr = np.arange(-15, 6, 1, dtype=float)  # SNR range: -15 to 5 dB, step 1 dB


# =========================
# Data Definitions
# =========================
# DM: DDIM (baseline - unconditional diffusion model)
nmse_dm = np.array([
    0.345148116, 0.28891772, 0.251942813, 0.224093035, 0.200874627,
    0.177105069, 0.162511826, 0.150072262, 0.142880231, 0.133797809,
    0.126413807, 0.119556457, 0.112782106, 0.106160231, 0.098546132,
    0.090614289, 0.082667232, 0.074467674, 0.067861624, 0.059568506,
    0.053351436
])

# FOG: DPS (Diffusion Posterior Sampling) with lambda=0.3
nmse_fog = np.array([
    0.303865641, 0.264427602, 0.228160739, 0.201456249, 0.181475192,
    0.171183541, 0.158067539, 0.148257226, 0.135497928, 0.129019246,
    0.122796707, 0.116692603, 0.109641783, 0.099133126, 0.091602869,
    0.083800338, 0.075959347, 0.068045378, 0.062228151, 0.054583624,
    0.04898889
])

# MOG_ORACLE_BETA: DPS-COV with oracle covariance, using beta_t scaling
nmse_mog_oracle_beta = np.array([
    0.221073493, 0.197115958, 0.177698806, 0.16211085,  0.149443194,
    0.138956174, 0.131949663, 0.125149399, 0.1190634,   0.113820717,
    0.108492777, 0.103040271, 0.097014181, 0.090125822, 0.083403312,
    0.076788463, 0.070107885, 0.063368112, 0.057537865, 0.051079109,
    0.045706157
])

# MOG_EST_BETA: DPS-COV with estimated covariance, using beta_t scaling
nmse_mog_est_beta = np.array([
    0.222036079, 0.198080599, 0.178040326, 0.161461473, 0.1493081,
    0.140089631, 0.131577834, 0.125402406, 0.118960463, 0.11412435,
    0.108456858, 0.102816306, 0.0971637,   0.089909866, 0.083364032,
    0.076852106, 0.070116624, 0.063395813, 0.05752657,  0.051180437,
    0.04566792
])

# MOG_ORACLE_SQRTBETA: DPS-COV with oracle covariance, using sqrt(beta_t) scaling
nmse_mog_oracle_sqrtbeta = np.array([
    0.179403812,
    0.156103522,
    0.139285162,
    0.128767744,
    0.120300412,
    0.114184693,
    0.109616555,
    0.105123289,
    0.101252347,
    0.097332999,
    0.093530826,
    0.08949618,
    0.085361965,
    0.079972036,
    0.074803598,
    0.069313399,
    0.063627906,
    0.057896528,
    0.05318734,
    0.047362495,
    0.042795561
])
# SOG: DPS-COV with oracle covariance, using sqrt(beta_t) scaling (alternative)
# Gram only 
mean_gram = np.array([
    1.748635e-01, 1.538816e-01, 1.387912e-01, 1.285072e-01, 1.205567e-01,
    1.153305e-01, 1.111535e-01, 1.070588e-01, 1.037800e-01, 1.005809e-01,
    9.726343e-02, 9.411472e-02, 9.060667e-02, 8.586457e-02, 8.137912e-02,
    7.657793e-02, 7.111976e-02, 6.517617e-02, 6.066789e-02, 5.418116e-02,
    4.941193e-02
])

# # Gram + naive likelihood
# mean_naive = np.array([
#     1.810029e-01, 1.575632e-01, 1.403509e-01, 1.293232e-01, 1.202357e-01,
#     1.139285e-01, 1.086584e-01, 1.044182e-01, 9.993476e-02, 9.589358e-02,
#     9.166604e-02, 8.727502e-02, 8.239515e-02, 7.671840e-02, 7.097632e-02,
#     6.502634e-02, 5.925589e-02, 5.369763e-02, 4.877206e-02, 4.397259e-02,
#     3.956686e-02
# ])

# Gram + gated likelihood
mean_gated = np.array([
    1.764893e-01, 1.529106e-01, 1.383094e-01, 1.280172e-01, 1.203346e-01,
    1.143455e-01, 1.093437e-01, 1.048839e-01, 1.006874e-01, 9.638820e-02,
    9.208596e-02, 8.760896e-02, 8.263041e-02, 7.669161e-02, 7.099428e-02,
    6.529442e-02, 5.927842e-02, 5.366533e-02, 4.882218e-02, 4.388248e-02,
    3.951930e-02
])

# Gram + gated likelihood (estimated / new results)
mean_gated_est = np.array([
    1.759127e-01, 1.538754e-01, 1.392316e-01, 1.280777e-01, 1.205633e-01,
    1.145857e-01, 1.096064e-01, 1.048951e-01, 1.004672e-01, 9.649553e-02,
    9.214246e-02, 8.755556e-02, 8.261013e-02, 7.682554e-02, 7.120566e-02,
    6.516127e-02, 5.936537e-02, 5.382114e-02, 4.885961e-02, 4.394368e-02,
    3.962712e-02
])

# Likelihood only (Exp H) — from terminal output, SNR -15..5 step 1
nmse_like_only_exp_h = np.array([
    3.120825e-01, 2.637613e-01, 2.285920e-01, 2.030980e-01, 1.810158e-01,
    1.627318e-01, 1.495765e-01, 1.386342e-01, 1.309456e-01, 1.229533e-01,
    1.156248e-01, 1.082929e-01, 1.007664e-01, 9.259009e-02, 8.497594e-02,
    7.772791e-02, 7.067811e-02, 6.403799e-02, 5.759114e-02, 5.179929e-02,
    4.622400e-02
])

# MOG_EST_SQRTBETA: DPS-COV with estimated covariance, using sqrt(beta_t) scaling
nmse_mog_est_sqrtbeta = np.array([
    0.17759876,
    0.155722782,
    0.139739186,
    0.128703699,
    0.120565183,
    0.114494249,
    0.109137014,
    0.105283365,
    0.101005979,
    0.097337,
    0.093609042,
    0.08976689,
    0.08526545,
    0.080025688,
    0.074939184,
    0.069546103,
    0.063844211,
    0.058028001,
    0.053305384,
    0.047430992,
    0.042837776
])

# =========================
# Figure 1: Main Comparison Plot (Best Configurations)
# =========================

plt.figure(figsize=(8.0, 5.5))

# Define colors for consistent styling (using more distinct colors)
color_dm = '#1f77b4'          # Blue
color_fog = '#ff7f0e'         # Orange
color_fog_cf = '#17becf'      # Cyan (match plot_3GPP_NLOS.py "Likelihood only")
color_mog_oracle = '#2ca02c'  # Green
color_mog_est = '#d62728'     # Red

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

# Plot DPS (SOG: Second-Order Guidance)
plt.semilogy(
    snr, mean_gram,
    marker='s', markersize=markersize, markevery=markevery,
    label='GRAM only', color=color_fog, linewidth=linewidth
)

# # Plot DPS-COV with oracle covariance, using beta_t scaling
# plt.semilogy(
#     snr, nmse_mog_oracle_beta,
#     marker='s', linestyle='--', markersize=markersize, markevery=markevery,
#     color=color_mog_oracle, linewidth=linewidth,
#     label='MOG (oracle, $\\beta_t$)'
# )

# # Plot DPS-COV with estimated covariance, using beta_t scaling
# plt.semilogy(
#     snr, nmse_mog_est_beta,
#     marker='^', linestyle='--', markersize=markersize, markevery=markevery,
#     color=color_mog_est, linewidth=linewidth,
#     label='MOG (est, $\\beta_t$)'
# )

# Plot DPS-COV with oracle covariance, using sqrt(beta_t) scaling
plt.semilogy(
    snr, mean_gated,
    marker='^', linestyle='-', markersize=markersize, markevery=markevery,
    color=color_mog_oracle, linewidth=linewidth,
    label='GRAM + Likelihood (oracle)'
)

plt.semilogy(
    snr, mean_gated_est,
    marker='d', linestyle='-', markersize=markersize, markevery=markevery,
    color=color_mog_est, linewidth=linewidth,
    label='GRAM + Likelihood (est)'
)

# Likelihood only (Exp H)
plt.semilogy(
    snr, nmse_like_only_exp_h,
    marker='o', linestyle='--', markersize=markersize, markevery=markevery,
    color=color_fog_cf, linewidth=linewidth, alpha=0.95,
    label='Likelihood only'
)

# # Plot DPS-COV with estimated covariance, using sqrt(beta_t) scaling
# plt.semilogy(
#     snr, mean_gated_est,
#     marker='d', linestyle='-', markersize=markersize, markevery=markevery,
#     color=color_mog_est, linewidth=linewidth,
#     label='GRAM + Likelihood (est)'
# )

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
plt.ylim([3.5e-2, 0.5])
plt.xticks(np.arange(-15, 6, 1))
plt.xlabel('SNR [dB]', fontsize=12, fontweight='bold')
plt.ylabel('NMSE', fontsize=12, fontweight='bold')
# plt.title('Quadriga LOS Channel: Performance Comparison', fontsize=13, fontweight='bold', pad=15)
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
output_path = output_dir / 'plot_Quadriga_LOS.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Main plot saved to '{output_path}'")

# Try to show if display is available
try:
    plt.show()
except Exception as e:
    print(f"Could not display plot: {e}")
    print(f"Plot has been saved to '{output_path}' instead")

