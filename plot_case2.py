"""
Plot Case 2: Quadriga LOS Channel Estimation Results

This script visualizes DPS and DPS-COV performance for Quadriga LOS channel.
It includes:
1. Main comparison plot (best configurations)
2. Relative improvement analysis (optional)

Channel Type: Quadriga LOS
Evaluation Methods: DDIM, DPS, DPS-COV (oracle), DPS-COV (estimated)
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
# DDIM (baseline - unconditional diffusion model)
# =========================
nmse_ddim = np.array([
    0.345148116, 0.28891772, 0.251942813, 0.224093035, 0.200874627,
    0.177105069, 0.162511826, 0.150072262, 0.142880231, 0.133797809,
    0.126413807, 0.119556457, 0.112782106, 0.106160231, 0.098546132,
    0.090614289, 0.082667232, 0.074467674, 0.067861624, 0.059568506,
    0.053351436
])

# =========================
# DPS: Diffusion Posterior Sampling with different lambda values
# =========================
# lambda: DPS likelihood guidance strength
dps = {
    0.1: np.array([
        0.319575399, 0.271066517, 0.237106591, 0.208670065, 0.187938437,
        0.16949001,  0.155688986, 0.145193726, 0.135046825, 0.128196225,
        0.121868066, 0.11606399,  0.110192604, 0.102007106, 0.095205598,
        0.087896116, 0.080318801, 0.072397031, 0.066530727, 0.058640901,
        0.052806117
    ]),
    0.2: np.array([
        0.311842859, 0.266778231, 0.232819363, 0.204627231, 0.183313832,
        0.168201834, 0.155146509, 0.145362616, 0.133696377, 0.127636284,
        0.121558607, 0.115514696, 0.109256737, 0.099916458, 0.09262275,
        0.085301876, 0.077634946, 0.069721848, 0.064148828, 0.056255501,
        0.050650369
    ]),
    0.3: np.array([
        0.303865641, 0.264427602, 0.228160739, 0.201456249, 0.181475192,
        0.171183541, 0.158067539, 0.148257226, 0.135497928, 0.129019246,
        0.122796707, 0.116692603, 0.109641783, 0.099133126, 0.091602869,
        0.083800338, 0.075959347, 0.068045378, 0.062228151, 0.054583624,
        0.04898889
    ]),
    0.5: np.array([
        0.300170869, 0.264399409, 0.232493117, 0.207948685, 0.188121408,
        0.183985814, 0.170905545, 0.161193997, 0.143730924, 0.136753589,
        0.129617706, 0.121541746, 0.113486014, 0.10050372,  0.09226077,
        0.083568588, 0.07536038,  0.067234635, 0.060855683, 0.053447388,
        0.047466796
    ]),
    0.7: np.array([
        0.305702925, 0.277131349, 0.244039893, 0.221183032, 0.202636614,
        0.20312272,  0.190254837, 0.179946721, 0.157568991, 0.14944835,
        0.141384467, 0.132665217, 0.122876674, 0.109244511, 0.099773929,
        0.089967042, 0.080610767, 0.071765013, 0.063585617, 0.056258123,
        0.049517862
    ]),
}

# =========================
# DPS-COV: DPS with covariance guidance
# =========================
# cov_lambda: Covariance guidance strength
# Oracle: Uses perfect covariance matrix H H^H
dps_cov_oracle = {
    1e-2: np.array([
        0.810444653, 0.743544698, 0.672205567, 0.605228782, 0.53387183,
        0.469858706, 0.395752937, 0.331485242, 0.258817077, 0.221941456,
        0.186213776, 0.154335901, 0.129488483, 0.103662513, 0.087395661,
        0.074221827, 0.064255051, 0.056054432, 0.050471734, 0.044316433,
        0.03969137
    ]),
    2e-3: np.array([
        0.379974782, 0.334146082, 0.300218374, 0.262522638, 0.228146881,
        0.19510293,  0.16832222,  0.146423623, 0.128684506, 0.114909977,
        0.104526915, 0.095776021, 0.088663489, 0.081587106, 0.075522706,
        0.069401778, 0.063615724, 0.05798123,  0.052884944, 0.04750894,
        0.042741969
    ]),
    1e-3: np.array([
        0.281218857, 0.239462942, 0.206281051, 0.180644929, 0.159108654,
        0.143177912, 0.130674914, 0.120734617, 0.113157175, 0.106999032,
        0.101375967, 0.09594918,  0.090497732, 0.08405745,  0.078351848,
        0.072356619, 0.066454366, 0.060431574, 0.055089839, 0.049384758,
        0.04430702
    ]),
    5e-4: np.array([
        0.221073493, 0.197115958, 0.177698806, 0.16211085,  0.149443194,
        0.138956174, 0.131949663, 0.125149399, 0.1190634,   0.113820717,
        0.108492777, 0.103040271, 0.097014181, 0.090125822, 0.083403312,
        0.076788463, 0.070107885, 0.063368112, 0.057537865, 0.051079109,
        0.045706157
    ]),
    1e-4: np.array([
        0.329013348, 0.30106169,  0.271612555, 0.24815774,  0.226134375,
        0.212751195, 0.195733503, 0.180707976, 0.161953002, 0.150016755,
        0.138916388, 0.127744764, 0.116882242, 0.102927297, 0.093292423,
        0.083953656, 0.075232297, 0.066932902, 0.060403984, 0.053095128,
        0.047143504
    ]),
}

# Estimated: Uses time-averaged covariance estimate (best cov_lambda)
dps_cov_est_best = np.array([
    0.222036079, 0.198080599, 0.178040326, 0.161461473, 0.1493081,
    0.140089631, 0.131577834, 0.125402406, 0.118960463, 0.11412435,
    0.108456858, 0.102816306, 0.0971637,   0.089909866, 0.083364032,
    0.076852106, 0.070116624, 0.063395813, 0.05752657,  0.051180437,
    0.04566792
])

dps_cov_oracle_sqrtbeta = np.array([
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
cbd_oracle = np.array([
    0.219811067,
    0.19705236,
    0.179879472,
    0.165927768,
    0.153186992,
    0.143654242,
    0.135986388,
    0.129242808,
    0.123257726,
    0.118758313,
    0.114668503,
    0.11050991,
    0.105952352,
    0.099880569,
    0.094165929,
    0.08781185,
    0.080954589,
    0.073548339,
    0.06811215,
    0.060045335,
    0.054383531
])

# =========================
# Figure 1: Main Comparison Plot (Best Configurations)
# =========================
# Best hyperparameters found through grid search
best_lambda = 0.3        # Best DPS lambda
best_cov_lambda = 5e-4  # Best DPS-COV cov_lambda

plt.figure(figsize=(7.6, 5.0))

# Plot baseline: DDIM (unconditional diffusion model)
plt.semilogy(snr, nmse_ddim, marker='o', label='DM')

# Plot DPS with best lambda
#(lambda0.3)
plt.semilogy(
    snr, dps[best_lambda],
    marker='s',
    label=rf'DPS'
)

# Plot DPS-COV with oracle covariance (perfect H H^H)
#(lambda0.3, cov_lambda0.0005)
plt.semilogy(
    snr, dps_cov_oracle[best_cov_lambda],
    marker='^',
    label=rf'DPS-COV ($\lambda=0.3$, $\lambda_{{cov}}=0.0005$)'
)

# Plot DPS-COV with estimated covariance (time-averaged)
# plt.semilogy(
#     snr, dps_cov_est_best,
#     marker='v', linestyle='--',
#     label=rf'DPS-COV (est, $\lambda={best_lambda}$, $\lambda_{{cov}}={best_cov_lambda:g}$)'
# )

# Plot CBD with oracle covariance (perfect H H^H)
plt.semilogy(
    snr, cbd_oracle,
    marker='x',
    label=rf'DPS-COV ($\lambda=0$, $\lambda_{{cov}}=0.0005$)'
)

# Plot CBD with estimated covariance (time-averaged)
plt.semilogy(
    snr, dps_cov_oracle_sqrtbeta,
    marker='v', linestyle='--',
    label=rf'DPS-COV (sqrt_beta_t)'
)

# Formatting
plt.xlim([-15, 5])
plt.ylim([3.5e-2, 0.5])
plt.xticks(np.arange(-15, 6, 1))
plt.xlabel('SNR [dB]')
plt.ylabel('NMSE')
plt.title('Quadriga LOS Channel: DPS Performance Comparison')
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save the figure instead of showing (for WSL/headless environments)
output_path = output_dir / 'plot_case2_main.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Main plot saved to '{output_path}'")

# Try to show if display is available
try:
    plt.show()
except Exception as e:
    print(f"Could not display plot: {e}")
    print(f"Plot has been saved to '{output_path}' instead")

# =========================
# Figure 2: DPS Lambda Sweep (Optional - Uncomment to plot)
# =========================
# Uncomment the following block to visualize DPS performance across different lambda values
# plt.figure(figsize=(7.6, 5.0))
# plt.semilogy(snr, nmse_ddim, marker='o', label='DDIM (baseline)')
# 
# for lbd, arr in sorted(dps.items()):
#     if lbd == best_lambda:
#         plt.semilogy(snr, arr, marker='s', linewidth=2.2, label=f'DPS (λ={lbd})')
#     else:
#         plt.semilogy(snr, arr, marker='.', label=f'DPS (λ={lbd})')
# 
# plt.xlim([-15, 5])
# plt.ylim([3.5e-2, 5e-1])
# plt.xticks(np.arange(-15, 6, 1))
# plt.xlabel('SNR [dB]')
# plt.ylabel('NMSE')
# plt.title('Quadriga LOS: DPS Lambda Sweep')
# plt.grid(True, which='both', linestyle=':', linewidth=0.5)
# plt.legend()
# plt.tight_layout()
# output_path = output_dir / 'plot_case2_dps_sweep.png'
# plt.savefig(output_path, dpi=150, bbox_inches='tight')
# plt.show()

# =========================
# Figure 3: DPS-COV Cov_Lambda Sweep (Optional - Uncomment to plot)
# =========================
# Uncomment the following block to visualize DPS-COV performance across different cov_lambda values
# plt.figure(figsize=(7.6, 5.0))
# plt.semilogy(snr, dps[best_lambda], marker='s', label=f'DPS (λ={best_lambda})')
# 
# for cov_lbd, arr in sorted(dps_cov_oracle.items()):
#     if cov_lbd == best_cov_lambda:
#         plt.semilogy(snr, arr, marker='^', linewidth=2.2, label=f'DPS-COV (λ_cov={cov_lbd:g})')
#     else:
#         plt.semilogy(snr, arr, marker='.', label=f'DPS-COV (λ_cov={cov_lbd:g})')
# 
# plt.xlim([-15, 5])
# plt.ylim([3.5e-2, 1])
# plt.xticks(np.arange(-15, 6, 1))
# plt.xlabel('SNR [dB]')
# plt.ylabel('NMSE')
# plt.title('Quadriga LOS: DPS-COV Cov_Lambda Sweep (Oracle)')
# plt.grid(True, which='both', linestyle=':', linewidth=0.5)
# plt.legend()
# plt.tight_layout()
# output_path = output_dir / 'plot_case2_cov_sweep.png'
# plt.savefig(output_path, dpi=150, bbox_inches='tight')
# plt.show()

# =========================
# Figure 4: Relative Improvement Analysis
# =========================
# Computes relative improvement: (NMSE_DDIM - NMSE_method) / NMSE_DDIM
# Positive values indicate improvement over DDIM baseline

rel_imp_dps = (nmse_ddim - dps[best_lambda]) / nmse_ddim
rel_imp_cov_oracle = (nmse_ddim - dps_cov_oracle[best_cov_lambda]) / nmse_ddim
rel_imp_cov_est = (nmse_ddim - dps_cov_est_best) / nmse_ddim

plt.figure(figsize=(7.6, 5.0))
plt.plot(snr, rel_imp_dps, marker='o', label=f'DDIM → DPS (λ={best_lambda})')
plt.plot(snr, rel_imp_cov_oracle, marker='^', 
         label=f'DDIM → DPS-COV (oracle, λ_cov={best_cov_lambda:g})')
plt.plot(snr, rel_imp_cov_est, marker='v', linestyle='--', 
         label=f'DDIM → DPS-COV (est, λ_cov={best_cov_lambda:g})')

plt.xlim([-15, 5])
plt.xticks(np.arange(-15, 6, 1))
plt.ylim([0.0, 0.45])
plt.xlabel('SNR [dB]')
plt.ylabel('Relative Improvement over DDIM')
plt.title('Quadriga LOS: Relative Performance Improvement')
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save the figure
output_path = output_dir / 'plot_case2_improvement.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Improvement plot saved to '{output_path}'")

# Try to show if display is available
try:
    plt.show()
except Exception as e:
    print(f"Could not display plot: {e}")
    print(f"Plot has been saved to '{output_path}' instead")
