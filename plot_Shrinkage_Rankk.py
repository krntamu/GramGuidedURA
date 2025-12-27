"""
Shrinkage and Rank-k Comparison Plot

This script visualizes the performance comparison of different post-processing methods
for covariance estimation with different numbers of time samples (Nd).

Methods compared:
- Nd = 300 raw: Baseline with 300 time samples
- Nd = 220 raw: Baseline with 220 time samples
- Nd = 220 full: Shrinkage + rank-k (both methods applied)
- Nd = 220 gated: Hybrid approach (shrinkage ON for SNR <= -10, rank-k only for SNR > -10)

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
from matplotlib.ticker import AutoMinorLocator

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
# Nd = 220: Raw (no post-processing)
mean_220_raw = np.array([
    3.747334e-01, 3.194034e-01, 2.761921e-01, 2.409284e-01, 2.137017e-01,
    1.907941e-01, 1.737239e-01, 1.603506e-01, 1.480202e-01, 1.396968e-01,
    1.309815e-01, 1.261423e-01, 1.214035e-01, 1.171525e-01, 1.155838e-01,
    1.117680e-01, 1.101513e-01, 1.097575e-01, 1.075924e-01, 1.059124e-01,
    1.059387e-01,
])

# Nd = 220: Shrinkage only
mean_220_shrink = np.array([
    3.226444e-01, 2.822485e-01, 2.489000e-01, 2.220586e-01, 1.999786e-01,
    1.817359e-01, 1.674183e-01, 1.570658e-01, 1.474600e-01, 1.409518e-01,
    1.342418e-01, 1.288488e-01, 1.253967e-01, 1.210628e-01, 1.208661e-01,
    1.161332e-01, 1.141773e-01, 1.146029e-01, 1.118255e-01, 1.120767e-01,
    1.115572e-01,
])

# Nd = 220: Rank-k only
mean_220_rankk = np.array([
    3.278779e-01, 2.853085e-01, 2.507288e-01, 2.230458e-01, 2.005815e-01,
    1.817338e-01, 1.670303e-01, 1.557518e-01, 1.451300e-01, 1.378653e-01,
    1.298759e-01, 1.252211e-01, 1.212985e-01, 1.167020e-01, 1.152893e-01,
    1.118217e-01, 1.099334e-01, 1.097645e-01, 1.073738e-01, 1.058730e-01,
    1.060072e-01,
])

# Nd = 220: Shrinkage + rank-k (both methods applied)
mean_220_both = np.array([
    3.167824e-01, 2.767370e-01, 2.453972e-01, 2.196880e-01, 1.988525e-01,
    1.813820e-01, 1.676099e-01, 1.583559e-01, 1.486650e-01, 1.426815e-01,
    1.363771e-01, 1.312425e-01, 1.275987e-01, 1.238022e-01, 1.239695e-01,
    1.190377e-01, 1.170286e-01, 1.176712e-01, 1.148596e-01, 1.155845e-01,
    1.148775e-01,
])

# Nd = 300: Raw (no post-processing)
mean_300_raw = np.array([
    3.207073e-01, 2.736122e-01, 2.360233e-01, 2.071134e-01, 1.830738e-01,
    1.638661e-01, 1.490375e-01, 1.362237e-01, 1.265290e-01, 1.194821e-01,
    1.130204e-01, 1.081138e-01, 1.043980e-01, 1.001950e-01, 9.801685e-02,
    9.650956e-02, 9.439316e-02, 9.363927e-02, 9.170141e-02, 9.139542e-02,
    9.156374e-02,
])

std_300_raw = np.array([
    1.075816e-02, 1.044882e-02, 1.134666e-02, 1.283100e-02, 1.303898e-02,
    1.401721e-02, 1.483250e-02, 1.624969e-02, 1.624667e-02, 1.726945e-02,
    1.827748e-02, 1.845139e-02, 1.903628e-02, 2.035533e-02, 1.980732e-02,
    2.086036e-02, 2.056813e-02, 2.027438e-02, 2.000103e-02, 2.090345e-02,
    2.202424e-02,
])

# =========================
# Hybrid Rule
# =========================
# Hybrid approach: Use both shrinkage and rank-k for SNR <= -10, rank-k only for SNR > -10
switch_snr = -10
use_both = snr <= switch_snr
mean_220_hybrid = np.where(use_both, mean_220_both, mean_220_rankk)

def shrink_gate(snr_db: np.ndarray, lo=-15, hi=-10) -> np.ndarray:
    """
    Create a gate for shrinkage activation based on SNR range.
    
    Args:
        snr_db: SNR values in dB
        lo: Lower bound for shrinkage activation (default: -15)
        hi: Upper bound for shrinkage activation (default: -10)
    
    Returns:
        Boolean array indicating where shrinkage should be active
    """
    return (snr_db >= lo) & (snr_db <= hi)

gate = shrink_gate(snr, lo=-15, hi=-10)
mean_220_gated = np.where(gate, mean_220_both, mean_220_rankk)

# =========================
# Figure: Shrinkage and Rank-k Comparison
# =========================

plt.figure(figsize=(8.0, 5.5))

ax = plt.gca()

# Define colors for consistent styling
color_300_raw = '#1f77b4'      # Blue
color_220_raw = '#ff7f0e'      # Orange
color_220_both = '#2ca02c'     # Green
color_220_gated = '#d62728'    # Red

# Line width and marker settings
linewidth = 2.0
linewidth_gated = 3.4
markersize = 7
markevery = 2  # Show marker every 2 points to reduce clutter

# Add shaded region for low SNR (where shrinkage is active)
ax.axvspan(-15, -10, alpha=0.1, color='gray', zorder=0, label='_nolegend_')

# Add text annotation for shrinkage region (centered in the -15 to -10 region)
# Position: -12.5 is the middle of [-15, -10], convert to axis coordinates
# x-axis range: -15 to 5 (20 units), -12.5 is at position 2.5/20 = 0.125 from left
shrinkage_center_x = (-12.5 - (-15)) / (5 - (-15))  # Convert to [0, 1] range
ax.text(
    shrinkage_center_x, 0.95,
    "Shrinkage ON",
    transform=ax.transAxes,
    ha="center", va="top", fontsize=11,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8, edgecolor='gray', linewidth=1)
)

# Plot Nd = 300 raw (baseline with more samples)
ax.plot(
    snr, mean_300_raw,
    marker="o", markersize=markersize, markevery=markevery,
    linewidth=linewidth, label="Nd = 300 raw",
    color=color_300_raw, zorder=2
)

# Plot Nd = 220 raw (baseline with fewer samples)
ax.plot(
    snr, mean_220_raw,
    linestyle=":", linewidth=linewidth+0.2, alpha=0.85,
    marker="^", markersize=markersize-1, markevery=markevery,
    markerfacecolor="white", markeredgewidth=1.5, markeredgecolor=color_220_raw,
    label="Nd = 220 raw",
    color=color_220_raw, zorder=10
)

# Plot Nd = 220 full (shrinkage + rank-k, both methods)
ax.plot(
    snr, mean_220_both,
    linestyle="--", linewidth=linewidth, alpha=0.75,
    marker="D", markersize=markersize-2, markevery=markevery,
    label="Nd = 220 full",
    color=color_220_both, zorder=4
)

# Plot Nd = 220 gated (hybrid approach: shrinkage ON for SNR <= -10)
ax.plot(
    snr, mean_220_gated,
    linestyle="-", linewidth=linewidth_gated, alpha=0.95,
    marker="s", markersize=markersize-0.5, markevery=markevery,
    label="Nd = 220 gated",
    color=color_220_gated, zorder=6
)

# Formatting
ax.set_xlim([-15, 5])
ax.set_xticks(snr)
ax.set_xlabel("SNR [dB]", fontsize=12, fontweight='bold')
ax.set_ylabel("Mean relative error", fontsize=12, fontweight='bold')
# ax.set_title("Shrinkage and Rank-k Post-processing Comparison", fontsize=13, fontweight='bold', pad=15)

# Grid styling
ax.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.6)
ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.4)

# Improve legend styling
legend = ax.legend(
    loc="upper right",
    fontsize=10,
    frameon=True,
    framealpha=0.95,
    fancybox=True,
    shadow=True,
    edgecolor='gray'
)
legend.get_frame().set_linewidth(1.2)

# Improve tick labels
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)

plt.tight_layout()

# Save the figure instead of showing (for WSL/headless environments)
output_path = output_dir / 'plot_Shrinkage_Rankk.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Plot saved to '{output_path}'")

# Try to show if display is available
try:
    plt.show()
except Exception as e:
    print(f"Could not display plot: {e}")
    print(f"Plot has been saved to '{output_path}' instead")

