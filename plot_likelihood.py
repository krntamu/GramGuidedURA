import numpy as np
import matplotlib
import os
from pathlib import Path

# Headless support
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# =========================
# Output directory
# =========================
# Keep consistent with other plotting scripts
output_dir = Path('results') / 'useful_results'
output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# SNR grid
# =========================
snr = np.arange(-15, 6, 1, dtype=float)

# =========================
# Data (replace if needed)
# =========================

# Gram only (baseline)
mean_gram = np.array([
    1.748635e-01, 1.538816e-01, 1.387912e-01, 1.285072e-01, 1.205567e-01,
    1.153305e-01, 1.111535e-01, 1.070588e-01, 1.037800e-01, 1.005809e-01,
    9.726343e-02, 9.411472e-02, 9.060667e-02, 8.586457e-02, 8.137912e-02,
    7.657793e-02, 7.111976e-02, 6.517617e-02, 6.066789e-02, 5.418116e-02,
    4.941193e-02
])

# Gram + naive likelihood
mean_naive = np.array([
    1.810029e-01, 1.575632e-01, 1.403509e-01, 1.293232e-01, 1.202357e-01,
    1.139285e-01, 1.086584e-01, 1.044182e-01, 9.993476e-02, 9.589358e-02,
    9.166604e-02, 8.727502e-02, 8.239515e-02, 7.671840e-02, 7.097632e-02,
    6.502634e-02, 5.925589e-02, 5.369763e-02, 4.877206e-02, 4.397259e-02,
    3.956686e-02
])

# Gram + gated likelihood
mean_gated = np.array([
    1.764893e-01, 1.529106e-01, 1.383094e-01, 1.280172e-01, 1.203346e-01,
    1.143455e-01, 1.093437e-01, 1.048839e-01, 1.006874e-01, 9.638820e-02,
    9.208596e-02, 8.760896e-02, 8.263041e-02, 7.669161e-02, 7.099428e-02,
    6.529442e-02, 5.927842e-02, 5.366533e-02, 4.882218e-02, 4.388248e-02,
    3.951930e-02
])

# =========================
# Plot
# =========================
plt.figure(figsize=(8.0, 5.5))
ax = plt.gca()

# Highlight ultra-low SNR region
ax.axvspan(-15, -10, alpha=0.08, color='gray', zorder=0)

# Plot settings
lw = 2.2
lw_gated = 2.2
ms = 7
markevery = 2

# Curves
ax.plot(
    snr, mean_gram,
    marker='o', markersize=ms, markevery=markevery,
    linewidth=lw, label='Gram only',
    zorder=3
)

ax.plot(
    snr, mean_naive,
    linestyle='--',
    marker='^', markersize=ms-1, markevery=markevery,
    linewidth=lw, label='Gram + naive likelihood',
    zorder=4
)

ax.plot(
    snr, mean_gated,
    linestyle='-',
    marker='s', markersize=ms,
    markevery=markevery,
    linewidth=lw_gated,
    label='Gram + SNR-aware likelihood',
    zorder=6
)

# Formatting
ax.set_xlim([-15, 5])
ax.set_xticks(snr)
ax.set_xlabel("SNR [dB]", fontsize=12, fontweight='bold')
ax.set_ylabel("Mean NMSE", fontsize=12, fontweight='bold')

# Use log scale on y-axis (consistent with other NMSE plots)
ax.set_yscale('log')
ax.set_ylim([3e-2, 2.5e-1])

ax.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.6)
ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.4)

legend = ax.legend(
    loc='upper right',
    fontsize=10,
    frameon=True,
    fancybox=True,
    framealpha=0.95
)
legend.get_frame().set_linewidth(1.1)

plt.tight_layout()

# Save
output_path = output_dir / 'plot_likelihood.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Plot saved to '{output_path}'")

# Try to show if display is available
try:
    plt.show()
except Exception as e:
    print(f"Could not display plot: {e}")
    print(f"Plot has been saved to '{output_path}' instead")
