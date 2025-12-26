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

# ===== SNR grid =====
snr = np.arange(-15, 6, 1, dtype=float)

# ===== Genie (LMMSE) =====
lmmse_genie = np.array([
    0.654871892, 0.614210594, 0.571697074, 0.528890459, 0.485627649,
    0.44325059,  0.402287903, 0.361847727, 0.323922238, 0.28793767,
    0.25426745,  0.223241458, 0.194940835, 0.169181755, 0.145924788,
    0.125577844, 0.10756921,  0.091593267, 0.077720248, 0.06574859,
    0.055502986
])

# ===== DPS (lbd = 0.1) =====
nmse_dps = np.array([
    0.953405201, 0.923455656, 0.883167565, 0.831169307, 0.767774642,
    0.697487533, 0.620056272, 0.542053878, 0.466282755, 0.400504202,
    0.343076289, 0.292655855, 0.250314415, 0.211691454, 0.180478558,
    0.153672323, 0.13030912,  0.11008589,  0.094051592, 0.078737386,
    0.066486768
])

# ===== DPS + COV (lbd = 0.1, cov_lbd = 0.01) =====
nmse_dps_cov = np.array([
    0.942350209, 0.854864717, 0.764093459, 0.67789036,  0.594796062,
    0.524424136, 0.454492867, 0.390313804, 0.3288095,   0.279698402,
    0.237899065, 0.202439368, 0.172186002, 0.146007895, 0.124765158,
    0.10631869,  0.090707541, 0.07726799,  0.066082083, 0.056151293,
    0.047963966
])

# ===== DDIM =====
nmse_ddim = np.array([
    0.956674993, 0.932828784, 0.898368776, 0.851259053, 0.79441011,
    0.727668762, 0.654289067, 0.574952424, 0.499485046, 0.427704483,
    0.364863485, 0.309671074, 0.262683958, 0.220814109, 0.1872347,
    0.157922253, 0.133347675, 0.112339757, 0.095159866, 0.079534099,
    0.067043602
])
# ===== DPS + COV (estimated) =====
nmse_dps_cov_est = np.array([
    0.948509872, 0.860948324, 0.767477989, 0.679588377, 0.594359457,
    0.525992751, 0.456567824, 0.390627414, 0.32931754,  0.281405777,
    0.238511965, 0.202675998, 0.172925904, 0.146775916, 0.125355572,
    0.106843434, 0.091108084, 0.077716768, 0.066501848, 0.056575477,
    0.048371933
])

nmse_dps_cov_sqrtbeta = np.array([
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
# ===== Plot =====

plt.figure(figsize=(7.2, 4.8))

plt.semilogy(snr, nmse_ddim,         marker='o', label='DM')
plt.semilogy(snr, nmse_dps,          marker='s', label='DPS')
plt.semilogy(snr, nmse_dps_cov,      marker='^', label='CGD (beta_t)')
# plt.semilogy(snr, nmse_dps_cov_est,  marker='v', linestyle='--', label='DPS-COV (est)')
plt.semilogy(snr, nmse_dps_cov_sqrtbeta,  marker='v', linestyle='--', label='CGD (sqrt_beta_t)')
plt.semilogy(snr, lmmse_genie,       marker='x', label='LMMSE')


plt.xlim([-15, 5])
plt.ylim([4e-2, 1])

plt.xticks(np.arange(-15, 6, 1))
plt.xlabel('SNR [dB]')
plt.ylabel('Normalized MSE')
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save the figure instead of showing (for WSL/headless environments)
output_path = output_dir / 'nmse_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to '{output_path}'")

# Try to show if display is available
try:
    plt.show()
except Exception as e:
    print(f"Could not display plot: {e}")
    print(f"Plot has been saved to '{output_path}' instead")
