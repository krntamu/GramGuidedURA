"""
Robustness Check: Impact of H H^H Side Information Quality on DPS Performance

This script evaluates how the quality of estimated covariance matrix H H^H affects
DPS-COV performance compared to oracle (perfect) covariance. It compares:
1. Relative degradation (percentage increase in NMSE)
2. dB gap (difference in dB scale)

The analysis helps determine the minimum number of time samples (N_d) needed
for covariance estimation to achieve acceptable performance.
"""

import numpy as np
import matplotlib
import os

# Set backend based on environment
# In WSL, if DISPLAY is not set, use non-interactive backend
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
    print("Note: Using non-interactive backend. Plot will be saved to file.")

import matplotlib.pyplot as plt

# ===== SNR grid =====
snr = np.array([-15, -10, -5, 0, 5], dtype=float)

# ===== NMSE results for different covariance estimation qualities =====
# Oracle: Perfect covariance matrix H H^H (ground truth)
# Est_Nd: Estimated covariance using N_d time samples
nmse_oracle = np.array([
    0.796062529,
    0.451896101,
    0.217503831,
    0.093928896,
    0.038828023
])

nmse_est_2000 = np.array([
    0.806137979,
    0.452589452,
    0.218173802,
    0.0946135,
    0.039409269
])

nmse_est_1000 = np.array([
    0.814643025,
    0.455318838,
    0.218401507,
    0.095189795,
    0.039915375
])

nmse_est_500 = np.array([
    0.83594209,
    0.458399355,
    0.220765606,
    0.096367344,
    0.041031949
])

nmse_est_300 = np.array([
    0.864319742,
    0.463583112,
    0.222807258,
    0.097921535,
    0.042554442
])

nmse_est_200 = np.array([
    0.909750223,
    0.472836733,
    0.225805134,
    0.100164622,
    0.044441957
])

nmse_est_100 = np.array([
    1.118835092,
    0.495506793,
    0.234834179,
    0.106134653,
    0.050113477
])
plt.figure(figsize=(7, 5))

plt.semilogy(snr, nmse_oracle, marker='o', linewidth=2,
             label='DPS-COV (oracle $H H^H$)')

plt.semilogy(snr, nmse_est_2000, marker='s',
             label='Est $H H^H$, $N_d=2000$')

plt.semilogy(snr, nmse_est_1000, marker='^',
             label='Est $H H^H$, $N_d=1000$')

plt.semilogy(snr, nmse_est_500, marker='v',
             label='Est $H H^H$, $N_d=500$')

plt.semilogy(snr, nmse_est_300, marker='D',
             label='Est $H H^H$, $N_d=300$')

plt.semilogy(snr, nmse_est_200, marker='x',
             label='Est $H H^H$, $N_d=200$')

plt.semilogy(snr, nmse_est_100, marker='*',
             label='Est $H H^H$, $N_d=100$')

plt.xlabel('SNR (dB)')
plt.ylabel('NMSE')
plt.title('Impact of $H H^H$ Side Information Quality on DPS')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save the figure instead of showing (for WSL/headless environments)
plt.savefig('robust_check.png', dpi=150, bbox_inches='tight')
print("Plot saved to 'robust_check.png'")

# Try to show if display is available
try:
    plt.show()
except Exception as e:
    print(f"Could not display plot: {e}")
    print("Plot has been saved to 'robust_check.png' instead")

print("\n" + "="*70)
print("ANALYSIS: Comparing Estimated vs Oracle Covariance Performance")
print("="*70)

# ============================================================================
# Comparison Method 1: Relative Degradation (Percentage-based)
# ============================================================================
# Measures: (NMSE_estimated - NMSE_oracle) / NMSE_oracle
# Interpretation: Percentage increase in NMSE compared to oracle
# Useful for: Understanding relative performance loss

print("\n--- Method 1: Relative Degradation Analysis ---")

# Define SNR region of interest
# Adjust this to focus on specific SNR ranges (e.g., low SNR: snr <= 0)
snr_region = (snr >= -15) & (snr <= 5)

# Threshold: maximum acceptable relative degradation (e.g., 10% = 0.10)
tau = 0.10  # 10% relative degradation threshold

# Dictionary mapping N_d (number of time samples) to corresponding NMSE results
candidates = {
    2000: nmse_est_2000,
    1000: nmse_est_1000,
    500:  nmse_est_500,
    300:  nmse_est_300,
    200:  nmse_est_200,
    100:  nmse_est_100,
}

oracle = nmse_oracle

# Compute relative degradation for each N_d
report = []
for nd in sorted(candidates.keys(), reverse=True):
    nmse = candidates[nd]
    # Relative degradation: (NMSE_estimated - NMSE_oracle) / NMSE_oracle
    deg = (nmse - oracle) / oracle
    worst = np.max(deg[snr_region])  # Worst-case degradation in SNR region
    avg = np.mean(deg[snr_region])   # Average degradation in SNR region
    report.append((nd, worst, avg))

# Print results table
print(f"\nSNR Region: [{snr[snr_region][0]:.0f}, {snr[snr_region][-1]:.0f}] dB")
print(f"Threshold: {tau*100:.0f}% relative degradation")
print("\nN_d | worst_rel_deg | avg_rel_deg (in chosen SNR region)")
print("-"*65)
for nd, worst, avg in sorted(report, key=lambda x: x[0], reverse=True):
    print(f"{nd:4d} | {worst:13.4f} | {avg:11.4f}")

# Find minimal N_d that meets the threshold
ok = [nd for nd, worst, avg in report if worst <= tau]
if ok:
    print(f"\n✅ Minimal N_d meeting worst-case <= {tau*100:.0f}%: {min(ok)}")
else:
    print(f"\n❌ No N_d meets worst-case <= {tau*100:.0f}% in this SNR region.")

# ============================================================================
# Comparison Method 2: dB Gap (Logarithmic Scale)
# ============================================================================
# Measures: 10*log10(NMSE_estimated) - 10*log10(NMSE_oracle)
# Interpretation: Difference in dB scale (more intuitive for signal processing)
# Useful for: Understanding performance gap in decibel units

print("\n--- Method 2: dB Gap Analysis ---")

# Define SNR region of interest (can be different from Method 1)
snr_region = (snr >= -15) & (snr <= 5)

# Threshold: maximum acceptable dB gap (e.g., 0.5 dB)
tau_db = 0.5  # dB gap threshold (typical values: 0.3, 0.5, 1.0)

# Dictionary mapping N_d to corresponding NMSE results
candidates = {
    2000: nmse_est_2000,
    1000: nmse_est_1000,
    500:  nmse_est_500,
    300:  nmse_est_300,
    200:  nmse_est_200,
    100:  nmse_est_100,
}

oracle = nmse_oracle

# Compute dB gap for each N_d
report_db = []
for nd in sorted(candidates.keys(), reverse=True):
    nmse = candidates[nd]
    
    # dB gap: Δ_dB = 10*log10(NMSE_estimated) - 10*log10(NMSE_oracle)
    # Positive values indicate worse performance than oracle
    delta_db = 10.0 * np.log10(nmse) - 10.0 * np.log10(oracle)
    
    worst_db = np.max(delta_db[snr_region])  # Worst-case dB gap in SNR region
    avg_db = np.mean(delta_db[snr_region])    # Average dB gap in SNR region
    report_db.append((nd, worst_db, avg_db))

# Print results table
print(f"\nSNR Region: [{snr[snr_region][0]:.0f}, {snr[snr_region][-1]:.0f}] dB")
print(f"Threshold: {tau_db:.2f} dB gap")
print("\nN_d | worst_db_gap | avg_db_gap (dB) (in chosen SNR region)")
print("-"*70)
for nd, worst_db, avg_db in sorted(report_db, key=lambda x: x[0], reverse=True):
    print(f"{nd:4d} | {worst_db:12.4f} | {avg_db:11.4f}")

# Find minimal N_d that meets the threshold
ok_db = [nd for nd, worst_db, avg_db in report_db if worst_db <= tau_db]
if ok_db:
    print(f"\n✅ Minimal N_d meeting worst-case <= {tau_db:.2f} dB: {min(ok_db)}")
else:
    print(f"\n❌ No N_d meets worst-case <= {tau_db:.2f} dB in this SNR region.")

print("\n" + "="*70)