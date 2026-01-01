"""
Plotting script for Jacobian ablation experiments (A/B/C).

This script loads results from experiments A, B, and C and creates
overlayed NMSE curves for comparison.
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results_from_csv(csv_file: Path) -> tuple:
    """
    Load results from CSV file.
    
    Returns
    -------
    snrs : np.ndarray
        SNR values in dB
    nmse : np.ndarray
        NMSE values
    """
    snrs = []
    nmse = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snrs.append(float(row['SNR']))
            nmse.append(float(row['nmse_dm_dps']))
    return np.array(snrs), np.array(nmse)


def find_result_files(base_dir: Path, exp_key: str, method: str = 'dps', 
                       dps_lambda: float = 0.1) -> Path:
    """
    Find the result CSV file for a given experiment.
    
    Parameters
    ----------
    base_dir : Path
        Base directory containing results
    exp_key : str
        Experiment key ('A', 'B', or 'C')
    method : str
        Method name (default: 'dps')
    dps_lambda : float
        DPS lambda value (default: 0.1)
    
    Returns
    -------
    Path
        Path to the CSV file
    """
    # Base pattern for all files
    base_pattern = f"*_method={method.upper()}*_dps_lambda={dps_lambda}*_best.csv"
    all_files = list(base_dir.glob(base_pattern))
    
    if exp_key == 'A':
        # Experiment A: files should NOT have _exp= suffix
        files = [f for f in all_files if '_exp=' not in f.name]
    else:
        # Experiments B/C: files should have _exp={exp_key} suffix
        files = [f for f in all_files if f'_exp={exp_key}' in f.name]
    
    if not files:
        raise FileNotFoundError(
            f"Could not find result file for exp_key={exp_key}, "
            f"method={method}, dps_lambda={dps_lambda}.\n"
            f"  Searched in: {base_dir}\n"
            f"  Pattern: {base_pattern}\n"
            f"  Found {len(all_files)} matching files, but none for exp_key={exp_key}"
        )
    
    # If multiple files, use the most recent one
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0]


def plot_ablation_results(base_dir: Path, method: str = 'dps', 
                          dps_lambda: float = 0.1, output_file: Path = None):
    """
    Plot ablation results for experiments A, B, and C.
    
    Parameters
    ----------
    base_dir : Path
        Base directory containing results
    method : str
        Method name (default: 'dps')
    dps_lambda : float
        DPS lambda value (default: 0.1)
    output_file : Path, optional
        Output file path. If None, saves to base_dir.
    """
    # Load results for each experiment
    results = {}
    for exp_key in ['A', 'B', 'C']:
        try:
            csv_file = find_result_files(base_dir, exp_key, method, dps_lambda)
            snrs, nmse = load_results_from_csv(csv_file)
            results[exp_key] = {'snrs': snrs, 'nmse': nmse}
            print(f"Loaded {exp_key}: {len(snrs)} SNR points from {csv_file.name}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
    
    if not results:
        raise ValueError("No results found for any experiment!")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    
    # Plot each experiment
    colors = {'A': 'C0', 'B': 'C1', 'C': 'C2'}
    labels = {
        'A': 'Baseline (no Jacobian)',
        'B': 'Scalar Jacobian (1/√ᾱ_t)',
        'C': 'Autograd Gold (full Jacobian)'
    }
    markers = {'A': 'o', 'B': '^', 'C': 's'}
    linestyles = {'A': '-', 'B': '--', 'C': '-.'}
    
    for exp_key in ['A', 'B', 'C']:
        if exp_key not in results:
            continue
        data = results[exp_key]
        ax.plot(
            data['snrs'], 
            data['nmse'],
            color=colors[exp_key],
            label=labels[exp_key],
            marker=markers[exp_key],
            linestyle=linestyles[exp_key],
            linewidth=2.0,
            markersize=6,
            markevery=max(1, len(data['snrs']) // 10),
        )
    
    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('NMSE', fontsize=12, fontweight='bold')
    ax.set_title('Jacobian Ablation: Likelihood Guidance Comparison', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.3, which='both')
    ax.grid(True, linestyle=':', alpha=0.2, which='minor')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10, framealpha=0.9, shadow=True, 
              fancybox=True, edgecolor='black')
    
    plt.tight_layout()
    
    # Save plot
    if output_file is None:
        output_file = base_dir / f'ablation_jacobian_method={method}_lambda={dps_lambda}.png'
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*70)
    print("Summary Table: Mean NMSE at each SNR")
    print("="*70)
    print(f"{'SNR (dB)':>10} ", end="")
    for exp_key in ['A', 'B', 'C']:
        if exp_key in results:
            print(f"{'Exp ' + exp_key:>15} ", end="")
    print()
    print("-"*70)
    
    # Find common SNR values
    all_snrs = set()
    for exp_key in results:
        all_snrs.update(results[exp_key]['snrs'])
    common_snrs = sorted(all_snrs)
    
    for snr in common_snrs:
        print(f"{snr:>10.1f} ", end="")
        for exp_key in ['A', 'B', 'C']:
            if exp_key not in results:
                continue
            data = results[exp_key]
            # Find closest SNR
            idx = np.argmin(np.abs(data['snrs'] - snr))
            nmse_val = data['nmse'][idx]
            print(f"{nmse_val:>15.6e} ", end="")
        print()
    
    print("="*70)
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot Jacobian ablation results')
    parser.add_argument(
        '--base_dir',
        type=str,
        default='results/dm_dps',
        help='Base directory containing result files'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='dps',
        help='Method name (default: dps)'
    )
    parser.add_argument(
        '--dps_lambda',
        type=float,
        default=0.1,
        help='DPS lambda value (default: 0.1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: auto-generated)'
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    output_file = Path(args.output) if args.output else None
    
    plot_ablation_results(base_dir, args.method, args.dps_lambda, output_file)


if __name__ == '__main__':
    main()

