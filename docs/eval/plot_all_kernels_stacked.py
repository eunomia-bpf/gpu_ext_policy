#!/usr/bin/env python3
"""
Plot policy comparison with stacked bars showing concurrent vs solo execution.

Visualization:
- Bottom segment: Time when both processes are running together (contention)
- Top segment: Time when only one process is running alone

Usage:
    python plot_all_kernels_stacked.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 26,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.dpi': 150,
})

# Selected configurations to plot
SELECTED_CONFIGS = [
    ('no_policy', None, None, 'No Policy'),
    ('prefetch_pid_tree', 0, 20, 'Prefetch(0,20)'),
    ('prefetch_pid_tree', 20, 80, 'Prefetch(20,80)'),
    ('prefetch_eviction_pid', 20, 80, 'Evict(20,80)'),
]


def load_data(csv_path):
    """Load and preprocess the CSV data."""
    df = pd.read_csv(csv_path)
    return df


def get_selected_rows(df):
    """Filter dataframe to only selected configurations."""
    rows = []
    for policy, hp, lp, label in SELECTED_CONFIGS:
        if policy == 'no_policy':
            row = df[df['policy'] == 'no_policy']
        else:
            row = df[(df['policy'] == policy) &
                     (df['high_param'] == hp) &
                     (df['low_param'] == lp)]
        if len(row) > 0:
            r = row.iloc[0].to_dict()
            r['label'] = label
            rows.append(r)
    return rows


def print_improvements(data):
    """Print improvement ratios compared to no_policy."""
    print("\n" + "=" * 80)
    print("IMPROVEMENT RATIOS (vs No Policy)")
    print("=" * 80)

    for kernel_name, df in data.items():
        print(f"\n### {kernel_name} ###")
        rows = get_selected_rows(df)

        baseline = None
        for r in rows:
            if r['label'] == 'No Policy':
                baseline = r
                break

        if baseline is None:
            print("  No baseline found")
            continue

        baseline_total = max(baseline['high_latency_s'], baseline['low_latency_s'])

        print(f"  {'Config':<20} {'Both(s)':<10} {'LowOnly(s)':<10} {'Total(s)':<10} {'Impr':<12}")
        print(f"  {'-'*62}")

        for r in rows:
            high = r['high_latency_s']
            low = r['low_latency_s']
            both_time = min(high, low)
            solo_time = abs(high - low)
            total = max(high, low)
            total_impr = (baseline_total - total) / baseline_total * 100

            print(f"  {r['label']:<20} {both_time:<10.1f} {solo_time:<10.1f} {total:<10.1f} {total_impr:>+10.1f}%")


def plot_kernel_subplot(ax, df, kernel_name):
    """Plot a single kernel's data with stacked bars."""
    rows = get_selected_rows(df)

    if not rows:
        ax.set_title(f"{kernel_name} (No Data)")
        return ax

    labels = [r['label'] for r in rows]

    # Calculate stacked bar segments
    both_running = []  # min(high, low) - time when both are running
    solo_running = []  # |high - low| - time when only one is running

    for r in rows:
        high = r['high_latency_s']
        low = r['low_latency_s']
        both_running.append(min(high, low))
        solo_running.append(abs(high - low))

    x = np.arange(len(labels))
    width = 0.6

    # Plot stacked bars
    bars1 = ax.bar(x, both_running, width,
                   label='Both Running', color='#e74c3c', alpha=0.85)
    bars2 = ax.bar(x, solo_running, width, bottom=both_running,
                   label='Only Low Running', color='#3498db', alpha=0.85)

    # Add baseline lines
    single_1x = df[df['policy'] == 'single_1x']['high_latency_s'].values
    single_2x = df[df['policy'] == 'single_2x']['high_latency_s'].values

    if len(single_1x) > 0:
        ax.axhline(y=single_1x[0], color='#2ecc71', linestyle='--', linewidth=2.5,
                   label=f'Single 1x ({single_1x[0]:.1f}s)')
    if len(single_2x) > 0:
        ax.axhline(y=single_2x[0], color='#9b59b6', linestyle='--', linewidth=2.5,
                   label=f'Single 2x ({single_2x[0]:.1f}s)')

    ax.set_ylabel('Completion Time (s)')
    ax.set_title(kernel_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')

    # Set y-axis limit with some padding
    max_val = max([b + s for b, s in zip(both_running, solo_running)])
    ax.set_ylim(0, max_val * 1.15)

    return ax


def main():
    base_dir = Path(__file__).parent

    # Define kernel data paths
    kernels = [
        ('Hotspot', base_dir / 'results_hotspot'),
        ('GEMM', base_dir / 'results_gemm'),
        ('K-Means', base_dir / 'results_kmeans'),
    ]

    # Load all data
    data = {}
    for kernel_name, result_dir in kernels:
        csv_files = list(result_dir.glob('policy_comparison_*.csv'))
        if not csv_files:
            print(f"Warning: No CSV found in {result_dir}")
            continue
        csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
        data[kernel_name] = load_data(csv_path)
        print(f"Loaded {kernel_name}: {csv_path}")

    if not data:
        print("Error: No data loaded")
        return

    # Print improvement ratios
    print_improvements(data)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Plot each kernel
    for idx, (kernel_name, df) in enumerate(data.items()):
        plot_kernel_subplot(axes[idx], df, kernel_name)

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save
    output_path = base_dir / 'all_kernels_stacked'
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{output_path}.png', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}.pdf/png")


if __name__ == "__main__":
    main()
