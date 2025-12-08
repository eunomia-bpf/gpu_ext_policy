#!/usr/bin/env python3
"""
Plot policy comparison across all kernels (hotspot, gemm, kmeans).
Creates a figure with 3 subplots showing selected policy configurations.

Usage:
    python plot_all_kernels.py
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
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.dpi': 150,
})

# Selected configurations to plot (policy, high_param, low_param, label, is_sched)
SELECTED_CONFIGS = [
    ('no_policy', None, None, 'No Policy', False),
    ('sched_timeslice', 1000000, 200, 'Scheduler', True),
    ('prefetch_pid_tree', 0, 20, 'Prefetch(0,20)', False),
    ('prefetch_pid_tree', 20, 80, 'Prefetch(20,80)', False),
    ('prefetch_eviction_pid', 20, 80, 'Evict(20,80)', False),
]


def load_data(csv_path):
    """Load and preprocess the CSV data."""
    df = pd.read_csv(csv_path)
    return df


def get_selected_rows(df, sched_df=None):
    """Filter dataframe to only selected configurations."""
    rows = []
    for policy, hp, lp, label, is_sched in SELECTED_CONFIGS:
        source_df = sched_df if is_sched and sched_df is not None else df
        if source_df is None:
            continue
        if policy == 'no_policy':
            row = source_df[source_df['policy'] == 'no_policy']
        else:
            row = source_df[(source_df['policy'] == policy) &
                     (source_df['high_param'] == hp) &
                     (source_df['low_param'] == lp)]
        if len(row) > 0:
            r = row.iloc[0].to_dict()
            r['label'] = label
            rows.append(r)
    return rows


def print_improvements(data, sched_data):
    """Print improvement ratios compared to no_policy."""
    print("\n" + "=" * 80)
    print("IMPROVEMENT RATIOS (vs No Policy)")
    print("=" * 80)

    for kernel_name, df in data.items():
        print(f"\n### {kernel_name} ###")
        sched_df = sched_data.get(kernel_name)
        rows = get_selected_rows(df, sched_df)

        # Find no_policy baseline
        baseline = None
        for r in rows:
            if r['label'] == 'No Policy':
                baseline = r
                break

        if baseline is None:
            print("  No baseline found")
            continue

        baseline_high = baseline['high_latency_s']
        baseline_low = baseline['low_latency_s']
        baseline_total = max(baseline_high, baseline_low)

        print(f"  {'Config':<20} {'High(s)':<10} {'Low(s)':<10} {'Total(s)':<10} {'High Impr':<12} {'Low Impr':<12} {'Total Impr':<12}")
        print(f"  {'-'*86}")

        for r in rows:
            high = r['high_latency_s']
            low = r['low_latency_s']
            total = max(high, low)

            high_impr = (baseline_high - high) / baseline_high * 100
            low_impr = (baseline_low - low) / baseline_low * 100
            total_impr = (baseline_total - total) / baseline_total * 100

            print(f"  {r['label']:<20} {high:<10.1f} {low:<10.1f} {total:<10.1f} "
                  f"{high_impr:>+10.1f}% {low_impr:>+10.1f}% {total_impr:>+10.1f}%")


def plot_kernel_subplot(ax, df, kernel_name, sched_df=None):
    """Plot a single kernel's data on the given axes."""
    rows = get_selected_rows(df, sched_df)

    if not rows:
        ax.set_title(f"{kernel_name} (No Data)")
        return ax

    labels = [r['label'] for r in rows]
    high_vals = [r['high_latency_s'] for r in rows]
    low_vals = [r['low_latency_s'] for r in rows]

    x = np.arange(len(labels))
    width = 0.35

    # Plot bars
    bars1 = ax.bar(x - width/2, high_vals, width,
                   label='High Priority', color='#2ecc71', alpha=0.85)
    bars2 = ax.bar(x + width/2, low_vals, width,
                   label='Low Priority', color='#e74c3c', alpha=0.85)

    # Add baseline lines
    single_1x = df[df['policy'] == 'single_1x']['high_latency_s'].values

    if len(single_1x) > 0:
        ax.axhline(y=single_1x[0], color='#3498db', linestyle='--', linewidth=2,
                   label='Single 1x')
        # Theoretical optimum: 2 × Single 1x
        theoretical_opt = single_1x[0] * 2
        ax.axhline(y=theoretical_opt, color='#9b59b6', linestyle='--', linewidth=2,
                   label='2×Single 1x')

    ax.set_ylabel('Completion Time (s)')
    ax.set_title(kernel_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')

    # Set y-axis limit with some padding
    max_val = max(high_vals + low_vals)
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

    # Load scheduler data
    sched_data = {}
    for kernel_name, result_dir in kernels:
        csv_files = list(result_dir.glob('sched_comparison_*.csv'))
        if csv_files:
            csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
            sched_data[kernel_name] = load_data(csv_path)
            print(f"Loaded Scheduler - {kernel_name}: {csv_path}")

    if not data:
        print("Error: No data loaded")
        return

    # Print improvement ratios
    print_improvements(data, sched_data)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Plot each kernel
    for idx, (kernel_name, df) in enumerate(data.items()):
        sched_df = sched_data.get(kernel_name)
        plot_kernel_subplot(axes[idx], df, kernel_name, sched_df)

    # Add legend only to the first subplot (or use a shared legend)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save
    output_path = base_dir / 'all_kernels_comparison'
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{output_path}.png', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}.pdf/png")


if __name__ == "__main__":
    main()
