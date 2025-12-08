#!/usr/bin/env python3
"""
Analyze policy comparison results for rand_stream workload.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read data
DATA_FILE = Path(__file__).parent / "policy_comparison_20251206_204358.csv"
OUTPUT_DIR = Path(__file__).parent


def load_and_clean_data():
    """Load CSV and remove outliers."""
    df = pd.read_csv(DATA_FILE)

    # Create config label
    df['config'] = df['policy'] + '_' + df['high_param'].astype(str) + '/' + df['low_param'].astype(str)

    # Remove extreme outliers:
    # - bandwidth > 100 GB/s (measurement error)
    # - median < 10ms (measurement error)
    df_clean = df[
        (df['high_bw_gbps'] < 100) & (df['low_bw_gbps'] < 100) &
        (df['high_median_ms'] > 10) & (df['low_median_ms'] > 10)
    ]

    print(f"Total rows: {len(df)}, After removing outliers: {len(df_clean)}")
    print(f"Removed {len(df) - len(df_clean)} outlier rows")

    return df, df_clean


def compute_summary(df):
    """Compute summary statistics per config."""
    summary = df.groupby('config').agg({
        'high_median_ms': ['mean', 'std', 'count'],
        'high_bw_gbps': ['mean', 'std'],
        'low_median_ms': ['mean', 'std'],
        'low_bw_gbps': ['mean', 'std'],
    }).round(2)

    # Flatten column names
    summary.columns = ['_'.join(col) for col in summary.columns]

    # Calculate metrics
    summary['bw_ratio'] = (summary['high_bw_gbps_mean'] / summary['low_bw_gbps_mean']).round(3)
    summary['total_bw_gbps'] = (summary['high_bw_gbps_mean'] + summary['low_bw_gbps_mean']).round(2)

    return summary


def plot_comparison(df_clean):
    """Plot bandwidth and latency comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    configs = df_clean['config'].unique()
    x = np.arange(len(configs))
    width = 0.35

    # Compute means
    means = df_clean.groupby('config').agg({
        'high_bw_gbps': 'mean',
        'low_bw_gbps': 'mean',
        'high_median_ms': 'mean',
        'low_median_ms': 'mean',
    }).reindex(configs)

    stds = df_clean.groupby('config').agg({
        'high_bw_gbps': 'std',
        'low_bw_gbps': 'std',
        'high_median_ms': 'std',
        'low_median_ms': 'std',
    }).reindex(configs)

    # Plot 1: Bandwidth
    ax1 = axes[0]
    ax1.bar(x - width/2, means['high_bw_gbps'], width,
            yerr=stds['high_bw_gbps'], label='High Priority',
            color='steelblue', capsize=3)
    ax1.bar(x + width/2, means['low_bw_gbps'], width,
            yerr=stds['low_bw_gbps'], label='Low Priority',
            color='coral', capsize=3)

    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title('Bandwidth Comparison - rand_stream workload')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Latency
    ax2 = axes[1]
    ax2.bar(x - width/2, means['high_median_ms'], width,
            yerr=stds['high_median_ms'], label='High Priority',
            color='steelblue', capsize=3)
    ax2.bar(x + width/2, means['low_median_ms'], width,
            yerr=stds['low_median_ms'], label='Low Priority',
            color='coral', capsize=3)

    ax2.set_ylabel('Median Latency (ms)')
    ax2.set_title('Latency Comparison - rand_stream workload')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rand_comparison.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'rand_comparison.png'}")
    plt.close()


def plot_total_throughput(summary):
    """Plot total throughput comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by total bandwidth
    sorted_summary = summary.sort_values('total_bw_gbps', ascending=True)

    configs = sorted_summary.index
    total_bw = sorted_summary['total_bw_gbps']

    colors = ['green' if bw > 30 else 'orange' if bw > 25 else 'red' for bw in total_bw]

    bars = ax.barh(configs, total_bw, color=colors)

    # Add baseline line
    if 'no_policy_50/50' in summary.index:
        baseline = summary.loc['no_policy_50/50', 'total_bw_gbps']
        ax.axvline(x=baseline, color='black', linestyle='--', label=f'Baseline ({baseline:.1f})')
        ax.legend()

    ax.set_xlabel('Total Bandwidth (GB/s)')
    ax.set_title('Total Throughput by Policy - rand_stream')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, bw in zip(bars, total_bw):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{bw:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rand_throughput.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'rand_throughput.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("Policy Comparison Analysis - rand_stream workload")
    print("=" * 60)

    # Load data
    df_all, df_clean = load_and_clean_data()

    # Compute summary
    print("\n--- Summary Statistics ---")
    summary = compute_summary(df_clean)
    print(summary.to_string())

    # Save summary
    summary.to_csv(OUTPUT_DIR / 'rand_summary.csv')
    print(f"\nSaved: {OUTPUT_DIR / 'rand_summary.csv'}")

    # Generate plots
    print("\n--- Generating Plots ---")
    plot_comparison(df_clean)
    plot_total_throughput(summary)

    # Throughput report
    print("\n" + "=" * 60)
    print("THROUGHPUT REPORT - rand_stream")
    print("=" * 60)
    print(f"{'Policy':<30} {'High BW':<10} {'Low BW':<10} {'Total BW':<10} {'Ratio':<8} {'vs Base'}")
    print("-" * 80)

    sorted_configs = summary.sort_values('total_bw_gbps', ascending=False)

    baseline_total = None
    if 'no_policy_50/50' in summary.index:
        baseline_total = summary.loc['no_policy_50/50', 'total_bw_gbps']

    for config in sorted_configs.index:
        high_bw = summary.loc[config, 'high_bw_gbps_mean']
        low_bw = summary.loc[config, 'low_bw_gbps_mean']
        total_bw = summary.loc[config, 'total_bw_gbps']
        ratio = summary.loc[config, 'bw_ratio']

        change_str = ""
        if baseline_total and config != 'no_policy_50/50':
            change = (total_bw - baseline_total) / baseline_total * 100
            change_str = f"({change:+.1f}%)"

        print(f"{config:<30} {high_bw:<10.2f} {low_bw:<10.2f} {total_bw:<10.2f} {ratio:<8.3f} {change_str}")

    if baseline_total:
        print(f"\nBaseline (no_policy) total BW: {baseline_total:.2f} GB/s")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS - rand_stream vs seq_stream")
    print("=" * 60)

    best_config = sorted_configs.index[0]
    best_total = sorted_configs.iloc[0]['total_bw_gbps']
    print(f"\nBest policy: {best_config} (Total BW = {best_total:.2f} GB/s)")

    if baseline_total:
        improvement = (best_total - baseline_total) / baseline_total * 100
        print(f"Improvement over baseline: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
