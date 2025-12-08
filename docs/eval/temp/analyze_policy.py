#!/usr/bin/env python3
"""
Analyze policy comparison results and generate plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read data
DATA_FILE = Path(__file__).parent / "policy_comparison_20251206_210915.csv"
OUTPUT_DIR = Path(__file__).parent

def load_and_clean_data():
    """Load CSV and remove outliers."""
    df = pd.read_csv(DATA_FILE)

    # Create config label
    df['config'] = df['policy'] + '_' + df['high_param'].astype(str) + '/' + df['low_param'].astype(str)

    # Remove extreme outliers using multiple criteria:
    # 1. median > 10000ms (timeout/stall)
    # 2. median < 10ms (measurement error)
    # 3. bandwidth > 100 GB/s (impossible for this hardware)
    # 4. bandwidth < 0.5 GB/s (severe degradation)
    df_clean = df[
        (df['high_median_ms'] < 10000) & (df['low_median_ms'] < 10000) &
        (df['high_median_ms'] > 10) & (df['low_median_ms'] > 10) &
        (df['high_bw_gbps'] < 100) & (df['low_bw_gbps'] < 100) &
        (df['high_bw_gbps'] > 0.5) & (df['low_bw_gbps'] > 0.5)
    ]

    print(f"Total rows: {len(df)}, After removing outliers: {len(df_clean)}")
    print(f"Removed {len(df) - len(df_clean)} outlier rows")

    return df, df_clean


def compute_summary(df):
    """Compute summary statistics per config."""
    summary = df.groupby('config').agg({
        'high_median_ms': ['mean', 'std', 'min', 'max'],
        'high_bw_gbps': ['mean', 'std'],
        'low_median_ms': ['mean', 'std', 'min', 'max'],
        'low_bw_gbps': ['mean', 'std'],
    }).round(2)

    # Flatten column names
    summary.columns = ['_'.join(col) for col in summary.columns]

    # Calculate high/low ratio (differentiation)
    summary['bw_ratio'] = (summary['high_bw_gbps_mean'] / summary['low_bw_gbps_mean']).round(3)
    summary['latency_ratio'] = (summary['high_median_ms_mean'] / summary['low_median_ms_mean']).round(3)

    # Calculate total throughput (high + low)
    summary['total_bw_gbps'] = (summary['high_bw_gbps_mean'] + summary['low_bw_gbps_mean']).round(2)

    return summary


def plot_bandwidth_comparison(df_clean):
    """Plot bandwidth comparison across policies."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Get unique configs
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
    bars1 = ax1.bar(x - width/2, means['high_bw_gbps'], width,
                    yerr=stds['high_bw_gbps'], label='High Priority',
                    color='steelblue', capsize=3)
    bars2 = ax1.bar(x + width/2, means['low_bw_gbps'], width,
                    yerr=stds['low_bw_gbps'], label='Low Priority',
                    color='coral', capsize=3)

    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title('Bandwidth Comparison by Policy Configuration')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Latency
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, means['high_median_ms'], width,
                    yerr=stds['high_median_ms'], label='High Priority',
                    color='steelblue', capsize=3)
    bars4 = ax2.bar(x + width/2, means['low_median_ms'], width,
                    yerr=stds['low_median_ms'], label='Low Priority',
                    color='coral', capsize=3)

    ax2.set_ylabel('Median Latency (ms)')
    ax2.set_title('Latency Comparison by Policy Configuration')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'policy_comparison_bars.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'policy_comparison_bars.png'}")
    plt.close()


def plot_differentiation(df_clean):
    """Plot how well each policy differentiates high vs low priority."""
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = df_clean['config'].unique()

    # Compute ratio for each config
    ratios = []
    for config in configs:
        subset = df_clean[df_clean['config'] == config]
        # Bandwidth ratio: high / low (>1 means high priority gets more bandwidth)
        ratio = subset['high_bw_gbps'].mean() / subset['low_bw_gbps'].mean()
        ratios.append(ratio)

    colors = ['green' if r > 1.1 else 'red' if r < 0.9 else 'gray' for r in ratios]

    x = np.arange(len(configs))
    bars = ax.bar(x, ratios, color=colors)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Equal (ratio=1)')
    ax.set_ylabel('Bandwidth Ratio (High/Low)')
    ax.set_title('Policy Differentiation: High Priority vs Low Priority\n(Green: High favored, Red: Low favored)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'policy_differentiation.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'policy_differentiation.png'}")
    plt.close()


def plot_boxplot(df_clean):
    """Plot boxplot of bandwidth distribution per config."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    configs = df_clean['config'].unique()

    # High priority bandwidth
    ax1 = axes[0]
    data_high = [df_clean[df_clean['config'] == c]['high_bw_gbps'].values for c in configs]
    bp1 = ax1.boxplot(data_high, labels=configs, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title('High Priority Bandwidth Distribution')
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Low priority bandwidth
    ax2 = axes[1]
    data_low = [df_clean[df_clean['config'] == c]['low_bw_gbps'].values for c in configs]
    bp2 = ax2.boxplot(data_low, labels=configs, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)
    ax2.set_ylabel('Bandwidth (GB/s)')
    ax2.set_title('Low Priority Bandwidth Distribution')
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'policy_boxplot.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'policy_boxplot.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("Policy Comparison Analysis")
    print("=" * 60)

    # Load data
    df_all, df_clean = load_and_clean_data()

    # Compute summary
    print("\n--- Summary Statistics (outliers removed) ---")
    summary = compute_summary(df_clean)
    print(summary.to_string())

    # Save summary to CSV
    summary.to_csv(OUTPUT_DIR / 'policy_summary.csv')
    print(f"\nSaved: {OUTPUT_DIR / 'policy_summary.csv'}")

    # Generate plots
    print("\n--- Generating Plots ---")
    plot_bandwidth_comparison(df_clean)
    plot_differentiation(df_clean)
    plot_boxplot(df_clean)

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Find best differentiation
    best_config = summary['bw_ratio'].idxmax()
    best_ratio = summary.loc[best_config, 'bw_ratio']
    print(f"\nBest differentiation: {best_config} (ratio={best_ratio})")

    # Throughput report
    print("\n--- THROUGHPUT REPORT ---")
    print(f"{'Policy':<35} {'High BW':<12} {'Low BW':<12} {'Total BW':<12} {'Ratio':<8}")
    print("-" * 80)

    # Sort by total bandwidth
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

        print(f"{config:<35} {high_bw:<12.2f} {low_bw:<12.2f} {total_bw:<12.2f} {ratio:<8.3f} {change_str}")

    # Compare with baseline
    if 'no_policy_50/50' in summary.index:
        print(f"\nBaseline (no_policy) total BW: {baseline_total:.2f} GB/s")


if __name__ == "__main__":
    main()
