#!/usr/bin/env python3
"""
Plot policy comparison results for multi-tenant GPU memory experiments.

This script visualizes the impact of different prefetch policies on:
1. Per-process execution time (median_ms)
2. Total completion time (latency_s)
3. Bandwidth utilization (bw_gbps)

Usage:
    python plot_policy_comparison.py results_hotspot/policy_comparison_*.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_data(csv_path):
    """Load and preprocess the CSV data."""
    df = pd.read_csv(csv_path)

    # Create a config label for each row
    def make_label(row):
        if pd.isna(row['high_param']) or row['high_param'] == '':
            return row['policy']
        return f"{row['policy']}({int(row['high_param'])},{int(row['low_param'])})"

    df['config'] = df.apply(make_label, axis=1)
    return df


def plot_completion_time_comparison(df, output_dir):
    """
    Plot 1: Completion time comparison - bar chart showing high vs low priority latency.
    This shows QoS differentiation capability.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Filter to multi-tenant configs (exclude single_*)
    mt_df = df[~df['policy'].str.startswith('single_')].copy()

    configs = mt_df['config'].tolist()
    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, mt_df['high_latency_s'], width,
                   label='High Priority Process', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, mt_df['low_latency_s'], width,
                   label='Low Priority Process', color='#e74c3c', alpha=0.8)

    # Add single_1x baseline line
    single_1x = df[df['policy'] == 'single_1x']['high_latency_s'].values
    if len(single_1x) > 0:
        ax.axhline(y=single_1x[0], color='blue', linestyle='--', linewidth=2,
                   label=f'Single Process Baseline ({single_1x[0]:.1f}s)')

    ax.set_ylabel('Completion Time (seconds)')
    ax.set_xlabel('Policy Configuration')
    ax.set_title('Multi-Tenant GPU: Per-Process Completion Time by Policy')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, max(mt_df['high_latency_s'].max(), mt_df['low_latency_s'].max()) * 1.15)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'completion_time_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'completion_time_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: completion_time_comparison.pdf/png")


def plot_median_time_comparison(df, output_dir):
    """
    Plot 2: Median execution time (ms) - shows per-iteration performance.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    mt_df = df[~df['policy'].str.startswith('single_')].copy()

    configs = mt_df['config'].tolist()
    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, mt_df['high_median_ms'], width,
                   label='High Priority', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, mt_df['low_median_ms'], width,
                   label='Low Priority', color='#e67e22', alpha=0.8)

    # Add single_1x baseline
    single_1x = df[df['policy'] == 'single_1x']['high_median_ms'].values
    if len(single_1x) > 0:
        ax.axhline(y=single_1x[0], color='green', linestyle='--', linewidth=2,
                   label=f'Ideal (single process): {single_1x[0]:.0f}ms')

    ax.set_ylabel('Median Execution Time (ms)')
    ax.set_xlabel('Policy Configuration')
    ax.set_title('Multi-Tenant GPU: Per-Iteration Execution Time')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / 'median_time_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'median_time_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: median_time_comparison.pdf/png")


def plot_bandwidth_comparison(df, output_dir):
    """
    Plot 3: Bandwidth utilization comparison.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    mt_df = df[~df['policy'].str.startswith('single_')].copy()

    configs = mt_df['config'].tolist()
    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, mt_df['high_bw_gbps'], width,
                   label='High Priority', color='#9b59b6', alpha=0.8)
    bars2 = ax.bar(x + width/2, mt_df['low_bw_gbps'], width,
                   label='Low Priority', color='#1abc9c', alpha=0.8)

    # Add single_1x baseline
    single_1x = df[df['policy'] == 'single_1x']['high_bw_gbps'].values
    if len(single_1x) > 0:
        ax.axhline(y=single_1x[0], color='red', linestyle='--', linewidth=2,
                   label=f'Peak BW (single): {single_1x[0]:.0f} GB/s')

    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_xlabel('Policy Configuration')
    ax.set_title('Multi-Tenant GPU: Memory Bandwidth Utilization')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'bandwidth_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'bandwidth_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: bandwidth_comparison.pdf/png")


def plot_qos_summary(df, output_dir):
    """
    Plot 4: QoS summary - scatter plot showing tradeoff between high and low priority performance.
    X-axis: Low priority speedup vs no_policy
    Y-axis: High priority speedup vs no_policy
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Get no_policy baseline
    no_policy = df[df['policy'] == 'no_policy'].iloc[0]
    baseline_high = no_policy['high_latency_s']
    baseline_low = no_policy['low_latency_s']

    # Get single_1x ideal
    single_1x = df[df['policy'] == 'single_1x'].iloc[0]
    ideal_latency = single_1x['high_latency_s']

    # Filter prefetch_pid_tree configs
    prefetch_df = df[df['policy'] == 'prefetch_pid_tree'].copy()

    # Calculate speedup (baseline / actual, higher is better)
    prefetch_df['high_speedup'] = baseline_high / prefetch_df['high_latency_s']
    prefetch_df['low_speedup'] = baseline_low / prefetch_df['low_latency_s']

    # Color by parameter difference
    colors = []
    for _, row in prefetch_df.iterrows():
        diff = row['low_param'] - row['high_param']
        colors.append(diff)

    scatter = ax.scatter(prefetch_df['low_speedup'], prefetch_df['high_speedup'],
                         c=colors, cmap='RdYlGn', s=150, alpha=0.8, edgecolors='black')

    # Add labels
    for _, row in prefetch_df.iterrows():
        label = f"({int(row['high_param'])},{int(row['low_param'])})"
        ax.annotate(label, (row['low_speedup'], row['high_speedup']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Add reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No policy baseline')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

    # Add ideal point
    ideal_high_speedup = baseline_high / ideal_latency
    ideal_low_speedup = baseline_low / ideal_latency
    ax.scatter([ideal_low_speedup], [ideal_high_speedup], marker='*', s=300,
               c='gold', edgecolors='black', label='Ideal (single process)', zorder=5)

    ax.set_xlabel('Low Priority Speedup (vs no_policy)')
    ax.set_ylabel('High Priority Speedup (vs no_policy)')
    ax.set_title('QoS Tradeoff: Policy Parameter Impact\n(params: high_throttle, low_throttle)')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Priority Difference (low_param - high_param)')

    ax.legend(loc='lower right')
    ax.set_xlim(0, max(prefetch_df['low_speedup'].max(), ideal_low_speedup) * 1.1)
    ax.set_ylim(0, max(prefetch_df['high_speedup'].max(), ideal_high_speedup) * 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / 'qos_tradeoff.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'qos_tradeoff.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: qos_tradeoff.pdf/png")


def plot_total_completion_time(df, output_dir):
    """
    Plot 5: Total system completion time (max of high and low latency).
    Shows overall system efficiency.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Calculate total completion time
    df_plot = df.copy()
    df_plot['total_latency'] = df_plot.apply(
        lambda r: max(r['high_latency_s'], r['low_latency_s']) if r['low_latency_s'] > 0 else r['high_latency_s'],
        axis=1
    )

    # Sort by total latency
    df_plot = df_plot.sort_values('total_latency')

    colors = []
    for _, row in df_plot.iterrows():
        if row['policy'].startswith('single_'):
            colors.append('#95a5a6')  # gray for single
        elif row['policy'] == 'no_policy':
            colors.append('#e74c3c')  # red for no_policy
        else:
            colors.append('#2ecc71')  # green for policies

    bars = ax.barh(df_plot['config'], df_plot['total_latency'], color=colors, alpha=0.8)

    ax.set_xlabel('Total Completion Time (seconds)')
    ax.set_ylabel('Configuration')
    ax.set_title('System Efficiency: Total Completion Time\n(time until both processes finish)')

    # Add value labels
    for bar, val in zip(bars, df_plot['total_latency']):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}s', va='center', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#95a5a6', label='Single process (baseline)'),
        Patch(facecolor='#e74c3c', label='No policy (contention)'),
        Patch(facecolor='#2ecc71', label='With policy'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'total_completion_time.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'total_completion_time.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: total_completion_time.pdf/png")


def print_summary_table(df):
    """Print a summary table of key metrics."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Get baselines
    single_1x = df[df['policy'] == 'single_1x'].iloc[0]
    no_policy = df[df['policy'] == 'no_policy'].iloc[0]

    print(f"\nBaselines:")
    print(f"  single_1x: {single_1x['high_latency_s']:.2f}s (ideal, no contention)")
    print(f"  no_policy: {no_policy['high_latency_s']:.2f}s high, {no_policy['low_latency_s']:.2f}s low")

    print(f"\n{'Config':<25} {'High(s)':<10} {'Low(s)':<10} {'Total(s)':<10} {'Speedup':<10}")
    print("-" * 65)

    for _, row in df.iterrows():
        if row['policy'].startswith('single_'):
            continue

        total = max(row['high_latency_s'], row['low_latency_s'])
        baseline_total = max(no_policy['high_latency_s'], no_policy['low_latency_s'])
        speedup = baseline_total / total if total > 0 else 0

        print(f"{row['config']:<25} {row['high_latency_s']:<10.2f} {row['low_latency_s']:<10.2f} {total:<10.2f} {speedup:<10.2f}x")

    print("\nKey Insights:")

    # Find best config for low priority
    prefetch_df = df[df['policy'] == 'prefetch_pid_tree']
    best_low = prefetch_df.loc[prefetch_df['low_latency_s'].idxmin()]
    print(f"  Best for Low Priority: {best_low['config']} -> {best_low['low_latency_s']:.2f}s "
          f"(vs {no_policy['low_latency_s']:.2f}s no_policy, {no_policy['low_latency_s']/best_low['low_latency_s']:.1f}x speedup)")

    # Find best config for high priority
    best_high = prefetch_df.loc[prefetch_df['high_latency_s'].idxmin()]
    print(f"  Best for High Priority: {best_high['config']} -> {best_high['high_latency_s']:.2f}s "
          f"(vs {no_policy['high_latency_s']:.2f}s no_policy, {no_policy['high_latency_s']/best_high['high_latency_s']:.1f}x speedup)")

    # Find best overall
    prefetch_df = prefetch_df.copy()
    prefetch_df['total'] = prefetch_df.apply(lambda r: max(r['high_latency_s'], r['low_latency_s']), axis=1)
    best_total = prefetch_df.loc[prefetch_df['total'].idxmin()]
    baseline_total = max(no_policy['high_latency_s'], no_policy['low_latency_s'])
    print(f"  Best Overall: {best_total['config']} -> {best_total['total']:.2f}s "
          f"(vs {baseline_total:.2f}s no_policy, {baseline_total/best_total['total']:.1f}x speedup)")


def plot_selected_configs_vertical(df, output_dir):
    """
    Plot selected configurations vertically with baselines as dashed lines.
    Only: no_policy, (0,0), (20,80)
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    # Larger fonts
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14})

    # Select specific configurations
    selected = [
        ('no_policy', 50, 50),
        ('prefetch_pid_tree', 0, 0),
        ('prefetch_pid_tree', 20, 80),
    ]
    labels = ['No Policy', '(0,0)', '(20,80)']

    # Filter and order data
    plot_data = []
    for (policy, hp, lp), label in zip(selected, labels):
        if policy == 'no_policy':
            row = df[df['policy'] == 'no_policy'].iloc[0]
        else:
            row = df[(df['policy'] == policy) &
                     (df['high_param'] == hp) &
                     (df['low_param'] == lp)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
        plot_data.append({
            'label': label,
            'high': row['high_latency_s'],
            'low': row['low_latency_s'],
        })

    x = np.arange(len(plot_data))
    width = 0.35

    # Plot vertical bars
    bars1 = ax.bar(x - width/2, [d['high'] for d in plot_data], width,
                   label='High Priority', color='#2ecc71', alpha=0.85)
    bars2 = ax.bar(x + width/2, [d['low'] for d in plot_data], width,
                   label='Low Priority', color='#e74c3c', alpha=0.85)

    # Add baseline lines
    single_1x = df[df['policy'] == 'single_1x']['high_latency_s'].values[0]
    single_2x = df[df['policy'] == 'single_2x']['high_latency_s'].values[0]

    ax.axhline(y=single_1x, color='#3498db', linestyle='--', linewidth=2,
               label=f'Single 1x ({single_1x:.1f}s)')
    ax.axhline(y=single_2x, color='#9b59b6', linestyle='--', linewidth=2,
               label=f'Single 2x ({single_2x:.1f}s)')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Completion Time (s)', fontsize=13)
    ax.set_xlabel('Policy', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([d['label'] for d in plot_data], fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max([d['high'] for d in plot_data] + [d['low'] for d in plot_data]) * 1.2)

    plt.tight_layout()
    plt.savefig(output_dir / 'policy_selected_vertical.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'policy_selected_vertical.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: policy_selected_vertical.pdf/png")


def main():
    if len(sys.argv) < 2:
        # Default to most recent CSV in results_hotspot
        csv_files = list(Path(__file__).parent.glob("results_hotspot/policy_comparison_*.csv"))
        if not csv_files:
            print("Usage: python plot_policy_comparison.py <csv_file>")
            sys.exit(1)
        csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent CSV: {csv_path}")
    else:
        csv_path = Path(sys.argv[1])

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    # Load data
    df = load_data(csv_path)
    print(f"Loaded {len(df)} configurations from {csv_path}")

    # Output directory
    output_dir = csv_path.parent

    # Generate plots
    plot_completion_time_comparison(df, output_dir)
    plot_median_time_comparison(df, output_dir)
    plot_bandwidth_comparison(df, output_dir)
    plot_qos_summary(df, output_dir)
    plot_total_completion_time(df, output_dir)
    plot_selected_configs_vertical(df, output_dir)

    # Print summary
    print_summary_table(df)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
