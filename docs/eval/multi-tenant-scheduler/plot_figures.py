#!/usr/bin/env python3
"""
Generate publication-quality figures for multi-tenant scheduler evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# OSDI-style formatting
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 150,
    'axes.linewidth': 1,
    'lines.linewidth': 2,
})

DATA_DIR = Path(__file__).parent / "simple_test_results"
OUTPUT_DIR = Path(__file__).parent
NUM_RUNS = 10

def load_data(mode, workload_type):
    """Load latency data for a mode and workload type."""
    latencies = []
    num_procs = 2 if workload_type == 'lc' else 4
    for run in range(NUM_RUNS):
        for proc in range(num_procs):
            csv_file = DATA_DIR / f"{mode}_run{run}_{workload_type}_{proc}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                latencies.extend(df['launch_latency_ms'].values * 1000)  # to µs
    return np.array(latencies)

def load_per_run_p99(mode, workload_type):
    """Load per-run P99 values."""
    p99_values = []
    num_procs = 2 if workload_type == 'lc' else 4
    for run in range(NUM_RUNS):
        run_latencies = []
        for proc in range(num_procs):
            csv_file = DATA_DIR / f"{mode}_run{run}_{workload_type}_{proc}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                run_latencies.extend(df['launch_latency_ms'].values * 1000)
        if run_latencies:
            p99_values.append(np.percentile(run_latencies, 99))
    return np.array(p99_values)

# =============================================================================
# Figure 1: CCDF (Complementary CDF) - Best for showing tail differences
# =============================================================================
def plot_ccdf():
    """Plot CCDF on log-log scale to emphasize tail behavior."""
    native_lc = load_data("native", "lc")
    policy_lc = load_data("policy", "lc")

    fig, ax = plt.subplots(figsize=(6, 4))

    for data, label, color, ls in [
        (native_lc, "Native", "#E74C3C", "--"),
        (policy_lc, "BPF Policy", "#2ECC71", "-"),
    ]:
        sorted_data = np.sort(data)
        ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # Avoid log(0)
        ccdf = np.maximum(ccdf, 1e-4)
        ax.plot(sorted_data, ccdf, label=label, color=color, linestyle=ls)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Launch Latency (µs)')
    ax.set_ylabel('CCDF (1 - CDF)')
    ax.set_title('LC Launch Latency Tail Distribution')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Add P99 reference line
    ax.axhline(y=0.01, color='gray', linestyle=':', alpha=0.7)
    ax.text(20, 0.012, 'P99', color='gray', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_ccdf.pdf", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig_ccdf.png", bbox_inches='tight', dpi=150)
    print("Saved fig_ccdf.pdf/png")
    plt.close()

# =============================================================================
# Figure 2: Per-run P99 Box Plot - Shows variance reduction clearly
# =============================================================================
def plot_per_run_p99_boxplot():
    """Box plot of per-run P99 values - best for showing variance reduction."""
    native_p99 = load_per_run_p99("native", "lc")
    policy_p99 = load_per_run_p99("policy", "lc")

    fig, ax = plt.subplots(figsize=(5, 4))

    # Box plot
    bp = ax.boxplot(
        [native_p99, policy_p99],
        labels=['Native', 'BPF Policy'],
        patch_artist=True,
        widths=0.5,
    )

    # Colors
    colors = ['#E74C3C', '#2ECC71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Scatter individual points
    for i, (data, color) in enumerate([(native_p99, '#E74C3C'), (policy_p99, '#2ECC71')]):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.6, color=color, s=30, zorder=3)

    ax.set_ylabel('Per-run P99 Latency (µs)')
    ax.set_title('LC P99 Latency Across 10 Runs')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics annotation
    native_mean = np.mean(native_p99)
    policy_mean = np.mean(policy_p99)
    native_std = np.std(native_p99)
    policy_std = np.std(policy_p99)

    stats_text = (
        f"Native:  mean={native_mean:.0f}µs, std={native_std:.0f}µs\n"
        f"Policy:  mean={policy_mean:.0f}µs, std={policy_std:.0f}µs\n"
        f"Reduction: {(1 - policy_mean/native_mean)*100:.1f}%"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_p99_boxplot.pdf", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig_p99_boxplot.png", bbox_inches='tight', dpi=150)
    print("Saved fig_p99_boxplot.pdf/png")
    plt.close()

# =============================================================================
# Figure 3: Combined figure with CDF + Tail zoom
# =============================================================================
def plot_cdf_with_tail():
    """Two-panel figure: full CDF + tail zoom."""
    native_lc = load_data("native", "lc")
    policy_lc = load_data("policy", "lc")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Full CDF
    for data, label, color, ls in [
        (native_lc, "Native", "#E74C3C", "--"),
        (policy_lc, "BPF Policy", "#2ECC71", "-"),
    ]:
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax1.plot(sorted_data, cdf, label=label, color=color, linestyle=ls)

    ax1.set_xscale('log')
    ax1.set_xlabel('Launch Latency (µs)')
    ax1.set_ylabel('CDF')
    ax1.set_title('(a) Full Distribution')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.02)
    ax1.axhline(y=0.99, color='gray', linestyle=':', alpha=0.7)

    # Right: Tail only (P95+)
    for data, label, color, ls in [
        (native_lc, "Native", "#E74C3C", "--"),
        (policy_lc, "BPF Policy", "#2ECC71", "-"),
    ]:
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # Only show P95+
        mask = cdf >= 0.95
        ax2.plot(sorted_data[mask], cdf[mask], label=label, color=color, linestyle=ls)

    ax2.set_xscale('log')
    ax2.set_xlabel('Launch Latency (µs)')
    ax2.set_ylabel('CDF')
    ax2.set_title('(b) Tail Distribution (P95+)')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.95, 1.002)
    ax2.axhline(y=0.99, color='gray', linestyle=':', alpha=0.7)
    ax2.text(ax2.get_xlim()[1] * 0.7, 0.988, 'P99', color='gray', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_cdf_tail.pdf", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig_cdf_tail.png", bbox_inches='tight', dpi=150)
    print("Saved fig_cdf_tail.pdf/png")
    plt.close()

# =============================================================================
# Figure 4: Bar chart comparing key metrics
# =============================================================================
def plot_summary_bars():
    """Bar chart summarizing key metrics."""
    native_p99 = load_per_run_p99("native", "lc")
    policy_p99 = load_per_run_p99("policy", "lc")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: P99 mean comparison
    means = [np.mean(native_p99), np.mean(policy_p99)]
    stds = [np.std(native_p99), np.std(policy_p99)]
    x = [0, 1]
    colors = ['#E74C3C', '#2ECC71']
    bars = ax1.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Native', 'BPF Policy'])
    ax1.set_ylabel('Per-run P99 Mean (µs)')
    ax1.set_title('(a) LC P99 Latency')
    ax1.set_yscale('log')

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, mean + std + 50,
                f'{mean:.0f}µs', ha='center', va='bottom', fontsize=10)

    # Right: P99 std comparison (variance)
    stds_only = [np.std(native_p99), np.std(policy_p99)]
    bars2 = ax2.bar(x, stds_only, color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Native', 'BPF Policy'])
    ax2.set_ylabel('Per-run P99 Std Dev (µs)')
    ax2.set_title('(b) LC P99 Variance')
    ax2.set_yscale('log')

    for bar, std in zip(bars2, stds_only):
        ax2.text(bar.get_x() + bar.get_width()/2, std + 20,
                f'{std:.0f}µs', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_summary.pdf", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig_summary.png", bbox_inches='tight', dpi=150)
    print("Saved fig_summary.pdf/png")
    plt.close()

def calc_throughput(mode):
    """Calculate throughput for BE workload."""
    total_kernels = 0
    max_end_time = 0
    min_start_time = float('inf')

    for run in range(NUM_RUNS):
        for proc in range(4):  # 4 BE processes
            csv_file = DATA_DIR / f"{mode}_run{run}_be_{proc}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                total_kernels += len(df)
                max_end_time = max(max_end_time, df['end_time_ms'].max())
                min_start_time = min(min_start_time, df['enqueue_time_ms'].min())

    duration_s = (max_end_time - min_start_time) / 1000
    return total_kernels / duration_s if duration_s > 0 else 0

def calc_throughput(mode):
    """Calculate throughput using same method as analyze_results.py.

    Throughput = total_kernels / total_kernel_duration(seconds)
    """
    total_kernels = 0
    total_duration_ms = 0

    for run in range(NUM_RUNS):
        for proc in range(4):  # 4 BE processes
            csv_file = DATA_DIR / f"{mode}_run{run}_be_{proc}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                total_kernels += len(df)
                total_duration_ms += df['duration_ms'].sum()

    if total_duration_ms > 0:
        return total_kernels / (total_duration_ms / 1000)  # kernels per second
    return 0

# =============================================================================
# Figure 5: Main result - LC P99 + BE Throughput (TWO PANEL - RECOMMENDED)
# =============================================================================
def plot_main_result():
    """Two-panel figure showing both LC P99 reduction and BE throughput preservation."""
    native_lc_p99 = load_per_run_p99("native", "lc")
    policy_lc_p99 = load_per_run_p99("policy", "lc")
    native_be_tput = calc_throughput("native")
    policy_be_tput = calc_throughput("policy")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['#E74C3C', '#2ECC71']

    # Panel (a): LC P99 - Bar chart
    native_mean = np.mean(native_lc_p99)
    policy_mean = np.mean(policy_lc_p99)
    reduction = (1 - policy_mean / native_mean) * 100

    x = [0, 1]
    bars1 = ax1.bar(x, [native_mean, policy_mean],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Native', 'BPF Policy'])
    ax1.set_ylabel('Launch Latency (µs)')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, native_mean * 1.15)

    # Add value labels on bars
    ax1.text(0, native_mean * 1.02, f'{native_mean:.0f}',
             ha='center', va='bottom', fontsize=14)
    ax1.text(1, policy_mean * 1.5, f'{policy_mean:.0f}',
             ha='center', va='bottom', fontsize=14)

    ax1.set_title(f'(a) LC P99 Launch Latency  [-{reduction:.0f}%]')

    # Panel (b): BE Throughput - Bar chart
    native_tput = native_be_tput  # Already a single value
    policy_tput = policy_be_tput
    tput_change = (policy_tput - native_tput) / native_tput * 100
    sign = '+' if tput_change >= 0 else ''

    bars2 = ax2.bar(x, [native_tput, policy_tput],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Native', 'BPF Policy'])
    ax2.set_ylabel('Throughput (kernels/s)')
    ax2.grid(True, alpha=0.3, axis='y')

    max_tput = max(native_tput, policy_tput)
    ax2.set_ylim(0, max_tput * 1.15)

    # Add value labels
    ax2.text(0, native_tput * 1.02,
             f'{native_tput:.2f}', ha='center', va='bottom', fontsize=14)
    ax2.text(1, policy_tput * 1.02,
             f'{policy_tput:.2f}', ha='center', va='bottom', fontsize=14)

    ax2.set_title(f'(b) BE Throughput  [{sign}{tput_change:.1f}%]')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_main_result.pdf", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig_main_result.png", bbox_inches='tight', dpi=150)
    print("Saved fig_main_result.pdf/png")
    plt.close()

# =============================================================================
# Figure 6: Normalized comparison (single panel)
# =============================================================================
def plot_normalized():
    """Single panel showing normalized comparison to baseline."""
    native_lc_p99 = load_per_run_p99("native", "lc")
    policy_lc_p99 = load_per_run_p99("policy", "lc")
    native_be_tput = calc_throughput("native")
    policy_be_tput = calc_throughput("policy")

    # Normalize to Native = 100%
    lc_p99_native = 100
    lc_p99_policy = np.mean(policy_lc_p99) / np.mean(native_lc_p99) * 100
    be_tput_native = 100
    be_tput_policy = policy_be_tput / native_be_tput * 100

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(2)
    width = 0.35

    bars1 = ax.bar(x - width/2, [lc_p99_native, be_tput_native], width,
                   label='Native', color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x + width/2, [lc_p99_policy, be_tput_policy], width,
                   label='BPF Policy', color='#2ECC71', alpha=0.8)

    ax.set_ylabel('Normalized to Native (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(['LC P99 Latency\n(lower is better)', 'BE Throughput\n(higher is better)'])
    ax.legend()
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    ax.text(0 + width/2, lc_p99_policy + 3, f'{lc_p99_policy:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(1 + width/2, be_tput_policy + 3, f'{be_tput_policy:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 130)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_normalized.pdf", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig_normalized.png", bbox_inches='tight', dpi=150)
    print("Saved fig_normalized.pdf/png")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate evaluation figures')
    parser.add_argument('--all', action='store_true', help='Generate all figures')
    parser.add_argument('--ccdf', action='store_true', help='Generate CCDF plot')
    parser.add_argument('--boxplot', action='store_true', help='Generate P99 boxplot')
    parser.add_argument('--cdf', action='store_true', help='Generate CDF with tail zoom')
    parser.add_argument('--summary', action='store_true', help='Generate summary bars')
    parser.add_argument('--normalized', action='store_true', help='Generate normalized comparison')
    args = parser.parse_args()

    # Load and print summary stats
    native_p99 = load_per_run_p99("native", "lc")
    policy_p99 = load_per_run_p99("policy", "lc")
    native_tput = calc_throughput("native")
    policy_tput = calc_throughput("policy")

    print(f"LC P99 - Native: mean={np.mean(native_p99):.1f}µs, std={np.std(native_p99):.1f}µs")
    print(f"LC P99 - Policy: mean={np.mean(policy_p99):.1f}µs, std={np.std(policy_p99):.1f}µs")
    print(f"LC P99 reduction: {(1 - np.mean(policy_p99)/np.mean(native_p99))*100:.1f}%")
    print()
    print(f"BE Throughput - Native: {native_tput:.2f} kernels/s")
    print(f"BE Throughput - Policy: {policy_tput:.2f} kernels/s")
    print(f"BE Throughput change: {(policy_tput/native_tput-1)*100:+.1f}%")
    print()

    # Always generate main result
    plot_main_result()

    # Optional figures
    if args.all or args.ccdf:
        plot_ccdf()
    if args.all or args.boxplot:
        plot_per_run_p99_boxplot()
    if args.all or args.cdf:
        plot_cdf_with_tail()
    if args.all or args.summary:
        plot_summary_bars()
    if args.all or args.normalized:
        plot_normalized()

    print("\nDone!")

if __name__ == "__main__":
    main()
