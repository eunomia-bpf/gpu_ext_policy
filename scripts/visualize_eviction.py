#!/usr/bin/env python3
"""
Visualize chunk eviction patterns from trace data.

This script analyzes the chunk_trace CSV data to understand:
1. Chunk lifecycle: activation, access, and eviction timing
2. Eviction patterns: which chunks get evicted and when
3. Temporal patterns: access frequency and recency
4. Policy effectiveness: LRU vs FIFO behavior
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import argparse
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_and_prepare_data(filename):
    """Load CSV and prepare data structures."""
    print(f"[1/6] Loading data from {filename}...")

    df = pd.read_csv(filename, on_bad_lines='skip', low_memory=False)
    print(f"  → Read {len(df)} rows")

    print(f"  → Converting time_ms column...")
    df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
    df = df[df['time_ms'].notna()].copy()

    print(f"  → Loaded {len(df)} valid events")
    print(f"  → Time range: {df['time_ms'].min():.0f} - {df['time_ms'].max():.0f} ms")

    return df

def analyze_chunk_lifecycle(df):
    """Analyze when chunks are activated, used, and evicted."""
    print(f"\n[2/6] Analyzing chunk lifecycle...")

    chunk_events = df[df['hook_type'] != 'EVICTION_PREPARE'].copy()
    eviction_events = df[df['hook_type'] == 'EVICTION_PREPARE'].copy()

    print(f"  → Chunk events: {len(chunk_events):,}")
    print(f"  → Eviction events: {len(eviction_events):,}")

    # Track each chunk's lifecycle (simplified for memory efficiency)
    chunk_lifecycle = defaultdict(lambda: {
        'first_seen': None,
        'last_seen': None,
        'activate_time': None,
        'access_count': 0,
        'evictions_during_lifetime': 0
    })

    # Process chunk events
    print(f"  → Processing chunk events...")
    total = len(chunk_events)
    for idx, row in enumerate(chunk_events.itertuples()):
        if idx % 100000 == 0:
            print(f"    Progress: {idx:,}/{total:,} ({idx/total*100:.1f}%)")

        chunk_addr = row.chunk_addr
        time_ms = row.time_ms
        hook_type = row.hook_type

        if chunk_lifecycle[chunk_addr]['first_seen'] is None:
            chunk_lifecycle[chunk_addr]['first_seen'] = time_ms

        if hook_type == 'ACTIVATE':
            chunk_lifecycle[chunk_addr]['activate_time'] = time_ms

        chunk_lifecycle[chunk_addr]['access_count'] += 1
        chunk_lifecycle[chunk_addr]['last_seen'] = time_ms

    print(f"  → Unique chunks: {len(chunk_lifecycle):,}")

    # Simplified eviction pressure calculation (skip for large datasets)
    if len(eviction_events) < 100000:
        print(f"  → Calculating eviction pressure...")
        for idx, row in enumerate(eviction_events.itertuples()):
            if idx % 10000 == 0:
                print(f"    Progress: {idx:,}/{len(eviction_events):,}")
            eviction_time = row.time_ms
            for chunk_addr, info in chunk_lifecycle.items():
                if (info['first_seen'] <= eviction_time <= info['last_seen']):
                    info['evictions_during_lifetime'] += 1
    else:
        print(f"  → Skipping eviction pressure (too many evictions)")

    return chunk_lifecycle, eviction_events

def plot_eviction_timeline(df, output_file='eviction_timeline.png'):
    """Plot eviction events over time."""
    print(f"\n[3/6] Generating eviction timeline plot...")

    eviction_events = df[df['hook_type'] == 'EVICTION_PREPARE'].copy()

    if len(eviction_events) == 0:
        print("No eviction events found")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Eviction events over time
    eviction_times = eviction_events['time_ms'].values
    eviction_counts = np.arange(1, len(eviction_times) + 1)

    ax1.plot(eviction_times, eviction_counts, linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Cumulative Evictions')
    ax1.set_title('Cumulative Eviction Events Over Time')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Eviction rate (events per second)
    window_ms = 100  # 100ms window
    bins = np.arange(0, eviction_times.max() + window_ms, window_ms)
    hist, edges = np.histogram(eviction_times, bins=bins)
    eviction_rate = hist / (window_ms / 1000)  # events per second

    ax2.plot(edges[:-1], eviction_rate, linewidth=1, alpha=0.8)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Eviction Rate (events/sec)')
    ax2.set_title(f'Eviction Rate Over Time (window: {window_ms}ms)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_file}")
    plt.close()

def plot_chunk_reuse_distribution(chunk_lifecycle, output_file='chunk_reuse.png'):
    """Plot distribution of chunk reuse patterns."""
    print(f"\n[4/6] Generating chunk reuse distribution plot...")

    reuse_counts = [info['access_count'] for info in chunk_lifecycle.values()]
    lifetimes = [(info['last_seen'] - info['first_seen'])
                 for info in chunk_lifecycle.values()]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Reuse count histogram
    axes[0, 0].hist(reuse_counts, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Number of Accesses')
    axes[0, 0].set_ylabel('Number of Chunks')
    axes[0, 0].set_title('Distribution of Chunk Reuse Counts')
    axes[0, 0].axvline(np.median(reuse_counts), color='red', linestyle='--',
                       label=f'Median: {np.median(reuse_counts):.0f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Lifetime histogram
    axes[0, 1].hist(lifetimes, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Lifetime (ms)')
    axes[0, 1].set_ylabel('Number of Chunks')
    axes[0, 1].set_title('Distribution of Chunk Lifetimes')
    axes[0, 1].axvline(np.median(lifetimes), color='red', linestyle='--',
                       label=f'Median: {np.median(lifetimes):.0f}ms')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Reuse vs Lifetime scatter
    sample_size = min(5000, len(reuse_counts))
    sample_indices = np.random.choice(len(reuse_counts), sample_size, replace=False)
    sample_reuse = [reuse_counts[i] for i in sample_indices]
    sample_lifetime = [lifetimes[i] for i in sample_indices]

    axes[1, 0].scatter(sample_lifetime, sample_reuse, alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Lifetime (ms)')
    axes[1, 0].set_ylabel('Number of Accesses')
    axes[1, 0].set_title(f'Reuse vs Lifetime (sample: {sample_size} chunks)')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Eviction pressure
    eviction_pressure = [info['evictions_during_lifetime']
                         for info in chunk_lifecycle.values()]
    axes[1, 1].hist(eviction_pressure, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Eviction Events During Lifetime')
    axes[1, 1].set_ylabel('Number of Chunks')
    axes[1, 1].set_title('Eviction Pressure per Chunk')
    axes[1, 1].axvline(np.median(eviction_pressure), color='red', linestyle='--',
                       label=f'Median: {np.median(eviction_pressure):.0f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_file}")
    plt.close()

def plot_temporal_access_heatmap(df, output_file='temporal_heatmap.png'):
    """Plot temporal access patterns as a heatmap."""
    print(f"\n[5/6] Generating temporal access heatmap...")

    chunk_events = df[df['hook_type'] != 'EVICTION_PREPARE'].copy()

    # Sample chunks for visualization (too many chunks will make it unreadable)
    unique_chunks = chunk_events['chunk_addr'].unique()
    sample_size = min(100, len(unique_chunks))
    sampled_chunks = np.random.choice(unique_chunks, sample_size, replace=False)

    # Create time bins (e.g., 100ms buckets)
    max_time = df['time_ms'].max()
    time_bins = np.linspace(0, max_time, 100)

    # Build access matrix: chunks x time_bins
    access_matrix = np.zeros((sample_size, len(time_bins) - 1))

    for i, chunk_addr in enumerate(sampled_chunks):
        chunk_times = chunk_events[chunk_events['chunk_addr'] == chunk_addr]['time_ms'].values
        hist, _ = np.histogram(chunk_times, bins=time_bins)
        access_matrix[i, :] = hist

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(access_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    # Set labels
    ax.set_xlabel('Time Bin (ms)')
    ax.set_ylabel('Chunk ID (sampled)')
    ax.set_title(f'Temporal Access Pattern Heatmap ({sample_size} sampled chunks)')

    # Set x-axis ticks to show time
    num_ticks = 10
    tick_positions = np.linspace(0, len(time_bins) - 2, num_ticks)
    tick_labels = [f'{time_bins[int(pos)]:.0f}' for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Access Count')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_file}")
    plt.close()

def plot_access_pattern_timeline(df, output_file='access_pattern.png'):
    """Plot different hook types over time."""
    print(f"\n[6/6] Generating access pattern timeline...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    hook_types = ['ACTIVATE', 'POPULATE', 'EVICTION_PREPARE']
    colors = ['blue', 'green', 'red']

    for ax, hook_type, color in zip(axes, hook_types, colors):
        events = df[df['hook_type'] == hook_type]
        times = events['time_ms'].values

        # Create histogram
        bins = np.linspace(0, df['time_ms'].max(), 200)
        hist, edges = np.histogram(times, bins=bins)

        # Plot as area
        ax.fill_between(edges[:-1], hist, alpha=0.5, color=color, label=hook_type)
        ax.plot(edges[:-1], hist, linewidth=1, color=color, alpha=0.8)

        ax.set_ylabel(f'{hook_type}\nCount')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ms)')
    axes[0].set_title('Hook Event Timeline')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_file}")
    plt.close()

def plot_inter_access_distribution(df, output_file='inter_access.png'):
    """Plot inter-access time distribution."""
    print(f"\n[Bonus] Generating inter-access time distribution...")

    chunk_events = df[df['hook_type'] != 'EVICTION_PREPARE'].copy()

    # Calculate inter-access times for each chunk
    inter_access_times = []

    for chunk_addr in chunk_events['chunk_addr'].unique():
        chunk_times = chunk_events[chunk_events['chunk_addr'] == chunk_addr]['time_ms'].values
        if len(chunk_times) > 1:
            chunk_times = np.sort(chunk_times)
            gaps = np.diff(chunk_times)
            inter_access_times.extend(gaps)

    inter_access_times = np.array(inter_access_times)

    if len(inter_access_times) == 0:
        print("No inter-access times found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Histogram
    axes[0, 0].hist(inter_access_times, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Inter-access Time (ms)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Inter-access Time Distribution')
    axes[0, 0].axvline(np.median(inter_access_times), color='red', linestyle='--',
                       label=f'Median: {np.median(inter_access_times):.2f}ms')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Log-scale histogram
    axes[0, 1].hist(inter_access_times, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Inter-access Time (ms)')
    axes[0, 1].set_ylabel('Frequency (log scale)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Inter-access Time Distribution (log scale)')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: CDF
    sorted_times = np.sort(inter_access_times)
    cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
    axes[1, 0].plot(sorted_times, cdf, linewidth=2)
    axes[1, 0].set_xlabel('Inter-access Time (ms)')
    axes[1, 0].set_ylabel('CDF')
    axes[1, 0].set_title('Cumulative Distribution Function')
    axes[1, 0].grid(True, alpha=0.3)

    # Add percentile lines
    for percentile in [50, 90, 95, 99]:
        value = np.percentile(inter_access_times, percentile)
        axes[1, 0].axvline(value, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].text(value, 0.5, f'P{percentile}', rotation=90, va='bottom')

    # Plot 4: Statistics box
    axes[1, 1].axis('off')
    stats_text = f"""
    Inter-access Time Statistics:

    Count:      {len(inter_access_times):,}
    Mean:       {inter_access_times.mean():.2f} ms
    Median:     {np.median(inter_access_times):.2f} ms
    Std Dev:    {inter_access_times.std():.2f} ms
    Min:        {inter_access_times.min():.2f} ms
    Max:        {inter_access_times.max():.2f} ms

    Percentiles:
    P50:        {np.percentile(inter_access_times, 50):.2f} ms
    P90:        {np.percentile(inter_access_times, 90):.2f} ms
    P95:        {np.percentile(inter_access_times, 95):.2f} ms
    P99:        {np.percentile(inter_access_times, 99):.2f} ms

    Temporal Locality:
    ≤ 1ms:      {np.sum(inter_access_times <= 1) / len(inter_access_times) * 100:.1f}%
    ≤ 10ms:     {np.sum(inter_access_times <= 10) / len(inter_access_times) * 100:.1f}%
    ≤ 100ms:    {np.sum(inter_access_times <= 100) / len(inter_access_times) * 100:.1f}%
    ≤ 1000ms:   {np.sum(inter_access_times <= 1000) / len(inter_access_times) * 100:.1f}%
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize chunk eviction patterns')
    parser.add_argument('input', nargs='?', default='/tmp/chunk_trace_10s.log',
                        help='Input CSV file (default: /tmp/chunk_trace_10s.log)')
    parser.add_argument('-o', '--output-dir', default='/tmp',
                        help='Output directory for plots (default: /tmp)')
    args = parser.parse_args()

    try:
        # Load data
        df = load_and_prepare_data(args.input)

        # Analyze lifecycle
        chunk_lifecycle, eviction_events = analyze_chunk_lifecycle(df)

        print(f"\nTotal unique chunks: {len(chunk_lifecycle)}")
        print(f"Total eviction events: {len(eviction_events)}")

        # Generate visualizations
        import os
        output_dir = args.output_dir

        plot_eviction_timeline(df, os.path.join(output_dir, 'eviction_timeline.png'))
        plot_access_pattern_timeline(df, os.path.join(output_dir, 'access_pattern.png'))
        plot_chunk_reuse_distribution(chunk_lifecycle, os.path.join(output_dir, 'chunk_reuse.png'))
        # Skip memory-intensive plots for large datasets
        # plot_temporal_access_heatmap(df, os.path.join(output_dir, 'temporal_heatmap.png'))
        # plot_inter_access_distribution(df, os.path.join(output_dir, 'inter_access.png'))

        print("\n" + "="*80)
        print("Visualization complete!")
        print(f"All plots saved to: {output_dir}")
        print("="*80)

    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
