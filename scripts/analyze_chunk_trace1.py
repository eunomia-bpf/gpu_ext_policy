#!/usr/bin/env python3
"""
Analyze chunk trace data to understand memory access patterns and suggest optimal eviction policies.

This script analyzes the BPF trace data from chunk_trace to understand:
1. Chunk lifetime and reuse patterns
2. Access frequency and temporal locality
3. VA block access patterns
4. Optimal eviction policy recommendations
"""

import pandas as pd
import numpy as np
import argparse
import sys
from collections import defaultdict

def load_trace_data(filename):
    """Load trace data from CSV file."""
    print(f"Loading trace data from {filename}...")

    # Read CSV, skipping the summary lines at the end
    df = pd.read_csv(filename, on_bad_lines='skip', low_memory=False)

    # Convert time_ms to numeric, filter out non-numeric rows
    df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
    df = df[df['time_ms'].notna()].copy()

    # Convert cpu and va_page_index to numeric
    df['cpu'] = pd.to_numeric(df['cpu'], errors='coerce')
    df['va_page_index'] = pd.to_numeric(df['va_page_index'], errors='coerce')

    print(f"Loaded {len(df)} valid events")
    print(f"Time range: {df['time_ms'].min():.0f}ms - {df['time_ms'].max():.0f}ms ({(df['time_ms'].max() - df['time_ms'].min())/1000:.1f}s)")

    return df

def analyze_hook_distribution(df):
    """Analyze the distribution of different hook types."""
    print("\n" + "="*80)
    print("HOOK TYPE DISTRIBUTION")
    print("="*80)

    hook_counts = df['hook_type'].value_counts()
    total = len(df)

    for hook, count in hook_counts.items():
        percentage = (count / total) * 100
        print(f"{hook:20s}: {count:10d} ({percentage:5.1f}%)")

    return hook_counts

def analyze_chunk_lifecycle(df):
    """Analyze chunk lifecycle: from ACTIVATE to last access."""
    print("\n" + "="*80)
    print("CHUNK LIFECYCLE ANALYSIS")
    print("="*80)

    # Filter out EVICTION_PREPARE events (they don't have chunk addresses)
    chunk_events = df[df['hook_type'] != 'EVICTION_PREPARE'].copy()

    # Group by chunk address
    chunk_groups = chunk_events.groupby('chunk_addr')

    lifetimes = []
    reuse_counts = []
    first_access_type = []

    for chunk_addr, events in chunk_groups:
        events_sorted = events.sort_values('time_ms')

        if len(events_sorted) >= 1:
            # Calculate lifetime (first to last access)
            lifetime = events_sorted['time_ms'].max() - events_sorted['time_ms'].min()
            lifetimes.append(lifetime)

            # Count number of accesses
            reuse_counts.append(len(events_sorted))

            # Record first access type
            first_access_type.append(events_sorted.iloc[0]['hook_type'])

    lifetimes = np.array(lifetimes)
    reuse_counts = np.array(reuse_counts)

    print(f"Total unique chunks: {len(chunk_groups)}")
    print(f"Chunks with accesses: {len(lifetimes)}")

    # First access type distribution
    from collections import Counter
    first_access_counter = Counter(first_access_type)
    print(f"\nFirst access type distribution:")
    for access_type, count in first_access_counter.most_common():
        pct = (count / len(first_access_type)) * 100
        print(f"  {access_type:15s}: {count:6d} ({pct:5.1f}%)")

    if len(lifetimes) > 0:
        print(f"\nLifetime statistics (ms):")
        print(f"  Mean:   {lifetimes.mean():.2f}")
        print(f"  Median: {np.median(lifetimes):.2f}")
        print(f"  Min:    {lifetimes.min():.2f}")
        print(f"  Max:    {lifetimes.max():.2f}")
        print(f"  StdDev: {lifetimes.std():.2f}")

        print(f"\nReuse count statistics:")
        print(f"  Mean:   {reuse_counts.mean():.2f}")
        print(f"  Median: {np.median(reuse_counts):.2f}")
        print(f"  Min:    {reuse_counts.min():.0f}")
        print(f"  Max:    {reuse_counts.max():.0f}")

        # Distribution of reuse counts
        print(f"\nReuse distribution:")
        for i in [1, 2, 5, 10, 20, 50, 100]:
            count = np.sum(reuse_counts >= i)
            if len(reuse_counts) > 0:
                pct = (count / len(reuse_counts)) * 100
                print(f"  >={i:3d} accesses: {count:6d} chunks ({pct:5.1f}%)")

    return lifetimes, reuse_counts

def analyze_temporal_patterns(df):
    """Analyze temporal access patterns."""
    print("\n" + "="*80)
    print("TEMPORAL ACCESS PATTERNS")
    print("="*80)

    chunk_events = df[df['hook_type'] != 'EVICTION_PREPARE'].copy()

    # Calculate inter-access times for each chunk
    inter_access_times = []

    for chunk_addr, events in chunk_events.groupby('chunk_addr'):
        if len(events) > 1:
            times = events['time_ms'].sort_values().values
            gaps = np.diff(times)
            inter_access_times.extend(gaps)

    if len(inter_access_times) > 0:
        inter_access_times = np.array(inter_access_times)

        print(f"Total inter-access intervals: {len(inter_access_times)}")
        print(f"\nInter-access time statistics (ms):")
        print(f"  Mean:   {inter_access_times.mean():.2f}")
        print(f"  Median: {np.median(inter_access_times):.2f}")
        print(f"  Min:    {inter_access_times.min():.2f}")
        print(f"  Max:    {inter_access_times.max():.2f}")

        # Show distribution
        print(f"\nTemporal locality (inter-access time):")
        for threshold in [1, 10, 100, 1000, 10000]:
            count = np.sum(inter_access_times <= threshold)
            pct = (count / len(inter_access_times)) * 100
            print(f"  <={threshold:5d}ms: {count:8d} ({pct:5.1f}%)")
    else:
        inter_access_times = np.array([])

    return inter_access_times

def analyze_va_block_patterns(df):
    """Analyze VA block access patterns."""
    print("\n" + "="*80)
    print("VA BLOCK ACCESS PATTERNS")
    print("="*80)

    # Filter events with valid VA blocks (not empty string and not EVICTION_PREPARE)
    va_events = df[(df['va_block'].notna()) &
                   (df['va_block'] != '') &
                   (df['hook_type'] != 'EVICTION_PREPARE')].copy()

    print(f"Events with VA block info: {len(va_events)} ({len(va_events)/len(df)*100:.1f}%)")

    if len(va_events) > 0:
        # Count accesses per VA block
        va_block_accesses = va_events.groupby('va_block').size()

        print(f"\nUnique VA blocks: {len(va_block_accesses)}")
        print(f"Accesses per VA block:")
        print(f"  Mean:   {va_block_accesses.mean():.2f}")
        print(f"  Median: {va_block_accesses.median():.2f}")
        print(f"  Min:    {va_block_accesses.min():.0f}")
        print(f"  Max:    {va_block_accesses.max():.0f}")

        # Hot VA blocks (top accessed)
        print(f"\nTop 10 most accessed VA blocks:")
        for va_block, count in va_block_accesses.nlargest(10).items():
            pct = (count / len(va_events)) * 100
            print(f"  {va_block}: {count:6d} accesses ({pct:5.1f}%)")

def analyze_eviction_pressure(df):
    """Analyze eviction pressure and frequency."""
    print("\n" + "="*80)
    print("EVICTION PRESSURE ANALYSIS")
    print("="*80)

    eviction_events = df[df['hook_type'] == 'EVICTION_PREPARE']

    if len(eviction_events) > 0:
        print(f"Total eviction events: {len(eviction_events)}")

        # Calculate time between evictions
        eviction_times = eviction_events['time_ms'].values
        if len(eviction_times) > 1:
            eviction_gaps = np.diff(eviction_times)

            print(f"\nTime between evictions (ms):")
            print(f"  Mean:   {eviction_gaps.mean():.2f}")
            print(f"  Median: {np.median(eviction_gaps):.2f}")
            print(f"  Min:    {eviction_gaps.min():.2f}")
            print(f"  Max:    {eviction_gaps.max():.2f}")

            # Eviction rate over time
            total_time_s = (df['time_ms'].max() - df['time_ms'].min()) / 1000
            if total_time_s > 0:
                eviction_rate = len(eviction_events) / total_time_s
                print(f"\nEviction rate: {eviction_rate:.2f} evictions/second")

                # Memory pressure indicator
                activate_count = len(df[df['hook_type'] == 'ACTIVATE'])
                if activate_count > 0:
                    eviction_ratio = len(eviction_events) / activate_count
                    print(f"Eviction/Activation ratio: {eviction_ratio:.2f}")
                    if eviction_ratio > 0.9:
                        print("  → HIGH memory pressure (almost 1:1 eviction/activation)")
                    elif eviction_ratio > 0.5:
                        print("  → MODERATE memory pressure")
                    else:
                        print("  → LOW memory pressure")

def recommend_policy(lifetimes, reuse_counts, inter_access_times):
    """Recommend optimal eviction policy based on analysis."""
    print("\n" + "="*80)
    print("EVICTION POLICY RECOMMENDATIONS")
    print("="*80)

    if len(lifetimes) == 0 or len(reuse_counts) == 0:
        print("Insufficient data for recommendations")
        return

    # Calculate key metrics
    mean_lifetime = lifetimes.mean()
    median_reuse = np.median(reuse_counts)
    mean_reuse = reuse_counts.mean()
    high_reuse_ratio = np.sum(reuse_counts >= 5) / len(reuse_counts)

    if len(inter_access_times) > 0:
        temporal_locality = np.sum(inter_access_times <= 1000) / len(inter_access_times)
        median_gap = np.median(inter_access_times)
    else:
        temporal_locality = 0
        median_gap = 0

    print("\nKey metrics:")
    print(f"  Mean chunk lifetime:        {mean_lifetime:.2f} ms")
    print(f"  Mean reuse count:           {mean_reuse:.2f}")
    print(f"  Median reuse count:         {median_reuse:.0f}")
    print(f"  High reuse ratio (>=5x):    {high_reuse_ratio*100:.1f}%")
    print(f"  Temporal locality (<1s):    {temporal_locality*100:.1f}%")
    print(f"  Median inter-access gap:    {median_gap:.2f} ms")

    print("\n" + "-"*80)
    print("Policy recommendations:")
    print("-"*80)

    # LRU vs LFU vs FIFO analysis
    if high_reuse_ratio > 0.8 and temporal_locality > 0.8:
        print("\n✓ LRU (Least Recently Used) - HIGHLY RECOMMENDED")
        print("  Rationale:")
        print("    - Very high temporal locality (>80%)")
        print("    - High reuse ratio (>80%)")
        print("    - Chunks accessed recently are likely to be accessed again soon")
        print("  Expected benefit: 25-35% reduction in evictions vs FIFO")

    elif high_reuse_ratio > 0.6 and temporal_locality > 0.6:
        print("\n✓ LRU (Least Recently Used) - RECOMMENDED")
        print("  Rationale:")
        print("    - Good temporal locality")
        print("    - Good reuse patterns")
        print("  Expected benefit: 15-25% reduction in evictions vs FIFO")

    elif high_reuse_ratio > 0.5:
        print("\n✓ LFU (Least Frequently Used) - RECOMMENDED")
        print("  Rationale:")
        print("    - Moderate reuse ratio")
        print("    - Frequency-based access patterns more important than recency")
        print("  Expected benefit: 10-20% reduction in evictions vs FIFO")

    else:
        print("\n✓ FIFO (First In First Out) - BASELINE")
        print("  Rationale:")
        print("    - Low temporal locality and reuse patterns")
        print("    - Simple policy may be adequate")
        print("  Consider: Current default policy should work reasonably well")

    # Additional recommendations based on specific patterns
    print("\n" + "-"*80)
    print("Additional insights:")
    print("-"*80)

    if mean_reuse > 50:
        print("\n→ Very high reuse detected (mean > 50):")
        print("  - Consider implementing multi-queue approach (hot/warm/cold)")
        print("  - Adaptive aging for frequently accessed chunks")
        print("  - May benefit from ARC (Adaptive Replacement Cache)")

    if temporal_locality > 0.7:
        print("\n→ Strong temporal locality detected (>70%):")
        print("  - Time-based eviction will significantly outperform frequency-based")
        print("  - LRU or CLOCK algorithm highly recommended")
        print("  - Consider implementing working set detection")

    if median_gap < 50:
        print("\n→ Very short inter-access times detected (median < 50ms):")
        print("  - Chunks are being accessed in tight loops")
        print("  - Consider longer eviction delays or hot-set protection")
        print("  - May benefit from chunk pinning for hot data")

    if high_reuse_ratio < 0.3:
        print("\n→ Low reuse detected (<30%):")
        print("  - Many chunks accessed only once or twice")
        print("  - Consider evicting single-access chunks more aggressively")
        print("  - Sequential scan detection may help")

def main():
    parser = argparse.ArgumentParser(description='Analyze chunk trace data')
    parser.add_argument('input', nargs='?', default='/tmp/chunk_trace_10s.log',
                        help='Input CSV file (default: /tmp/chunk_trace_10s.log)')
    args = parser.parse_args()

    try:
        # Load data
        df = load_trace_data(args.input)

        # Run analyses
        analyze_hook_distribution(df)
        lifetimes, reuse_counts = analyze_chunk_lifecycle(df)
        inter_access_times = analyze_temporal_patterns(df)
        analyze_va_block_patterns(df)
        analyze_eviction_pressure(df)

        # Recommendations
        recommend_policy(lifetimes, reuse_counts, inter_access_times)

        print("\n" + "="*80)
        print("Analysis complete!")
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
