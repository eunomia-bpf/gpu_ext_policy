#!/usr/bin/env python3
"""
Analyze chunk_trace CSV output

This script analyzes the CSV output from chunk_trace tool which tracks:
- GPU memory chunk lifecycle events (ACTIVATE, POPULATE, DEPOPULATE, EVICTION_PREPARE)
- VA block mapping information (when available)

CSV Format:
time_ms,hook_type,cpu,chunk_addr,list_addr,va_block,va_start,va_end,va_page_index
"""

import sys
import csv
import argparse
from collections import defaultdict, Counter
from statistics import mean, median


def parse_hex(value):
    """Parse hex value, return 0 if empty or invalid"""
    if not value or value == '':
        return 0
    try:
        return int(value, 16) if value.startswith('0x') else int(value)
    except ValueError:
        return 0


def analyze_csv(filename):
    """Main analysis function"""

    # Data structures
    events = []
    hook_counts = Counter()
    chunk_events = defaultdict(list)  # chunk_addr -> list of (time, hook_type)
    va_block_stats = {
        'total': 0,
        'with_va': 0,
        'without_va': 0,
        'va_blocks': set(),  # unique VA block addresses
        'va_ranges': set(),  # unique (va_start, va_end) pairs
    }

    # Read CSV
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip summary lines
            if not row.get('time_ms') or not row['time_ms'][0].isdigit():
                break

            try:
                time_ms = int(row['time_ms'])
                hook_type = row['hook_type']
                cpu = int(row['cpu'])
                chunk_addr = parse_hex(row['chunk_addr'])
                list_addr = parse_hex(row['list_addr'])
                va_block = parse_hex(row['va_block'])
                va_start = parse_hex(row['va_start'])
                va_end = parse_hex(row['va_end'])
                va_page_index = int(row['va_page_index']) if row['va_page_index'] else 0

                event = {
                    'time_ms': time_ms,
                    'hook_type': hook_type,
                    'cpu': cpu,
                    'chunk_addr': chunk_addr,
                    'list_addr': list_addr,
                    'va_block': va_block,
                    'va_start': va_start,
                    'va_end': va_end,
                    'va_page_index': va_page_index,
                }

                events.append(event)
                hook_counts[hook_type] += 1

                # Track chunk lifecycle
                if hook_type != 'EVICTION_PREPARE' and chunk_addr != 0:
                    chunk_events[chunk_addr].append((time_ms, hook_type))

                # Track VA block info
                if hook_type != 'EVICTION_PREPARE':
                    va_block_stats['total'] += 1
                    if va_block != 0:
                        va_block_stats['with_va'] += 1
                        va_block_stats['va_blocks'].add(va_block)
                        if va_start != 0 and va_end != 0:
                            va_block_stats['va_ranges'].add((va_start, va_end))
                    else:
                        va_block_stats['without_va'] += 1

            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid row: {e}", file=sys.stderr)
                continue

    if not events:
        print("Error: No valid events found in CSV", file=sys.stderr)
        return

    # Print analysis
    print("=" * 80)
    print("CHUNK TRACE ANALYSIS")
    print("=" * 80)
    print()

    # Basic statistics
    print("BASIC STATISTICS")
    print("-" * 80)
    print(f"Total events:              {len(events):,}")
    print(f"Time range:                {events[0]['time_ms']} - {events[-1]['time_ms']} ms")
    print(f"Duration:                  {events[-1]['time_ms'] - events[0]['time_ms']} ms")
    print(f"Unique chunks tracked:     {len(chunk_events):,}")
    print()

    # Hook type distribution
    print("HOOK TYPE DISTRIBUTION")
    print("-" * 80)
    total_hooks = sum(hook_counts.values())
    for hook, count in sorted(hook_counts.items()):
        pct = 100.0 * count / total_hooks if total_hooks > 0 else 0
        print(f"{hook:20s} {count:8,} ({pct:5.1f}%)")
    print("-" * 80)
    print(f"{'TOTAL':20s} {total_hooks:8,}")
    print()

    # VA block tracking
    print("VA BLOCK TRACKING")
    print("-" * 80)
    if va_block_stats['total'] > 0:
        coverage = 100.0 * va_block_stats['with_va'] / va_block_stats['total']
        print(f"Events with VA info:       {va_block_stats['with_va']:,} ({coverage:.1f}%)")
        print(f"Events without VA info:    {va_block_stats['without_va']:,}")
        print(f"Unique VA blocks:          {len(va_block_stats['va_blocks']):,}")
        print(f"Unique VA ranges:          {len(va_block_stats['va_ranges']):,}")

        if va_block_stats['with_va'] == 0:
            print()
            print("⚠️  WARNING: No VA block information captured!")
            print("   Possible reasons:")
            print("   1. Struct offset incorrect (va_block pointer not at offset 40)")
            print("   2. All chunks have NULL va_block pointers")
            print("   3. Memory read failed in BPF program")
    else:
        print("No VA block tracking data (all EVICTION_PREPARE events)")
    print()

    # Chunk lifecycle analysis
    print("CHUNK LIFECYCLE ANALYSIS")
    print("-" * 80)

    activate_counts = Counter()
    populate_counts = Counter()
    chunk_lifetimes = []

    for chunk_addr, lifecycle in chunk_events.items():
        # Sort by time
        lifecycle.sort()

        # Count ACTIVATE and POPULATE
        activates = sum(1 for _, hook in lifecycle if hook == 'ACTIVATE')
        populates = sum(1 for _, hook in lifecycle if hook == 'POPULATE')

        activate_counts[activates] += 1
        populate_counts[populates] += 1

        # Calculate lifetime
        if len(lifecycle) > 1:
            lifetime = lifecycle[-1][0] - lifecycle[0][0]
            chunk_lifetimes.append(lifetime)

    print("ACTIVATE per chunk:")
    for count in sorted(activate_counts.keys()):
        num_chunks = activate_counts[count]
        print(f"  {count:2d} ACTIVATEs: {num_chunks:6,} chunks")

    print()
    print("POPULATE per chunk:")
    for count in sorted(populate_counts.keys())[:10]:  # Top 10
        num_chunks = populate_counts[count]
        print(f"  {count:2d} POPULATEs: {num_chunks:6,} chunks")
    if len(populate_counts) > 10:
        print(f"  ... and {len(populate_counts) - 10} more")

    if chunk_lifetimes:
        print()
        print("Chunk lifetime statistics (first to last event):")
        print(f"  Mean:                  {mean(chunk_lifetimes):.2f} ms")
        print(f"  Median:                {median(chunk_lifetimes):.2f} ms")
        print(f"  Min:                   {min(chunk_lifetimes):.2f} ms")
        print(f"  Max:                   {max(chunk_lifetimes):.2f} ms")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze chunk_trace CSV output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s trace.csv
  %(prog)s /tmp/chunk_trace_va_5s.csv
        """
    )
    parser.add_argument('csv_file', help='CSV file from chunk_trace')

    args = parser.parse_args()

    try:
        analyze_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
