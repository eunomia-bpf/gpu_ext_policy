#!/usr/bin/env python3
"""
Analyze and visualize GPU memory access and thread divergence benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    try:
        df = pd.read_csv('results.csv')
    except FileNotFoundError:
        print("Error: results.csv not found. Run ./bench first.")
        sys.exit(1)

    # Separate memory and divergence tests
    mem_df = df[df['test_type'] == 'memory']
    div_df = df[df['test_type'] == 'divergence']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Memory Coalescing - Bandwidth vs Stride
    ax1 = axes[0]
    ax1.bar(range(len(mem_df)), mem_df['bandwidth_gbps'], color='steelblue')
    ax1.set_xticks(range(len(mem_df)))
    ax1.set_xticklabels([f"stride={p}" for p in mem_df['parameter']])
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_xlabel('Access Pattern')
    ax1.set_title('Memory Coalescing: Bandwidth vs Stride')
    ax1.grid(axis='y', alpha=0.3)

    # Add efficiency annotations
    for i, (_, row) in enumerate(mem_df.iterrows()):
        ax1.annotate(f"{row['efficiency']*100:.1f}%",
                     (i, row['bandwidth_gbps']),
                     ha='center', va='bottom', fontsize=9)

    # Plot 2: Thread Divergence - Slowdown vs Factor
    ax2 = axes[1]
    baseline_time = div_df.iloc[0]['time_ms']
    slowdowns = div_df['time_ms'] / baseline_time

    ax2.bar(range(len(div_df)), slowdowns, color='coral')
    ax2.set_xticks(range(len(div_df)))
    ax2.set_xticklabels([f"div={p}" for p in div_df['parameter']])
    ax2.set_ylabel('Slowdown (x)')
    ax2.set_xlabel('Divergence Factor')
    ax2.set_title('Thread Divergence: Slowdown vs Factor')
    ax2.grid(axis='y', alpha=0.3)

    # Add slowdown annotations
    for i, s in enumerate(slowdowns):
        ax2.annotate(f"{s:.1f}x",
                     (i, s),
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print("Chart saved to: results.png")

    # Print summary
    print("\n=== Summary ===")
    print(f"Memory: stride=1 → stride=32 bandwidth drop: "
          f"{mem_df.iloc[0]['bandwidth_gbps']:.1f} → {mem_df.iloc[-1]['bandwidth_gbps']:.1f} GB/s "
          f"({mem_df.iloc[-1]['efficiency']/mem_df.iloc[0]['efficiency']*100:.1f}% of baseline)")
    print(f"Divergence: div=1 → div=32 slowdown: {slowdowns.iloc[-1]:.1f}x")

if __name__ == '__main__':
    main()
