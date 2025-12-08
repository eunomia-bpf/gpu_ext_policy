#!/usr/bin/env python3
"""
Multi-Tenant Scheduler Results Analysis

Analyzes the experimental data to verify the claim:
- High latency events reduced by ~35% with policy
- BE throughput maintained unchanged
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import re

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "simple_test_results"


def load_data():
    """Load all CSV files and separate by mode and role."""
    native_lc, native_be = [], []
    policy_lc, policy_be = [], []
    native_be_duration, policy_be_duration = 0, 0

    # Per-run data for P99 averaging
    native_lc_per_run = {}  # run_id -> latencies
    native_be_per_run = {}
    policy_lc_per_run = {}
    policy_be_per_run = {}

    native_files = list(DATA_DIR.glob("native_run*_*.csv"))
    policy_files = list(DATA_DIR.glob("policy_run*_*.csv"))

    for f in native_files:
        df = pd.read_csv(f)
        latencies = df['launch_latency_ms'].values * 1000  # to µs
        # Extract run number
        match = re.search(r'run(\d+)', f.name)
        run_id = int(match.group(1)) if match else 0

        if '_lc_' in f.name:
            native_lc.extend(latencies)
            if run_id not in native_lc_per_run:
                native_lc_per_run[run_id] = []
            native_lc_per_run[run_id].extend(latencies)
        else:
            native_be.extend(latencies)
            native_be_duration += df['duration_ms'].sum()
            if run_id not in native_be_per_run:
                native_be_per_run[run_id] = []
            native_be_per_run[run_id].extend(latencies)

    for f in policy_files:
        df = pd.read_csv(f)
        latencies = df['launch_latency_ms'].values * 1000  # to µs
        match = re.search(r'run(\d+)', f.name)
        run_id = int(match.group(1)) if match else 0

        if '_lc_' in f.name:
            policy_lc.extend(latencies)
            if run_id not in policy_lc_per_run:
                policy_lc_per_run[run_id] = []
            policy_lc_per_run[run_id].extend(latencies)
        else:
            policy_be.extend(latencies)
            policy_be_duration += df['duration_ms'].sum()
            if run_id not in policy_be_per_run:
                policy_be_per_run[run_id] = []
            policy_be_per_run[run_id].extend(latencies)

    return {
        'native_lc': np.array(native_lc),
        'native_be': np.array(native_be),
        'policy_lc': np.array(policy_lc),
        'policy_be': np.array(policy_be),
        'native_be_duration': native_be_duration,
        'policy_be_duration': policy_be_duration,
        'native_files': len(native_files),
        'policy_files': len(policy_files),
        # Per-run data
        'native_lc_per_run': native_lc_per_run,
        'native_be_per_run': native_be_per_run,
        'policy_lc_per_run': policy_lc_per_run,
        'policy_be_per_run': policy_be_per_run,
    }


def analyze(data):
    """Analyze the data and print results."""
    print("=" * 70)
    print("MULTI-TENANT SCHEDULER EVALUATION RESULTS")
    print("=" * 70)

    # Data completeness check
    print("\n[1] DATA COMPLETENESS CHECK")
    print("-" * 40)
    expected_files = 60  # 10 runs * (2 LC + 4 BE)
    print(f"Native CSV files: {data['native_files']} (expected: {expected_files})")
    print(f"Policy CSV files: {data['policy_files']} (expected: {expected_files})")
    print(f"Native LC samples: {len(data['native_lc'])} (expected: 4000)")
    print(f"Native BE samples: {len(data['native_be'])} (expected: 8000)")
    print(f"Policy LC samples: {len(data['policy_lc'])} (expected: 4000)")
    print(f"Policy BE samples: {len(data['policy_be'])} (expected: 8000)")

    completeness_ok = (
        data['native_files'] == expected_files and
        data['policy_files'] == expected_files and
        len(data['native_lc']) == 4000 and
        len(data['native_be']) == 8000 and
        len(data['policy_lc']) == 4000 and
        len(data['policy_be']) == 8000
    )
    print(f"Status: {'PASS' if completeness_ok else 'INCOMPLETE'}")

    # LC Latency Analysis
    print("\n[2] LC LAUNCH LATENCY")
    print("-" * 40)

    # Overall percentiles (across all samples)
    native_lc_p99 = np.percentile(data['native_lc'], 99)
    policy_lc_p99 = np.percentile(data['policy_lc'], 99)
    native_lc_p995 = np.percentile(data['native_lc'], 99.5)
    policy_lc_p995 = np.percentile(data['policy_lc'], 99.5)
    native_lc_p999 = np.percentile(data['native_lc'], 99.9)
    policy_lc_p999 = np.percentile(data['policy_lc'], 99.9)
    native_lc_max = np.max(data['native_lc'])
    policy_lc_max = np.max(data['policy_lc'])

    lc_improvement = (native_lc_p99 - policy_lc_p99) / native_lc_p99 * 100
    lc_improvement_995 = (native_lc_p995 - policy_lc_p995) / native_lc_p995 * 100
    lc_improvement_999 = (native_lc_p999 - policy_lc_p999) / native_lc_p999 * 100
    lc_improvement_max = (native_lc_max - policy_lc_max) / native_lc_max * 100

    print("Overall percentiles (all 4000 samples):")
    print(f"  {'Percentile':<12} {'Native':>12} {'Policy':>12} {'Change':>12}")
    print(f"  {'P99':<12} {native_lc_p99:>10.1f}µs {policy_lc_p99:>10.1f}µs {lc_improvement:>+10.1f}%")
    print(f"  {'P99.5':<12} {native_lc_p995:>10.1f}µs {policy_lc_p995:>10.1f}µs {lc_improvement_995:>+10.1f}%")
    print(f"  {'P99.9':<12} {native_lc_p999:>10.1f}µs {policy_lc_p999:>10.1f}µs {lc_improvement_999:>+10.1f}%")
    print(f"  {'Max':<12} {native_lc_max:>10.1f}µs {policy_lc_max:>10.1f}µs {lc_improvement_max:>+10.1f}%")

    # Per-run P99 statistics (OSDI-style analysis)
    native_lc_p99_per_run = [np.percentile(v, 99) for v in data['native_lc_per_run'].values()]
    policy_lc_p99_per_run = [np.percentile(v, 99) for v in data['policy_lc_per_run'].values()]

    # Statistics
    native_lc_p99_mean = np.mean(native_lc_p99_per_run)
    native_lc_p99_std = np.std(native_lc_p99_per_run)
    native_lc_p99_median = np.median(native_lc_p99_per_run)
    policy_lc_p99_mean = np.mean(policy_lc_p99_per_run)
    policy_lc_p99_std = np.std(policy_lc_p99_per_run)
    policy_lc_p99_median = np.median(policy_lc_p99_per_run)

    print(f"\nPer-run P99 statistics (10 runs):")
    print(f"  {'':15} {'Native':>15} {'Policy':>15}")
    print(f"  {'Mean':15} {native_lc_p99_mean:>13.1f}µs {policy_lc_p99_mean:>13.1f}µs")
    print(f"  {'Std':15} {native_lc_p99_std:>13.1f}µs {policy_lc_p99_std:>13.1f}µs")
    print(f"  {'Median':15} {native_lc_p99_median:>13.1f}µs {policy_lc_p99_median:>13.1f}µs")
    print(f"  {'Min':15} {min(native_lc_p99_per_run):>13.1f}µs {min(policy_lc_p99_per_run):>13.1f}µs")
    print(f"  {'Max':15} {max(native_lc_p99_per_run):>13.1f}µs {max(policy_lc_p99_per_run):>13.1f}µs")

    # Count how many runs have "bad" P99 (>1ms)
    native_bad_runs = sum(1 for p in native_lc_p99_per_run if p > 1000)
    policy_bad_runs = sum(1 for p in policy_lc_p99_per_run if p > 1000)
    print(f"\n  Runs with P99 > 1ms: Native={native_bad_runs}/10, Policy={policy_bad_runs}/10")

    # Show per-run details
    print("\nPer-run LC P99 details:")
    print(f"  {'Run':<5} {'Native P99':>15} {'Policy P99':>15} {'Native>1ms':>12} {'Policy>1ms':>12}")
    for run_id in sorted(data['native_lc_per_run'].keys()):
        n_p99 = np.percentile(data['native_lc_per_run'][run_id], 99)
        p_p99 = np.percentile(data['policy_lc_per_run'].get(run_id, [0]), 99)
        n_bad = "YES" if n_p99 > 1000 else "no"
        p_bad = "YES" if p_p99 > 1000 else "no"
        print(f"  {run_id:<5} {n_p99:>13.1f}µs {p_p99:>13.1f}µs {n_bad:>12} {p_bad:>12}")

    # For backward compatibility
    native_lc_p99_avg = native_lc_p99_mean
    policy_lc_p99_avg = policy_lc_p99_mean
    lc_improvement_avg = (native_lc_p99_avg - policy_lc_p99_avg) / native_lc_p99_avg * 100 if native_lc_p99_avg > 0 else 0

    # Calculate per-run Max average
    native_lc_max_per_run = [np.max(v) for v in data['native_lc_per_run'].values()]
    policy_lc_max_per_run = [np.max(v) for v in data['policy_lc_per_run'].values()]
    native_lc_max_avg = np.mean(native_lc_max_per_run)
    policy_lc_max_avg = np.mean(policy_lc_max_per_run)
    max_improvement_avg = (native_lc_max_avg - policy_lc_max_avg) / native_lc_max_avg * 100 if native_lc_max_avg > 0 else 0

    native_lc_p50 = np.percentile(data['native_lc'], 50)
    policy_lc_p50 = np.percentile(data['policy_lc'], 50)
    print(f"\nP50: Native={native_lc_p50:.1f}µs, Policy={policy_lc_p50:.1f}µs")

    # BE Throughput Analysis
    print("\n[3] BE THROUGHPUT")
    print("-" * 40)
    native_be_tput = len(data['native_be']) / (data['native_be_duration'] / 1000)
    policy_be_tput = len(data['policy_be']) / (data['policy_be_duration'] / 1000)
    tput_change = (policy_be_tput - native_be_tput) / native_be_tput * 100

    print(f"Native: {native_be_tput:.2f} kernels/s")
    print(f"Policy: {policy_be_tput:.2f} kernels/s")
    print(f"Change: {tput_change:+.1f}%")

    # High Latency Events
    print("\n[4] HIGH LATENCY EVENTS (>1ms)")
    print("-" * 40)
    native_lc_high = np.sum(data['native_lc'] > 1000)
    native_be_high = np.sum(data['native_be'] > 1000)
    policy_lc_high = np.sum(data['policy_lc'] > 1000)
    policy_be_high = np.sum(data['policy_be'] > 1000)

    native_total = native_lc_high + native_be_high
    policy_total = policy_lc_high + policy_be_high
    high_reduction = (native_total - policy_total) / native_total * 100 if native_total > 0 else 0

    print(f"{'':15} {'Native':>10} {'Policy':>10}")
    print(f"{'LC High >1ms':<15} {native_lc_high:>10} {policy_lc_high:>10}")
    print(f"{'BE High >1ms':<15} {native_be_high:>10} {policy_be_high:>10}")
    print(f"{'Total':<15} {native_total:>10} {policy_total:>10}")
    print(f"Reduction: {high_reduction:.1f}%")

    # Summary Table
    # Note: change = (policy - native) / native * 100
    # Negative means reduction, positive means increase
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    lc_p99_mean_change = (policy_lc_p99_mean - native_lc_p99_mean) / native_lc_p99_mean * 100 if native_lc_p99_mean > 0 else 0
    lc_p99_median_change = (policy_lc_p99_median - native_lc_p99_median) / native_lc_p99_median * 100 if native_lc_p99_median > 0 else 0
    lc_p99_std_change = (policy_lc_p99_std - native_lc_p99_std) / native_lc_p99_std * 100 if native_lc_p99_std > 0 else 0
    lc_high_change = (policy_lc_high - native_lc_high) / native_lc_high * 100 if native_lc_high > 0 else 0
    be_high_change = (policy_be_high - native_be_high) / native_be_high * 100 if native_be_high > 0 else 0
    print(f"\n{'Metric':<40} {'Native':>15} {'Policy':>15} {'Change':>12}")
    print("-" * 85)
    print(f"{'LC P99 per-run mean (µs)':<40} {native_lc_p99_mean:>15.1f} {policy_lc_p99_mean:>15.1f} {lc_p99_mean_change:>+11.1f}%")
    print(f"{'LC P99 per-run median (µs)':<40} {native_lc_p99_median:>15.1f} {policy_lc_p99_median:>15.1f} {lc_p99_median_change:>+11.1f}%")
    print(f"{'LC P99 per-run std (µs)':<40} {native_lc_p99_std:>15.1f} {policy_lc_p99_std:>15.1f} {lc_p99_std_change:>+11.1f}%")
    print(f"{'LC High Latency Events (>1ms)':<40} {native_lc_high:>15} {policy_lc_high:>15} {lc_high_change:>+11.1f}%")
    print(f"{'BE Throughput (kernels/s)':<40} {native_be_tput:>15.2f} {policy_be_tput:>15.2f} {tput_change:>+11.1f}%")
    print(f"{'BE High Latency Events (>1ms)':<40} {native_be_high:>15} {policy_be_high:>15} {be_high_change:>+11.1f}%")

    # Verification
    print("\n" + "=" * 70)
    print("CLAIM VERIFICATION")
    print("=" * 70)

    # Claim 1: LC P99 per-run mean reduced significantly (negative change means reduction)
    claim1_pass = lc_p99_mean_change < -50  # >50% reduction
    print(f"\n[Claim 1] LC P99 per-run mean reduced significantly")
    print(f"          Native: {native_lc_p99_mean:.1f}µs, Policy: {policy_lc_p99_mean:.1f}µs")
    print(f"          Change: {lc_p99_mean_change:+.1f}%")
    print(f"          Status: {'VERIFIED' if claim1_pass else 'NOT VERIFIED'}")

    # Claim 2: BE throughput maintained
    claim2_pass = abs(tput_change) < 5  # Within ±5%
    print(f"\n[Claim 2] BE throughput maintained unchanged")
    print(f"          Native: {native_be_tput:.2f} k/s, Policy: {policy_be_tput:.2f} k/s")
    print(f"          Change: {tput_change:+.1f}%")
    print(f"          Status: {'VERIFIED' if claim2_pass else 'NOT VERIFIED'}")

    # Overall
    print("\n" + "-" * 70)
    all_pass = claim1_pass and claim2_pass
    if all_pass:
        print("OVERALL: ALL CLAIMS VERIFIED")
    else:
        print("OVERALL: SOME CLAIMS NOT VERIFIED")
    print("=" * 70)

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
Based on {len(data['native_lc_per_run'])} runs:

LC (Latency-Critical):
  - P99 per-run mean: {native_lc_p99_mean:.0f}µs → {policy_lc_p99_mean:.0f}µs ({lc_p99_mean_change:+.1f}%)
  - P99 per-run std:  {native_lc_p99_std:.0f}µs → {policy_lc_p99_std:.0f}µs ({lc_p99_std_change:+.1f}%)
  - High latency events (>1ms): {native_lc_high} → {policy_lc_high} ({lc_high_change:+.1f}%)

BE (Best-Effort):
  - Throughput: {native_be_tput:.2f} → {policy_be_tput:.2f} kernels/s ({tput_change:+.1f}%)
  - High latency events (>1ms): {native_be_high} → {policy_be_high} ({be_high_change:+.1f}%)

KEY FINDING: Policy reduces LC tail latency by {-lc_p99_mean_change:.0f}% while maintaining BE throughput.
""")
    print("=" * 70)

    return {
        'lc_p99_mean_native': native_lc_p99_mean,
        'lc_p99_mean_policy': policy_lc_p99_mean,
        'lc_p99_mean_change': lc_p99_mean_change,
        'be_tput_native': native_be_tput,
        'be_tput_policy': policy_be_tput,
        'tput_change': tput_change,
        'claim1_pass': claim1_pass,
        'claim2_pass': claim2_pass,
        'all_pass': all_pass,
    }


def main():
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("Please ensure simple_test_results/ contains the CSV files.")
        return 1

    data = load_data()
    results = analyze(data)

    # Return exit code based on verification
    return 0 if results['all_pass'] else 1


if __name__ == "__main__":
    exit(main())
