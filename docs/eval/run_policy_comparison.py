#!/usr/bin/env python3
"""
Policy Comparison Evaluation Script

Tests different eviction/prefetch policies with uvmbench workloads.

Usage:
  python run_policy_comparison.py                    # Start fresh
  python run_policy_comparison.py --resume file.csv # Resume from existing CSV
"""

import subprocess
import tempfile
import time
import re
import os
import signal
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path("/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy")
SRC = BASE_DIR / "src"
UVM = BASE_DIR / "microbench" / "memory" / "uvmbench"
OUT = Path(__file__).parent / "results"

# Benchmark parameters
SIZE_FACTOR = 0.6
ITERATIONS = 1
# KERNEL = "rand_stream"
# KERNEL = "seq_stream"
KERNEL = "gemm"
NUM_ROUNDS = 10

# Policy configurations to test
POLICIES = [
    # (policy_name, policy_binary, configs)
    # configs = [(high_param, low_param), ...]
    ("no_policy", None, [(50, 50)]),
    ("eviction_pid_quota", "eviction_pid_quota", [(50, 50), (80, 20), (90, 10)]),
    ("eviction_fifo_chance", "eviction_fifo_chance", [(0, 0), (3, 0), (5, 0), (8, 1)]),
    # eviction_freq_pid_decay: -P = high decay (1=always protected), -L = low decay (larger=less protected)
    ("eviction_freq_pid_decay", "eviction_freq_pid_decay", [(1, 1), (1, 10), (1, 5)]),
    ("prefetch_pid_tree", "prefetch_pid_tree", [(0, 0), (50, 50), (20, 80), (0, 20), (0, 40)]),
]


def cleanup_processes():
    """Kill any existing policy processes and cleanup struct_ops."""
    subprocess.run(["sudo", "pkill", "-f", "eviction_|prefetch_pid"],
                   capture_output=True)
    cleanup_tool = SRC / "cleanup_struct_ops_tool"
    if cleanup_tool.exists():
        subprocess.run(["sudo", str(cleanup_tool)], capture_output=True)
    time.sleep(1)


def run_uvmbench(output_file):
    """Start a uvmbench process."""
    cmd = [
        str(UVM),
        f"--size_factor={SIZE_FACTOR}",
        "--mode=uvm",
        f"--iterations={ITERATIONS}",
        f"--kernel={KERNEL}",
    ]
    with open(output_file, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc


def parse_uvmbench_output(output_file):
    """Parse uvmbench output to extract median time and bandwidth."""
    median_ms = 0.0
    bw_gbps = 0.0

    try:
        with open(output_file, 'r') as f:
            content = f.read()

        # Parse "Median time: X.XXX ms"
        match = re.search(r'Median time:\s+([\d.]+)', content)
        if match:
            median_ms = float(match.group(1))

        # Parse "Bandwidth: X.XX GB/s"
        match = re.search(r'Bandwidth:\s+([\d.]+)', content)
        if match:
            bw_gbps = float(match.group(1))
    except Exception as e:
        print(f"  Warning: Failed to parse {output_file}: {e}")

    return median_ms, bw_gbps


def run_experiment(policy_name, policy_binary, high_param, low_param, round_idx):
    """Run a single experiment with the given policy configuration."""

    cleanup_processes()

    # Create temp files for output
    high_output = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    low_output = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    high_output.close()
    low_output.close()

    policy_proc = None
    policy_output = None

    try:
        # Start both uvmbench processes
        high_proc = run_uvmbench(high_output.name)
        low_proc = run_uvmbench(low_output.name)

        # Start policy process if needed
        if policy_binary:
            policy_path = SRC / policy_binary
            policy_output = OUT / f"{policy_binary}_{high_param}_{low_param}_r{round_idx+1}.txt"

            cmd = [
                "sudo", str(policy_path),
                "-p", str(high_proc.pid), "-P", str(high_param),
                "-l", str(low_proc.pid), "-L", str(low_param),
            ]

            with open(policy_output, 'w') as f:
                policy_proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            time.sleep(1)

        # Wait for uvmbench to complete
        high_proc.wait()
        low_proc.wait()

        # Stop policy process
        if policy_proc:
            policy_proc.send_signal(signal.SIGINT)
            try:
                policy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                policy_proc.kill()
                policy_proc.wait()

        # Parse results
        high_median, high_bw = parse_uvmbench_output(high_output.name)
        low_median, low_bw = parse_uvmbench_output(low_output.name)

        return {
            'high_median_ms': high_median,
            'high_bw_gbps': high_bw,
            'low_median_ms': low_median,
            'low_bw_gbps': low_bw,
        }

    finally:
        # Cleanup temp files
        os.unlink(high_output.name)
        os.unlink(low_output.name)


def warmup():
    """Run warmup iteration."""
    print("=== WARMUP ===")
    cmd = [
        str(UVM),
        f"--size_factor={SIZE_FACTOR}",
        "--mode=uvm",
        "--iterations=2",
        f"--kernel={KERNEL}",
    ]
    subprocess.run(cmd, capture_output=True)
    time.sleep(2)


def load_completed_tests(csv_path):
    """Load completed tests from existing CSV file.

    Returns a set of (policy_name, high_param, low_param, round) tuples.
    """
    completed = set()
    if not csv_path or not Path(csv_path).exists():
        return completed

    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 7:
            policy = parts[0]
            high_param = int(parts[1])
            low_param = int(parts[2])
            round_num = int(parts[7]) if len(parts) > 7 else int(parts[6])
            completed.add((policy, high_param, low_param, round_num))

    return completed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Policy Comparison Evaluation')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to existing CSV file to resume from')
    return parser.parse_args()


def main():
    args = parse_args()

    # Check uvmbench exists
    if not UVM.exists():
        print(f"Error: {UVM} not found")
        sys.exit(1)

    # Create output directory
    OUT.mkdir(parents=True, exist_ok=True)

    # Handle resume mode
    completed_tests = set()
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"Error: Resume file not found: {args.resume}")
            sys.exit(1)
        completed_tests = load_completed_tests(resume_path)
        csv_path = resume_path
        print(f"Resuming from: {csv_path}")
        print(f"Found {len(completed_tests)} completed tests")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUT / f"policy_comparison_{timestamp}.csv"
        # Write CSV header for new file
        with open(csv_path, 'w') as f:
            f.write("policy,high_param,low_param,high_median_ms,high_bw_gbps,low_median_ms,low_bw_gbps,round\n")

    print("=" * 60)
    print("Policy Comparison Evaluation")
    print("=" * 60)
    print(f"Output: {csv_path}")
    print()

    # Count remaining tests
    remaining = 0
    for policy_name, policy_binary, configs in POLICIES:
        for high_param, low_param in configs:
            for round_idx in range(NUM_ROUNDS):
                if (policy_name, high_param, low_param, round_idx + 1) not in completed_tests:
                    remaining += 1
    print(f"Remaining tests: {remaining}")
    print()

    # Warmup
    warmup()

    # Run all experiments
    results = []

    for policy_name, policy_binary, configs in POLICIES:
        for high_param, low_param in configs:
            for round_idx in range(NUM_ROUNDS):
                # Skip completed tests
                if (policy_name, high_param, low_param, round_idx + 1) in completed_tests:
                    print(f"=== SKIP {policy_name} {high_param}/{low_param} R{round_idx+1} (already done) ===")
                    continue

                exp_name = f"{policy_name} {high_param}/{low_param} R{round_idx+1}"
                print(f"=== {exp_name} ===")

                result = run_experiment(policy_name, policy_binary, high_param, low_param, round_idx)

                # Print result
                print(f"  H:{result['high_median_ms']:.2f}ms {result['high_bw_gbps']:.2f}GB/s "
                      f"L:{result['low_median_ms']:.2f}ms {result['low_bw_gbps']:.2f}GB/s")

                # Write to CSV
                with open(csv_path, 'a') as f:
                    f.write(f"{policy_name},{high_param},{low_param},"
                           f"{result['high_median_ms']},{result['high_bw_gbps']},"
                           f"{result['low_median_ms']},{result['low_bw_gbps']},"
                           f"{round_idx+1}\n")

                results.append({
                    'policy': policy_name,
                    'high_param': high_param,
                    'low_param': low_param,
                    'round': round_idx + 1,
                    **result,
                })

    # Cleanup
    cleanup_processes()

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Load all results from CSV for summary (including previously completed)
    from collections import defaultdict
    all_results = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 7:
            all_results.append({
                'policy': parts[0],
                'high_param': int(parts[1]),
                'low_param': int(parts[2]),
                'high_median_ms': float(parts[3]),
                'high_bw_gbps': float(parts[4]),
                'low_median_ms': float(parts[5]),
                'low_bw_gbps': float(parts[6]),
            })

    # Group by policy and config
    grouped = defaultdict(list)
    for r in all_results:
        key = (r['policy'], r['high_param'], r['low_param'])
        grouped[key].append(r)

    print(f"{'Policy':<25} {'Params':<10} {'H_Med(ms)':<12} {'H_BW(GB/s)':<12} {'L_Med(ms)':<12} {'L_BW(GB/s)':<12}")
    print("-" * 85)

    for (policy, hp, lp), runs in grouped.items():
        h_med_avg = sum(r['high_median_ms'] for r in runs) / len(runs)
        h_bw_avg = sum(r['high_bw_gbps'] for r in runs) / len(runs)
        l_med_avg = sum(r['low_median_ms'] for r in runs) / len(runs)
        l_bw_avg = sum(r['low_bw_gbps'] for r in runs) / len(runs)

        print(f"{policy:<25} {hp}/{lp:<8} {h_med_avg:<12.2f} {h_bw_avg:<12.2f} {l_med_avg:<12.2f} {l_bw_avg:<12.2f}")

    print()
    print(f"Results saved to: {csv_path}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
