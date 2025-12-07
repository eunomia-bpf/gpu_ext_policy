#!/usr/bin/env python3
"""
Policy Comparison Evaluation Script

Tests different eviction/prefetch policies with uvmbench workloads.
"""

import subprocess
import tempfile
import time
import re
import os
import signal
import sys
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


def main():
    # Check uvmbench exists
    if not UVM.exists():
        print(f"Error: {UVM} not found")
        sys.exit(1)

    # Create output directory
    OUT.mkdir(parents=True, exist_ok=True)

    # Create CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT / f"policy_comparison_{timestamp}.csv"

    print("=" * 60)
    print("Policy Comparison Evaluation")
    print("=" * 60)
    print(f"Output: {csv_path}")
    print()

    # Write CSV header
    with open(csv_path, 'w') as f:
        f.write("policy,high_param,low_param,high_median_ms,high_bw_gbps,low_median_ms,low_bw_gbps,round\n")

    # Warmup
    warmup()

    # Run all experiments
    results = []

    for policy_name, policy_binary, configs in POLICIES:
        for high_param, low_param in configs:
            for round_idx in range(NUM_ROUNDS):
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

    # Group by policy and config
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
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
