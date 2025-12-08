#!/usr/bin/env python3
"""
Policy Comparison Evaluation Script

Tests different eviction/prefetch policies with uvmbench workloads.

Usage:
  python run_policy_comparison.py                           # Default: hotspot, size_factor=0.6
  python run_policy_comparison.py --resume file.csv         # Resume from existing CSV
  python run_policy_comparison.py --output results_gemm     # Custom output directory

Example configurations:
  sudo python run_policy_comparison.py --kernel hotspot --size-factor 0.6 --output results_hotspot
  sudo python run_policy_comparison.py --kernel gemm --size-factor 0.6 --output results_gemm
  sudo python run_policy_comparison.py --kernel kmeans_sparse --size-factor 0.9 --output results_kmeans
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
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"

# Benchmark parameters (defaults, can be overridden via command line)
DEFAULT_SIZE_FACTOR = 0.6
DEFAULT_KERNEL = "hotspot"
ITERATIONS = 1
NUM_ROUNDS = 1

# Available kernels: rand_stream, seq_stream, hotspot, gemm, kmeans_sparse

# Policy configurations to test
POLICIES = [
    # (policy_name, policy_binary, configs)
    # configs = [(high_param, low_param), ...]
    ("no_policy", None, [(50, 50)]),
    # ("eviction_pid_quota", "eviction_pid_quota", [(50, 50), (80, 20), (90, 10)]),
    # ("eviction_fifo_chance", "eviction_fifo_chance", [(0, 0), (3, 0), (5, 0), (8, 1)]),
    # ("eviction_fifo_chance", "eviction_fifo_chance", [(0, 0), (5, 0)]),
    # eviction_freq_pid_decay: -P = high decay (1=always protected), -L = low decay (larger=less protected)
    ("eviction_freq_pid_decay", "eviction_freq_pid_decay", [(1, 1), (1, 10)]),
    # ("eviction_freq_pid_decay", "eviction_freq_pid_decay", [(1, 1), (1, 10)]),
    # ("prefetch_pid_tree", "prefetch_pid_tree", [(0, 0), (50, 50), (20, 80), (0, 40), (40, 40), (60, 60), (80, 80)]),
    ("prefetch_pid_tree", "prefetch_pid_tree", [(0, 0), (0, 20), (20, 80)]),
    # ("prefetch_pid_tree", "prefetch_pid_tree", [(20, 80)]),
    ("prefetch_eviction_pid", "prefetch_eviction_pid", [(20, 80)]),
]

# Single process configurations (no policy needed)
# (config_name, size_factor_multiplier)
SINGLE_PROCESS_CONFIGS = [
    ("single_1x", 1),      # SIZE_FACTOR * 1
    ("single_2x", 2),      # SIZE_FACTOR * 2
]


def cleanup_processes():
    """Kill any existing policy processes and cleanup struct_ops."""
    subprocess.run(["sudo", "pkill", "-f", "eviction_|prefetch_pid"],
                   capture_output=True)
    cleanup_tool = SRC / "cleanup_struct_ops_tool"
    if cleanup_tool.exists():
        subprocess.run(["sudo", str(cleanup_tool)], capture_output=True)
    time.sleep(1)


def run_uvmbench(output_file, size_factor, kernel):
    """Start a uvmbench process."""
    cmd = [
        str(UVM),
        f"--size_factor={size_factor}",
        "--mode=uvm",
        f"--iterations={ITERATIONS}",
        f"--kernel={kernel}",
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


def run_experiment(policy_name, policy_binary, high_param, low_param, round_idx, size_factor, kernel, output_dir):
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
        # Record start time
        start_time = time.time()

        # Start both uvmbench processes
        high_proc = run_uvmbench(high_output.name, size_factor, kernel)
        low_proc = run_uvmbench(low_output.name, size_factor, kernel)

        # Start policy process if needed
        if policy_binary:
            policy_path = SRC / policy_binary
            policy_output = output_dir / f"{policy_binary}_{high_param}_{low_param}_r{round_idx+1}.txt"

            cmd = [
                "sudo", str(policy_path),
                "-p", str(high_proc.pid), "-P", str(high_param),
                "-l", str(low_proc.pid), "-L", str(low_param),
            ]

            with open(policy_output, 'w') as f:
                policy_proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            time.sleep(1)

        # Wait for uvmbench to complete, record end times
        high_end_time = None
        low_end_time = None

        while high_proc.poll() is None or low_proc.poll() is None:
            if high_end_time is None and high_proc.poll() is not None:
                high_end_time = time.time()
            if low_end_time is None and low_proc.poll() is not None:
                low_end_time = time.time()
            time.sleep(0.01)

        # Ensure end times are recorded
        if high_end_time is None:
            high_end_time = time.time()
        if low_end_time is None:
            low_end_time = time.time()

        # Calculate latency (seconds)
        high_latency = high_end_time - start_time
        low_latency = low_end_time - start_time

        # Calculate throughput (iterations per second)
        high_throughput = ITERATIONS / high_latency if high_latency > 0 else 0
        low_throughput = ITERATIONS / low_latency if low_latency > 0 else 0

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
            'high_latency_s': high_latency,
            'high_throughput': high_throughput,
            'low_median_ms': low_median,
            'low_bw_gbps': low_bw,
            'low_latency_s': low_latency,
            'low_throughput': low_throughput,
        }

    finally:
        # Cleanup temp files
        os.unlink(high_output.name)
        os.unlink(low_output.name)


def run_single_experiment(config_name, size_multiplier, round_idx, size_factor, kernel):
    """Run a single process experiment without any policy."""

    cleanup_processes()

    # Create temp file for output
    output = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    output.close()

    try:
        # Calculate actual size factor
        actual_size_factor = size_factor * size_multiplier

        # Record start time
        start_time = time.time()

        # Start single uvmbench process
        proc = run_uvmbench(output.name, actual_size_factor, kernel)

        # Wait for uvmbench to complete
        proc.wait()
        end_time = time.time()

        # Calculate latency (seconds)
        latency = end_time - start_time

        # Calculate throughput (iterations per second)
        throughput = ITERATIONS / latency if latency > 0 else 0

        # Parse results
        median_ms, bw_gbps = parse_uvmbench_output(output.name)

        return {
            'median_ms': median_ms,
            'bw_gbps': bw_gbps,
            'latency_s': latency,
            'throughput': throughput,
        }

    finally:
        # Cleanup temp file
        os.unlink(output.name)


def warmup(size_factor, kernel):
    """Run warmup iteration."""
    print("=== WARMUP ===")
    cmd = [
        str(UVM),
        f"--size_factor={size_factor}",
        "--mode=uvm",
        "--iterations=2",
        f"--kernel={kernel}",
    ]
    subprocess.run(cmd, capture_output=True)
    time.sleep(2)


def load_completed_tests(csv_path):
    """Load completed tests from existing CSV file.

    Returns a set of (policy_name, high_param, low_param, round) tuples.
    For single process tests, high_param and low_param will be empty strings.
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
        if len(parts) >= 12:
            policy = parts[0]
            high_param = parts[1]  # Keep as string (may be empty for single process)
            low_param = parts[2]
            round_num = int(parts[11])
            completed.add((policy, high_param, low_param, round_num))

    return completed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Policy Comparison Evaluation')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to existing CSV file to resume from')
    parser.add_argument('--size-factor', type=float, default=DEFAULT_SIZE_FACTOR,
                        help=f'Size factor for uvmbench (default: {DEFAULT_SIZE_FACTOR})')
    parser.add_argument('--kernel', type=str, default=DEFAULT_KERNEL,
                        choices=['rand_stream', 'seq_stream', 'hotspot', 'gemm', 'kmeans_sparse'],
                        help=f'Kernel to run (default: {DEFAULT_KERNEL})')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help=f'Output directory for results (default: results)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Check uvmbench exists
    if not UVM.exists():
        print(f"Error: {UVM} not found")
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).parent / output_dir
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

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
        csv_path = output_dir / f"policy_comparison_{timestamp}.csv"
        # Write CSV header for new file
        with open(csv_path, 'w') as f:
            f.write("policy,high_param,low_param,high_median_ms,high_bw_gbps,high_latency_s,high_throughput,low_median_ms,low_bw_gbps,low_latency_s,low_throughput,round\n")

    # Get configuration from args
    size_factor = args.size_factor
    kernel = args.kernel

    print("=" * 60)
    print("Policy Comparison Evaluation")
    print("=" * 60)
    print(f"Kernel: {kernel}, Size Factor: {size_factor}")
    print(f"Output: {csv_path}")
    print()

    # Count remaining tests
    remaining = 0
    # Count dual-process tests
    for policy_name, policy_binary, configs in POLICIES:
        for high_param, low_param in configs:
            for round_idx in range(NUM_ROUNDS):
                if (policy_name, str(high_param), str(low_param), round_idx + 1) not in completed_tests:
                    remaining += 1
    # Count single-process tests
    for config_name, size_multiplier in SINGLE_PROCESS_CONFIGS:
        for round_idx in range(NUM_ROUNDS):
            if (config_name, "", "", round_idx + 1) not in completed_tests:
                remaining += 1
    print(f"Remaining tests: {remaining}")
    print()

    # Warmup
    warmup(size_factor, kernel)

    # Run all experiments
    results = []

    # Run single-process experiments first
    for config_name, size_multiplier in SINGLE_PROCESS_CONFIGS:
        for round_idx in range(NUM_ROUNDS):
            # Skip completed tests
            if (config_name, "", "", round_idx + 1) in completed_tests:
                print(f"=== SKIP {config_name} R{round_idx+1} (already done) ===")
                continue

            exp_name = f"{config_name} (size={size_factor * size_multiplier}) R{round_idx+1}"
            print(f"=== {exp_name} ===")

            result = run_single_experiment(config_name, size_multiplier, round_idx, size_factor, kernel)

            # Print result
            print(f"  {result['median_ms']:.2f}ms {result['bw_gbps']:.2f}GB/s "
                  f"latency={result['latency_s']:.2f}s throughput={result['throughput']:.2f}/s")

            # Write to CSV (leave high_param, low_param, low_* fields empty)
            with open(csv_path, 'a') as f:
                f.write(f"{config_name},,,"
                       f"{result['median_ms']},{result['bw_gbps']},"
                       f"{result['latency_s']},{result['throughput']},"
                       f",,,,{round_idx+1}\n")

            results.append({
                'policy': config_name,
                'high_param': '',
                'low_param': '',
                'round': round_idx + 1,
                'high_median_ms': result['median_ms'],
                'high_bw_gbps': result['bw_gbps'],
                'high_latency_s': result['latency_s'],
                'high_throughput': result['throughput'],
                'low_median_ms': 0,
                'low_bw_gbps': 0,
                'low_latency_s': 0,
                'low_throughput': 0,
            })

    # Run dual-process experiments
    for policy_name, policy_binary, configs in POLICIES:
        for high_param, low_param in configs:
            for round_idx in range(NUM_ROUNDS):
                # Skip completed tests
                if (policy_name, str(high_param), str(low_param), round_idx + 1) in completed_tests:
                    print(f"=== SKIP {policy_name} {high_param}/{low_param} R{round_idx+1} (already done) ===")
                    continue

                exp_name = f"{policy_name} {high_param}/{low_param} R{round_idx+1}"
                print(f"=== {exp_name} ===")

                result = run_experiment(policy_name, policy_binary, high_param, low_param, round_idx, size_factor, kernel, output_dir)

                # Print result
                print(f"  H:{result['high_median_ms']:.2f}ms {result['high_bw_gbps']:.2f}GB/s "
                      f"lat={result['high_latency_s']:.2f}s tput={result['high_throughput']:.2f}/s")
                print(f"  L:{result['low_median_ms']:.2f}ms {result['low_bw_gbps']:.2f}GB/s "
                      f"lat={result['low_latency_s']:.2f}s tput={result['low_throughput']:.2f}/s")

                # Write to CSV
                with open(csv_path, 'a') as f:
                    f.write(f"{policy_name},{high_param},{low_param},"
                           f"{result['high_median_ms']},{result['high_bw_gbps']},"
                           f"{result['high_latency_s']},{result['high_throughput']},"
                           f"{result['low_median_ms']},{result['low_bw_gbps']},"
                           f"{result['low_latency_s']},{result['low_throughput']},"
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
        if len(parts) >= 12:
            policy = parts[0]
            # Handle single process tests (empty params)
            high_param = parts[1] if parts[1] else ''
            low_param = parts[2] if parts[2] else ''
            high_median = float(parts[3]) if parts[3] else 0.0
            high_bw = float(parts[4]) if parts[4] else 0.0
            high_latency = float(parts[5]) if parts[5] else 0.0
            high_throughput = float(parts[6]) if parts[6] else 0.0
            low_median = float(parts[7]) if parts[7] else 0.0
            low_bw = float(parts[8]) if parts[8] else 0.0
            low_latency = float(parts[9]) if parts[9] else 0.0
            low_throughput = float(parts[10]) if parts[10] else 0.0
            all_results.append({
                'policy': policy,
                'high_param': high_param,
                'low_param': low_param,
                'high_median_ms': high_median,
                'high_bw_gbps': high_bw,
                'high_latency_s': high_latency,
                'high_throughput': high_throughput,
                'low_median_ms': low_median,
                'low_bw_gbps': low_bw,
                'low_latency_s': low_latency,
                'low_throughput': low_throughput,
            })

    # Group by policy and config
    grouped = defaultdict(list)
    for r in all_results:
        key = (r['policy'], r['high_param'], r['low_param'])
        grouped[key].append(r)

    print(f"{'Policy':<22} {'Params':<8} {'H_Lat(s)':<10} {'H_Tput':<10} {'L_Lat(s)':<10} {'L_Tput':<10}")
    print("-" * 75)

    for (policy, hp, lp), runs in grouped.items():
        h_lat_avg = sum(r['high_latency_s'] for r in runs) / len(runs)
        h_tput_avg = sum(r['high_throughput'] for r in runs) / len(runs)
        l_lat_avg = sum(r['low_latency_s'] for r in runs) / len(runs)
        l_tput_avg = sum(r['low_throughput'] for r in runs) / len(runs)

        # Format params display
        if hp == '' and lp == '':
            params_str = "(single)"
        else:
            params_str = f"{hp}/{lp}"

        print(f"{policy:<22} {params_str:<8} {h_lat_avg:<10.2f} {h_tput_avg:<10.2f} {l_lat_avg:<10.2f} {l_tput_avg:<10.2f}")

    print()
    print(f"Results saved to: {csv_path}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
