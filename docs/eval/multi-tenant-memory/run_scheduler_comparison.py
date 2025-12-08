#!/usr/bin/env python3
"""
Scheduler Policy Comparison Evaluation Script

Tests scheduler timeslice policies with the SAME uvmbench workloads as memory policy.
This allows comparison: Does scheduler alone solve memory contention?

Usage:
  python run_scheduler_comparison.py --kernel hotspot --size-factor 0.6 --output results_hotspot
  python run_scheduler_comparison.py --kernel gemm --size-factor 0.6 --output results_gemm
  python run_scheduler_comparison.py --kernel kmeans_sparse --size-factor 0.9 --output results_kmeans
"""

import subprocess
import tempfile
import time
import re
import os
import signal
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path("/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy")
SRC = BASE_DIR / "src"
UVM = BASE_DIR / "microbench" / "memory" / "uvmbench"
SCHED_POLICY = SRC / "gpu_sched_set_timeslices"
CLEANUP_TOOL = SRC / "cleanup_struct_ops_tool"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results_sched"

# Benchmark parameters
DEFAULT_SIZE_FACTOR = 0.6
DEFAULT_KERNEL = "hotspot"
ITERATIONS = 1
NUM_ROUNDS = 1

# Scheduler timeslice settings (microseconds)
HIGH_TIMESLICE = 1_000_000  # 1 second for high priority
LOW_TIMESLICE = 200         # 200 microseconds for low priority

# Scheduler policy configurations to test
SCHED_POLICIES = [
    # (policy_name, use_sched_policy, high_timeslice, low_timeslice)
    ("no_policy", False, None, None),
    ("sched_timeslice", True, HIGH_TIMESLICE, LOW_TIMESLICE),
]

# Single process configurations
SINGLE_PROCESS_CONFIGS = [
    ("single_1x", 1),
    # ("single_2x", 2),
]


def cleanup_processes():
    """Kill any lingering processes."""
    subprocess.run(["pkill", "-9", "-f", "uvmbench"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "uvmbench_high"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "uvmbench_low"], capture_output=True)
    if CLEANUP_TOOL.exists():
        subprocess.run(["sudo", str(CLEANUP_TOOL)], capture_output=True)
    time.sleep(1)


def parse_output(output_file):
    """Parse uvmbench output to extract metrics."""
    median_ms = 0
    bw_gbps = 0

    try:
        with open(output_file, 'r') as f:
            content = f.read()

        match = re.search(r'Median time:\s*([\d.]+)\s*ms', content)
        if match:
            median_ms = float(match.group(1))

        match = re.search(r'Bandwidth:\s*([\d.]+)\s*GB/s', content)
        if match:
            bw_gbps = float(match.group(1))
    except Exception as e:
        print(f"  Warning: Failed to parse {output_file}: {e}")

    return median_ms, bw_gbps


def run_uvmbench(output_file, size_factor, kernel, binary_name="uvmbench"):
    """Start a uvmbench process with optional custom binary name."""
    # Create symlink with custom name for scheduler policy identification
    uvm_link = Path(f"/tmp/{binary_name}")
    if uvm_link.exists():
        uvm_link.unlink()
    uvm_link.symlink_to(UVM)

    cmd = [
        str(uvm_link),
        f"--size_factor={size_factor}",
        "--mode=uvm",
        f"--iterations={ITERATIONS}",
        f"--kernel={kernel}",
    ]
    with open(output_file, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc


def run_experiment(policy_name, use_sched_policy, high_ts, low_ts, round_idx, size_factor, kernel, output_dir):
    """Run a single experiment with scheduler policy."""
    cleanup_processes()

    high_output = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    low_output = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    high_output.close()
    low_output.close()

    sched_proc = None

    try:
        start_time = time.time()

        # Start both uvmbench processes with different names for scheduler identification
        high_proc = run_uvmbench(high_output.name, size_factor, kernel, "uvmbench_high")
        low_proc = run_uvmbench(low_output.name, size_factor, kernel, "uvmbench_low")

        # Start scheduler policy if needed
        if use_sched_policy and SCHED_POLICY.exists():
            time.sleep(0.5)  # Wait for processes to start

            cmd = [
                "sudo", str(SCHED_POLICY),
                "-p", f"uvmbench_high:{high_ts}",
                "-p", f"uvmbench_low:{low_ts}",
            ]
            sched_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(0.5)

            if sched_proc.poll() is not None:
                print("  Warning: Scheduler policy failed to start")

        # Wait for completion
        high_proc.wait()
        high_end = time.time()
        low_proc.wait()
        low_end = time.time()

        high_latency = high_end - start_time
        low_latency = low_end - start_time

        # Parse results
        high_median, high_bw = parse_output(high_output.name)
        low_median, low_bw = parse_output(low_output.name)

        return {
            'policy': policy_name,
            'high_param': high_ts if high_ts else '',
            'low_param': low_ts if low_ts else '',
            'high_median_ms': high_median,
            'high_bw_gbps': high_bw,
            'high_latency_s': high_latency,
            'high_throughput': 1.0 / high_latency if high_latency > 0 else 0,
            'low_median_ms': low_median,
            'low_bw_gbps': low_bw,
            'low_latency_s': low_latency,
            'low_throughput': 1.0 / low_latency if low_latency > 0 else 0,
            'round': round_idx + 1,
        }

    finally:
        if sched_proc:
            sched_proc.send_signal(signal.SIGINT)
            try:
                sched_proc.wait(timeout=5)
            except:
                sched_proc.kill()

        # Cleanup
        if CLEANUP_TOOL.exists():
            subprocess.run(["sudo", str(CLEANUP_TOOL)], capture_output=True)

        for f in [high_output.name, low_output.name]:
            try:
                os.unlink(f)
            except:
                pass

        # Remove symlinks
        for name in ["uvmbench_high", "uvmbench_low"]:
            try:
                Path(f"/tmp/{name}").unlink()
            except:
                pass


def run_single_experiment(config_name, size_multiplier, round_idx, size_factor, kernel):
    """Run a single process experiment."""
    cleanup_processes()

    output = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    output.close()

    try:
        actual_size_factor = size_factor * size_multiplier
        start_time = time.time()

        proc = run_uvmbench(output.name, actual_size_factor, kernel)
        proc.wait()
        end_time = time.time()

        latency = end_time - start_time
        median_ms, bw_gbps = parse_output(output.name)

        return {
            'policy': config_name,
            'high_param': '',
            'low_param': '',
            'high_median_ms': median_ms,
            'high_bw_gbps': bw_gbps,
            'high_latency_s': latency,
            'high_throughput': 1.0 / latency if latency > 0 else 0,
            'low_median_ms': '',
            'low_bw_gbps': '',
            'low_latency_s': '',
            'low_throughput': '',
            'round': round_idx + 1,
        }
    finally:
        try:
            os.unlink(output.name)
        except:
            pass
        try:
            Path("/tmp/uvmbench").unlink()
        except:
            pass


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Scheduler Policy Comparison Evaluation')
    parser.add_argument('--size-factor', type=float, default=DEFAULT_SIZE_FACTOR,
                        help=f'Size factor for uvmbench (default: {DEFAULT_SIZE_FACTOR})')
    parser.add_argument('--kernel', type=str, default=DEFAULT_KERNEL,
                        choices=['rand_stream', 'seq_stream', 'hotspot', 'gemm', 'kmeans_sparse'],
                        help=f'Kernel to run (default: {DEFAULT_KERNEL})')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for results (default: results_sched)')
    return parser.parse_args()


def main():
    args = parse_args()

    if not UVM.exists():
        print(f"Error: {UVM} not found")
        sys.exit(1)

    if not SCHED_POLICY.exists():
        print(f"Error: {SCHED_POLICY} not found")
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).parent / output_dir
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    size_factor = args.size_factor
    kernel = args.kernel

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"sched_comparison_{timestamp}.csv"

    with open(csv_path, 'w') as f:
        f.write("policy,high_param,low_param,high_median_ms,high_bw_gbps,high_latency_s,high_throughput,low_median_ms,low_bw_gbps,low_latency_s,low_throughput,round\n")

    print("=" * 60)
    print("Scheduler Policy Comparison Evaluation")
    print("=" * 60)
    print(f"Kernel: {kernel}, Size Factor: {size_factor}")
    print(f"Timeslice: High={HIGH_TIMESLICE}µs, Low={LOW_TIMESLICE}µs")
    print(f"Output: {csv_path}")
    print()

    warmup(size_factor, kernel)

    # Run single process baselines
    print("\n=== SINGLE PROCESS BASELINES ===")
    for config_name, size_multiplier in SINGLE_PROCESS_CONFIGS:
        for round_idx in range(NUM_ROUNDS):
            exp_name = f"{config_name} (size={size_factor * size_multiplier}) R{round_idx+1}"
            print(f"=== {exp_name} ===")

            result = run_single_experiment(config_name, size_multiplier, round_idx, size_factor, kernel)

            print(f"  {result['high_latency_s']:.2f}s, {result['high_median_ms']:.2f}ms, {result['high_bw_gbps']:.2f}GB/s")

            with open(csv_path, 'a') as f:
                f.write(f"{result['policy']},{result['high_param']},{result['low_param']},"
                       f"{result['high_median_ms']},{result['high_bw_gbps']},{result['high_latency_s']},{result['high_throughput']},"
                       f"{result['low_median_ms']},{result['low_bw_gbps']},{result['low_latency_s']},{result['low_throughput']},"
                       f"{result['round']}\n")

    # Run scheduler policy experiments
    print("\n=== SCHEDULER POLICY EXPERIMENTS ===")
    for policy_name, use_sched, high_ts, low_ts in SCHED_POLICIES:
        for round_idx in range(NUM_ROUNDS):
            ts_str = f"{high_ts}/{low_ts}" if high_ts else "none"
            exp_name = f"{policy_name} (ts={ts_str}) R{round_idx+1}"
            print(f"=== {exp_name} ===")

            result = run_experiment(policy_name, use_sched, high_ts, low_ts, round_idx, size_factor, kernel, output_dir)

            print(f"  H:{result['high_median_ms']:.2f}ms {result['high_bw_gbps']:.2f}GB/s "
                  f"lat={result['high_latency_s']:.2f}s")
            print(f"  L:{result['low_median_ms']:.2f}ms {result['low_bw_gbps']:.2f}GB/s "
                  f"lat={result['low_latency_s']:.2f}s")

            with open(csv_path, 'a') as f:
                f.write(f"{result['policy']},{result['high_param']},{result['low_param']},"
                       f"{result['high_median_ms']},{result['high_bw_gbps']},{result['high_latency_s']},{result['high_throughput']},"
                       f"{result['low_median_ms']},{result['low_bw_gbps']},{result['low_latency_s']},{result['low_throughput']},"
                       f"{result['round']}\n")

    print()
    print(f"Results saved to: {csv_path}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
