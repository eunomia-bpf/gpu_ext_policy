#!/usr/bin/env python3
"""
Single Process Policy Evaluation Script

Tests a single policy with uvmbench (single process, not multi-tenant).

Usage:
  python run_single_policy.py --policy prefetch_pid_tree -P 50
  python run_single_policy.py --policy eviction_fifo_chance -P 5
  python run_single_policy.py  # No policy baseline
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
KERNEL = "gemm"


def get_available_policies():
    """Scan SRC for available policy executables."""
    policies = []
    for f in os.listdir(SRC):
        path = SRC / f
        if ((f.startswith('eviction_') or f.startswith('prefetch_')) and
            path.is_file() and
            os.access(path, os.X_OK) and
            not f.endswith(('.c', '.o', '.h', '.bpf.c'))):
            policies.append(f)
    return sorted(policies)


def cleanup_processes():
    """Kill any existing policy processes and cleanup struct_ops."""
    subprocess.run(["sudo", "pkill", "-f", "eviction_|prefetch_"],
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

        match = re.search(r'Median time:\s+([\d.]+)', content)
        if match:
            median_ms = float(match.group(1))

        match = re.search(r'Bandwidth:\s+([\d.]+)', content)
        if match:
            bw_gbps = float(match.group(1))
    except Exception as e:
        print(f"  Warning: Failed to parse {output_file}: {e}")

    return median_ms, bw_gbps


def run_experiment(policy_binary, param, round_idx):
    """Run a single experiment with the given policy configuration."""

    cleanup_processes()

    # Create temp file for output
    uvm_output = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    uvm_output.close()

    policy_proc = None
    policy_output = None

    try:
        # Start uvmbench process
        uvm_proc = run_uvmbench(uvm_output.name)

        # Start policy process if needed
        if policy_binary:
            policy_path = SRC / policy_binary
            policy_output = OUT / f"{policy_binary}_{param}_r{round_idx+1}.txt"

            # Build command: -p PID -P param
            cmd = [
                "sudo", str(policy_path),
                "-p", str(uvm_proc.pid), "-P", str(param),
            ]

            with open(policy_output, 'w') as f:
                policy_proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            time.sleep(1)

        # Wait for uvmbench to complete
        uvm_proc.wait()

        # Stop policy process
        if policy_proc:
            policy_proc.send_signal(signal.SIGINT)
            try:
                policy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                policy_proc.kill()
                policy_proc.wait()

        # Parse results
        median_ms, bw_gbps = parse_uvmbench_output(uvm_output.name)

        return {
            'median_ms': median_ms,
            'bw_gbps': bw_gbps,
        }

    finally:
        os.unlink(uvm_output.name)


def warmup():
    """Run warmup iteration."""
    print("=== WARMUP ===")
    cmd = [
        str(UVM),
        f"--size_factor={SIZE_FACTOR}",
        "--mode=uvm",
        "--iterations=1",
        f"--kernel={KERNEL}",
    ]
    subprocess.run(cmd, capture_output=True)
    time.sleep(2)


def main():
    available_policies = get_available_policies()

    parser = argparse.ArgumentParser(
        description='Single Process Policy Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available policies:
{chr(10).join(f'  {p}' for p in available_policies) if available_policies else '  (none found)'}

Examples:
  %(prog)s                                      # No policy baseline
  %(prog)s --policy prefetch_pid_tree -P 50     # Prefetch with param=50
  %(prog)s --policy prefetch_none               # Disable prefetch
  %(prog)s --policy eviction_fifo_chance -P 5   # Eviction policy
  %(prog)s -r 10                                # Run 10 rounds
""")
    parser.add_argument('--policy', type=str, default=None,
                        help='Policy executable name (default: none)')
    parser.add_argument('-P', '--param', type=int, default=50,
                        help='Policy parameter (default: 50)')
    parser.add_argument('-r', '--rounds', type=int, default=1,
                        help='Number of rounds to run (default: 1)')
    parser.add_argument('-k', '--kernel', type=str, default='gemm',
                        choices=['seq_stream', 'rand_stream', 'pointer_chase', 'gemm'],
                        help='Access pattern kernel (default: gemm)')
    parser.add_argument('-s', '--size-factor', type=float, default=0.6,
                        help='Size factor for uvmbench (default: 0.6)')
    parser.add_argument('-i', '--iterations', type=int, default=1,
                        help='Iterations per run (default: 1)')
    parser.add_argument('--list-policies', action='store_true',
                        help='List available policies and exit')
    parser.add_argument('--no-warmup', action='store_true',
                        help='Skip warmup')
    args = parser.parse_args()

    if args.list_policies:
        print(f"Available policies in {SRC}:")
        for p in available_policies:
            print(f"  {p}")
        return 0

    # Update global params
    global SIZE_FACTOR, ITERATIONS, KERNEL
    SIZE_FACTOR = args.size_factor
    ITERATIONS = args.iterations
    KERNEL = args.kernel

    # Validate policy
    if args.policy:
        policy_path = SRC / args.policy
        if not policy_path.exists():
            print(f"Error: Policy '{args.policy}' not found at {policy_path}")
            print(f"Available: {', '.join(available_policies)}")
            return 1

    # Check uvmbench exists
    if not UVM.exists():
        print(f"Error: {UVM} not found")
        return 1

    # Create output directory
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Single Process Policy Evaluation")
    print("=" * 60)
    print(f"Policy:     {args.policy or '(none)'}")
    if args.policy:
        print(f"Parameter:  {args.param}")
    print(f"Kernel:     {KERNEL}")
    print(f"Size:       {SIZE_FACTOR}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Rounds:     {args.rounds}")
    print()

    # Warmup
    if not args.no_warmup:
        warmup()

    # Run experiments
    results = []

    for round_idx in range(args.rounds):
        policy_name = args.policy or "no_policy"
        exp_name = f"{policy_name} P={args.param} R{round_idx+1}"
        print(f"=== {exp_name} ===")

        result = run_experiment(args.policy, args.param, round_idx)

        print(f"  {result['median_ms']:.2f}ms {result['bw_gbps']:.2f}GB/s")

        results.append({
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

    avg_median = sum(r['median_ms'] for r in results) / len(results)
    avg_bw = sum(r['bw_gbps'] for r in results) / len(results)

    if args.rounds > 1:
        min_median = min(r['median_ms'] for r in results)
        max_median = max(r['median_ms'] for r in results)
        print(f"Median time: {avg_median:.2f}ms (min={min_median:.2f}, max={max_median:.2f})")
    else:
        print(f"Median time: {avg_median:.2f}ms")
    print(f"Bandwidth:   {avg_bw:.2f}GB/s")
    print("=== DONE ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())
