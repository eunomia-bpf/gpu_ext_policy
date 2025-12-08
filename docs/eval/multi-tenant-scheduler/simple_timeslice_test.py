#!/usr/bin/env python3
"""
Simple Timeslice Policy Test

Goal: Create maximum GPU contention to show timeslice differentiation.

Key insight: Both groups must submit kernels simultaneously and continuously
to create enough contention for the timeslice policy to show effect.
"""

import subprocess
import time
import shutil
import signal
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MULTI_STREAM_DIR = SCRIPT_DIR.parent
BENCH_PATH = MULTI_STREAM_DIR / "multi_stream_bench"
STRUCT_OPS_PATH = Path("/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src/gpu_sched_set_timeslices")
CLEANUP_TOOL = Path("/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src/cleanup_struct_ops_tool")

# Create separate binaries for each group
BENCH_LC = MULTI_STREAM_DIR / "bench_lc"
BENCH_BE = MULTI_STREAM_DIR / "bench_be"

# Test parameters - longer kernels to trigger preemption
NUM_STREAMS = 4
NUM_KERNELS = 50  # Fewer kernels but longer each
WORKLOAD_SIZE = 32 * 1024 * 1024  # 32M elements (~50-100ms kernel)
KERNEL_TYPE = "compute"

# Process counts - balanced for clear comparison
LC_PROCS = 2
BE_PROCS = 4

# Timeslice settings (microseconds)
LC_TIMESLICE = 1_000_000  # 1 second
BE_TIMESLICE = 200        # 200 microseconds


def setup():
    """Setup benchmark binaries."""
    if not BENCH_PATH.exists():
        print(f"Error: {BENCH_PATH} not found")
        sys.exit(1)

    for target in [BENCH_LC, BENCH_BE]:
        if target.exists():
            target.unlink()
        shutil.copy2(BENCH_PATH, target)
        target.chmod(0o755)
    print(f"Created bench_lc and bench_be")


def cleanup():
    """Cleanup binaries and struct_ops."""
    for target in [BENCH_LC, BENCH_BE]:
        if target.exists():
            target.unlink()

    if CLEANUP_TOOL.exists():
        subprocess.run(["sudo", str(CLEANUP_TOOL)], capture_output=True)


def run_bench(bench_path, output_file):
    """Start a benchmark process."""
    cmd = [
        str(bench_path),
        "-s", str(NUM_STREAMS),
        "-k", str(NUM_KERNELS),
        "-w", str(WORKLOAD_SIZE),
        "-t", KERNEL_TYPE,
        "-o", str(output_file),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def analyze_results(lc_files, be_files):
    """Analyze launch latency from CSV files."""
    import pandas as pd
    import numpy as np

    lc_data = []
    be_data = []

    for f in lc_files:
        if f.exists():
            df = pd.read_csv(f)
            lc_data.extend(df['launch_latency_ms'].values * 1000)  # to µs

    for f in be_files:
        if f.exists():
            df = pd.read_csv(f)
            be_data.extend(df['launch_latency_ms'].values * 1000)

    lc_data = np.array(lc_data)
    be_data = np.array(be_data)

    if len(lc_data) > 0 and len(be_data) > 0:
        lc_p99 = np.percentile(lc_data, 99)
        be_p99 = np.percentile(be_data, 99)
        lc_p50 = np.percentile(lc_data, 50)
        be_p50 = np.percentile(be_data, 50)

        return {
            'lc_n': len(lc_data),
            'lc_p50': lc_p50,
            'lc_p99': lc_p99,
            'be_n': len(be_data),
            'be_p50': be_p50,
            'be_p99': be_p99,
        }
    return None


def run_experiment(use_policy=False, run_id=0):
    """Run a single experiment."""
    output_dir = SCRIPT_DIR / "simple_test_results"
    output_dir.mkdir(exist_ok=True)

    prefix = "policy" if use_policy else "native"

    # Start struct_ops if using policy
    struct_ops_proc = None
    if use_policy:
        subprocess.run(["sudo", str(CLEANUP_TOOL)], capture_output=True)
        time.sleep(0.5)

        cmd = [
            "sudo", str(STRUCT_OPS_PATH),
            "-p", f"bench_lc:{LC_TIMESLICE}",
            "-p", f"bench_be:{BE_TIMESLICE}",
        ]
        struct_ops_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1)

        if struct_ops_proc.poll() is not None:
            print("  struct_ops failed to start!")
            return None

    # Prepare output files
    lc_files = []
    be_files = []
    procs = []

    # Start ALL processes as quickly as possible (interleaved)
    for i in range(max(LC_PROCS, BE_PROCS)):
        if i < LC_PROCS:
            f = output_dir / f"{prefix}_run{run_id}_lc_{i}.csv"
            lc_files.append(f)
            procs.append(run_bench(BENCH_LC, f))

        if i < BE_PROCS:
            f = output_dir / f"{prefix}_run{run_id}_be_{i}.csv"
            be_files.append(f)
            procs.append(run_bench(BENCH_BE, f))

    # Wait for all to complete
    for p in procs:
        p.wait()

    # Stop struct_ops
    if struct_ops_proc:
        struct_ops_proc.send_signal(signal.SIGINT)
        try:
            stdout, _ = struct_ops_proc.communicate(timeout=5)
            # Print stats
            output = stdout.decode()
            if "policy_hit" in output:
                for line in output.split('\n'):
                    if 'policy_hit' in line or 'timeslice_mod' in line:
                        print(f"    {line.strip()}")
        except:
            struct_ops_proc.kill()
        subprocess.run(["sudo", str(CLEANUP_TOOL)], capture_output=True)

    return analyze_results(lc_files, be_files)


def main():
    print("=" * 60)
    print("Simple Timeslice Policy Test")
    print("=" * 60)
    print(f"Config: {LC_PROCS} LC + {BE_PROCS} BE procs, {NUM_STREAMS} streams, {NUM_KERNELS} kernels")
    print(f"Timeslice: LC={LC_TIMESLICE}µs, BE={BE_TIMESLICE}µs")
    print()

    setup()

    NUM_RUNS = 10

    try:
        # Run native (no policy) experiments
        print("--- Native (no policy) ---")
        native_results = []
        for i in range(NUM_RUNS):
            result = run_experiment(use_policy=False, run_id=i)
            if result:
                native_results.append(result)
                ratio = result['lc_p99'] / result['be_p99'] if result['be_p99'] > 0 else 0
                print(f"  Run {i}: LC P99={result['lc_p99']:.1f}µs, BE P99={result['be_p99']:.1f}µs, ratio={ratio:.2f}x")
            time.sleep(1)

        print()

        # Run policy experiments
        print("--- gBPF Policy (LC=1s, BE=200µs) ---")
        policy_results = []
        for i in range(NUM_RUNS):
            result = run_experiment(use_policy=True, run_id=i)
            if result:
                policy_results.append(result)
                ratio = result['lc_p99'] / result['be_p99'] if result['be_p99'] > 0 else 0
                status = "EFFECTIVE!" if ratio < 0.5 or ratio > 2 else ""
                print(f"  Run {i}: LC P99={result['lc_p99']:.1f}µs, BE P99={result['be_p99']:.1f}µs, ratio={ratio:.2f}x {status}")
            time.sleep(1)

        # Summary
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        if native_results:
            avg_native_lc = sum(r['lc_p99'] for r in native_results) / len(native_results)
            avg_native_be = sum(r['be_p99'] for r in native_results) / len(native_results)
            print(f"Native avg:  LC P99={avg_native_lc:.1f}µs, BE P99={avg_native_be:.1f}µs")

        if policy_results:
            avg_policy_lc = sum(r['lc_p99'] for r in policy_results) / len(policy_results)
            avg_policy_be = sum(r['be_p99'] for r in policy_results) / len(policy_results)
            print(f"Policy avg:  LC P99={avg_policy_lc:.1f}µs, BE P99={avg_policy_be:.1f}µs")

            # Check if any run showed significant effect
            effective_runs = [r for r in policy_results
                            if r['lc_p99'] / r['be_p99'] < 0.3 or r['lc_p99'] / r['be_p99'] > 3]
            print(f"\nEffective runs: {len(effective_runs)}/{len(policy_results)}")

            if effective_runs:
                print("Policy CAN work, but effect depends on kernel scheduling timing.")
            else:
                print("No significant effect observed in this run.")

    finally:
        cleanup()


if __name__ == "__main__":
    main()
