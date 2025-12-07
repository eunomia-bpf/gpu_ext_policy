#!/usr/bin/env python3
"""
Benchmark script for prefetch policies.
Tests different parameters and finds optimal configuration.

Usage:
    python3 benchmark_prefetch.py                    # Full brute-force test
    python3 benchmark_prefetch.py --smart            # Smart search (golden section)
    python3 benchmark_prefetch.py --adaptive-range 30 50   # Custom adaptive range
    python3 benchmark_prefetch.py --direction-range 64 256 # Custom direction range
"""

import subprocess
import time
import re
import os
import csv
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

# Paths
BPF_DIR = "/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src"
BENCH_DIR = "/home/yunwei37/workspace/gpu/co-processor-demo/memory/micro"
UVMBENCH = f"{BENCH_DIR}/uvmbench"

# Test parameters
BENCH_TIMEOUT = 60  # seconds
BPF_STARTUP_WAIT = 2  # seconds to wait for BPF to load
BENCH_ARGS = ["--kernel=seq_device_prefetch", "--mode=uvm", "--size_factor=1.2", "--iterations=1"]

# Smart search parameters
COARSE_POINTS = [0, 25, 50, 75, 100]
FINE_PRECISION = 3
VALIDATION_RUNS = 3


@dataclass
class BenchResult:
    bandwidth_gbps: float
    time_ms: float
    success: bool
    error: Optional[str] = None


result_cache: Dict[str, BenchResult] = {}


def run_benchmark(timeout: int = BENCH_TIMEOUT) -> BenchResult:
    """Run uvmbench and parse results."""
    try:
        result = subprocess.run(
            [UVMBENCH] + BENCH_ARGS,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=BENCH_DIR
        )

        match = re.search(r"Bandwidth:\s+([\d.]+)\s+GB/s", result.stdout)
        time_match = re.search(r"Median time:\s+([\d.]+)\s+ms", result.stdout)

        if match and time_match:
            return BenchResult(
                bandwidth_gbps=float(match.group(1)),
                time_ms=float(time_match.group(1)),
                success=True
            )
        else:
            return BenchResult(0, 0, False, f"Parse error: {result.stdout[:200]}")

    except subprocess.TimeoutExpired:
        return BenchResult(0, 0, False, "Timeout")
    except Exception as e:
        return BenchResult(0, 0, False, str(e))


def start_bpf_policy(policy: str, args: List[str]) -> subprocess.Popen:
    """Start a BPF policy in background."""
    cmd = [f"{BPF_DIR}/{policy}"] + args
    proc = subprocess.Popen(
        ["sudo"] + cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    time.sleep(BPF_STARTUP_WAIT)
    return proc


def stop_bpf_policy(proc: subprocess.Popen):
    """Stop a BPF policy gracefully."""
    if proc.poll() is None:
        subprocess.run(["sudo", "kill", "-INT", str(proc.pid)], capture_output=True)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            subprocess.run(["sudo", "kill", "-9", str(proc.pid)], capture_output=True)
    time.sleep(1)


def test_baseline() -> BenchResult:
    """Test without any BPF policy (kernel default)."""
    print("Testing baseline (no BPF policy)...")
    return run_benchmark()


def test_policy(policy: str, args: List[str] = None) -> BenchResult:
    """Test a BPF policy with given arguments."""
    args = args or []
    print(f"Testing {policy} {' '.join(args)}...")
    proc = start_bpf_policy(policy, args)
    result = run_benchmark()
    stop_bpf_policy(proc)
    return result


def test_adaptive_tree_iter(threshold: int, use_cache: bool = True, verbose: bool = True) -> BenchResult:
    """Test prefetch_adaptive_tree_iter with given threshold."""
    cache_key = f"adaptive_t{threshold}"

    if use_cache and cache_key in result_cache:
        return result_cache[cache_key]

    if verbose:
        print(f"Testing prefetch_adaptive_tree_iter -t {threshold}...", end=" ", flush=True)

    proc = start_bpf_policy("prefetch_adaptive_tree_iter", ["-t", str(threshold)])
    result = run_benchmark()
    stop_bpf_policy(proc)

    if verbose:
        if result.success:
            print(f"{result.bandwidth_gbps:.2f} GB/s")
        else:
            print(f"FAILED ({result.error})")

    result_cache[cache_key] = result
    return result


def test_direction(direction: str, num_pages: int, use_cache: bool = True, verbose: bool = True) -> BenchResult:
    """Test prefetch_direction with given parameters."""
    cache_key = f"direction_{direction}_n{num_pages}"

    if use_cache and cache_key in result_cache:
        return result_cache[cache_key]

    if verbose:
        print(f"Testing prefetch_direction -d {direction} -n {num_pages}...", end=" ", flush=True)

    proc = start_bpf_policy("prefetch_direction", ["-d", direction, "-n", str(num_pages)])
    result = run_benchmark()
    stop_bpf_policy(proc)

    if verbose:
        if result.success:
            print(f"{result.bandwidth_gbps:.2f} GB/s")
        else:
            print(f"FAILED ({result.error})")

    result_cache[cache_key] = result
    return result


def golden_section_search(test_fn, low: int, high: int, precision: int = FINE_PRECISION) -> Tuple[int, float]:
    """
    Golden section search to find optimal parameter.
    Assumes unimodal function (one peak).
    Returns (best_param, best_bandwidth).
    """
    phi = 0.618

    x1 = int(high - phi * (high - low))
    x2 = int(low + phi * (high - low))

    r1 = test_fn(x1)
    r2 = test_fn(x2)

    f1 = r1.bandwidth_gbps if r1.success else 0
    f2 = r2.bandwidth_gbps if r2.success else 0

    while (high - low) > precision:
        if f1 > f2:
            high = x2
            x2 = x1
            f2 = f1
            x1 = int(high - phi * (high - low))
            r1 = test_fn(x1)
            f1 = r1.bandwidth_gbps if r1.success else 0
        else:
            low = x1
            x1 = x2
            f1 = f2
            x2 = int(low + phi * (high - low))
            r2 = test_fn(x2)
            f2 = r2.bandwidth_gbps if r2.success else 0

    if f1 > f2:
        return x1, f1
    else:
        return x2, f2


def save_results(results: List, suffix: str = ""):
    """Save results to CSV and Markdown."""
    base_path = os.path.dirname(__file__)
    csv_path = f"{base_path}/benchmark_results{suffix}.csv"
    md_path = f"{base_path}/benchmark_results{suffix}.md"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "params", "bandwidth_gbps", "time_ms", "success", "error"])
        for policy, params, res in results:
            writer.writerow([policy, params, res.bandwidth_gbps, res.time_ms, res.success, res.error or ""])

    # Find best results
    best_adaptive = max(
        [(p, params, r) for p, params, r in results if "adaptive" in p and r.success],
        key=lambda x: x[2].bandwidth_gbps,
        default=None
    )
    best_direction = max(
        [(p, params, r) for p, params, r in results if "direction" in p and r.success],
        key=lambda x: x[2].bandwidth_gbps,
        default=None
    )

    with open(md_path, "w") as f:
        f.write("# Prefetch Policy Benchmark Results\n\n")
        f.write(f"Benchmark: `{' '.join(BENCH_ARGS)}`\n\n")

        f.write("## Results\n\n")
        f.write("| Policy | Params | Bandwidth (GB/s) | Time (ms) |\n")
        f.write("|--------|--------|------------------|----------|\n")
        for policy, params, res in results:
            if res.success:
                f.write(f"| {policy} | {params} | {res.bandwidth_gbps:.2f} | {res.time_ms:.1f} |\n")
            else:
                f.write(f"| {policy} | {params} | FAILED | {res.error} |\n")

        f.write(f"\n## Best Configuration\n\n")
        if best_adaptive:
            f.write(f"- **prefetch_adaptive_tree_iter**: `{best_adaptive[1]}` ({best_adaptive[2].bandwidth_gbps:.2f} GB/s)\n")
        if best_direction:
            f.write(f"- **prefetch_direction**: `{best_direction[1]}` ({best_direction[2].bandwidth_gbps:.2f} GB/s)\n")

    print(f"\nResults saved to: {csv_path}")
    print(f"Report saved to: {md_path}")


def run_full_benchmark():
    """Run full brute-force benchmark suite."""
    results = []

    print("=" * 60)
    print("Prefetch Policy Benchmark - Full Test")
    print("=" * 60)
    print(f"Benchmark: {' '.join(BENCH_ARGS)}")
    print(f"Timeout: {BENCH_TIMEOUT}s")
    print()

    # Baseline tests
    print("-" * 40)
    print("Phase 1: Baseline Tests")
    print("-" * 40)

    baseline = test_baseline()
    results.append(("baseline", "-", baseline))
    print(f"  -> {baseline.bandwidth_gbps:.2f} GB/s, {baseline.time_ms:.1f} ms")

    none_result = test_policy("prefetch_none")
    results.append(("prefetch_none", "-", none_result))
    print(f"  -> {none_result.bandwidth_gbps:.2f} GB/s, {none_result.time_ms:.1f} ms")

    always_max = test_policy("prefetch_always_max")
    results.append(("prefetch_always_max", "-", always_max))
    print(f"  -> {always_max.bandwidth_gbps:.2f} GB/s, {always_max.time_ms:.1f} ms")

    # adaptive_tree_iter
    print()
    print("-" * 40)
    print("Phase 2: prefetch_adaptive_tree_iter -t [0-100]")
    print("-" * 40)

    thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    best_threshold = 0
    best_threshold_bw = 0

    for t in thresholds:
        result = test_adaptive_tree_iter(t)
        results.append(("adaptive_tree_iter", f"t={t}", result))
        if result.success:
            print(f"  t={t:3d}: {result.bandwidth_gbps:.2f} GB/s, {result.time_ms:.1f} ms")
            if result.bandwidth_gbps > best_threshold_bw:
                best_threshold_bw = result.bandwidth_gbps
                best_threshold = t
        else:
            print(f"  t={t:3d}: FAILED - {result.error}")

    print(f"\n  Best threshold: {best_threshold} ({best_threshold_bw:.2f} GB/s)")

    # prefetch_direction
    print()
    print("-" * 40)
    print("Phase 3: prefetch_direction -d forward -n [0-256]")
    print("-" * 40)

    num_pages_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_num_pages = 0
    best_num_pages_bw = 0

    for n in num_pages_list:
        result = test_direction("forward", n)
        results.append(("direction_forward", f"n={n}", result))
        if result.success:
            print(f"  n={n:4d}: {result.bandwidth_gbps:.2f} GB/s, {result.time_ms:.1f} ms")
            if result.bandwidth_gbps > best_num_pages_bw:
                best_num_pages_bw = result.bandwidth_gbps
                best_num_pages = n
        else:
            print(f"  n={n:4d}: FAILED - {result.error}")

    print(f"\n  Best num_pages: {best_num_pages} ({best_num_pages_bw:.2f} GB/s)")

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Baseline:              {baseline.bandwidth_gbps:.2f} GB/s")
    print(f"prefetch_none:         {none_result.bandwidth_gbps:.2f} GB/s")
    print(f"prefetch_always_max:   {always_max.bandwidth_gbps:.2f} GB/s")
    print(f"Best adaptive (t={best_threshold}):  {best_threshold_bw:.2f} GB/s")
    print(f"Best direction (n={best_num_pages}): {best_num_pages_bw:.2f} GB/s")

    save_results(results)


def run_smart_benchmark():
    """Run smart benchmark with golden section search."""
    results = []

    print("=" * 60)
    print("Prefetch Policy Benchmark - Smart Search")
    print("=" * 60)
    print(f"Benchmark: {' '.join(BENCH_ARGS)}")
    print(f"Timeout: {BENCH_TIMEOUT}s")
    print()

    # Baseline
    print("-" * 40)
    print("Baseline Test")
    print("-" * 40)
    baseline = test_baseline()
    results.append(("baseline", "-", baseline))
    baseline_bw = baseline.bandwidth_gbps if baseline.success else 0
    print(f"  -> {baseline_bw:.2f} GB/s")

    # === Adaptive: Coarse scan + Golden section ===
    print()
    print("=" * 50)
    print("Phase 1: Coarse Scan (adaptive_tree_iter)")
    print("=" * 50)

    coarse_results = []
    for t in COARSE_POINTS:
        result = test_adaptive_tree_iter(t)
        results.append(("adaptive_tree_iter", f"t={t}", result))
        if result.success:
            coarse_results.append((t, result.bandwidth_gbps))

    if coarse_results:
        coarse_results.sort(key=lambda x: x[1], reverse=True)
        best_t, best_bw = coarse_results[0]

        idx = COARSE_POINTS.index(best_t)
        low = COARSE_POINTS[max(0, idx - 1)]
        high = COARSE_POINTS[min(len(COARSE_POINTS) - 1, idx + 1)]

        print(f"\nBest coarse: t={best_t} ({best_bw:.2f} GB/s)")
        print(f"Refining in range [{low}, {high}]...")

        print()
        print("-" * 50)
        print("Phase 2: Fine Search (golden section)")
        print("-" * 50)

        def test_adaptive_wrapper(t):
            r = test_adaptive_tree_iter(t)
            results.append(("adaptive_tree_iter", f"t={t}", r))
            return r

        refined_t, refined_bw = golden_section_search(test_adaptive_wrapper, low, high)

        print(f"\nRefined best: t={refined_t} ({refined_bw:.2f} GB/s)")

        # Validation
        print()
        print("-" * 50)
        print(f"Phase 3: Validation ({VALIDATION_RUNS} runs)")
        print("-" * 50)

        bandwidths = []
        for i in range(VALIDATION_RUNS):
            result = test_adaptive_tree_iter(refined_t, use_cache=False)
            results.append(("adaptive_tree_iter_validation", f"t={refined_t}_run{i+1}", result))
            if result.success:
                bandwidths.append(result.bandwidth_gbps)

        if bandwidths:
            avg_bw = sum(bandwidths) / len(bandwidths)
            print(f"\nValidation: avg={avg_bw:.2f} GB/s (min={min(bandwidths):.2f}, max={max(bandwidths):.2f})")
            best_adaptive = (refined_t, avg_bw)
        else:
            best_adaptive = (refined_t, refined_bw)
    else:
        best_adaptive = (0, 0)

    # === Direction: Scan powers of 2 ===
    print()
    print("=" * 50)
    print("Phase 4: Scan (prefetch_direction)")
    print("=" * 50)

    points = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    dir_results = []

    for n in points:
        result = test_direction("forward", n)
        results.append(("direction_forward", f"n={n}", result))
        if result.success:
            dir_results.append((n, result.bandwidth_gbps))

    if dir_results:
        dir_results.sort(key=lambda x: x[1], reverse=True)
        best_n, best_n_bw = dir_results[0]

        print(f"\nBest: n={best_n} ({best_n_bw:.2f} GB/s)")

        # Validation
        print()
        print("-" * 50)
        print(f"Phase 5: Validation ({VALIDATION_RUNS} runs)")
        print("-" * 50)

        bandwidths = []
        for i in range(VALIDATION_RUNS):
            result = test_direction("forward", best_n, use_cache=False)
            results.append(("direction_forward_validation", f"n={best_n}_run{i+1}", result))
            if result.success:
                bandwidths.append(result.bandwidth_gbps)

        if bandwidths:
            avg_bw = sum(bandwidths) / len(bandwidths)
            print(f"\nValidation: avg={avg_bw:.2f} GB/s (min={min(bandwidths):.2f}, max={max(bandwidths):.2f})")
            best_direction = (best_n, avg_bw)
        else:
            best_direction = (best_n, best_n_bw)
    else:
        best_direction = (0, 0)

    # Summary
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Baseline:                    {baseline_bw:.2f} GB/s")
    print(f"Best adaptive (t={best_adaptive[0]:3d}):      {best_adaptive[1]:.2f} GB/s")
    print(f"Best direction (n={best_direction[0]:3d}):     {best_direction[1]:.2f} GB/s")

    if best_adaptive[1] > 0 and baseline_bw > 0:
        speedup = (best_adaptive[1] / baseline_bw - 1) * 100
        print(f"\nBest speedup: {speedup:+.1f}% with adaptive_tree_iter -t {best_adaptive[0]}")

    save_results(results, "_smart")


def run_adaptive_range(min_t: int, max_t: int):
    """Run golden section search on adaptive in given range."""
    results = []

    print("=" * 60)
    print(f"Adaptive Golden Section Search [{min_t}, {max_t}]")
    print("=" * 60)
    print()

    def test_adaptive_wrapper(t):
        r = test_adaptive_tree_iter(t)
        results.append(("adaptive_tree_iter", f"t={t}", r))
        return r

    best_t, best_bw = golden_section_search(test_adaptive_wrapper, min_t, max_t)

    print(f"\nBest: t={best_t} ({best_bw:.2f} GB/s)")

    # Validation
    print()
    print(f"Validation ({VALIDATION_RUNS} runs):")
    bandwidths = []
    for i in range(VALIDATION_RUNS):
        result = test_adaptive_tree_iter(best_t, use_cache=False)
        results.append(("adaptive_tree_iter_validation", f"t={best_t}_run{i+1}", result))
        if result.success:
            bandwidths.append(result.bandwidth_gbps)

    if bandwidths:
        avg_bw = sum(bandwidths) / len(bandwidths)
        print(f"  avg={avg_bw:.2f} GB/s (min={min(bandwidths):.2f}, max={max(bandwidths):.2f})")

    save_results(results, f"_adaptive_{min_t}_{max_t}")


def run_direction_range(min_n: int, max_n: int):
    """Run golden section search on direction in given range."""
    results = []

    print("=" * 60)
    print(f"Direction Golden Section Search [{min_n}, {max_n}]")
    print("=" * 60)
    print()

    def test_direction_wrapper(n):
        r = test_direction("forward", n)
        results.append(("direction_forward", f"n={n}", r))
        return r

    best_n, best_bw = golden_section_search(test_direction_wrapper, min_n, max_n)

    print(f"\nBest: n={best_n} ({best_bw:.2f} GB/s)")

    # Validation
    print()
    print(f"Validation ({VALIDATION_RUNS} runs):")
    bandwidths = []
    for i in range(VALIDATION_RUNS):
        result = test_direction("forward", best_n, use_cache=False)
        results.append(("direction_forward_validation", f"n={best_n}_run{i+1}", result))
        if result.success:
            bandwidths.append(result.bandwidth_gbps)

    if bandwidths:
        avg_bw = sum(bandwidths) / len(bandwidths)
        print(f"  avg={avg_bw:.2f} GB/s (min={min(bandwidths):.2f}, max={max(bandwidths):.2f})")

    save_results(results, f"_direction_{min_n}_{max_n}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark prefetch policies")
    parser.add_argument("--smart", action="store_true",
                        help="Use smart search (coarse scan + golden section)")
    parser.add_argument("--adaptive-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                        help="Golden section search on adaptive in [MIN, MAX]")
    parser.add_argument("--direction-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                        help="Golden section search on direction in [MIN, MAX]")

    args = parser.parse_args()

    if args.adaptive_range:
        run_adaptive_range(args.adaptive_range[0], args.adaptive_range[1])
    elif args.direction_range:
        run_direction_range(args.direction_range[0], args.direction_range[1])
    elif args.smart:
        run_smart_benchmark()
    else:
        run_full_benchmark()


if __name__ == "__main__":
    main()
