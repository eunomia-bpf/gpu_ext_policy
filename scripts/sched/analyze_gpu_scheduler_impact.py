#!/usr/bin/env python3
"""
CUDA + CPU Scheduler Impact Analysis

Analyzes the impact of CPU scheduler on GPU workload performance by examining:
1. Scheduling delays before kernel launches
2. Off-CPU time during GPU synchronization
3. Context switch frequency and patterns
4. Potential performance bottlenecks

Usage:
    python analyze_gpu_scheduler_impact.py trace.csv [--output report.md]
"""

import pandas as pd
import argparse
import sys
from collections import defaultdict
from datetime import datetime


class GPUSchedulerAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.processes = {}
        self.metrics = defaultdict(dict)

    def load_data(self):
        """Load trace data from CSV"""
        print(f"Loading trace data from {self.csv_file}...")
        self.df = pd.read_csv(self.csv_file)
        print(f"Loaded {len(self.df)} events")

        # Group by PID
        for pid in self.df['pid'].unique():
            self.processes[pid] = self.df[self.df['pid'] == pid].copy()
            self.processes[pid] = self.processes[pid].sort_values('timestamp_ns')

        print(f"Found {len(self.processes)} GPU processes")

    def analyze_launch_delays(self):
        """Analyze scheduling delays before kernel launches"""
        print("\n=== Analyzing Kernel Launch Delays ===")

        for pid, events in self.processes.items():
            launches = events[events['event_type'].str.contains('LaunchKernel', na=False)]
            sched_switches = events[events['event_type'] == 'schedSwitch']

            if len(launches) == 0:
                continue

            delays_before_launch = []

            for _, launch in launches.iterrows():
                launch_time = launch['timestamp_ns']

                # Find the most recent OFF-CPU event before this launch
                offcpu_before = sched_switches[
                    (sched_switches['timestamp_ns'] < launch_time) &
                    (sched_switches['last_offcpu_ns'] > 0)
                ]

                if len(offcpu_before) > 0:
                    last_offcpu = offcpu_before.iloc[-1]

                    # Find when it came back ON-CPU
                    oncpu_after = sched_switches[
                        (sched_switches['timestamp_ns'] > last_offcpu['timestamp_ns']) &
                        (sched_switches['timestamp_ns'] <= launch_time) &
                        (sched_switches['last_oncpu_ns'] > 0)
                    ]

                    if len(oncpu_after) > 0:
                        oncpu_time = oncpu_after.iloc[0]['timestamp_ns']
                        delay = launch_time - oncpu_time
                        delays_before_launch.append(delay)

            if delays_before_launch:
                self.metrics[pid]['launch_delays_ns'] = delays_before_launch
                self.metrics[pid]['avg_launch_delay_us'] = sum(delays_before_launch) / len(delays_before_launch) / 1000
                self.metrics[pid]['max_launch_delay_us'] = max(delays_before_launch) / 1000

                print(f"  PID {pid}: Avg launch delay: {self.metrics[pid]['avg_launch_delay_us']:.2f} µs, "
                      f"Max: {self.metrics[pid]['max_launch_delay_us']:.2f} µs")

    def analyze_sync_impact(self):
        """Analyze off-CPU time during cudaDeviceSynchronize"""
        print("\n=== Analyzing Sync Operation Impact ===")

        for pid, events in self.processes.items():
            sync_enters = events[events['event_type'] == 'syncEnter']
            sync_exits = events[events['event_type'] == 'syncExit']
            sched_switches = events[events['event_type'] == 'schedSwitch']

            if len(sync_enters) == 0:
                continue

            sync_metrics = []

            for _, enter in sync_enters.iterrows():
                enter_time = enter['timestamp_ns']

                # Find corresponding exit
                exits_after = sync_exits[sync_exits['timestamp_ns'] > enter_time]
                if len(exits_after) == 0:
                    continue

                exit_event = exits_after.iloc[0]
                exit_time = exit_event['timestamp_ns']
                total_sync_time = exit_time - enter_time

                # Find all sched switches during sync
                switches_during = sched_switches[
                    (sched_switches['timestamp_ns'] >= enter_time) &
                    (sched_switches['timestamp_ns'] <= exit_time)
                ]

                # Calculate off-CPU time
                offcpu_time = 0
                offcpu_start = None

                for _, switch in switches_during.iterrows():
                    if switch['last_offcpu_ns'] > 0:  # Going OFF-CPU
                        offcpu_start = switch['timestamp_ns']
                    elif switch['last_oncpu_ns'] > 0 and offcpu_start is not None:  # Coming ON-CPU
                        offcpu_time += switch['timestamp_ns'] - offcpu_start
                        offcpu_start = None

                # Handle case where still off-CPU at sync exit
                if offcpu_start is not None:
                    offcpu_time += exit_time - offcpu_start

                sync_metrics.append({
                    'total_time_us': total_sync_time / 1000,
                    'offcpu_time_us': offcpu_time / 1000,
                    'offcpu_ratio': offcpu_time / total_sync_time if total_sync_time > 0 else 0,
                    'num_switches': len(switches_during)
                })

            if sync_metrics:
                avg_offcpu_ratio = sum(m['offcpu_ratio'] for m in sync_metrics) / len(sync_metrics)
                avg_switches = sum(m['num_switches'] for m in sync_metrics) / len(sync_metrics)
                total_sync_time = sum(m['total_time_us'] for m in sync_metrics)
                total_offcpu_time = sum(m['offcpu_time_us'] for m in sync_metrics)

                self.metrics[pid]['sync_count'] = len(sync_metrics)
                self.metrics[pid]['avg_sync_offcpu_ratio'] = avg_offcpu_ratio
                self.metrics[pid]['avg_switches_per_sync'] = avg_switches
                self.metrics[pid]['total_sync_time_ms'] = total_sync_time / 1000
                self.metrics[pid]['total_offcpu_in_sync_ms'] = total_offcpu_time / 1000

                print(f"  PID {pid}: {len(sync_metrics)} sync operations")
                print(f"    Avg off-CPU ratio: {avg_offcpu_ratio*100:.1f}%")
                print(f"    Avg context switches per sync: {avg_switches:.1f}")
                print(f"    Total sync time: {total_sync_time/1000:.2f} ms")
                print(f"    Total off-CPU time: {total_offcpu_time/1000:.2f} ms")

    def analyze_overall_scheduling(self):
        """Analyze overall scheduling behavior"""
        print("\n=== Overall Scheduling Analysis ===")

        for pid, events in self.processes.items():
            launches = events[events['event_type'].str.contains('LaunchKernel', na=False)]
            switches = events[events['event_type'] == 'schedSwitch']

            if len(events) == 0:
                continue

            total_time = events['timestamp_ns'].max() - events['timestamp_ns'].min()

            # Calculate total off-CPU time
            total_offcpu_time = 0
            offcpu_start = None

            for _, switch in switches.iterrows():
                if switch['last_offcpu_ns'] > 0:
                    offcpu_start = switch['timestamp_ns']
                elif switch['last_oncpu_ns'] > 0 and offcpu_start is not None:
                    total_offcpu_time += switch['timestamp_ns'] - offcpu_start
                    offcpu_start = None

            num_switches = len(switches)
            switch_freq = num_switches / (total_time / 1e9) if total_time > 0 else 0  # switches per second

            self.metrics[pid]['total_runtime_ms'] = total_time / 1e6
            self.metrics[pid]['total_offcpu_time_ms'] = total_offcpu_time / 1e6
            self.metrics[pid]['offcpu_ratio'] = total_offcpu_time / total_time if total_time > 0 else 0
            self.metrics[pid]['num_launches'] = len(launches)
            self.metrics[pid]['num_switches'] = num_switches
            self.metrics[pid]['switch_freq_hz'] = switch_freq

            print(f"  PID {pid} ({events['comm'].iloc[0]}):")
            print(f"    Total runtime: {total_time/1e6:.2f} ms")
            print(f"    Total off-CPU: {total_offcpu_time/1e6:.2f} ms ({total_offcpu_time/total_time*100:.1f}%)")
            print(f"    Kernel launches: {len(launches)}")
            print(f"    Context switches: {num_switches} ({switch_freq:.2f} Hz)")

    def calculate_performance_impact(self):
        """Calculate estimated performance impact"""
        print("\n=== Performance Impact Estimation ===")

        for pid, metrics in self.metrics.items():
            impacts = []

            # Impact from launch delays
            if 'avg_launch_delay_us' in metrics and metrics['num_launches'] > 0:
                total_launch_delay = metrics['avg_launch_delay_us'] * metrics['num_launches']
                impacts.append(f"Launch delays: ~{total_launch_delay/1000:.2f} ms total")

            # Impact from sync off-CPU time
            if 'total_offcpu_in_sync_ms' in metrics:
                impacts.append(f"Off-CPU during sync: {metrics['total_offcpu_in_sync_ms']:.2f} ms")

            # Overall scheduling overhead
            if 'offcpu_ratio' in metrics:
                overhead_pct = metrics['offcpu_ratio'] * 100
                impacts.append(f"Overall scheduling overhead: {overhead_pct:.1f}% off-CPU time")

            if impacts:
                print(f"\n  PID {pid}:")
                for impact in impacts:
                    print(f"    - {impact}")

                # Simple performance impact score (0-100, higher = worse)
                score = min(100, metrics.get('offcpu_ratio', 0) * 100 +
                           metrics.get('switch_freq_hz', 0) * 2)
                severity = "LOW" if score < 20 else "MEDIUM" if score < 50 else "HIGH"
                print(f"    Scheduler Impact Score: {score:.1f}/100 [{severity}]")

    def generate_report(self, output_file=None):
        """Generate markdown report"""
        lines = []
        lines.append("# CUDA + CPU Scheduler Impact Analysis Report\n")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Trace File:** {self.csv_file}\n")
        lines.append(f"**Total Events:** {len(self.df)}\n")
        lines.append(f"**GPU Processes:** {len(self.processes)}\n")
        lines.append("\n---\n")

        for pid, metrics in self.metrics.items():
            comm = self.processes[pid]['comm'].iloc[0]
            lines.append(f"\n## Process: {comm} (PID {pid})\n")

            # Overall metrics
            lines.append("### Overall Metrics\n")
            if 'total_runtime_ms' in metrics:
                lines.append(f"- **Total Runtime:** {metrics['total_runtime_ms']:.2f} ms\n")
                lines.append(f"- **Off-CPU Time:** {metrics['total_offcpu_time_ms']:.2f} ms "
                           f"({metrics['offcpu_ratio']*100:.1f}%)\n")
            if 'num_launches' in metrics:
                lines.append(f"- **Kernel Launches:** {metrics['num_launches']}\n")
            if 'num_switches' in metrics:
                lines.append(f"- **Context Switches:** {metrics['num_switches']} "
                           f"({metrics['switch_freq_hz']:.2f} Hz)\n")

            # Launch delays
            if 'avg_launch_delay_us' in metrics:
                lines.append("\n### Kernel Launch Analysis\n")
                lines.append(f"- **Average Launch Delay:** {metrics['avg_launch_delay_us']:.2f} µs\n")
                lines.append(f"- **Maximum Launch Delay:** {metrics['max_launch_delay_us']:.2f} µs\n")

            # Sync analysis
            if 'sync_count' in metrics:
                lines.append("\n### Synchronization Analysis\n")
                lines.append(f"- **Number of Syncs:** {metrics['sync_count']}\n")
                lines.append(f"- **Average Off-CPU Ratio:** {metrics['avg_sync_offcpu_ratio']*100:.1f}%\n")
                lines.append(f"- **Average Switches per Sync:** {metrics['avg_switches_per_sync']:.1f}\n")
                lines.append(f"- **Total Sync Time:** {metrics['total_sync_time_ms']:.2f} ms\n")
                lines.append(f"- **Total Off-CPU in Sync:** {metrics['total_offcpu_in_sync_ms']:.2f} ms\n")

            # Performance impact
            lines.append("\n### Performance Impact\n")
            score = min(100, metrics.get('offcpu_ratio', 0) * 100 +
                       metrics.get('switch_freq_hz', 0) * 2)
            severity = "🟢 LOW" if score < 20 else "🟡 MEDIUM" if score < 50 else "🔴 HIGH"
            lines.append(f"- **Scheduler Impact Score:** {score:.1f}/100 [{severity}]\n")

            # Recommendations
            lines.append("\n### Recommendations\n")
            if metrics.get('offcpu_ratio', 0) > 0.3:
                lines.append("- ⚠️ High off-CPU ratio detected. Consider:\n")
                lines.append("  - Using CPU affinity to pin GPU process to specific cores\n")
                lines.append("  - Increasing process priority (nice value)\n")
                lines.append("  - Reducing other CPU load during GPU operations\n")

            if metrics.get('switch_freq_hz', 0) > 100:
                lines.append("- ⚠️ High context switch frequency. Consider:\n")
                lines.append("  - Batch GPU operations to reduce syscall overhead\n")
                lines.append("  - Use asynchronous CUDA streams\n")

            if metrics.get('avg_sync_offcpu_ratio', 0) > 0.5:
                lines.append("- ⚠️ Process spends >50% off-CPU during sync. Consider:\n")
                lines.append("  - Using CUDA events instead of synchronize when possible\n")
                lines.append("  - Overlapping CPU and GPU work\n")

            if score < 20:
                lines.append("- ✅ Minimal scheduler impact detected. No action needed.\n")

            lines.append("\n---\n")

        report = "".join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\nReport written to: {output_file}")

        return report

    def run_analysis(self, output_file=None):
        """Run complete analysis"""
        self.load_data()
        self.analyze_launch_delays()
        self.analyze_sync_impact()
        self.analyze_overall_scheduling()
        self.calculate_performance_impact()

        report = self.generate_report(output_file)

        if not output_file:
            print("\n" + "="*80)
            print(report)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze CPU scheduler impact on GPU workload performance'
    )
    parser.add_argument('trace_csv', help='Path to trace CSV file from cuda_sched_trace')
    parser.add_argument('-o', '--output', help='Output markdown report file')

    args = parser.parse_args()

    analyzer = GPUSchedulerAnalyzer(args.trace_csv)
    analyzer.run_analysis(args.output)


if __name__ == '__main__':
    main()
