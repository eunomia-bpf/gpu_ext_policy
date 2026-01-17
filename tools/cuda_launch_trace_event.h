// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */

#ifndef __CUDA_LAUNCH_TRACE_EVENT_H
#define __CUDA_LAUNCH_TRACE_EVENT_H

#define TASK_COMM_LEN 16

// Event types
enum hook_type {
    HOOK_CULAUNCHKERNEL = 1,
    HOOK_CUDALAUNCHKERNEL = 2,
    HOOK_SYNC_ENTER = 3,
    HOOK_SYNC_EXIT = 4,
    HOOK_SCHED_SWITCH = 5,
    HOOK_HARDIRQ_ENTRY = 6,
    HOOK_HARDIRQ_EXIT = 7,
    HOOK_SOFTIRQ_ENTRY = 8,
    HOOK_SOFTIRQ_EXIT = 9,
};

// Event structure for CUDA kernel launch
struct cuda_launch_event {
    __u64 timestamp_ns;
    __u32 pid;
    __u32 tid;
    char comm[TASK_COMM_LEN];
    __u32 hook_type;

    // Kernel launch parameters
    __u64 function;          // Kernel function pointer
    __u32 grid_dim_x;
    __u32 grid_dim_y;
    __u32 grid_dim_z;
    __u32 block_dim_x;
    __u32 block_dim_y;
    __u32 block_dim_z;
    __u32 shared_mem_bytes;
    __u64 stream;            // CUDA stream pointer

    // Scheduler impact tracking
    __u64 last_offcpu_ns;    // Last time switched off CPU
    __u64 last_oncpu_ns;     // Last time switched on CPU
    __u64 total_offcpu_ns;   // Total off-CPU time since launch/sync_enter
    __u32 switch_count;      // Number of context switches
    __u32 cpu_id;            // CPU ID where event occurred
};

// Sync tracking event
struct sync_event {
    __u64 timestamp_ns;
    __u32 pid;
    __u32 tid;
    char comm[TASK_COMM_LEN];
    __u32 hook_type;         // HOOK_SYNC_ENTER or HOOK_SYNC_EXIT

    __u64 duration_ns;       // For exit: total sync duration
    __u64 offcpu_time_ns;    // Time spent off-CPU during sync
    __u32 switch_count;      // Context switches during sync
    __u32 cpu_id;
};

// IRQ tracking event
struct irq_event {
    __u64 timestamp_ns;
    __u32 pid;               // PID of interrupted GPU process
    __u32 tid;
    char comm[TASK_COMM_LEN];
    __u32 hook_type;         // HOOK_HARDIRQ_ENTRY/EXIT or HOOK_SOFTIRQ_ENTRY/EXIT

    __u32 irq;               // IRQ number (for hardirq) or vec_nr (for softirq)
    __u32 cpu_id;
    __u64 duration_ns;       // Only valid for EXIT events
    char irq_name[32];       // IRQ handler name (for hardirq)
};

#endif /* __CUDA_LAUNCH_TRACE_EVENT_H */
