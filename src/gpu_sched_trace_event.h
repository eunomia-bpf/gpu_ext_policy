// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */

#ifndef __GPU_SCHED_TRACE_EVENT_H
#define __GPU_SCHED_TRACE_EVENT_H

// Hook types for GPU scheduler
#define HOOK_TASK_INIT      1   // TSG (channel group) creation
#define HOOK_BIND           2   // TSG bind to hardware runlist
#define HOOK_TOKEN_REQUEST  3   // Work submit token request (for sync)
#define HOOK_TASK_DESTROY   4   // TSG destruction

// Event structure shared between BPF and userspace
struct gpu_sched_event {
    __u64 timestamp_ns;     // Event timestamp
    __u32 hook_type;        // Type of hook (HOOK_TASK_INIT, etc.)
    __u32 cpu;              // CPU where event occurred

    // Process info (from BPF helpers)
    __u32 pid;              // Process ID (thread ID)
    __u32 tgid;             // Thread group ID (main process PID)
    char comm[16];          // Process name

    // Common fields
    __u64 tsg_id;           // TSG (channel group) ID

    // task_init specific fields
    __u32 engine_type;      // Engine type (GRAPHICS, COPY, etc.)
    __u64 timeslice_us;     // Timeslice in microseconds
    __u32 interleave_level; // Interleave level (LOW/MEDIUM/HIGH)
    __u32 runlist_id;       // Runlist ID

    // bind specific fields
    __u32 channel_count;    // Number of channels in TSG
    __u32 allow;            // Whether binding was allowed

    // token_request specific fields
    __u32 channel_id;       // Channel ID
    __u32 token;            // Work submit token
};

// Engine types (from NVIDIA driver)
#define ENGINE_TYPE_GRAPHICS    0
#define ENGINE_TYPE_COPY        1
#define ENGINE_TYPE_NVDEC       2
#define ENGINE_TYPE_NVENC       3
#define ENGINE_TYPE_NVJPEG      4

// Interleave levels (from NVA06C_CTRL_INTERLEAVE_LEVEL_*)
#define INTERLEAVE_LEVEL_LOW    1
#define INTERLEAVE_LEVEL_MEDIUM 2
#define INTERLEAVE_LEVEL_HIGH   3

#endif /* __GPU_SCHED_TRACE_EVENT_H */
