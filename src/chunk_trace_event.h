// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */

#ifndef __CHUNK_TRACE_EVENT_H
#define __CHUNK_TRACE_EVENT_H

// Hook types
#define HOOK_ACTIVATE 1
#define HOOK_POPULATE 2
#define HOOK_EVICTION_PREPARE 3

// Event structure shared between BPF and userspace
struct hook_event {
    __u64 timestamp_ns;
    __u32 hook_type;
    __u32 cpu;
    __u32 pid;            // Current kernel thread PID
    __u32 owner_pid;      // Process that owns this VA block (from va_block->cpu.fault_authorized.first_pid)
    __u64 va_space;       // va_space pointer - unique per process
    __u64 chunk_addr;
    __u64 list_addr;
    __u64 va_block;       // VA block pointer
    __u64 va_start;       // VA block start address
    __u64 va_end;         // VA block end address
    __u32 va_page_index;  // Page index within VA block
};

#endif /* __CHUNK_TRACE_EVENT_H */
