// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * Chunk Trace Tool - Trace BPF hook calls using kprobes
 *
 * Replicates trace_bpf_hooks_only.bt functionality:
 * - Traces only the BPF hook wrapper functions
 * - Outputs timestamp, hook type, chunk address, list address
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "GPL";

// Ring buffer for outputting events
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

// Event structure
struct hook_event {
    u64 timestamp_ns;
    u32 hook_type;
    u32 cpu;
    u64 chunk_addr;
    u64 list_addr;
};

// Hook types
#define HOOK_ACTIVATE 1
#define HOOK_POPULATE 2
#define HOOK_DEPOPULATE 3
#define HOOK_EVICTION_PREPARE 4

// Statistics counters
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 5);
    __type(key, u32);
    __type(value, u64);
} stats SEC(".maps");

#define STAT_ACTIVATE 0
#define STAT_POPULATE 1
#define STAT_DEPOPULATE 2
#define STAT_EVICTION_PREPARE 3
#define STAT_DROPPED 4

static __always_inline void inc_stat(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&stats, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

static __always_inline void submit_event(u32 hook_type, u64 chunk, u64 list)
{
    struct hook_event *e;

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->hook_type = hook_type;
    e->chunk_addr = chunk;
    e->list_addr = list;

    bpf_ringbuf_submit(e, 0);
}

/* Hook 1: Activate (chunk becomes evictable) */
SEC("kprobe/uvm_bpf_call_pmm_chunk_activate")
int BPF_KPROBE(trace_activate, void *pmm, void *chunk, void *list)
{
    inc_stat(STAT_ACTIVATE);
    submit_event(HOOK_ACTIVATE, (u64)chunk, (u64)list);
    return 0;
}

/* Hook 2: Populate (chunk gets first resident page) */
SEC("kprobe/uvm_bpf_call_pmm_chunk_populate")
int BPF_KPROBE(trace_populate, void *pmm, void *chunk, void *list)
{
    inc_stat(STAT_POPULATE);
    submit_event(HOOK_POPULATE, (u64)chunk, (u64)list);
    return 0;
}

/* Hook 3: Depopulate (chunk loses all resident pages) */
SEC("kprobe/uvm_bpf_call_pmm_chunk_depopulate")
int BPF_KPROBE(trace_depopulate, void *pmm, void *chunk, void *list)
{
    inc_stat(STAT_DEPOPULATE);
    submit_event(HOOK_DEPOPULATE, (u64)chunk, (u64)list);
    return 0;
}

/* Hook 4: Eviction prepare (before selecting chunk to evict) */
SEC("kprobe/uvm_bpf_call_pmm_eviction_prepare")
int BPF_KPROBE(trace_eviction_prepare, void *pmm, void *used_list, void *unused_list)
{
    inc_stat(STAT_EVICTION_PREPARE);
    // For eviction_prepare, chunk_addr stores used_list, list_addr stores unused_list
    submit_event(HOOK_EVICTION_PREPARE, (u64)used_list, (u64)unused_list);
    return 0;
}
