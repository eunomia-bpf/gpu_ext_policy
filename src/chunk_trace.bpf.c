// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * Chunk Trace Tool - Trace BPF hook calls using kprobes
 *
 * Traces BPF hook wrapper functions and extracts VA block information:
 * - Timestamp, hook type, chunk address, list address
 * - VA block pointer, VA start/end addresses, page index
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "chunk_trace_event.h"
#include "uvm_types.h"

char LICENSE[] SEC("license") = "GPL";

// Ring buffer for outputting events
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

// Statistics counters
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 4);
    __type(key, u32);
    __type(value, u64);
} stats SEC(".maps");

#define STAT_ACTIVATE 0
#define STAT_POPULATE 1
#define STAT_EVICTION_PREPARE 2
#define STAT_DROPPED 3

static __always_inline void inc_stat(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&stats, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

static __always_inline void submit_event(u32 hook_type, u64 chunk, u64 list)
{
    struct hook_event *e;
    u64 va_block_ptr = 0;
    u64 va_start = 0;
    u64 va_end = 0;
    u32 va_page_index = 0;
    u32 owner_pid = 0;
    u64 va_space_ptr = 0;

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return;
    }

    // Use BPF CO-RE to read VA block information from chunk
    if (chunk != 0 && hook_type != HOOK_EVICTION_PREPARE) {
        struct uvm_gpu_chunk_struct *gpu_chunk = (struct uvm_gpu_chunk_struct *)chunk;
        uvm_va_block_t *va_block;

        // Read va_block pointer using CO-RE
        va_block = BPF_CORE_READ(gpu_chunk, va_block);
        va_block_ptr = (u64)va_block;

        if (va_block_ptr != 0) {
            // Read VA block start and end addresses using CO-RE
            va_start = BPF_CORE_READ(va_block, start);
            va_end = BPF_CORE_READ(va_block, end);

            // Read va_space pointer: va_block->managed_range->va_range.va_space
            uvm_va_range_managed_t *managed_range = BPF_CORE_READ(va_block, managed_range);
            if (managed_range != 0) {
                uvm_va_space_t *va_space = BPF_CORE_READ(managed_range, va_range.va_space);
                va_space_ptr = (u64)va_space;

                // Read owner PID: va_space->va_space_mm.mm->owner->tgid
                if (va_space != 0) {
                    struct mm_struct *mm = BPF_CORE_READ(va_space, va_space_mm.mm);
                    if (mm != 0) {
                        struct task_struct *owner = BPF_CORE_READ(mm, owner);
                        if (owner != 0) {
                            owner_pid = BPF_CORE_READ(owner, tgid);
                        }
                    }
                }
            }

            // Read va_block_page_index from chunk
            // This is a bitfield, so we need to read the whole struct then extract
            struct uvm_gpu_chunk_struct chunk_copy;
            bpf_probe_read_kernel(&chunk_copy, sizeof(chunk_copy), gpu_chunk);
            va_page_index = chunk_copy.va_block_page_index;
        }
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->owner_pid = owner_pid;
    e->va_space = va_space_ptr;
    e->hook_type = hook_type;
    e->chunk_addr = chunk;
    e->list_addr = list;
    e->va_block = va_block_ptr;
    e->va_start = va_start;
    e->va_end = va_end;
    e->va_page_index = va_page_index;

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

/* Hook 2: Used (chunk gets accessed/used) */
SEC("kprobe/uvm_bpf_call_pmm_chunk_used")
int BPF_KPROBE(trace_used, void *pmm, void *chunk, void *list)
{
    inc_stat(STAT_POPULATE);
    submit_event(HOOK_POPULATE, (u64)chunk, (u64)list);
    return 0;
}

/* Hook 3: Eviction prepare (before selecting chunk to evict) */
SEC("kprobe/uvm_bpf_call_pmm_eviction_prepare")
int BPF_KPROBE(trace_eviction_prepare, void *pmm, void *used_list, void *unused_list)
{
    inc_stat(STAT_EVICTION_PREPARE);
    // For eviction_prepare, chunk_addr stores used_list, list_addr stores unused_list
    submit_event(HOOK_EVICTION_PREPARE, (u64)used_list, (u64)unused_list);
    return 0;
}
