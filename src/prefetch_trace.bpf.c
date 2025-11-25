// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * Prefetch Trace Tool - Trace prefetch BPF hook calls using kprobes
 *
 * Traces prefetch-related BPF hook wrapper functions:
 * - uvm_bpf_call_before_compute_prefetch: Called before computing prefetch region
 * - uvm_bpf_call_on_tree_iter: Called on each tree iteration
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "prefetch_trace_event.h"
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

#define STAT_BEFORE_COMPUTE 0
#define STAT_ON_TREE_ITER 1
#define STAT_DROPPED 2

static __always_inline void inc_stat(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&stats, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

/*
 * Hook 1: uvm_bpf_call_before_compute_prefetch
 *
 * Function signature:
 * enum uvm_bpf_action uvm_bpf_call_before_compute_prefetch(
 *     uvm_page_index_t page_index,
 *     uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
 *     uvm_va_block_region_t *max_prefetch_region,
 *     uvm_va_block_region_t *result_region)
 */
SEC("kprobe/uvm_bpf_call_before_compute_prefetch")
int BPF_KPROBE(trace_before_compute,
               u32 page_index,
               void *bitmap_tree,
               uvm_va_block_region_t *max_prefetch_region,
               uvm_va_block_region_t *result_region)
{
    struct prefetch_event *e;

    inc_stat(STAT_BEFORE_COMPUTE);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->hook_type = HOOK_PREFETCH_BEFORE_COMPUTE;
    e->page_index = page_index;

    // Read max_prefetch_region
    if (max_prefetch_region) {
        e->max_region_first = BPF_CORE_READ(max_prefetch_region, first);
        e->max_region_outer = BPF_CORE_READ(max_prefetch_region, outer);
    } else {
        e->max_region_first = 0;
        e->max_region_outer = 0;
    }

    // result_region will be filled by the BPF program, we trace entry
    e->result_region_first = 0;
    e->result_region_outer = 0;

    // Not used for before_compute
    e->current_region_first = 0;
    e->current_region_outer = 0;
    e->counter = 0;
    e->selected = 0;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Hook 1 return: Capture result_region after call
 */
SEC("kretprobe/uvm_bpf_call_before_compute_prefetch")
int BPF_KRETPROBE(trace_before_compute_ret, int ret)
{
    struct prefetch_event *e;

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->hook_type = HOOK_PREFETCH_BEFORE_COMPUTE_RET;
    e->page_index = 0;
    e->max_region_first = 0;
    e->max_region_outer = 0;
    e->result_region_first = 0;
    e->result_region_outer = 0;
    e->current_region_first = 0;
    e->current_region_outer = 0;
    e->counter = 0;
    e->selected = (u32)ret;  // Store return value (action)

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Hook 2: uvm_bpf_call_on_tree_iter
 *
 * Function signature:
 * enum uvm_bpf_action uvm_bpf_call_on_tree_iter(
 *     uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
 *     uvm_va_block_region_t *max_prefetch_region,
 *     uvm_va_block_region_t *current_region,
 *     unsigned int counter,
 *     uvm_va_block_region_t *prefetch_region)
 */
SEC("kprobe/uvm_bpf_call_on_tree_iter")
int BPF_KPROBE(trace_on_tree_iter,
               void *bitmap_tree,
               uvm_va_block_region_t *max_prefetch_region,
               uvm_va_block_region_t *current_region,
               unsigned int counter,
               uvm_va_block_region_t *prefetch_region)
{
    struct prefetch_event *e;

    inc_stat(STAT_ON_TREE_ITER);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->hook_type = HOOK_PREFETCH_ON_TREE_ITER;
    e->page_index = 0;

    // Read max_prefetch_region
    if (max_prefetch_region) {
        e->max_region_first = BPF_CORE_READ(max_prefetch_region, first);
        e->max_region_outer = BPF_CORE_READ(max_prefetch_region, outer);
    } else {
        e->max_region_first = 0;
        e->max_region_outer = 0;
    }

    // Read current_region
    if (current_region) {
        e->current_region_first = BPF_CORE_READ(current_region, first);
        e->current_region_outer = BPF_CORE_READ(current_region, outer);
    } else {
        e->current_region_first = 0;
        e->current_region_outer = 0;
    }

    e->counter = counter;
    e->result_region_first = 0;
    e->result_region_outer = 0;
    e->selected = 0;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Hook 2 return: Capture return value (selected or not)
 */
SEC("kretprobe/uvm_bpf_call_on_tree_iter")
int BPF_KRETPROBE(trace_on_tree_iter_ret, int ret)
{
    struct prefetch_event *e;

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->hook_type = HOOK_PREFETCH_ON_TREE_ITER_RET;
    e->page_index = 0;
    e->max_region_first = 0;
    e->max_region_outer = 0;
    e->current_region_first = 0;
    e->current_region_outer = 0;
    e->counter = 0;
    e->result_region_first = 0;
    e->result_region_outer = 0;
    e->selected = (u32)ret;  // Store return value

    bpf_ringbuf_submit(e, 0);
    return 0;
}
