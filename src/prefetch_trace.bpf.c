// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * Prefetch Trace Tool - Trace prefetch BPF hook calls using kprobes
 *
 * Traces uvm_bpf_call_before_compute_prefetch to understand prefetch behavior
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
    __uint(max_entries, 3);
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

// Count set bits in a 64-bit word
static __always_inline u32 popcount64(u64 x)
{
    u32 count = 0;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        count += (x >> i) & 1;
    }
    return count;
}

/*
 * Hook: uvm_bpf_call_before_compute_prefetch
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
               uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
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

    // Read max_prefetch_region using CO-RE
    if (max_prefetch_region) {
        e->max_region_first = BPF_CORE_READ(max_prefetch_region, first);
        e->max_region_outer = BPF_CORE_READ(max_prefetch_region, outer);
    } else {
        e->max_region_first = 0;
        e->max_region_outer = 0;
    }

    // Read bitmap_tree info using CO-RE
    if (bitmap_tree) {
        e->tree_offset = BPF_CORE_READ(bitmap_tree, offset);
        e->tree_leaf_count = BPF_CORE_READ(bitmap_tree, leaf_count);
        e->tree_level_count = BPF_CORE_READ(bitmap_tree, level_count);

        // Count total pages accessed (popcount of bitmap)
        u32 total = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            u64 bits = BPF_CORE_READ(bitmap_tree, pages.bitmap[i]);
            total += popcount64(bits);
        }
        e->pages_accessed = total;
    } else {
        e->tree_offset = 0;
        e->tree_leaf_count = 0;
        e->tree_level_count = 0;
        e->pages_accessed = 0;
    }

    bpf_ringbuf_submit(e, 0);
    return 0;
}
