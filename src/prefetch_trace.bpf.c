// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * Prefetch Trace Tool - Trace prefetch calls using kprobes
 *
 * Traces uvm_perf_prefetch_get_hint_va_block to understand prefetch behavior
 * with VA block information (virtual addresses)
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

#define STAT_GET_HINT 0
#define STAT_BEFORE_COMPUTE 1
#define STAT_ON_TREE_ITER 2
#define STAT_DROPPED 3

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
 * Hook: uvm_perf_prefetch_get_hint_va_block
 *
 * Function signature:
 * void uvm_perf_prefetch_get_hint_va_block(
 *     uvm_va_block_t *va_block,                      // arg1 (rdi)
 *     uvm_va_block_context_t *va_block_context,      // arg2 (rsi)
 *     uvm_processor_id_t new_residency,              // arg3 (rdx)
 *     uvm_page_mask_t *faulted_pages,                // arg4 (rcx)
 *     uvm_va_block_region_t faulted_region,          // arg5 (r8) - passed by value (32-bit)
 *     uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,  // arg6 (r9)
 *     uvm_perf_prefetch_hint_t *out_hint             // arg7 (stack)
 * )
 *
 * Note: faulted_region is passed by value as a 32-bit struct (2x u16),
 * so on x86_64 it comes in register r8
 * Note: arg7 is on stack, we don't need it for tracing
 */
SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(trace_get_hint_va_block,
               uvm_va_block_t *va_block,
               void *va_block_context,
               u32 new_residency,
               void *faulted_pages,
               u32 faulted_region_packed,  /* first:16 | outer:16 */
               uvm_perf_prefetch_bitmap_tree_t *bitmap_tree)
{
    struct prefetch_event *e;

    inc_stat(STAT_GET_HINT);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->hook_type = HOOK_PREFETCH_GET_HINT;

    // Extract faulted_region (packed as first:16, outer:16)
    e->faulted_first = faulted_region_packed & 0xFFFF;
    e->faulted_outer = (faulted_region_packed >> 16) & 0xFFFF;
    e->page_index = e->faulted_first;  // Use faulted_first as page_index

    // Read VA block info using CO-RE
    if (va_block) {
        e->va_block = (u64)va_block;
        e->va_start = BPF_CORE_READ(va_block, start);
        e->va_end = BPF_CORE_READ(va_block, end);
    } else {
        e->va_block = 0;
        e->va_start = 0;
        e->va_end = 0;
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

    // max_region not available in this hook, set to 0
    e->max_region_first = 0;
    e->max_region_outer = 0;

    bpf_ringbuf_submit(e, 0);
    return 0;
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
 *
 * Note: This hook doesn't have va_block, so no virtual address info
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

    // No VA block info in this hook
    e->va_block = 0;
    e->va_start = 0;
    e->va_end = 0;
    e->faulted_first = 0;
    e->faulted_outer = 0;

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
