// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */

#ifndef __PREFETCH_TRACE_EVENT_H
#define __PREFETCH_TRACE_EVENT_H

// Hook types for prefetch tracing
#define HOOK_PREFETCH_BEFORE_COMPUTE 1
#define HOOK_PREFETCH_ON_TREE_ITER 2
#define HOOK_PREFETCH_GET_HINT 3

// Event structure shared between BPF and userspace
struct prefetch_event {
    __u64 timestamp_ns;
    __u32 cpu;
    __u32 hook_type;

    // Page fault info
    __u32 page_index;           // Triggering page index

    // max_prefetch_region
    __u32 max_region_first;     // max_prefetch_region.first
    __u32 max_region_outer;     // max_prefetch_region.outer

    // bitmap_tree info
    __u32 tree_offset;          // bitmap_tree->offset
    __u32 tree_leaf_count;      // bitmap_tree->leaf_count
    __u32 tree_level_count;     // bitmap_tree->level_count
    __u32 pages_accessed;       // popcount of bitmap_tree->pages (how many pages already accessed)

    // VA block info (from uvm_perf_prefetch_get_hint_va_block)
    __u64 va_block;             // va_block pointer
    __u64 va_start;             // va_block->start
    __u64 va_end;               // va_block->end

    // faulted_region info
    __u32 faulted_first;        // faulted_region.first
    __u32 faulted_outer;        // faulted_region.outer
};

#endif /* __PREFETCH_TRACE_EVENT_H */
