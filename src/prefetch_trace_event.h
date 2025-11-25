// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */

#ifndef __PREFETCH_TRACE_EVENT_H
#define __PREFETCH_TRACE_EVENT_H

// Hook types for prefetch tracing
#define HOOK_PREFETCH_BEFORE_COMPUTE 1
#define HOOK_PREFETCH_ON_TREE_ITER 2
#define HOOK_PREFETCH_BEFORE_COMPUTE_RET 3
#define HOOK_PREFETCH_ON_TREE_ITER_RET 4

// Event structure shared between BPF and userspace
struct prefetch_event {
    __u64 timestamp_ns;
    __u32 hook_type;
    __u32 cpu;

    // For before_compute
    __u32 page_index;           // Triggering page index
    __u32 max_region_first;     // max_prefetch_region.first
    __u32 max_region_outer;     // max_prefetch_region.outer
    __u32 result_region_first;  // result_region.first (output)
    __u32 result_region_outer;  // result_region.outer (output)

    // For on_tree_iter
    __u32 current_region_first; // current_region.first
    __u32 current_region_outer; // current_region.outer
    __u32 counter;              // Access counter for this region
    __u32 selected;             // Whether this region was selected (return value)
};

#endif /* __PREFETCH_TRACE_EVENT_H */
