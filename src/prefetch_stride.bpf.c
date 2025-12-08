/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Stride-based Prefetch Policy
 *
 * Detects stride access patterns at page level and prefetches accordingly.
 * - Tracks access history per VA block
 * - Detects stable stride patterns
 * - Prefetches predicted next page(s) based on stride
 *
 * Confidence decay: when stride changes, confidence decreases by 1 instead
 * of resetting to 0, making it robust to occasional irregular accesses.
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"
#include "trace_helper.h"

char _license[] SEC("license") = "GPL";

/* Configuration keys */
#define CONFIG_CONFIDENCE_THRESHOLD  0  /* Min confidence to trigger prefetch */
#define CONFIG_PREFETCH_PAGES        1  /* Number of pages to prefetch */
#define CONFIG_MAX_STRIDE            2  /* Maximum allowed stride */

/* Configuration map */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 8);
    __type(key, u32);
    __type(value, u64);
} config SEC(".maps");

/* Stride state per VA block */
struct stride_state {
    s32 last_page;       /* Last accessed page_index (-1 = uninitialized) */
    s32 stride;          /* Detected stride */
    s32 confidence;      /* Confidence level (can go negative) */
    u32 total_faults;    /* Total page faults (stats) */
    u32 prefetch_count;  /* Prefetches issued (stats) */
    u32 stride_hits;     /* Times stride matched (stats) */
};

/* Per VA block stride tracking: va_block_ptr -> stride_state */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, u64);    /* va_block pointer */
    __type(value, struct stride_state);
} stride_map SEC(".maps");

/* Per-CPU cache for current VA block pointer */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);  /* va_block pointer */
} va_block_cache SEC(".maps");

/* Global statistics */
struct stride_stats {
    u64 total_faults;
    u64 prefetch_issued;
    u64 stride_detected;
    u64 no_prefetch;
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct stride_stats);
} global_stats SEC(".maps");

/* Helper: Get config value with default */
static __always_inline u64 get_config(u32 key, u64 default_val)
{
    u64 *val = bpf_map_lookup_elem(&config, &key);
    return val ? *val : default_val;
}

/* Helper: Update global stats */
static __always_inline void update_stats(bool prefetched)
{
    u32 key = 0;
    struct stride_stats *stats = bpf_map_lookup_elem(&global_stats, &key);
    if (stats) {
        __sync_fetch_and_add(&stats->total_faults, 1);
        if (prefetched) {
            __sync_fetch_and_add(&stats->prefetch_issued, 1);
        } else {
            __sync_fetch_and_add(&stats->no_prefetch, 1);
        }
    }
}

/*
 * Hook: uvm_perf_prefetch_get_hint_va_block (via kprobe)
 * Captures va_block pointer for stride tracking.
 */
SEC("kprobe/uvm_perf_prefetch_get_hint_va_block")
int BPF_KPROBE(prefetch_get_hint_va_block,
               uvm_va_block_t *va_block,
               void *va_block_context,
               u32 new_residency,
               void *faulted_pages,
               u32 faulted_region_packed,
               uvm_perf_prefetch_bitmap_tree_t *bitmap_tree)
{
    u32 key = 0;
    u64 *cached = bpf_map_lookup_elem(&va_block_cache, &key);
    if (cached) {
        *cached = (u64)va_block;
    }
    return 0;
}

/* Helper: Get cached va_block pointer */
static __always_inline u64 get_cached_va_block(void)
{
    u32 key = 0;
    u64 *cached = bpf_map_lookup_elem(&va_block_cache, &key);
    return cached ? *cached : 0;
}

/* Helper: Absolute value for s32 */
static __always_inline s32 abs_s32(s32 x)
{
    return x < 0 ? -x : x;
}

SEC("struct_ops/uvm_prefetch_before_compute")
int BPF_PROG(uvm_prefetch_before_compute,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *result_region)
{
    u64 va_block_ptr = get_cached_va_block();
    s32 confidence_threshold = (s32)get_config(CONFIG_CONFIDENCE_THRESHOLD, 2);
    u32 prefetch_pages = (u32)get_config(CONFIG_PREFETCH_PAGES, 2);
    s32 max_stride = (s32)get_config(CONFIG_MAX_STRIDE, 128);

    /* Read max_prefetch_region bounds */
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);

    /* Default: no prefetch */
    bpf_uvm_set_va_block_region(result_region, 0, 0);

    if (va_block_ptr == 0) {
        update_stats(false);
        return 1; /* BYPASS */
    }

    /* Lookup or create stride state for this va_block */
    struct stride_state *state = bpf_map_lookup_elem(&stride_map, &va_block_ptr);
    struct stride_state new_state = {
        .last_page = -1,
        .stride = 0,
        .confidence = 0,
        .total_faults = 0,
        .prefetch_count = 0,
        .stride_hits = 0,
    };

    if (!state) {
        /* First time seeing this va_block */
        new_state.last_page = (s32)page_index;
        new_state.total_faults = 1;
        bpf_map_update_elem(&stride_map, &va_block_ptr, &new_state, BPF_ANY);
        update_stats(false);
        return 1; /* BYPASS, no prefetch on first access */
    }

    /* Update stats */
    __sync_fetch_and_add(&state->total_faults, 1);

    /* Check if this is first real access (last_page was -1) */
    if (state->last_page < 0) {
        state->last_page = (s32)page_index;
        update_stats(false);
        return 1; /* BYPASS */
    }

    /* Calculate current stride */
    s32 current_stride = (s32)page_index - state->last_page;

    /* Update last_page */
    state->last_page = (s32)page_index;

    /* Handle stride = 0 (same page accessed again) */
    if (current_stride == 0) {
        update_stats(false);
        return 1; /* BYPASS */
    }

    /* Check if stride is within allowed range */
    if (abs_s32(current_stride) > max_stride) {
        /* Stride too large, decay confidence */
        if (state->confidence > 0)
            state->confidence--;
        update_stats(false);
        return 1; /* BYPASS */
    }

    /* Update stride and confidence */
    if (current_stride == state->stride) {
        /* Stride matches, increase confidence */
        state->confidence++;
        __sync_fetch_and_add(&state->stride_hits, 1);
    } else {
        /* Stride changed, decay confidence and update stride */
        if (state->confidence > 0)
            state->confidence--;
        state->stride = current_stride;
    }

    /* Check if we should prefetch */
    if (state->confidence >= confidence_threshold) {
        /* Predict next access */
        s32 predicted = (s32)page_index + state->stride;

        /* Calculate prefetch region */
        s32 pf_first, pf_outer;

        if (state->stride > 0) {
            /* Forward stride */
            pf_first = predicted;
            pf_outer = predicted + (s32)prefetch_pages;
        } else {
            /* Backward stride */
            pf_first = predicted - (s32)prefetch_pages + 1;
            pf_outer = predicted + 1;
        }

        /* Clamp to valid range */
        if (pf_first < (s32)max_first)
            pf_first = (s32)max_first;
        if (pf_outer > (s32)max_outer)
            pf_outer = (s32)max_outer;
        if (pf_first < 0)
            pf_first = 0;

        /* Only prefetch if we have a valid region */
        if (pf_first < pf_outer) {
            bpf_uvm_set_va_block_region(result_region,
                                        (uvm_page_index_t)pf_first,
                                        (uvm_page_index_t)pf_outer);
            __sync_fetch_and_add(&state->prefetch_count, 1);

            bpf_printk("stride_prefetch: page=%d, stride=%d, conf=%d, pf=[%d,%d)\n",
                       page_index, state->stride, state->confidence, pf_first, pf_outer);

            update_stats(true);
            return 1; /* BYPASS */
        }
    }

    update_stats(false);
    return 1; /* BYPASS */
}

/* Not used - we handle everything in before_compute */
SEC("struct_ops/uvm_prefetch_on_tree_iter")
int BPF_PROG(uvm_prefetch_on_tree_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_va_block_region_t *prefetch_region)
{
    return 0;
}

/* Dummy implementation for test trigger */
SEC("struct_ops/uvm_bpf_test_trigger_kfunc")
int BPF_PROG(uvm_bpf_test_trigger_kfunc, const char *buf, int len)
{
    return 0;
}

/* Define the struct_ops map */
SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops_stride = {
    .uvm_bpf_test_trigger_kfunc = (void *)uvm_bpf_test_trigger_kfunc,
    .uvm_prefetch_before_compute = (void *)uvm_prefetch_before_compute,
    .uvm_prefetch_on_tree_iter = (void *)uvm_prefetch_on_tree_iter,
};
