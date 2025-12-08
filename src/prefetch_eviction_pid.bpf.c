/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Combined PID-based Prefetch + Probabilistic LRU Eviction Policy
 *
 * Prefetch:
 *   - High priority PID gets lower threshold (more prefetch bandwidth)
 *   - Low priority PID gets higher threshold (less prefetch bandwidth)
 *
 * Eviction:
 *   - Uses probabilistic LRU based on priority / 10
 *   - priority = 0: every access moves to tail (always protected)
 *   - priority = 100: every 10 accesses moves to tail (less protected)
 *   - Lower priority value = more protection, higher = less protection
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"
#include "eviction_common.h"
#include "trace_helper.h"

char _license[] SEC("license") = "GPL";

/* Configuration map - shared for both prefetch and eviction */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 8);
    __type(key, u32);
    __type(value, u64);
} config SEC(".maps");

/* Per-PID statistics */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u32);  /* PID */
    __type(value, struct pid_chunk_stats);
} pid_stats SEC(".maps");

/* Per-CPU cache for current VA block's owner PID (for prefetch) */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);  /* owner_tgid */
} va_block_pid_cache SEC(".maps");

/* Active chunk tracking: chunk_ptr -> owner_pid (for eviction) */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, u64);    /* chunk pointer */
    __type(value, u32);  /* owner_pid */
} active_chunks SEC(".maps");

/* Per-chunk access counter: chunk_ptr -> access_count (for eviction) */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, u64);    /* chunk pointer */
    __type(value, u64);  /* access count */
} chunk_access_count SEC(".maps");

/* Helper: Get config value */
static __always_inline u64 get_config_u64(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&config, &key);
    return val ? *val : 0;
}

/* Helper: Get prefetch threshold for a specific PID (lower = more prefetch) */
static __always_inline u32 get_prefetch_threshold_for_pid(u32 pid)
{
    u64 *high_pid_ptr, *high_param_ptr;
    u64 *low_pid_ptr, *low_param_ptr;
    u64 *default_param_ptr;
    u32 key;

    /* Get high priority PID and its threshold */
    key = CONFIG_PRIORITY_PID;
    high_pid_ptr = bpf_map_lookup_elem(&config, &key);
    key = CONFIG_PRIORITY_PARAM;
    high_param_ptr = bpf_map_lookup_elem(&config, &key);

    if (high_pid_ptr && high_param_ptr && *high_pid_ptr == pid) {
        return (u32)*high_param_ptr;
    }

    /* Get low priority PID and its threshold */
    key = CONFIG_LOW_PRIORITY_PID;
    low_pid_ptr = bpf_map_lookup_elem(&config, &key);
    key = CONFIG_LOW_PRIORITY_PARAM;
    low_param_ptr = bpf_map_lookup_elem(&config, &key);

    if (low_pid_ptr && low_param_ptr && *low_pid_ptr == pid) {
        return (u32)*low_param_ptr;
    }

    /* Default threshold for other PIDs */
    key = CONFIG_DEFAULT_PARAM;
    default_param_ptr = bpf_map_lookup_elem(&config, &key);
    if (default_param_ptr) {
        return (u32)*default_param_ptr;
    }

    return 50;  /* Default 50% if not configured */
}

/*
 * Helper: Get eviction decay factor for a specific PID
 *
 * priority / 10 determines how many accesses before move_tail:
 *   priority = 0   -> decay_factor = 1 (every access moves to tail)
 *   priority = 50  -> decay_factor = 5 (every 5 accesses)
 *   priority = 100 -> decay_factor = 10 (every 10 accesses)
 *
 * Lower priority value = more protection (more frequent move to tail)
 */
static __always_inline u64 get_eviction_decay_for_pid(u32 pid)
{
    u64 priority_pid = get_config_u64(CONFIG_PRIORITY_PID);
    u64 low_priority_pid = get_config_u64(CONFIG_LOW_PRIORITY_PID);
    u64 priority;

    if (priority_pid != 0 && pid == (u32)priority_pid) {
        priority = get_config_u64(CONFIG_PRIORITY_PARAM);
    } else if (low_priority_pid != 0 && pid == (u32)low_priority_pid) {
        priority = get_config_u64(CONFIG_LOW_PRIORITY_PARAM);
    } else {
        priority = get_config_u64(CONFIG_DEFAULT_PARAM);
    }

    /* Convert priority to decay factor: priority / 10, minimum 1 */
    u64 decay_factor = priority / 10;
    if (decay_factor == 0)
        decay_factor = 1;

    return decay_factor;
}

/* Helper: Update PID stats for prefetch */
static __always_inline void update_prefetch_stats(u32 pid, bool allowed)
{
    struct pid_chunk_stats *stats, new_stats = {};

    stats = bpf_map_lookup_elem(&pid_stats, &pid);
    if (!stats) {
        new_stats.total_activate = 1;
        if (allowed)
            new_stats.policy_allow = 1;
        else
            new_stats.policy_deny = 1;
        bpf_map_update_elem(&pid_stats, &pid, &new_stats, BPF_ANY);
        return;
    }

    __sync_fetch_and_add(&stats->total_activate, 1);
    if (allowed)
        __sync_fetch_and_add(&stats->policy_allow, 1);
    else
        __sync_fetch_and_add(&stats->policy_deny, 1);
}

/* ============== PREFETCH HOOKS ============== */

/*
 * Hook: uvm_perf_prefetch_get_hint_va_block (via kprobe)
 *
 * Called BEFORE before_compute. Captures owner PID and stores in per-CPU cache.
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
    u32 *cached_pid = bpf_map_lookup_elem(&va_block_pid_cache, &key);
    if (!cached_pid)
        return 0;

    /* Use trace_helper.h to get owner PID */
    *cached_pid = get_owner_pid_from_va_block(va_block);
    return 0;
}

/* Helper: Get cached owner PID */
static __always_inline u32 get_cached_owner_pid(void)
{
    u32 key = 0;
    u32 *cached_pid = bpf_map_lookup_elem(&va_block_pid_cache, &key);
    return cached_pid ? *cached_pid : 0;
}

SEC("struct_ops/uvm_prefetch_before_compute")
int BPF_PROG(uvm_prefetch_before_compute,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *result_region)
{
    u32 owner_tgid = get_cached_owner_pid();
    u32 threshold = get_prefetch_threshold_for_pid(owner_tgid);

    bpf_printk("PID-Prefetch-Evict: pid=%u, page=%u, prefetch_thresh=%u%%\n",
               owner_tgid, page_index, threshold);

    /* Initialize result_region to empty */
    bpf_uvm_set_va_block_region(result_region, 0, 0);

    /* Return ENTER_LOOP to let driver iterate tree and call on_tree_iter */
    return 2; // UVM_BPF_ACTION_ENTER_LOOP
}

SEC("struct_ops/uvm_prefetch_on_tree_iter")
int BPF_PROG(uvm_prefetch_on_tree_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_va_block_region_t *prefetch_region)
{
    u32 owner_tgid = get_cached_owner_pid();
    u32 threshold = get_prefetch_threshold_for_pid(owner_tgid);

    /* Calculate subregion_pages from current_region */
    uvm_page_index_t first = BPF_CORE_READ(current_region, first);
    uvm_page_index_t outer = BPF_CORE_READ(current_region, outer);
    unsigned int subregion_pages = outer - first;

    /* Apply PID-based threshold: counter * 100 > subregion_pages * threshold
     *
     * Lower threshold -> easier to pass -> more prefetch -> more bandwidth
     * Higher threshold -> harder to pass -> less prefetch -> less bandwidth
     */
    bool allowed = (counter * 100 > subregion_pages * threshold);

    /* Update stats for this PID */
    if (owner_tgid > 0) {
        update_prefetch_stats(owner_tgid, allowed);
    }

    if (allowed) {
        bpf_uvm_set_va_block_region(prefetch_region, first, outer);
        return 1; // Selected this region
    }

    return 0; // Region doesn't meet threshold
}

/* ============== EVICTION HOOKS ============== */

SEC("struct_ops/uvm_pmm_chunk_activate")
int BPF_PROG(uvm_pmm_chunk_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    u32 owner_pid;
    struct pid_chunk_stats *stats;
    struct pid_chunk_stats new_stats = {0};
    u64 chunk_ptr = (u64)chunk;

    owner_pid = get_owner_pid_from_chunk(chunk);
    if (owner_pid == 0)
        return 0;

    /* Check if this chunk was already tracked (avoid double counting) */
    if (bpf_map_lookup_elem(&active_chunks, &chunk_ptr))
        return 0;

    /* Track this chunk as active */
    bpf_map_update_elem(&active_chunks, &chunk_ptr, &owner_pid, BPF_ANY);

    /* Update per-PID stats */
    stats = bpf_map_lookup_elem(&pid_stats, &owner_pid);
    if (stats) {
        __sync_fetch_and_add(&stats->current_count, 1);
        __sync_fetch_and_add(&stats->total_activate, 1);
    } else {
        new_stats.current_count = 1;
        new_stats.total_activate = 1;
        bpf_map_update_elem(&pid_stats, &owner_pid, &new_stats, BPF_ANY);
    }

    return 0;
}

SEC("struct_ops/uvm_pmm_chunk_used")
int BPF_PROG(uvm_pmm_chunk_used,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    u32 owner_pid;
    u64 decay_factor;
    struct pid_chunk_stats *stats;
    u64 chunk_ptr = (u64)chunk;
    u64 *access_count;
    u64 count;

    owner_pid = get_owner_pid_from_chunk(chunk);
    if (owner_pid == 0)
        return 0;

    /* Get per-PID stats */
    stats = bpf_map_lookup_elem(&pid_stats, &owner_pid);

    /* Update total_used for this PID */
    if (stats) {
        __sync_fetch_and_add(&stats->total_used, 1);
    }

    /* Get decay factor based on PID priority */
    decay_factor = get_eviction_decay_for_pid(owner_pid);

    /* Get and increment access count for this chunk */
    access_count = bpf_map_lookup_elem(&chunk_access_count, &chunk_ptr);
    if (access_count) {
        count = __sync_fetch_and_add(access_count, 1) + 1;
    } else {
        /* First access, initialize */
        u64 one = 1;
        bpf_map_update_elem(&chunk_access_count, &chunk_ptr, &one, BPF_ANY);
        count = 1;
    }

    /* Move tail only when access count reaches decay threshold */
    if (count % decay_factor == 0) {
        bpf_uvm_pmm_chunk_move_tail(chunk, list);
        if (stats) {
            __sync_fetch_and_add(&stats->policy_allow, 1);
        }
        bpf_printk("PID-Prefetch-Evict: pid=%u moved_tail (count=%llu, decay=%llu)\n",
                   owner_pid, count, decay_factor);
    } else {
        if (stats) {
            __sync_fetch_and_add(&stats->policy_deny, 1);
        }
    }

    return 1; /* BYPASS - don't let kernel do LRU move */
}

SEC("struct_ops/uvm_pmm_eviction_prepare")
int BPF_PROG(uvm_pmm_eviction_prepare,
             uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used,
             struct list_head *va_block_unused)
{
    struct list_head *first;
    uvm_gpu_chunk_t *chunk;
    u32 *tracked_pid;
    struct pid_chunk_stats *stats;
    u64 chunk_ptr;

    if (!va_block_used)
        return 0;

    /* Get the first entry in the va_block_used list (head of eviction) */
    first = BPF_CORE_READ(va_block_used, next);
    if (!first || first == va_block_used)
        return 0;

    /*
     * The list entry is embedded in uvm_gpu_chunk_t.list
     * uvm_gpu_root_chunk_t has chunk as first member (offset 0)
     * So: container_of(first, uvm_gpu_chunk_t, list)
     */
    chunk = (uvm_gpu_chunk_t *)((char *)first -
              __builtin_offsetof(struct uvm_gpu_chunk_struct, list));
    chunk_ptr = (u64)chunk;

    /* Only decrement if we tracked this chunk in activate */
    tracked_pid = bpf_map_lookup_elem(&active_chunks, &chunk_ptr);
    if (!tracked_pid)
        return 0;  /* Chunk was not tracked by us, don't decrement */

    /* Decrement current_count for the tracked PID */
    stats = bpf_map_lookup_elem(&pid_stats, tracked_pid);
    if (stats && stats->current_count > 0) {
        __sync_fetch_and_sub(&stats->current_count, 1);
    }

    /* Remove from tracking maps */
    bpf_map_delete_elem(&active_chunks, &chunk_ptr);
    bpf_map_delete_elem(&chunk_access_count, &chunk_ptr);

    return 0;
}

/* Dummy implementation for test trigger */
SEC("struct_ops/uvm_bpf_test_trigger_kfunc")
int BPF_PROG(uvm_bpf_test_trigger_kfunc, const char *buf, int len)
{
    return 0;
}

/* Define the struct_ops map - includes both prefetch and eviction hooks */
SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops_prefetch_eviction_pid = {
    .uvm_bpf_test_trigger_kfunc = (void *)uvm_bpf_test_trigger_kfunc,
    .uvm_prefetch_before_compute = (void *)uvm_prefetch_before_compute,
    .uvm_prefetch_on_tree_iter = (void *)uvm_prefetch_on_tree_iter,
    .uvm_pmm_chunk_activate = (void *)uvm_pmm_chunk_activate,
    .uvm_pmm_chunk_used = (void *)uvm_pmm_chunk_used,
    .uvm_pmm_eviction_prepare = (void *)uvm_pmm_eviction_prepare,
};
