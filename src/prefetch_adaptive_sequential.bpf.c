/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* Adaptive Sequential Prefetch Policy
 *
 * Instead of iterating the bitmap tree, this policy directly takes a portion
 * of max_prefetch_region based on a configurable percentage (0-100).
 *
 * Userspace can dynamically adjust the prefetch percentage based on:
 * - PCIe bandwidth utilization
 * - GPU memory pressure
 * - Workload characteristics
 *
 * percentage = 100: equivalent to always_max (prefetch entire region)
 * percentage = 0:   equivalent to prefetch_none (no prefetch)
 * percentage = 50:  prefetch half of the max region
 */

/* BPF map: Stores prefetch percentage (0-100) set by userspace */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);  // Prefetch percentage (0-100)
} prefetch_pct_map SEC(".maps");

/* Helper: Get prefetch percentage from userspace */
static __always_inline unsigned int get_prefetch_percentage(void)
{
    u32 key = 0;
    u32 *pct = bpf_map_lookup_elem(&prefetch_pct_map, &key);

    if (!pct)
        return 100;  // Default to 100% (always_max behavior) if not set

    return *pct;
}

SEC("struct_ops/uvm_prefetch_before_compute")
int BPF_PROG(uvm_prefetch_before_compute,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *result_region)
{
    u32 pct = get_prefetch_percentage();

    /* Read max_prefetch_region bounds */
    uvm_page_index_t max_first = BPF_CORE_READ(max_prefetch_region, first);
    uvm_page_index_t max_outer = BPF_CORE_READ(max_prefetch_region, outer);
    unsigned int total_pages = max_outer - max_first;

    /* Handle edge cases */
    if (pct == 0 || total_pages == 0) {
        /* No prefetch */
        bpf_uvm_set_va_block_region(result_region, 0, 0);
        return 1; /* UVM_BPF_ACTION_BYPASS */
    }

    if (pct >= 100) {
        /* Full prefetch - equivalent to always_max */
        bpf_uvm_set_va_block_region(result_region, max_first, max_outer);
        return 1; /* UVM_BPF_ACTION_BYPASS */
    }

    /* Calculate how many pages to prefetch based on percentage */
    unsigned int prefetch_pages = (total_pages * pct) / 100;
    if (prefetch_pages == 0)
        prefetch_pages = 1;  // At least 1 page if pct > 0

    /* Prefetch from the beginning of the region (sequential prefetch)
     * This assumes sequential access pattern from the fault page forward.
     * For different access patterns, other strategies could be implemented:
     * - Center around page_index
     * - Prefetch from page_index forward
     * - etc.
     */
    uvm_page_index_t new_outer = max_first + prefetch_pages;
    if (new_outer > max_outer)
        new_outer = max_outer;

    bpf_printk("adaptive_seq: page=%u, pct=%u%%, region=[%u,%u) of [%u,%u)\n",
               page_index, pct, max_first, new_outer, max_first, max_outer);

    bpf_uvm_set_va_block_region(result_region, max_first, new_outer);

    return 1; /* UVM_BPF_ACTION_BYPASS */
}

/* This hook is not used - we bypass tree iteration */
SEC("struct_ops/uvm_prefetch_on_tree_iter")
int BPF_PROG(uvm_prefetch_on_tree_iter,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_va_block_region_t *prefetch_region)
{
    /* Not used - we return BYPASS in before_compute */
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
struct uvm_gpu_ext uvm_ops_adaptive_sequential = {
    .uvm_bpf_test_trigger_kfunc = (void *)uvm_bpf_test_trigger_kfunc,
    .uvm_prefetch_before_compute = (void *)uvm_prefetch_before_compute,
    .uvm_prefetch_on_tree_iter = (void *)uvm_prefetch_on_tree_iter,
};
