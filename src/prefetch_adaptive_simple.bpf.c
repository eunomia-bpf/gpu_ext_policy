/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* Adaptive Threshold Prefetch Policy - Simple Version
 *
 * This version uses the prefetch_region pointer passed directly to on_tree_iter.
 * BPF can modify it via bpf_uvm_set_va_block_region() kfunc.
 */

/* BPF map: Track GPU memory usage for adaptive threshold */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} gpu_memory_usage SEC(".maps");

/* Helper: Calculate adaptive threshold based on memory usage */
static __always_inline unsigned int get_adaptive_threshold(void)
{
    u32 key = 0;
    u64 *mem_usage = bpf_map_lookup_elem(&gpu_memory_usage, &key);

    if (!mem_usage)
        return 51;  // Default 51%

    /* Adaptive logic:
     * - High memory usage (>90%): Conservative 75%
     * - Medium usage (50-90%): Default 51%
     * - Low usage (<50%): Aggressive 30%
     */
    if (*mem_usage > 90)
        return 75;
    else if (*mem_usage > 50)
        return 51;
    else
        return 30;
}

SEC("struct_ops/uvm_prefetch_before_compute")
int BPF_PROG(uvm_prefetch_before_compute,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *result_region)
{
    bpf_printk("Adaptive SIMPLE: page_index=%u, entering ENTER_LOOP mode\n", page_index);

    /* Initialize result_region to empty */
    bpf_uvm_set_va_block_region(result_region, 0, 0);

    /* Return ENTER_LOOP to let driver iterate tree and call on_tree_iter */
    return 2; // UVM_BPF_ACTION_ENTER_LOOP
}

SEC("struct_ops/uvm_prefetch_on_tree_iter")
int BPF_PROG(uvm_prefetch_on_tree_iter,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *current_region,
             unsigned int counter,
             uvm_va_block_region_t *prefetch_region)
{
    unsigned int threshold = get_adaptive_threshold();

    /* Calculate subregion_pages from current_region */
    uvm_page_index_t first = BPF_CORE_READ(current_region, first);
    uvm_page_index_t outer = BPF_CORE_READ(current_region, outer);
    unsigned int subregion_pages = outer - first;

    /* Apply adaptive threshold: counter * 100 > subregion_pages * threshold */
    if (counter * 100 > subregion_pages * threshold) {
        bpf_printk("Adaptive SIMPLE: counter=%u/%u (threshold=%u%%), selecting [%u,%u)\n",
                   counter, subregion_pages, threshold, first, outer);

        /* Update prefetch_region via kfunc - this is the key change!
         * Now prefetch_region is passed directly from driver, not from map
         */
        bpf_uvm_set_va_block_region(prefetch_region, first, outer);

        return 1; // Indicate we selected this region
    }

    return 0; // This region doesn't meet threshold
}

/* Dummy implementation for test trigger */
SEC("struct_ops/uvm_bpf_test_trigger_kfunc")
int BPF_PROG(uvm_bpf_test_trigger_kfunc, const char *buf, int len)
{
    return 0;
}

/* Define the struct_ops map */
SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops_adaptive_simple = {
    .uvm_bpf_test_trigger_kfunc = (void *)uvm_bpf_test_trigger_kfunc,
    .uvm_prefetch_before_compute = (void *)uvm_prefetch_before_compute,
    .uvm_prefetch_on_tree_iter = (void *)uvm_prefetch_on_tree_iter,
};
