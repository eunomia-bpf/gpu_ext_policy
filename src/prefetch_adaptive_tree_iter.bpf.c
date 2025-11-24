/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* Adaptive Threshold Prefetch Policy - PCIe Throughput Based
 *
 * Threshold is calculated in userspace based on GPU PCIe throughput.
 * Userspace monitors PCIe traffic and updates threshold_map every second.
 * BPF reads threshold and prints it.
 */

/* BPF map: Stores threshold computed by userspace (0-100) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);  // Threshold percentage (0-100)
} threshold_map SEC(".maps");

/* Helper: Get threshold from userspace */
static __always_inline unsigned int get_threshold(void)
{
    u32 key = 0;
    u32 *threshold = bpf_map_lookup_elem(&threshold_map, &key);

    if (!threshold)
        return 51;  // Default 51% if not set

    return *threshold;
}

SEC("struct_ops/uvm_prefetch_before_compute")
int BPF_PROG(uvm_prefetch_before_compute,
             uvm_page_index_t page_index,
             uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
             uvm_va_block_region_t *max_prefetch_region,
             uvm_va_block_region_t *result_region)
{
    u32 threshold = get_threshold();

    bpf_printk("Adaptive: page=%u, threshold=%u%%\n", page_index, threshold);

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
    unsigned int threshold = get_threshold();

    /* Calculate subregion_pages from current_region */
    uvm_page_index_t first = BPF_CORE_READ(current_region, first);
    uvm_page_index_t outer = BPF_CORE_READ(current_region, outer);
    unsigned int subregion_pages = outer - first;

    /* Apply adaptive threshold: counter * 100 > subregion_pages * threshold */
    if (counter * 100 > subregion_pages * threshold) {
        // bpf_printk("Adaptive: counter=%u/%u (threshold=%u%%), selecting [%u,%u)\n",
                //    counter, subregion_pages, threshold, first, outer);

        /* Update prefetch_region via kfunc */
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
struct uvm_gpu_ext uvm_ops_adaptive_tree_iter = {
    .uvm_bpf_test_trigger_kfunc = (void *)uvm_bpf_test_trigger_kfunc,
    .uvm_prefetch_before_compute = (void *)uvm_prefetch_before_compute,
    .uvm_prefetch_on_tree_iter = (void *)uvm_prefetch_on_tree_iter,
};
