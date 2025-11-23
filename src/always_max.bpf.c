/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "module/bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* Declare the external kfunc from nvidia-uvm module */
extern void bpf_uvm_set_va_block_region(void *region, u32 first, u32 outer) __ksym;

/* Always prefetch the maximum region policy
 * This is the simplest policy that always prefetches the entire max_prefetch_region
 */
SEC("struct_ops/uvm_prefetch_before_compute")
int BPF_PROG(uvm_prefetch_before_compute,
             u32 page_index,
             void *bitmap_tree,
             void *max_prefetch_region,
             void *result_region)
{
    bpf_printk("BPF always_max: page_index=%u\n", page_index);

    /* Read max_prefetch_region using bpf_probe_read_kernel */
    u32 max_first, max_outer;

    /* Read the first and outer fields from max_prefetch_region
     * Assuming uvm_va_block_region_t has two u32 fields: first and outer
     */
    if (bpf_probe_read_kernel(&max_first, sizeof(max_first), max_prefetch_region) != 0) {
        bpf_printk("Failed to read max_first\n");
        return 0; /* UVM_BPF_ACTION_DEFAULT */
    }

    if (bpf_probe_read_kernel(&max_outer, sizeof(max_outer),
                              max_prefetch_region + sizeof(u32)) != 0) {
        bpf_printk("Failed to read max_outer\n");
        return 0; /* UVM_BPF_ACTION_DEFAULT */
    }

    bpf_printk("BPF always_max: Setting prefetch region [%u, %u)\n",
               max_first, max_outer);

    /* Set result_region to the full max_prefetch_region */
    bpf_uvm_set_va_block_region(result_region, max_first, max_outer);

    /* Return BYPASS to skip default kernel computation */
    return 1; /* UVM_BPF_ACTION_BYPASS */
}

/* This hook is called on each tree iteration - not used in always_max policy */
SEC("struct_ops/uvm_prefetch_on_tree_iter")
int BPF_PROG(uvm_prefetch_on_tree_iter,
             u32 page_index,
             void *bitmap_tree,
             void *max_prefetch_region,
             void *current_region,
             unsigned int counter,
             unsigned int subregion_pages)
{
    /* Not used in always_max policy */
    return 0; /* UVM_BPF_ACTION_DEFAULT */
}

/* Dummy implementation for the old test trigger */
SEC("struct_ops/uvm_bpf_test_trigger_kfunc")
int BPF_PROG(uvm_bpf_test_trigger_kfunc, const char *buf, int len)
{
    return 0;
}

/* Define the struct_ops map */
SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops_always_max = {
    .uvm_bpf_test_trigger_kfunc = (void *)uvm_bpf_test_trigger_kfunc,
    .uvm_prefetch_before_compute = (void *)uvm_prefetch_before_compute,
    .uvm_prefetch_on_tree_iter = (void *)uvm_prefetch_on_tree_iter,
};
