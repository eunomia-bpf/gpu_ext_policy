/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* FIFO eviction policy for GPU memory management
 *
 * In FIFO (First-In-First-Out), the chunk that has been in memory the longest
 * is evicted first. This is implemented by:
 * - Moving newly populated chunks to the head of the list (highest priority for eviction)
 * - Keeping the default LRU behavior for activate
 */

SEC("struct_ops/uvm_pmm_chunk_activate")
int BPF_PROG(uvm_pmm_chunk_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    /* For FIFO, we use default behavior when a chunk becomes evictable */
    bpf_printk("BPF FIFO: chunk_activate (using default behavior)\n");
    return 0;
}

SEC("struct_ops/uvm_pmm_chunk_used")
int BPF_PROG(uvm_pmm_chunk_used,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    /* FIFO policy: skip this step
     */
    /* Return BYPASS to skip default kernel computation */
    return 1; /* UVM_BPF_ACTION_BYPASS */
}

SEC("struct_ops/uvm_pmm_eviction_prepare")
int BPF_PROG(uvm_pmm_eviction_prepare,
             uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used,
             struct list_head *va_block_unused)
{
    /* FIFO doesn't need special preparation before eviction
     * The list is already ordered by populate time due to move_head in populate
     */
    bpf_printk("BPF FIFO: eviction_prepare (no reordering needed)\n");
    return 0;
}

/* Define the struct_ops map */
SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops_fifo = {
    .uvm_bpf_test_trigger_kfunc = (void *)NULL,
    .uvm_prefetch_before_compute = (void *)NULL,
    .uvm_prefetch_on_tree_iter = (void *)NULL,
    .uvm_pmm_chunk_activate = (void *)uvm_pmm_chunk_activate,
    .uvm_pmm_chunk_used = (void *)uvm_pmm_chunk_used,
    .uvm_pmm_eviction_prepare = (void *)uvm_pmm_eviction_prepare,
};
