#ifndef _BPF_TESTMOD_H
#define _BPF_TESTMOD_H

/* Note: This header assumes uvm_types.h is included first to provide type definitions */


/* UVM GPU extension struct_ops definition */
struct uvm_gpu_ext {
	int (*uvm_bpf_test_trigger_kfunc)(const char *, int);
	int (*uvm_prefetch_before_compute)(uvm_page_index_t, uvm_perf_prefetch_bitmap_tree_t *, uvm_va_block_region_t *, uvm_va_block_region_t *);
	int (*uvm_prefetch_on_tree_iter)(uvm_perf_prefetch_bitmap_tree_t *, uvm_va_block_region_t *, uvm_va_block_region_t *, unsigned int, uvm_va_block_region_t *);
	
	int (*uvm_pmm_chunk_activate)(uvm_pmm_gpu_t *, uvm_gpu_chunk_t *, struct list_head *);
	int (*uvm_pmm_chunk_used)(uvm_pmm_gpu_t *, uvm_gpu_chunk_t *, struct list_head *);
	int (*uvm_pmm_eviction_prepare)(uvm_pmm_gpu_t *, struct list_head *, struct list_head *);
};


/* BPF kfuncs */
#ifndef BPF_NO_KFUNC_PROTOTYPES
#ifndef __ksym
#define __ksym __attribute__((section(".ksyms")))
#endif
#ifndef __weak
#define __weak __attribute__((weak))
#endif

/* Prefetch kfuncs */
extern void bpf_uvm_set_va_block_region(uvm_va_block_region_t *region, uvm_page_index_t first, uvm_page_index_t outer) __weak __ksym;
extern int bpf_uvm_strstr(const char *str, unsigned int str__sz, const char *substr, unsigned int substr__sz) __weak __ksym;

/* PMM eviction policy kfuncs */
extern void bpf_uvm_pmm_chunk_move_head(uvm_gpu_chunk_t *chunk, struct list_head *list) __weak __ksym;
extern void bpf_uvm_pmm_chunk_move_tail(uvm_gpu_chunk_t *chunk, struct list_head *list) __weak __ksym;

#endif

#endif /* _BPF_TESTMOD_H */