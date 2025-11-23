#ifndef _BPF_TESTMOD_H
#define _BPF_TESTMOD_H

/* Note: This header assumes uvm_types.h is included first to provide type definitions */

/* Shared struct_ops definition between kernel module and BPF program */
struct uvm_gpu_ext {
	int (*uvm_bpf_test_trigger_kfunc)(const char *, int);
	int (*uvm_prefetch_before_compute)(uvm_page_index_t, uvm_perf_prefetch_bitmap_tree_t *, uvm_va_block_region_t *, uvm_va_block_region_t *);
	int (*uvm_prefetch_on_tree_iter)(uvm_page_index_t, uvm_perf_prefetch_bitmap_tree_t *, uvm_va_block_region_t *, uvm_va_block_region_t *, unsigned int, uvm_va_block_region_t *);
};

/* BPF kfuncs */
#ifndef BPF_NO_KFUNC_PROTOTYPES
#ifndef __ksym
#define __ksym __attribute__((section(".ksyms")))
#endif
#ifndef __weak
#define __weak __attribute__((weak))
#endif
extern void bpf_uvm_set_va_block_region(uvm_va_block_region_t *region, uvm_page_index_t first, uvm_page_index_t outer) __weak __ksym;
extern int bpf_uvm_strstr(const char *str, unsigned int str__sz, const char *substr, unsigned int substr__sz) __weak __ksym;
#endif

#endif /* _BPF_TESTMOD_H */