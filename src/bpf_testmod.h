#ifndef _BPF_TESTMOD_H
#define _BPF_TESTMOD_H

#ifndef __KERNEL__
typedef unsigned int __u32;
typedef unsigned int u32;
#endif

/* Shared struct_ops definition between kernel module and BPF program */
struct uvm_gpu_ext {
	int (*uvm_bpf_test_trigger_kfunc)(const char *buf, int len);

	/* Prefetch hooks */
	int (*uvm_prefetch_before_compute)(
		u32 page_index,
		void *bitmap_tree,
		void *max_prefetch_region,
		void *result_region);

	int (*uvm_prefetch_on_tree_iter)(
		u32 page_index,
		void *bitmap_tree,
		void *max_prefetch_region,
		void *current_region,
		unsigned int counter,
		unsigned int subregion_pages);
};

#endif /* _BPF_TESTMOD_H */