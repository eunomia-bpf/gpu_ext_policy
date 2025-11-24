#ifndef __UVM_TYPES_H__
#define __UVM_TYPES_H__

/* Extract only UVM-specific types from nvidia-uvm.ko BTF */

typedef short unsigned int NvU16;
typedef NvU16 uvm_page_index_t;

typedef struct {
	uvm_page_index_t first;
	uvm_page_index_t outer;
} uvm_va_block_region_t;

typedef struct {
	long unsigned int bitmap[8];
} uvm_page_mask_t;

typedef struct uvm_perf_prefetch_bitmap_tree {
	uvm_page_mask_t pages;
	uvm_page_index_t offset;
	NvU16 leaf_count;
	unsigned char level_count;
} uvm_perf_prefetch_bitmap_tree_t;

/* PMM (Physical Memory Manager) types for eviction policy */

/* Forward declarations - opaque types for BPF */
typedef struct uvm_pmm_gpu_struct uvm_pmm_gpu_t;

/* uvm_va_block_struct - minimal definition for accessing start/end addresses */
struct uvm_va_block_struct {
	/* We don't know the exact internal layout, but based on empirical testing:
	 * - start address is at offset ~24
	 * - end address is at offset ~32
	 * Using CO-RE, BPF will relocate these offsets automatically */
	char _padding[24];
	unsigned long long start;  // VA block start address
	unsigned long long end;    // VA block end address
};

typedef struct uvm_va_block_struct uvm_va_block_t;

/* Full definition of uvm_gpu_chunk_struct - needed to access chunk->list field */
struct uvm_gpu_chunk_struct {
	unsigned long long address;
	/* We only need the list field for BPF, but include minimal structure */
	struct {
		unsigned int type : 2;
		unsigned int in_eviction : 1;
		unsigned int inject_split_error : 1;
		unsigned int is_zero : 1;
		unsigned int is_referenced : 1;
		unsigned int state : 3;
		unsigned int log2_size : 6;
		unsigned short va_block_page_index : 10;
		unsigned int gpu_index : 7;
	};
	struct list_head list;  /* This is what we need to access */
	uvm_va_block_t *va_block;  /* VA block using this chunk */
	void *parent;
	void *suballoc;
};

typedef struct uvm_gpu_chunk_struct uvm_gpu_chunk_t;

/* Note: list_head is already defined in vmlinux.h, no need to redefine it */

#endif /* __UVM_TYPES_H__ */
