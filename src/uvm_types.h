#ifndef __UVM_TYPES_H__
#define __UVM_TYPES_H__


#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = record)
#endif

#ifndef __ksym
#define __ksym __attribute__((section(".ksyms")))
#endif

#ifndef __weak
#define __weak __attribute__((weak))
#endif

#ifndef __bpf_fastcall
#if __has_attribute(bpf_fastcall)
#define __bpf_fastcall __attribute__((bpf_fastcall))
#else
#define __bpf_fastcall
#endif
#endif

/* Extract only UVM-specific types from nvidia-uvm.ko BTF */

typedef unsigned char NvU8;
typedef short unsigned int NvU16;
typedef unsigned int NvU32;
typedef unsigned long long NvU64;
typedef NvU16 uvm_page_index_t;

/*
 * uvm_va_block_region_t - VA block region descriptor
 * Used to specify a range of pages within a VA block
 */
typedef struct {
	uvm_page_index_t first;   /* First page index (inclusive) */
	uvm_page_index_t outer;   /* Last page index + 1 (exclusive) */
} uvm_va_block_region_t;

/*
 * uvm_page_mask_t - Bitmask for 512 pages (2MB VA block)
 */
typedef struct {
	unsigned long bitmap[8];  /* 8 * 64 = 512 bits */
} uvm_page_mask_t;

/*
 * uvm_perf_prefetch_bitmap_tree_t - Prefetch decision tree
 * Used to track page access patterns for prefetch decisions
 */
typedef struct uvm_perf_prefetch_bitmap_tree {
	uvm_page_mask_t pages;     /* Bitmap of accessed pages */
	uvm_page_index_t offset;   /* Offset within VA block */
	NvU16 leaf_count;          /* Number of leaf nodes */
	NvU8 level_count;          /* Tree depth */
} uvm_perf_prefetch_bitmap_tree_t;

/*
 * uvm_perf_prefetch_bitmap_tree_iter_t - Tree iterator
 */
typedef struct {
	signed char level_idx;
	uvm_page_index_t node_idx;
} uvm_perf_prefetch_bitmap_tree_iter_t;

/* PMM (Physical Memory Manager) types for eviction policy */

/* Forward declarations - opaque types for BPF */
typedef struct uvm_pmm_gpu_struct uvm_pmm_gpu_t;

/* Forward declarations for va_space chain */
struct mm_struct;
struct task_struct;

/* uvm_va_space_mm_struct - contains mm pointer */
struct uvm_va_space_mm_struct {
	struct mm_struct *mm;
	// ... other fields not needed
};
typedef struct uvm_va_space_mm_struct uvm_va_space_mm_t;

struct uvm_va_space_struct {
	char _padding[0x6840];  // Offset to va_space_mm (approximate, CO-RE will fix)
	uvm_va_space_mm_t va_space_mm;
};
typedef struct uvm_va_space_struct uvm_va_space_t;

/* uvm_va_range_struct - to access va_space */
struct uvm_va_range_struct {
	uvm_va_space_t *va_space;
	// ... other fields not needed
};
typedef struct uvm_va_range_struct uvm_va_range_t;

/* uvm_va_range_managed_struct - contains va_range as first member */
struct uvm_va_range_managed_struct {
	struct uvm_va_range_struct va_range;
	// ... other fields not needed
};
typedef struct uvm_va_range_managed_struct uvm_va_range_managed_t;

/* Nested structures for va_block->cpu.fault_authorized */
struct uvm_va_block_fault_authorized {
	unsigned long long first_fault_stamp;
	int first_pid;   // pid_t is int
	unsigned short page_index;
};

struct uvm_va_block_cpu_state {
	void *node_state;
	unsigned long allocated_bitmap[8];
	unsigned long resident_bitmap[8];
	unsigned long pte_bits[2][8];
	unsigned char ever_mapped;
	struct uvm_va_block_fault_authorized fault_authorized;
};

/* uvm_va_block_struct - definition for accessing start/end addresses, owner PID, and va_space */
struct uvm_va_block_struct {
	/* Using CO-RE, BPF will relocate these offsets automatically */
	char _kref[16];           // nv_kref_t
	char _lock[24];           // uvm_mutex_t (approximate)
	uvm_va_range_managed_t *managed_range;  // pointer to managed_range
	unsigned long long start;  // VA block start address
	unsigned long long end;    // VA block end address
	char _masks[192];          // processor masks (approximate)
	struct uvm_va_block_cpu_state cpu;  // CPU state including fault_authorized
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

#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute pop
#endif

#endif /* __UVM_TYPES_H__ */
